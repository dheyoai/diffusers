#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
import warnings
from typing import Tuple

import torch
import torch.nn.functional as F

# This module provides QWEN-2512 specific wrappers around the generic kernex
# CUDA kernels exposed via `kernel_lib`. It mirrors the Z-Image `kernel_ops`
# patterns but keeps QWEN-specific naming so it can evolve independently.

KERNELS_AVAILABLE = False
kernel_lib = None

_USED_KERNELS: set[str] = set()

# Set QWEN_KERNEL_DEBUG=1 to add cuda synchronize + print after each kernel call
# to diagnose which kernel hangs.
_KERNEL_DEBUG = os.environ.get("QWEN_KERNEL_DEBUG", "0") == "1"
_kernel_call_count = 0


if _KERNEL_DEBUG:
    def _debug_sync(kernel_name: str) -> None:
        """Synchronize CUDA and print which kernel just ran."""
        global _kernel_call_count
        _kernel_call_count += 1
        torch.cuda.synchronize()
        print(f"[QWEN_KERNEL_DEBUG] #{_kernel_call_count} {kernel_name} completed OK", flush=True)
else:
    def _debug_sync(kernel_name: str) -> None:  # noqa: ARG001
        """No-op when QWEN_KERNEL_DEBUG is off — avoids ~4,320 function calls per image."""
        pass


def _record_kernel(name: str) -> None:
    # Fast path: skip set.add when the kernel is already recorded.
    if name not in _USED_KERNELS:
        _USED_KERNELS.add(name)


def get_used_kernels() -> list[str]:
    return sorted(_USED_KERNELS)


def clear_used_kernels() -> None:
    """Reset the used-kernels set. Call between benchmark configurations."""
    _USED_KERNELS.clear()


def _get_cuda_stream() -> int:
    """Get the raw CUDA stream handle for PyTorch's current stream.

    This ensures custom kernels launch on the same CUDA stream as PyTorch
    operations, avoiding implicit synchronization from the legacy default
    stream (stream 0) which would serialize all GPU work.
    """
    return torch.cuda.current_stream().cuda_stream


# Try multiple paths to find kernel_lib built from QWEN-2512/kernex
_kernex_paths = [
    # Path relative to this file into QWEN-2512/kernex/bazel-bin/bindings
    os.path.join(os.path.dirname(__file__), "../../../../../kernex/bazel-bin/bindings"),
    # Environment variable override
    os.environ.get("KERNEX_BINDINGS_PATH", ""),
]

for bindings_path in _kernex_paths:
    if bindings_path and os.path.exists(bindings_path):
        if bindings_path not in sys.path:
            sys.path.insert(0, bindings_path)
        try:
            import kernel_lib as _kernel_lib  # type: ignore[import]  # noqa: F401

            kernel_lib = _kernel_lib
            KERNELS_AVAILABLE = True
            break
        except ImportError:
            continue

if not KERNELS_AVAILABLE:
    warnings.warn(
        "QWEN kernel_ops_qwen: custom CUDA kernels not available. Falling back to PyTorch ops. "
        "To enable kernels, build QWEN-2512/kernex with Bazel and ensure its bindings "
        "directory is in the Python path."
    )

# ============================================================================
# Output buffer cache for eager mode
#
# In eager mode (no torch.compile), we bypass the custom_op and call kernel_lib
# directly, reusing pre-allocated output tensors.  This eliminates ~5,400
# torch.empty_like allocations per image (10 per block × 540 block passes).
# Under torch.compile, we use the custom_op path instead — CUDA graphs already
# handle allocation reuse, so the buffer cache is not needed.
# ============================================================================

_output_buffers: dict = {}
_mlp_intermediate_buffer: dict = {}
_row_stats_buffer: dict = {}


def _get_output_buf(key: tuple, shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return a cached output buffer, allocating only on shape/device change."""
    buf = _output_buffers.get(key)
    if buf is not None and buf.shape == shape and buf.device == device:
        return buf
    buf = torch.empty(shape, dtype=dtype, device=device)
    _output_buffers[key] = buf
    return buf


def _get_row_stats_buf(M: int, device: torch.device) -> torch.Tensor:
    """Return a cached scratch buffer for per-row LayerNorm statistics (mean + inv_std)."""
    key = (M, device)
    buf = _row_stats_buffer.get(key)
    if buf is not None and buf.shape[0] >= M * 2:
        return buf
    buf = torch.empty(M * 2, dtype=torch.float32, device=device)
    _row_stats_buffer[key] = buf
    return buf


def clear_kernel_buffer_caches() -> None:
    """Free all kernel output buffer caches to reclaim VRAM.

    Call this between pipeline loads or when switching resolutions
    to avoid stale buffers occupying VRAM.
    """
    _output_buffers.clear()
    _mlp_intermediate_buffer.clear()
    _row_stats_buffer.clear()


# ============================================================================
# torch.library custom op registration for torch.compile compatibility
#
# Raw pybind11 calls are opaque to torch.compile and cause graph breaks at
# every kernel invocation (~2,160 per inference).  Registering kernels as
# custom ops lets torch.compile include them in the compiled graph and
# capture them in CUDA graphs (reduce-overhead mode).
# ============================================================================

if KERNELS_AVAILABLE and kernel_lib is not None:

    @torch.library.custom_op("qwen::layernorm_modulate_bf16", mutates_args=())
    def _layernorm_modulate_bf16_op(
        x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float
    ) -> torch.Tensor:
        B, S, _D = x.shape
        out = torch.empty_like(x)
        kernel_lib.layernorm_modulate_bf16_device_cu(
            int(x.data_ptr()), int(scale.data_ptr()), int(shift.data_ptr()),
            int(out.data_ptr()), B, S, eps, _get_cuda_stream(),
        )
        return out

    @_layernorm_modulate_bf16_op.register_fake
    def _(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float) -> torch.Tensor:
        return torch.empty_like(x)

    @torch.library.custom_op("qwen::qk_norm_perhead_bf16", mutates_args=())
    def _qk_norm_perhead_bf16_op(
        q: torch.Tensor, k: torch.Tensor,
        q_weight: torch.Tensor, k_weight: torch.Tensor,
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, H, head_dim = q.shape
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
        kernel_lib.qk_norm_perhead_bf16_device_cu(
            int(q.data_ptr()), int(k.data_ptr()),
            int(q_weight.data_ptr()), int(k_weight.data_ptr()),
            int(q_out.data_ptr()), int(k_out.data_ptr()),
            B, S, H, head_dim, eps, _get_cuda_stream(),
        )
        return q_out, k_out

    @_qk_norm_perhead_bf16_op.register_fake
    def _(
        q: torch.Tensor, k: torch.Tensor,
        q_weight: torch.Tensor, k_weight: torch.Tensor,
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.empty_like(q), torch.empty_like(k)

    @torch.library.custom_op("qwen::rope_3d_fused_bf16", mutates_args=())
    def _rope_3d_fused_bf16_op(
        x: torch.Tensor,
        grid_frame: int, grid_height: int, grid_width: int,
        theta: float,
        axes_dim_0: int, axes_dim_1: int, axes_dim_2: int,
        height_offset: int, width_offset: int,
    ) -> torch.Tensor:
        B, S_img, H, D = x.shape
        out = torch.empty_like(x)
        kernel_lib.rope_3d_optimized_bf16_device_cu(
            int(x.data_ptr()), int(out.data_ptr()),
            B, S_img, H, D,
            grid_frame, grid_height, grid_width,
            axes_dim_0, axes_dim_1, axes_dim_2,
            float(theta), height_offset, width_offset,
            _get_cuda_stream(),
        )
        return out

    @_rope_3d_fused_bf16_op.register_fake
    def _(
        x: torch.Tensor,
        grid_frame: int, grid_height: int, grid_width: int,
        theta: float,
        axes_dim_0: int, axes_dim_1: int, axes_dim_2: int,
        height_offset: int, width_offset: int,
    ) -> torch.Tensor:
        return torch.empty_like(x)

    @torch.library.custom_op("qwen::qk_norm_rope_3d_fused_bf16", mutates_args=())
    def _qk_norm_rope_3d_fused_bf16_op(
        q: torch.Tensor, k: torch.Tensor,
        q_weight: torch.Tensor, k_weight: torch.Tensor,
        grid_frame: int, grid_height: int, grid_width: int,
        theta: float, eps: float,
        axes_dim_0: int, axes_dim_1: int, axes_dim_2: int,
        height_offset: int, width_offset: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S_img, H, D = q.shape
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
        kernel_lib.qk_norm_rope_3d_fused_bf16_device_cu(
            int(q.data_ptr()), int(k.data_ptr()),
            int(q_weight.data_ptr()), int(k_weight.data_ptr()),
            int(q_out.data_ptr()), int(k_out.data_ptr()),
            B, S_img, H, D,
            grid_frame, grid_height, grid_width,
            axes_dim_0, axes_dim_1, axes_dim_2,
            float(theta), float(eps),
            height_offset, width_offset,
            _get_cuda_stream(),
        )
        return q_out, k_out

    @_qk_norm_rope_3d_fused_bf16_op.register_fake
    def _(
        q: torch.Tensor, k: torch.Tensor,
        q_weight: torch.Tensor, k_weight: torch.Tensor,
        grid_frame: int, grid_height: int, grid_width: int,
        theta: float, eps: float,
        axes_dim_0: int, axes_dim_1: int, axes_dim_2: int,
        height_offset: int, width_offset: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.empty_like(q), torch.empty_like(k)

    @torch.library.custom_op("qwen::layernorm_modulate_gemm_up_bf16", mutates_args=())
    def _layernorm_modulate_gemm_up_bf16_op(
        x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor,
        weight: torch.Tensor, bias: torch.Tensor, eps: float,
    ) -> torch.Tensor:
        B, S, D = x.shape
        N = weight.shape[0]
        M = B * S
        out = torch.empty(B, S, N, dtype=x.dtype, device=x.device)
        stats = torch.empty(M * 2, dtype=torch.float32, device=x.device)
        kernel_lib.layernorm_modulate_gemm_up_bf16_device_cu(
            int(x.data_ptr()), int(scale.data_ptr()), int(shift.data_ptr()),
            int(weight.data_ptr()), int(bias.data_ptr()),
            int(out.data_ptr()), int(stats.data_ptr()),
            B, S, D, N, eps, _get_cuda_stream(),
        )
        return out

    @_layernorm_modulate_gemm_up_bf16_op.register_fake
    def _(
        x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor,
        weight: torch.Tensor, bias: torch.Tensor, eps: float,
    ) -> torch.Tensor:
        B, S, D = x.shape
        N = weight.shape[0]
        return torch.empty(B, S, N, dtype=x.dtype, device=x.device)

    # ========================================================================
    # Decompositions for torch.compile / Inductor
    #
    # When torch.compile encounters these custom_ops during graph lowering,
    # it uses the decompositions below instead of the opaque CUDA kernel.
    # This lets Inductor generate fused Triton kernels and merge them with
    # surrounding operations — eliminating the fusion barrier overhead.
    #
    # In eager mode, the CUDA kernels still run (dispatch priority: CUDA > decomp).
    # ========================================================================

    try:
        from torch._decomp import register_decomposition as _register_decomp

        @_register_decomp(torch.ops.qwen.layernorm_modulate_bf16.default)
        def _lnm_decomp(
            x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float
        ) -> torch.Tensor:
            normed = F.layer_norm(x, [x.shape[-1]], eps=eps)
            return normed * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        @_register_decomp(torch.ops.qwen.qk_norm_perhead_bf16.default)
        def _qk_norm_decomp(
            q: torch.Tensor, k: torch.Tensor,
            q_weight: torch.Tensor, k_weight: torch.Tensor,
            eps: float,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            B, S, H, D = q.shape
            q_flat = q.reshape(-1, D)
            k_flat = k.reshape(-1, D)
            q_norm = q_flat * torch.rsqrt(q_flat.pow(2).mean(-1, keepdim=True) + eps) * q_weight
            k_norm = k_flat * torch.rsqrt(k_flat.pow(2).mean(-1, keepdim=True) + eps) * k_weight
            return q_norm.reshape(B, S, H, D), k_norm.reshape(B, S, H, D)

        @_register_decomp(torch.ops.qwen.rope_3d_fused_bf16.default)
        def _rope_3d_decomp(
            x: torch.Tensor,
            grid_frame: int, grid_height: int, grid_width: int,
            theta: float,
            axes_dim_0: int, axes_dim_1: int, axes_dim_2: int,
            height_offset: int, width_offset: int,
        ) -> torch.Tensor:
            B, S_img, H, D = x.shape
            device = x.device
            # Build position grids [frame, height, width] → flatten to [S_img]
            f_pos = torch.arange(grid_frame, device=device, dtype=torch.float32)
            h_pos = torch.arange(grid_height, device=device, dtype=torch.float32) - height_offset
            w_pos = torch.arange(grid_width, device=device, dtype=torch.float32) - width_offset
            gf, gh, gw = torch.meshgrid(f_pos, h_pos, w_pos, indexing="ij")
            pos_f = gf.reshape(-1)  # [S_img]
            pos_h = gh.reshape(-1)
            pos_w = gw.reshape(-1)
            # Compute frequencies per axis: freq_j = 1 / theta^(2j/dim)
            freq_t = 1.0 / torch.pow(theta, torch.arange(0, axes_dim_0, 2, device=device, dtype=torch.float32) / axes_dim_0)
            freq_h = 1.0 / torch.pow(theta, torch.arange(0, axes_dim_1, 2, device=device, dtype=torch.float32) / axes_dim_1)
            freq_w = 1.0 / torch.pow(theta, torch.arange(0, axes_dim_2, 2, device=device, dtype=torch.float32) / axes_dim_2)
            # Angles: position × frequency → [S_img, dim/2]
            angles = torch.cat([
                pos_f.unsqueeze(-1) * freq_t.unsqueeze(0),
                pos_h.unsqueeze(-1) * freq_h.unsqueeze(0),
                pos_w.unsqueeze(-1) * freq_w.unsqueeze(0),
            ], dim=-1)  # [S_img, D/2]
            cos_a = torch.cos(angles).unsqueeze(0).unsqueeze(2).to(x.dtype)  # [1, S_img, 1, D/2]
            sin_a = torch.sin(angles).unsqueeze(0).unsqueeze(2).to(x.dtype)
            # Apply complex rotation: (re + im·i) × e^(iθ)
            x_pairs = x.reshape(B, S_img, H, D // 2, 2)
            x_re, x_im = x_pairs[..., 0], x_pairs[..., 1]
            out_re = x_re * cos_a - x_im * sin_a
            out_im = x_re * sin_a + x_im * cos_a
            return torch.stack([out_re, out_im], dim=-1).reshape(B, S_img, H, D)

        @_register_decomp(torch.ops.qwen.qk_norm_rope_3d_fused_bf16.default)
        def _qk_norm_rope_3d_decomp(
            q: torch.Tensor, k: torch.Tensor,
            q_weight: torch.Tensor, k_weight: torch.Tensor,
            grid_frame: int, grid_height: int, grid_width: int,
            theta: float, eps: float,
            axes_dim_0: int, axes_dim_1: int, axes_dim_2: int,
            height_offset: int, width_offset: int,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            B, S_img, H, D = q.shape
            device = q.device
            # RMSNorm
            q_flat = q.reshape(-1, D)
            k_flat = k.reshape(-1, D)
            q_norm = q_flat * torch.rsqrt(q_flat.pow(2).mean(-1, keepdim=True) + eps) * q_weight
            k_norm = k_flat * torch.rsqrt(k_flat.pow(2).mean(-1, keepdim=True) + eps) * k_weight
            q_norm = q_norm.reshape(B, S_img, H, D)
            k_norm = k_norm.reshape(B, S_img, H, D)
            # 3D Axial RoPE
            f_pos = torch.arange(grid_frame, device=device, dtype=torch.float32)
            h_pos = torch.arange(grid_height, device=device, dtype=torch.float32) - height_offset
            w_pos = torch.arange(grid_width, device=device, dtype=torch.float32) - width_offset
            gf, gh, gw = torch.meshgrid(f_pos, h_pos, w_pos, indexing="ij")
            pos_f, pos_h, pos_w = gf.reshape(-1), gh.reshape(-1), gw.reshape(-1)
            freq_t = 1.0 / torch.pow(theta, torch.arange(0, axes_dim_0, 2, device=device, dtype=torch.float32) / axes_dim_0)
            freq_h = 1.0 / torch.pow(theta, torch.arange(0, axes_dim_1, 2, device=device, dtype=torch.float32) / axes_dim_1)
            freq_w = 1.0 / torch.pow(theta, torch.arange(0, axes_dim_2, 2, device=device, dtype=torch.float32) / axes_dim_2)
            angles = torch.cat([
                pos_f.unsqueeze(-1) * freq_t.unsqueeze(0),
                pos_h.unsqueeze(-1) * freq_h.unsqueeze(0),
                pos_w.unsqueeze(-1) * freq_w.unsqueeze(0),
            ], dim=-1)
            cos_a = torch.cos(angles).unsqueeze(0).unsqueeze(2).to(q.dtype)
            sin_a = torch.sin(angles).unsqueeze(0).unsqueeze(2).to(q.dtype)
            # Apply rotation to both Q and K
            def _apply_rope(x: torch.Tensor) -> torch.Tensor:
                x_pairs = x.reshape(B, S_img, H, D // 2, 2)
                x_re, x_im = x_pairs[..., 0], x_pairs[..., 1]
                out_re = x_re * cos_a - x_im * sin_a
                out_im = x_re * sin_a + x_im * cos_a
                return torch.stack([out_re, out_im], dim=-1).reshape(B, S_img, H, D)
            return _apply_rope(q_norm), _apply_rope(k_norm)

        @_register_decomp(torch.ops.qwen.layernorm_modulate_gemm_up_bf16.default)
        def _lnm_gemm_up_decomp(
            x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor,
            weight: torch.Tensor, bias: torch.Tensor, eps: float,
        ) -> torch.Tensor:
            normed = F.layer_norm(x, [x.shape[-1]], eps=eps)
            modulated = normed * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
            return F.linear(modulated, weight, bias)

    except Exception as e:
        warnings.warn(
            f"QWEN kernel_ops_qwen: decomposition registration failed: {e}. "
            "torch.compile will use opaque custom_ops (fusion barriers)."
        )

else:
    _layernorm_modulate_bf16_op = None
    _qk_norm_perhead_bf16_op = None
    _rope_3d_fused_bf16_op = None
    _qk_norm_rope_3d_fused_bf16_op = None
    _layernorm_modulate_gemm_up_bf16_op = None


def check_kernels_available() -> bool:
    """Check if custom CUDA kernels for QWEN are available."""
    return KERNELS_AVAILABLE


def qwen_rmsnorm_scale_kernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    QWEN wrapper for RMSNorm + Scale fused kernel.

    Args:
        x: [B, S, D]
        weight: [D]
        scale: [B, D] (will be broadcast to [B, 1, D])
    """
    if KERNELS_AVAILABLE and kernel_lib is not None:
        try:
            B, S, D = x.shape
            orig_dtype = x.dtype

            if x.dtype == torch.bfloat16 and x.is_contiguous():
                x_bf16 = x
            else:
                x_bf16 = x.to(dtype=torch.bfloat16).contiguous()

            if weight.dtype == torch.bfloat16 and weight.is_contiguous():
                weight_bf16 = weight
            else:
                weight_bf16 = weight.to(dtype=torch.bfloat16).contiguous()

            if scale.dtype == torch.bfloat16 and scale.is_contiguous():
                scale_bf16 = scale
            else:
                scale_bf16 = scale.to(dtype=torch.bfloat16).contiguous()

            out_bf16 = _get_output_buf(("rmsnorm_scale", x_bf16.shape), x_bf16.shape, x_bf16.dtype, x_bf16.device)
            kernel_lib.rmsnorm_scale_bf16_device_cu(  # type: ignore[attr-defined]
                int(x_bf16.data_ptr()),
                int(weight_bf16.data_ptr()),
                int(scale_bf16.data_ptr()),
                int(out_bf16.data_ptr()),
                B,
                S,
                D,
                eps,
                _get_cuda_stream(),
            )
            _record_kernel("qwen_rmsnorm_scale_bf16")
            _debug_sync("qwen_rmsnorm_scale_bf16")

            if orig_dtype == torch.bfloat16:
                return out_bf16
            return out_bf16.to(dtype=orig_dtype)
        except Exception as e:
            warnings.warn(f"QWEN rmsnorm+scale kernel failed, falling back to PyTorch: {e}")

    # PyTorch fallback: plain RMSNorm + scale
    variance = x.pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps) * weight
    return x_norm * scale.unsqueeze(1)


def qwen_qk_norm_perhead_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    QWEN wrapper for per-head RMSNorm over Q and K.

    Args:
        q, k: [B, S, H, head_dim]  — expected bf16 contiguous
        q_weight, k_weight: [head_dim]
    """
    if _qk_norm_perhead_bf16_op is not None:
        if torch.compiler.is_compiling():
            # Under torch.compile: use custom_op for graph compatibility.
            return _qk_norm_perhead_bf16_op(q, k, q_weight, k_weight, eps)
        # Eager: direct kernel_lib call with pre-allocated output buffers.
        B, S, H, head_dim = q.shape
        q_out = _get_output_buf(("qk_q", q.shape), q.shape, q.dtype, q.device)
        k_out = _get_output_buf(("qk_k", k.shape), k.shape, k.dtype, k.device)
        kernel_lib.qk_norm_perhead_bf16_device_cu(
            int(q.data_ptr()), int(k.data_ptr()),
            int(q_weight.data_ptr()), int(k_weight.data_ptr()),
            int(q_out.data_ptr()), int(k_out.data_ptr()),
            B, S, H, head_dim, eps, _get_cuda_stream(),
        )
        _record_kernel("qwen_qk_norm_perhead_bf16")
        _debug_sync("qwen_qk_norm_perhead_bf16")
        return q_out, k_out

    # PyTorch fallback
    B, S, H, head_dim = q.shape
    q_flat = q.reshape(-1, head_dim)
    k_flat = k.reshape(-1, head_dim)

    q_var = q_flat.pow(2).mean(-1, keepdim=True)
    q_norm = q_flat * torch.rsqrt(q_var + eps) * q_weight

    k_var = k_flat.pow(2).mean(-1, keepdim=True)
    k_norm = k_flat * torch.rsqrt(k_var + eps) * k_weight

    return q_norm.reshape(B, S, H, head_dim), k_norm.reshape(B, S, H, head_dim)


def qwen_rmsnorm_gated_residual_kernel(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    QWEN wrapper for RMSNorm + gated residual.

    Args:
        x, residual: [B, S, D]
        weight: [D]
        gate: [B, D] (will be broadcast to [B, 1, D])
    """
    if KERNELS_AVAILABLE and kernel_lib is not None:
        try:
            B, S, D = x.shape
            orig_dtype = x.dtype

            if x.dtype == torch.bfloat16 and x.is_contiguous():
                x_bf16 = x
            else:
                x_bf16 = x.to(dtype=torch.bfloat16).contiguous()

            if residual.dtype == torch.bfloat16 and residual.is_contiguous():
                residual_bf16 = residual
            else:
                residual_bf16 = residual.to(dtype=torch.bfloat16).contiguous()

            if weight.dtype == torch.bfloat16 and weight.is_contiguous():
                weight_bf16 = weight
            else:
                weight_bf16 = weight.to(dtype=torch.bfloat16).contiguous()

            if gate.dtype == torch.bfloat16 and gate.is_contiguous():
                gate_bf16 = gate
            else:
                gate_bf16 = gate.to(dtype=torch.bfloat16).contiguous()

            out_bf16 = _get_output_buf(("rmsnorm_gated_res", x_bf16.shape), x_bf16.shape, x_bf16.dtype, x_bf16.device)

            kernel_lib.rmsnorm_gated_residual_bf16_device_cu(  # type: ignore[attr-defined]
                int(x_bf16.data_ptr()),
                int(residual_bf16.data_ptr()),
                int(weight_bf16.data_ptr()),
                int(gate_bf16.data_ptr()),
                int(out_bf16.data_ptr()),
                B,
                S,
                D,
                eps,
                _get_cuda_stream(),
            )
            _record_kernel("qwen_rmsnorm_gated_residual_bf16")
            _debug_sync("qwen_rmsnorm_gated_residual_bf16")

            if orig_dtype == torch.bfloat16:
                return out_bf16
            return out_bf16.to(dtype=orig_dtype)
        except Exception as e:
            warnings.warn(f"QWEN rmsnorm_gated_residual kernel failed, falling back to PyTorch: {e}")

    # PyTorch fallback
    variance = x.pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps) * weight
    return x_norm * gate.unsqueeze(1) + residual


def qwen_layernorm_kernel(
    x: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    QWEN wrapper for batched LayerNorm (elementwise_affine=False).

    Args:
        x: [B, S, D] or any shape — last dim is normalized.
    """
    if KERNELS_AVAILABLE and kernel_lib is not None:
        try:
            orig_shape = x.shape
            orig_dtype = x.dtype
            D = orig_shape[-1]
            rows = x.numel() // D

            if x.dtype == torch.bfloat16 and x.is_contiguous():
                x_bf16 = x
            else:
                x_bf16 = x.to(dtype=torch.bfloat16).contiguous()

            out_bf16 = _get_output_buf(("layernorm", x_bf16.shape), x_bf16.shape, x_bf16.dtype, x_bf16.device)
            kernel_lib.layernorm_bf16_device_cu(  # type: ignore[attr-defined]
                int(x_bf16.data_ptr()),
                int(out_bf16.data_ptr()),
                rows,
                D,
                eps,
                _get_cuda_stream(),
            )
            _record_kernel("qwen_layernorm_bf16")
            _debug_sync("qwen_layernorm_bf16")

            if orig_dtype == torch.bfloat16:
                return out_bf16
            return out_bf16.to(dtype=orig_dtype)
        except Exception as e:
            warnings.warn(f"QWEN layernorm kernel failed, falling back to PyTorch: {e}")

    # PyTorch fallback
    return F.layer_norm(x, [x.shape[-1]], eps=eps)


def qwen_gelu_kernel(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    QWEN wrapper for GELU activation (tanh approximation).

    Args:
        x: any shape — element-wise operation.
    """
    if KERNELS_AVAILABLE and kernel_lib is not None:
        try:
            orig_dtype = x.dtype
            total_elements = x.numel()

            if x.dtype == torch.bfloat16 and x.is_contiguous():
                x_bf16 = x
            else:
                x_bf16 = x.to(dtype=torch.bfloat16).contiguous()

            out_bf16 = _get_output_buf(("gelu", x_bf16.shape), x_bf16.shape, x_bf16.dtype, x_bf16.device)
            kernel_lib.gelu_bf16_device_cu(  # type: ignore[attr-defined]
                int(x_bf16.data_ptr()),
                int(out_bf16.data_ptr()),
                total_elements,
                _get_cuda_stream(),
            )
            _record_kernel("qwen_gelu_bf16")
            _debug_sync("qwen_gelu_bf16")

            if orig_dtype == torch.bfloat16:
                return out_bf16
            return out_bf16.to(dtype=orig_dtype)
        except Exception as e:
            warnings.warn(f"QWEN gelu kernel failed, falling back to PyTorch: {e}")

    # PyTorch fallback
    return F.gelu(x, approximate="tanh")


def qwen_silu_kernel(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    QWEN wrapper for SiLU (Swish) activation.

    Args:
        x: any shape — element-wise operation.

    Note: The underlying ``silu_bf16_device_cu`` binding does not accept a stream
    handle parameter. Using it would cause implicit synchronization via the
    legacy default stream (stream 0), serializing all GPU work. This wrapper
    is provided for future use once the binding is updated with stream support.
    For now it always falls back to PyTorch ``F.silu``.
    """
    # Intentionally disabled: kernel_lib.silu_bf16_device_cu lacks stream handle,
    # which would cause implicit sync and hurt throughput.  Uncomment when the
    # binding is updated to accept a stream parameter.
    #
    # if KERNELS_AVAILABLE and kernel_lib is not None:
    #     try:
    #         orig_dtype = x.dtype
    #         total_elements = x.numel()
    #         x_bf16 = x if (x.dtype == torch.bfloat16 and x.is_contiguous()) else x.to(dtype=torch.bfloat16).contiguous()
    #         out_bf16 = torch.empty_like(x_bf16)
    #         kernel_lib.silu_bf16_device_cu(
    #             int(x_bf16.data_ptr()),
    #             int(out_bf16.data_ptr()),
    #             total_elements,
    #         )
    #         _record_kernel("qwen_silu_bf16")
    #         _debug_sync("qwen_silu_bf16")
    #         return out_bf16 if orig_dtype == torch.bfloat16 else out_bf16.to(dtype=orig_dtype)
    #     except Exception as e:
    #         warnings.warn(f"QWEN silu kernel failed, falling back to PyTorch: {e}")

    return F.silu(x)


def qwen_layernorm_modulate_kernel(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fused LayerNorm (elementwise_affine=False) + Modulation.
    output = LayerNorm(x) * (1 + scale) + shift

    Args:
        x: [B, S, D=3072]  — expected bf16 contiguous
        scale: [B, D] (will be broadcast to [B, 1, D])
        shift: [B, D] (will be broadcast to [B, 1, D])
    """
    if _layernorm_modulate_bf16_op is not None:
        if torch.compiler.is_compiling():
            # Under torch.compile: use custom_op for graph compatibility.
            return _layernorm_modulate_bf16_op(x, scale, shift, eps)
        # Eager: direct kernel_lib call with pre-allocated output buffer.
        B, S, _D = x.shape
        out = _get_output_buf(("lnm", x.shape), x.shape, x.dtype, x.device)
        kernel_lib.layernorm_modulate_bf16_device_cu(
            int(x.data_ptr()), int(scale.data_ptr()), int(shift.data_ptr()),
            int(out.data_ptr()), B, S, eps, _get_cuda_stream(),
        )
        _record_kernel("qwen_layernorm_modulate_bf16")
        _debug_sync("qwen_layernorm_modulate_bf16")
        return out

    # PyTorch fallback
    x_normed = F.layer_norm(x, [x.shape[-1]], eps=eps)
    return x_normed * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def qwen_layernorm_modulate_gemm_up_kernel(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fused LayerNorm + Modulation + Linear up-projection.

    Eliminates the HBM round-trip of the normalized+modulated intermediate
    by computing norm+modulate on-the-fly during WMMA GEMM A-tile loading.

    Equivalent to:
        normed = F.layer_norm(x, [D], eps=eps)
        modulated = normed * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        output = F.linear(modulated, weight, bias)

    Args:
        x:      [B, S, D]  hidden states (bf16 contiguous)
        scale:  [B, D]     modulation scale
        shift:  [B, D]     modulation shift
        weight: [N, D]     up-projection weight
        bias:   [N]        up-projection bias
        eps:    LayerNorm epsilon
    Returns:
        [B, S, N]  up-projected output (bf16)
    """
    if _layernorm_modulate_gemm_up_bf16_op is not None:
        if torch.compiler.is_compiling():
            return _layernorm_modulate_gemm_up_bf16_op(x, scale, shift, weight, bias, eps)
        # Eager: direct kernel_lib call with buffer reuse
        B, S, D = x.shape
        N = weight.shape[0]
        M = B * S
        out_shape = (B, S, N)
        out = _get_output_buf(("lnm_gemm_up", out_shape), out_shape, x.dtype, x.device)
        stats = _get_row_stats_buf(M, x.device)
        kernel_lib.layernorm_modulate_gemm_up_bf16_device_cu(
            int(x.data_ptr()), int(scale.data_ptr()), int(shift.data_ptr()),
            int(weight.data_ptr()), int(bias.data_ptr()),
            int(out.data_ptr()), int(stats.data_ptr()),
            B, S, D, N, eps, _get_cuda_stream(),
        )
        _record_kernel("qwen_layernorm_modulate_gemm_up_bf16")
        _debug_sync("qwen_layernorm_modulate_gemm_up_bf16")
        return out

    # PyTorch fallback
    x_normed = F.layer_norm(x, [x.shape[-1]], eps=eps)
    modulated = x_normed * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return F.linear(modulated, weight, bias)


def qwen_gate_residual_kernel(
    residual: torch.Tensor,
    gate: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Fused gate + residual: output = residual + gate * x

    Args:
        residual: [B, S, D=3072]
        gate: [B, 1, D] or [B, D] -- if 3D, squeezed to [B, D]
        x: [B, S, D=3072]
    """
    if KERNELS_AVAILABLE and kernel_lib is not None:
        try:
            B, S, D = residual.shape
            if D != 3072:
                raise ValueError(f"gate_residual kernel requires D=3072, got {D}")

            gate_2d = gate.squeeze(1) if gate.ndim == 3 else gate
            orig_dtype = residual.dtype

            if residual.dtype == torch.bfloat16 and residual.is_contiguous():
                res_bf16 = residual
            else:
                res_bf16 = residual.to(dtype=torch.bfloat16).contiguous()

            if gate_2d.dtype == torch.bfloat16 and gate_2d.is_contiguous():
                gate_bf16 = gate_2d
            else:
                gate_bf16 = gate_2d.to(dtype=torch.bfloat16).contiguous()

            if x.dtype == torch.bfloat16 and x.is_contiguous():
                x_bf16 = x
            else:
                x_bf16 = x.to(dtype=torch.bfloat16).contiguous()

            out_bf16 = _get_output_buf(("gate_residual", res_bf16.shape), res_bf16.shape, res_bf16.dtype, res_bf16.device)
            kernel_lib.gate_residual_bf16_device_cu(  # type: ignore[attr-defined]
                int(res_bf16.data_ptr()),
                int(gate_bf16.data_ptr()),
                int(x_bf16.data_ptr()),
                int(out_bf16.data_ptr()),
                B,
                S,
                _get_cuda_stream(),
            )
            _record_kernel("qwen_gate_residual_bf16")
            _debug_sync("qwen_gate_residual_bf16")

            if orig_dtype == torch.bfloat16:
                return out_bf16
            return out_bf16.to(dtype=orig_dtype)
        except Exception as e:
            warnings.warn(f"QWEN gate_residual kernel failed, falling back to PyTorch: {e}")

    # PyTorch fallback
    if gate.ndim == 2:
        gate = gate.unsqueeze(1)
    return residual + gate * x


def qwen_swiglu_kernel(
    x: torch.Tensor,
    w1_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w3_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Pure-PyTorch SwiGLU implementation (no custom CUDA kernel).

    Note: The QWEN-2512 transformer MLP uses GELU (not SwiGLU), so this
    function is not used in the main inference path. It is kept here as a
    utility for architectures that use SwiGLU-style MLPs. The actual MLP
    acceleration is handled via ``qwen_gelu_kernel`` which calls into the
    fused ``gelu_bf16_device_cu`` CUDA kernel.

    Shapes:
      x: [B, S, D]  with D=3072
      w1_weight, w3_weight: [inter_dim, D] with inter_dim=12288
      w2_weight: [D, inter_dim]
    """
    # Standard SwiGLU: (SiLU(x W1) * (x W3)) W2^T
    w1_out = F.linear(x, w1_weight)  # [B, S, inter_dim]
    w3_out = F.linear(x, w3_weight)  # [B, S, inter_dim]
    gated = F.silu(w1_out) * w3_out
    return F.linear(gated, w2_weight)


# ============================================================================
# QWEN-2512 fused device-pointer kernels
# These replace multiple PyTorch ops with a single CUDA kernel launch.
# ============================================================================

# Persistent intermediate buffer for fused MLP (avoids per-call allocation).
# _mlp_intermediate_buffer is declared near the top of the file alongside _output_buffers.


def _get_mlp_intermediate(B: int, S: int, inter_dim: int, device: torch.device) -> torch.Tensor:
    """Get or create a persistent intermediate buffer for fused MLP."""
    key = (B, S, inter_dim, device)
    buf = _mlp_intermediate_buffer.get(key)
    if buf is None or buf.shape != (B * S, inter_dim):
        buf = torch.empty(B * S, inter_dim, dtype=torch.bfloat16, device=device)
        _mlp_intermediate_buffer[key] = buf
    return buf


def qwen_mlp_fused_gelu_kernel(
    x: torch.Tensor,
    w1_weight: torch.Tensor,
    b1_bias: torch.Tensor,
    w2_weight: torch.Tensor,
    b2_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Fused MLP: Linear(D→inter_dim) + GELU + Linear(inter_dim→D) using
    WMMA tensor-core kernels.  Replaces the entire FeedForward pass.

    Args:
        x: [B, S, D=3072]
        w1_weight: [inter_dim, D]  (first linear weight)
        b1_bias:   [inter_dim]     (first linear bias)
        w2_weight: [D, inter_dim]  (second linear weight)
        b2_bias:   [D]             (second linear bias)
    """
    if KERNELS_AVAILABLE and kernel_lib is not None:
        try:
            B, S, D = x.shape
            inter_dim = w1_weight.shape[0]
            orig_dtype = x.dtype

            x_bf16 = x if (x.dtype == torch.bfloat16 and x.is_contiguous()) else x.to(torch.bfloat16).contiguous()
            w1_bf16 = w1_weight if (w1_weight.dtype == torch.bfloat16 and w1_weight.is_contiguous()) else w1_weight.to(torch.bfloat16).contiguous()
            b1_bf16 = b1_bias if (b1_bias.dtype == torch.bfloat16 and b1_bias.is_contiguous()) else b1_bias.to(torch.bfloat16).contiguous()
            w2_bf16 = w2_weight if (w2_weight.dtype == torch.bfloat16 and w2_weight.is_contiguous()) else w2_weight.to(torch.bfloat16).contiguous()
            b2_bf16 = b2_bias if (b2_bias.dtype == torch.bfloat16 and b2_bias.is_contiguous()) else b2_bias.to(torch.bfloat16).contiguous()

            out_bf16 = _get_output_buf(("mlp_fused", x_bf16.shape), x_bf16.shape, x_bf16.dtype, x_bf16.device)
            intermediate = _get_mlp_intermediate(B, S, inter_dim, x.device)

            kernel_lib.mlp_fused_gelu_bf16_4352x3072_device_cu(
                int(x_bf16.data_ptr()),
                int(w1_bf16.data_ptr()), int(b1_bf16.data_ptr()),
                int(w2_bf16.data_ptr()), int(b2_bf16.data_ptr()),
                int(out_bf16.data_ptr()), int(intermediate.data_ptr()),
                B, S, D, inter_dim,
            )
            _record_kernel("qwen_mlp_fused_gelu_bf16")
            _debug_sync("qwen_mlp_fused_gelu_bf16")

            return out_bf16 if orig_dtype == torch.bfloat16 else out_bf16.to(orig_dtype)
        except Exception as e:
            warnings.warn(f"QWEN mlp_fused_gelu kernel failed, falling back to PyTorch: {e}")

    # PyTorch fallback
    h = F.linear(x, w1_weight, b1_bias)
    h = F.gelu(h, approximate="tanh")
    return F.linear(h, w2_weight, b2_bias)


def qwen_qkv_rmsnorm_fused_kernel(
    x: torch.Tensor,
    q_weight: torch.Tensor, k_weight: torch.Tensor, v_weight: torch.Tensor,
    q_bias: torch.Tensor, k_bias: torch.Tensor, v_bias: torch.Tensor,
    q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor,
    num_heads: int,
    head_dim: int,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused QKV projection (WMMA GEMM) + per-head RMSNorm.

    Replaces: to_q/to_k/to_v linear projections + norm_q/norm_k RMSNorm.

    Args:
        x: [B, S, D=3072]
        q/k/v_weight: [D, D]  projection weights
        q/k/v_bias: [D]  projection biases
        q/k_norm_weight: [head_dim] per-head RMSNorm weights
    Returns:
        (q, k, v) each [B, S, H, head_dim] with Q and K already RMSNorm'd.
    """
    if KERNELS_AVAILABLE and kernel_lib is not None:
        try:
            B, S, D = x.shape
            H = num_heads
            Dh = head_dim
            orig_dtype = x.dtype

            x_bf16 = x if (x.dtype == torch.bfloat16 and x.is_contiguous()) else x.to(torch.bfloat16).contiguous()
            qw = q_weight if (q_weight.dtype == torch.bfloat16 and q_weight.is_contiguous()) else q_weight.to(torch.bfloat16).contiguous()
            kw = k_weight if (k_weight.dtype == torch.bfloat16 and k_weight.is_contiguous()) else k_weight.to(torch.bfloat16).contiguous()
            vw = v_weight if (v_weight.dtype == torch.bfloat16 and v_weight.is_contiguous()) else v_weight.to(torch.bfloat16).contiguous()
            qb = q_bias if (q_bias.dtype == torch.bfloat16 and q_bias.is_contiguous()) else q_bias.to(torch.bfloat16).contiguous()
            kb = k_bias if (k_bias.dtype == torch.bfloat16 and k_bias.is_contiguous()) else k_bias.to(torch.bfloat16).contiguous()
            vb = v_bias if (v_bias.dtype == torch.bfloat16 and v_bias.is_contiguous()) else v_bias.to(torch.bfloat16).contiguous()
            qnw = q_norm_weight if (q_norm_weight.dtype == torch.bfloat16 and q_norm_weight.is_contiguous()) else q_norm_weight.to(torch.bfloat16).contiguous()
            knw = k_norm_weight if (k_norm_weight.dtype == torch.bfloat16 and k_norm_weight.is_contiguous()) else k_norm_weight.to(torch.bfloat16).contiguous()

            _qkv_shape = (B, S, D)
            q_out = _get_output_buf(("qkv_q", _qkv_shape), _qkv_shape, torch.bfloat16, x.device)
            k_out = _get_output_buf(("qkv_k", _qkv_shape), _qkv_shape, torch.bfloat16, x.device)
            v_out = _get_output_buf(("qkv_v", _qkv_shape), _qkv_shape, torch.bfloat16, x.device)

            kernel_lib.qkv_rmsnorm_fused_bf16_4352x3072_device_cu(
                int(x_bf16.data_ptr()),
                int(qw.data_ptr()), int(kw.data_ptr()), int(vw.data_ptr()),
                int(qb.data_ptr()), int(kb.data_ptr()), int(vb.data_ptr()),
                int(qnw.data_ptr()), int(knw.data_ptr()),
                int(q_out.data_ptr()), int(k_out.data_ptr()), int(v_out.data_ptr()),
                B, S, D, H, Dh, eps,
            )
            _record_kernel("qwen_qkv_rmsnorm_fused_bf16")
            _debug_sync("qwen_qkv_rmsnorm_fused_bf16")

            # Reshape to [B, S, H, head_dim]
            q_out = q_out.unflatten(-1, (H, Dh))
            k_out = k_out.unflatten(-1, (H, Dh))
            v_out = v_out.unflatten(-1, (H, Dh))

            if orig_dtype != torch.bfloat16:
                q_out = q_out.to(orig_dtype)
                k_out = k_out.to(orig_dtype)
                v_out = v_out.to(orig_dtype)

            return q_out, k_out, v_out
        except Exception as e:
            warnings.warn(f"QWEN qkv_rmsnorm_fused kernel failed, falling back to PyTorch: {e}")

    # PyTorch fallback
    q = F.linear(x, q_weight, q_bias).unflatten(-1, (num_heads, head_dim))
    k = F.linear(x, k_weight, k_bias).unflatten(-1, (num_heads, head_dim))
    v = F.linear(x, v_weight, v_bias).unflatten(-1, (num_heads, head_dim))
    # Per-head RMSNorm on Q and K
    q_flat = q.reshape(-1, head_dim)
    q_var = q_flat.pow(2).mean(-1, keepdim=True)
    q = (q_flat * torch.rsqrt(q_var + eps) * q_norm_weight).reshape(q.shape)
    k_flat = k.reshape(-1, head_dim)
    k_var = k_flat.pow(2).mean(-1, keepdim=True)
    k = (k_flat * torch.rsqrt(k_var + eps) * k_norm_weight).reshape(k.shape)
    return q, k, v


def qwen_modulation_project_kernel(
    temb: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused modulation projection: SiLU(temb) @ W + bias -> 6 outputs.

    Replaces: nn.Sequential(nn.SiLU(), nn.Linear(D, 6*D)).

    Args:
        temb: [B, D]
        weight: [6*D, D]
        bias: [6*D]
    Returns:
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) each [B, D]
    """
    if KERNELS_AVAILABLE and kernel_lib is not None:
        try:
            B = temb.shape[0]
            D = temb.shape[-1]
            orig_dtype = temb.dtype

            temb_bf16 = temb if (temb.dtype == torch.bfloat16 and temb.is_contiguous()) else temb.to(torch.bfloat16).contiguous()
            w_bf16 = weight if (weight.dtype == torch.bfloat16 and weight.is_contiguous()) else weight.to(torch.bfloat16).contiguous()
            b_bf16 = bias if (bias.dtype == torch.bfloat16 and bias.is_contiguous()) else bias.to(torch.bfloat16).contiguous()

            shift_msa = torch.empty(B, D, dtype=torch.bfloat16, device=temb.device)
            scale_msa = torch.empty(B, D, dtype=torch.bfloat16, device=temb.device)
            gate_msa = torch.empty(B, D, dtype=torch.bfloat16, device=temb.device)
            shift_mlp = torch.empty(B, D, dtype=torch.bfloat16, device=temb.device)
            scale_mlp = torch.empty(B, D, dtype=torch.bfloat16, device=temb.device)
            gate_mlp = torch.empty(B, D, dtype=torch.bfloat16, device=temb.device)

            kernel_lib.modulation_project_bf16_device_cu(
                int(temb_bf16.data_ptr()), int(w_bf16.data_ptr()), int(b_bf16.data_ptr()),
                int(shift_msa.data_ptr()), int(scale_msa.data_ptr()), int(gate_msa.data_ptr()),
                int(shift_mlp.data_ptr()), int(scale_mlp.data_ptr()), int(gate_mlp.data_ptr()),
                B, D,
            )
            _record_kernel("qwen_modulation_project_bf16")
            _debug_sync("qwen_modulation_project_bf16")

            if orig_dtype != torch.bfloat16:
                return tuple(t.to(orig_dtype) for t in (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp))
            return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        except Exception as e:
            warnings.warn(f"QWEN modulation_project kernel failed, falling back to PyTorch: {e}")

    # PyTorch fallback
    h = F.silu(temb)
    h = F.linear(h, weight, bias)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = h.chunk(6, dim=-1)
    return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp


def qwen_rope_precomputed_kernel(
    x: torch.Tensor,
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
) -> torch.Tensor:
    """
    RoPE with precomputed cos/sin frequencies using a fused CUDA kernel.

    Args:
        x: [B, S, H, D] query or key tensor
        cos_freqs: [S, D//2] float32 cosine frequencies
        sin_freqs: [S, D//2] float32 sine frequencies
    """
    if KERNELS_AVAILABLE and kernel_lib is not None:
        try:
            B, S, H, D = x.shape
            orig_dtype = x.dtype

            x_bf16 = x if (x.dtype == torch.bfloat16 and x.is_contiguous()) else x.to(torch.bfloat16).contiguous()
            cos_f32 = cos_freqs if (cos_freqs.dtype == torch.float32 and cos_freqs.is_contiguous()) else cos_freqs.float().contiguous()
            sin_f32 = sin_freqs if (sin_freqs.dtype == torch.float32 and sin_freqs.is_contiguous()) else sin_freqs.float().contiguous()

            out_bf16 = _get_output_buf(("rope_precomp", x_bf16.shape), x_bf16.shape, x_bf16.dtype, x_bf16.device)

            kernel_lib.rope_apply_precomputed_bf16_device_cu(
                int(x_bf16.data_ptr()), int(out_bf16.data_ptr()),
                int(cos_f32.data_ptr()), int(sin_f32.data_ptr()),
                B, S, H, D,
            )
            _record_kernel("qwen_rope_precomputed_bf16")
            _debug_sync("qwen_rope_precomputed_bf16")

            return out_bf16 if orig_dtype == torch.bfloat16 else out_bf16.to(orig_dtype)
        except Exception as e:
            warnings.warn(f"QWEN rope_precomputed kernel failed, falling back to PyTorch: {e}")

    # PyTorch fallback: standard complex-number RoPE
    cos = cos_freqs[None, :, None, :]
    sin = sin_freqs[None, :, None, :]
    x_r, x_i = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    o_r = x_r * cos.to(x.dtype) - x_i * sin.to(x.dtype)
    o_i = x_r * sin.to(x.dtype) + x_i * cos.to(x.dtype)
    return torch.stack([o_r, o_i], dim=-1).flatten(-2)


def qwen_qk_norm_rope_3d_fused_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    grid_frame: int,
    grid_height: int,
    grid_width: int,
    theta: float,
    eps: float,
    axes_dim: Tuple[int, int, int],
    height_offset: int,
    width_offset: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused per-head QK RMSNorm + 3D Axial RoPE kernel for image tokens.

    Combines two separate operations into a single CUDA kernel launch,
    eliminating the intermediate tensor materialization between QK-norm
    and RoPE. This reduces HBM traffic by ~50% for these operations.

    Args:
        q, k: [B, S_img, H, D] query and key tensors (bf16)
        q_weight, k_weight: [D] per-head RMSNorm weight vectors
        grid_frame: number of frames in the image grid
        grid_height: height of the image grid
        grid_width: width of the image grid
        theta: RoPE base frequency (typically 10000.0)
        eps: RMSNorm epsilon
        axes_dim: (dim_time, dim_height, dim_width) real dimension sizes
        height_offset: centering offset for scale_rope
        width_offset: centering offset for scale_rope
    Returns:
        (q_out, k_out): normalized and RoPE-rotated Q and K tensors
    """
    if _qk_norm_rope_3d_fused_bf16_op is not None:
        B, S_img, H, D = q.shape
        # Fused kernel requires D==128 (block size is hardcoded).
        if D != 128:
            return None
        if torch.compiler.is_compiling():
            return _qk_norm_rope_3d_fused_bf16_op(
                q, k, q_weight, k_weight,
                grid_frame, grid_height, grid_width,
                theta, eps,
                axes_dim[0], axes_dim[1], axes_dim[2],
                height_offset, width_offset,
            )
        # Eager: direct kernel_lib call with pre-allocated output buffers.
        q_out = _get_output_buf(("fused_qk_q", q.shape), q.shape, q.dtype, q.device)
        k_out = _get_output_buf(("fused_qk_k", k.shape), k.shape, k.dtype, k.device)
        kernel_lib.qk_norm_rope_3d_fused_bf16_device_cu(
            int(q.data_ptr()), int(k.data_ptr()),
            int(q_weight.data_ptr()), int(k_weight.data_ptr()),
            int(q_out.data_ptr()), int(k_out.data_ptr()),
            B, S_img, H, D,
            grid_frame, grid_height, grid_width,
            axes_dim[0], axes_dim[1], axes_dim[2],
            float(theta), float(eps),
            height_offset, width_offset,
            _get_cuda_stream(),
        )
        _record_kernel("qwen_qk_norm_rope_3d_fused_bf16")
        _debug_sync("qwen_qk_norm_rope_3d_fused_bf16")
        return q_out, k_out

    # No fused kernel available — caller should fall back to separate ops
    return None


def qwen_rope_3d_fused_kernel(
    x: torch.Tensor,
    grid_frame: int,
    grid_height: int,
    grid_width: int,
    theta: float,
    axes_dim: Tuple[int, int, int],
    height_offset: int,
    width_offset: int,
) -> torch.Tensor:
    """
    Fused 3D axial RoPE kernel for image tokens.

    Computes sin/cos on-the-fly from grid coordinates, eliminating
    precomputed frequency tables and fp32 conversion overhead.
    Single-pass bf16 read-write.

    Args:
        x: [B, S_img, H, D] query or key tensor (bf16)
        grid_frame: number of frames in the image grid
        grid_height: height of the image grid
        grid_width: width of the image grid
        theta: RoPE base frequency (typically 10000.0)
        axes_dim: (dim_time, dim_height, dim_width) real dimension sizes
        height_offset: centering offset for scale_rope (height - height//2)
        width_offset: centering offset for scale_rope (width - width//2)
    """
    if _rope_3d_fused_bf16_op is not None:
        if torch.compiler.is_compiling():
            # Under torch.compile: use custom_op for graph compatibility.
            return _rope_3d_fused_bf16_op(
                x, grid_frame, grid_height, grid_width, theta,
                axes_dim[0], axes_dim[1], axes_dim[2],
                height_offset, width_offset,
            )
        # Eager: direct kernel_lib call with pre-allocated output buffer.
        B, S_img, H, D = x.shape
        out = _get_output_buf(("rope3d", x.shape), x.shape, x.dtype, x.device)
        kernel_lib.rope_3d_optimized_bf16_device_cu(
            int(x.data_ptr()), int(out.data_ptr()),
            B, S_img, H, D,
            grid_frame, grid_height, grid_width,
            axes_dim[0], axes_dim[1], axes_dim[2],
            float(theta), height_offset, width_offset,
            _get_cuda_stream(),
        )
        _record_kernel("qwen_rope_3d_fused_bf16")
        _debug_sync("qwen_rope_3d_fused_bf16")
        return out

    # No PyTorch fallback — caller should fall back to apply_rotary_emb_qwen
    return None

