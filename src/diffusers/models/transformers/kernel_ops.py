#!/usr/bin/env python3

# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Custom CUDA kernel operations wrapper for Z-Image transformer.

This module provides PyTorch-compatible wrappers for custom CUDA kernels
from the kernex project. It handles tensor conversions and provides
fallback to PyTorch operations if kernels are unavailable.

All active hot-path kernels consume device pointers only (no NumPy / host
round-trips). BF16 handling is done in-place where necessary, with
upstream modules expected to produce BF16 tensors on hot paths to minimize
casts.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Tuple

import torch
import torch.nn.functional as F

# Try to import kernel library
KERNELS_AVAILABLE = False
kernel_lib = None

# Track which custom kernels were actually invoked successfully in this process.
_USED_KERNELS: set[str] = set()


def _record_kernel(name: str) -> None:
    _USED_KERNELS.add(name)


def get_used_kernels() -> list[str]:
    """
    Return a sorted list of custom kernel wrappers that were successfully used.

    This is purely for introspection / logging during benchmarking.
    """
    return sorted(_USED_KERNELS)


# Try multiple paths to find kernel_lib
_kernex_paths = [
    # Path relative to workspace root
    os.path.join(os.path.dirname(__file__), "../../../../../../kernex/bazel-bin/bindings"),
    # Alternative: absolute path from workspace
    os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                    )
                )
            )
        ),
        "kernex/bazel-bin/bindings",
    ),
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
        "Custom CUDA kernels not available. Falling back to PyTorch operations. "
        "To enable kernels, build with: bazel build --config=cuda //bindings:kernel_lib "
        "and ensure the bindings directory is in the Python path."
    )


def adaln_modulation_kernel(
    adaln_input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    AdaLN modulation kernel wrapper.

    Args:
        adaln_input: Input tensor of shape [B, emb_dim]
        weight: Weight tensor of shape [4 * hidden_dim, emb_dim]
        bias: Bias tensor of shape [4 * hidden_dim]

    Returns:
        Tuple of (scale_msa, gate_msa, scale_mlp, gate_mlp), each of shape [B, 1, hidden_dim]
    """
    B = adaln_input.shape[0]
    hidden_dim = weight.shape[0] // 4
    emb_dim = weight.shape[1]

    if KERNELS_AVAILABLE and kernel_lib is not None:
        try:
            if adaln_input.dtype != torch.bfloat16 or not adaln_input.is_contiguous():
                adaln_bf16 = adaln_input.to(dtype=torch.bfloat16).contiguous()
            else:
                adaln_bf16 = adaln_input
            if weight.dtype != torch.bfloat16 or not weight.is_contiguous():
                weight_bf16 = weight.to(dtype=torch.bfloat16).contiguous()
            else:
                weight_bf16 = weight
            if bias.dtype != torch.bfloat16 or not bias.is_contiguous():
                bias_bf16 = bias.to(dtype=torch.bfloat16).contiguous()
            else:
                bias_bf16 = bias

            scale_msa = torch.empty(B, hidden_dim, dtype=torch.bfloat16, device=adaln_input.device)
            gate_msa = torch.empty(B, hidden_dim, dtype=torch.bfloat16, device=adaln_input.device)
            scale_mlp = torch.empty(B, hidden_dim, dtype=torch.bfloat16, device=adaln_input.device)
            gate_mlp = torch.empty(B, hidden_dim, dtype=torch.bfloat16, device=adaln_input.device)

            kernel_lib.adaln_modulation_bf16_device_cu(  # type: ignore[attr-defined]
                int(adaln_bf16.data_ptr()),
                int(weight_bf16.data_ptr()),
                int(bias_bf16.data_ptr()),
                int(scale_msa.data_ptr()),
                int(gate_msa.data_ptr()),
                int(scale_mlp.data_ptr()),
                int(gate_mlp.data_ptr()),
                B,
                emb_dim,
                hidden_dim,
            )
            _record_kernel("adaln_modulation_bf16")

            return (
                scale_msa.unsqueeze(1),
                gate_msa.unsqueeze(1),
                scale_mlp.unsqueeze(1),
                gate_mlp.unsqueeze(1),
            )
        except Exception as e:
            warnings.warn(f"AdaLN kernel failed, falling back to PyTorch: {e}")

    output = F.linear(adaln_input, weight, bias)  # [B, 4 * hidden_dim]
    output = output.unsqueeze(1)  # [B, 1, 4 * hidden_dim]
    scale_msa, gate_msa, scale_mlp, gate_mlp = output.chunk(4, dim=-1)
    gate_msa = torch.tanh(gate_msa)
    gate_mlp = torch.tanh(gate_mlp)
    scale_msa = 1.0 + scale_msa
    scale_mlp = 1.0 + scale_mlp
    return scale_msa, gate_msa, scale_mlp, gate_mlp


def rmsnorm_scale_kernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    RMSNorm + Scale kernel wrapper (optimized to avoid unnecessary conversions).

    Args:
        x: Input tensor of shape [B, S, D]
        weight: Weight tensor of shape [D]
        scale: Scale tensor of shape [B, D] (will be broadcast to [B, 1, D])
        eps: Epsilon for numerical stability

    Returns:
        Output tensor of shape [B, S, D]
    """
    if KERNELS_AVAILABLE and kernel_lib is not None:
        try:
            B, S, D = x.shape
            orig_dtype = x.dtype

            # Optimize: only convert dtype if needed, avoid redundant device transfers
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

            output_bf16 = torch.empty_like(x_bf16)
            kernel_lib.rmsnorm_scale_bf16_device_cu(  # type: ignore[attr-defined]
                int(x_bf16.data_ptr()),
                int(weight_bf16.data_ptr()),
                int(scale_bf16.data_ptr()),
                int(output_bf16.data_ptr()),
                B,
                S,
                D,
                eps,
            )
            _record_kernel("rmsnorm_scale_bf16")

            # Optimize: only convert back if dtype changed
            if orig_dtype == torch.bfloat16:
                return output_bf16
            return output_bf16.to(dtype=orig_dtype)
        except Exception as e:
            warnings.warn(f"Kernel call failed, falling back to PyTorch: {e}")
            variance = x.pow(2).mean(-1, keepdim=True)
            x_norm = x * torch.rsqrt(variance + eps) * weight
            return x_norm * scale.unsqueeze(1)

    # Fallback to PyTorch
    variance = x.pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps) * weight
    return x_norm * scale.unsqueeze(1)


def qkv_projection_kernel(
    x: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    QKV projection kernel wrapper.

    Note: For large matrix multiplications (3840x3840), PyTorch's F.linear uses
    highly optimized cuBLAS with tensor cores, which is much faster than our
    custom 16x16 tiled kernels. Since QKV projection is just 3 separate matmuls
    (no fusion benefit), we use PyTorch's optimized implementation.

    Args:
        x: Input tensor of shape [B, S, D]
        q_weight: Q weight tensor of shape [D, D]
        k_weight: K weight tensor of shape [D, D]
        v_weight: V weight tensor of shape [D, D]

    Returns:
        Tuple of (q, k, v), each of shape [B, S, D]
    """
    # Use PyTorch's optimized cuBLAS implementation for matrix multiplication
    # Custom kernels are only beneficial for fused operations (RMSNorm+scale, etc.)
    q = F.linear(x, q_weight)
    k = F.linear(x, k_weight)
    v = F.linear(x, v_weight)
    return q, k, v


def qk_norm_perhead_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    QK normalization per-head kernel wrapper (optimized to avoid unnecessary conversions).

    Args:
        q: Query tensor of shape [B, S, H, head_dim]
        k: Key tensor of shape [B, S, H, head_dim]
        q_weight: Q weight tensor of shape [head_dim]
        k_weight: K weight tensor of shape [head_dim]
        eps: Epsilon for numerical stability

    Returns:
        Tuple of (q_out, k_out), each of shape [B, S, H, head_dim]
    """
    if KERNELS_AVAILABLE and kernel_lib is not None:
        try:
            B, S, H, head_dim = q.shape
            orig_dtype = q.dtype

            # Optimize: only convert dtype if needed, avoid redundant device transfers
            if q.dtype == torch.bfloat16 and q.is_contiguous():
                q_bf16 = q
            else:
                q_bf16 = q.to(dtype=torch.bfloat16).contiguous()

            if k.dtype == torch.bfloat16 and k.is_contiguous():
                k_bf16 = k
            else:
                k_bf16 = k.to(dtype=torch.bfloat16).contiguous()

            if q_weight.dtype == torch.bfloat16 and q_weight.is_contiguous():
                q_weight_bf16 = q_weight
            else:
                q_weight_bf16 = q_weight.to(dtype=torch.bfloat16).contiguous()

            if k_weight.dtype == torch.bfloat16 and k_weight.is_contiguous():
                k_weight_bf16 = k_weight
            else:
                k_weight_bf16 = k_weight.to(dtype=torch.bfloat16).contiguous()

            q_out_bf16 = torch.empty_like(q_bf16)
            k_out_bf16 = torch.empty_like(k_bf16)
            kernel_lib.qk_norm_perhead_bf16_device_cu(  # type: ignore[attr-defined]
                int(q_bf16.data_ptr()),
                int(k_bf16.data_ptr()),
                int(q_weight_bf16.data_ptr()),
                int(k_weight_bf16.data_ptr()),
                int(q_out_bf16.data_ptr()),
                int(k_out_bf16.data_ptr()),
                B,
                S,
                H,
                head_dim,
                eps,
            )
            _record_kernel("qk_norm_perhead_bf16")

            # Optimize: only convert back if dtype changed
            if orig_dtype == torch.bfloat16:
                return q_out_bf16, k_out_bf16
            return q_out_bf16.to(dtype=orig_dtype), k_out_bf16.to(dtype=orig_dtype)
        except Exception as e:
            warnings.warn(f"Kernel call failed, falling back to PyTorch: {e}")
            B, S, H, head_dim = q.shape
            q_flat = q.reshape(-1, head_dim)
            k_flat = k.reshape(-1, head_dim)
            q_var = q_flat.pow(2).mean(-1, keepdim=True)
            q_norm = q_flat * torch.rsqrt(q_var + eps) * q_weight
            k_var = k_flat.pow(2).mean(-1, keepdim=True)
            k_norm = k_flat * torch.rsqrt(k_var + eps) * k_weight
            return q_norm.reshape(B, S, H, head_dim), k_norm.reshape(B, S, H, head_dim)

    # Fallback to PyTorch
    B, S, H, head_dim = q.shape
    q_flat = q.reshape(-1, head_dim)
    k_flat = k.reshape(-1, head_dim)

    q_var = q_flat.pow(2).mean(-1, keepdim=True)
    q_norm = q_flat * torch.rsqrt(q_var + eps) * q_weight

    k_var = k_flat.pow(2).mean(-1, keepdim=True)
    k_norm = k_flat * torch.rsqrt(k_var + eps) * k_weight

    return q_norm.reshape(B, S, H, head_dim), k_norm.reshape(B, S, H, head_dim)


def rmsnorm_gated_residual_kernel(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    RMSNorm + Gated Residual kernel wrapper (optimized to avoid unnecessary conversions).

    Args:
        x: Input tensor of shape [B, S, D]
        residual: Residual tensor of shape [B, S, D]
        weight: Weight tensor of shape [D]
        gate: Gate tensor of shape [B, D] (will be broadcast to [B, 1, D])
        eps: Epsilon for numerical stability

    Returns:
        Output tensor of shape [B, S, D]
    """
    if KERNELS_AVAILABLE and kernel_lib is not None:
        try:
            B, S, D = x.shape
            orig_dtype = x.dtype

            # Optimize: only convert dtype if needed, avoid redundant device transfers
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

            output_bf16 = torch.empty_like(x_bf16)
            kernel_lib.rmsnorm_gated_residual_bf16_device_cu(  # type: ignore[attr-defined]
                int(x_bf16.data_ptr()),
                int(residual_bf16.data_ptr()),
                int(weight_bf16.data_ptr()),
                int(gate_bf16.data_ptr()),
                int(output_bf16.data_ptr()),
                B,
                S,
                D,
                eps,
            )
            _record_kernel("rmsnorm_gated_residual_bf16")

            # Optimize: only convert back if dtype changed
            if orig_dtype == torch.bfloat16:
                return output_bf16
            return output_bf16.to(dtype=orig_dtype)
        except Exception as e:
            warnings.warn(f"Kernel call failed, falling back to PyTorch: {e}")
            variance = x.pow(2).mean(-1, keepdim=True)
            x_norm = x * torch.rsqrt(variance + eps) * weight
            return x_norm * gate.unsqueeze(1) + residual

    # Fallback to PyTorch
    variance = x.pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps) * weight
    return x_norm * gate.unsqueeze(1) + residual


def silu_and_mul_kernel(w1_out: torch.Tensor, w3_out: torch.Tensor) -> torch.Tensor:
    """
    Fused SiLU(w1_out) * w3_out kernel wrapper.

    Args:
        w1_out: Tensor of shape [B, S, inter_dim]
        w3_out: Tensor of shape [B, S, inter_dim]

    Returns:
        Tensor of shape [B, S, inter_dim]
    """
    if KERNELS_AVAILABLE and kernel_lib is not None:
        try:
            B, S, inter_dim = w1_out.shape
            orig_dtype = w1_out.dtype

            if w1_out.dtype != torch.bfloat16 or not w1_out.is_contiguous():
                w1_bf16 = w1_out.to(dtype=torch.bfloat16).contiguous()
            else:
                w1_bf16 = w1_out
            if w3_out.dtype != torch.bfloat16 or not w3_out.is_contiguous():
                w3_bf16 = w3_out.to(dtype=torch.bfloat16).contiguous()
            else:
                w3_bf16 = w3_out

            output_bf16 = torch.empty_like(w1_bf16)
            kernel_lib.silu_and_mul_bf16_device_cu(  # type: ignore[attr-defined]
                int(w1_bf16.data_ptr()),
                int(w3_bf16.data_ptr()),
                int(output_bf16.data_ptr()),
                B,
                S,
                inter_dim,
            )
            _record_kernel("silu_and_mul_bf16")

            if orig_dtype == torch.bfloat16:
                return output_bf16
            return output_bf16.to(dtype=orig_dtype)
        except Exception as e:
            warnings.warn(f"SiLU+Mul kernel failed, falling back to PyTorch: {e}")

    return F.silu(w1_out) * w3_out


def swiglu_kernel(
    x: torch.Tensor,
    w1_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w3_weight: torch.Tensor,
) -> torch.Tensor:
    """
    SwiGLU wrapper. GEMMs use PyTorch/cuBLAS; SiLU+Mul uses custom kernel when available.
    """
    w1_out = F.linear(x, w1_weight)  # [B, S, inter_dim]
    w3_out = F.linear(x, w3_weight)  # [B, S, inter_dim]
    gated = silu_and_mul_kernel(w1_out, w3_out)
    return F.linear(gated, w2_weight)  # [B, S, D]


def rope_attention_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """
    RoPE Attention kernel wrapper.

    Args:
        q: Query tensor of shape [B, S, H, head_dim]
        k: Key tensor of shape [B, S, H, head_dim]
        v: Value tensor of shape [B, S, H, head_dim]
        freqs_cis: RoPE frequencies tensor of shape [B, S, head_dim] (complex64, stored as real/imag pairs)

    Returns:
        Output tensor of shape [B, S, H * head_dim] (flattened)
    """
    if not KERNELS_AVAILABLE:
        # Fallback to PyTorch (simplified - full implementation would be complex)
        # This is a placeholder - the actual RoPE attention is handled in the attention processor
        warnings.warn("RoPE attention kernel not available, using PyTorch fallback")
        return v.reshape(v.shape[0], v.shape[1], -1)

    try:
        # Note: rope_attention_bf16 kernel is a placeholder and doesn't actually compute anything.
        # It would produce uninitialized/garbage values. For now, always use PyTorch fallback.
        # When properly implemented, follow the same pattern as other kernels above:
        # - Convert to bfloat16, contiguous
        # - Pre-allocate outputs
        # - Call device-pointer variant
        # - Cast back to original dtype
        raise NotImplementedError(
            "rope_attention_bf16 kernel not yet fully implemented - using PyTorch fallback"
        )
    except Exception as e:
        warnings.warn(f"Kernel call failed, falling back to PyTorch: {e}")
        # Fallback to PyTorch
        return v.reshape(v.shape[0], v.shape[1], -1)


def check_kernels_available() -> bool:
    """Check if custom CUDA kernels are available."""
    return KERNELS_AVAILABLE
