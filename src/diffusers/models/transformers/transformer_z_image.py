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

import math
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...models.attention_processor import Attention
from ...models.modeling_utils import ModelMixin
from ...models.normalization import RMSNorm
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention_dispatch import dispatch_attention_fn
from ..modeling_outputs import Transformer2DModelOutput

# Try to import kernel operations
try:
    from .kernel_ops import (
        adaln_modulation_kernel,
        rmsnorm_scale_kernel,
        qkv_projection_kernel,
        qk_norm_perhead_kernel,
        rmsnorm_gated_residual_kernel,
        swiglu_kernel,
        check_kernels_available,
    )
    KERNEL_OPS_AVAILABLE = True
except ImportError:
    KERNEL_OPS_AVAILABLE = False
    # Define dummy functions to avoid errors
    def adaln_modulation_kernel(*args, **kwargs):
        raise NotImplementedError("Kernel ops not available")
    def rmsnorm_scale_kernel(*args, **kwargs):
        raise NotImplementedError("Kernel ops not available")

# Optional timing instrumentation (enabled via env var)
ENABLE_TIMING = os.environ.get("ZIMAGE_ENABLE_TIMING", "0") == "1"
_timing_stats = {}  # Global dict to accumulate timing stats


def get_timing_stats() -> dict:
    """Get accumulated timing statistics."""
    return {k: {"mean": sum(v) / len(v), "count": len(v), "total": sum(v)} for k, v in _timing_stats.items()}


def reset_timing_stats():
    """Reset timing statistics."""
    global _timing_stats
    _timing_stats = {}


def print_timing_stats():
    """Print timing statistics summary."""
    if not ENABLE_TIMING:
        print("Timing is disabled. Set ZIMAGE_ENABLE_TIMING=1 to enable.")
        return
    
    stats = get_timing_stats()
    if not stats:
        print("No timing data collected.")
        return
    
    print("\n" + "="*80)
    print("Transformer Block Timing Statistics (ms)")
    print("="*80)
    
    # Group by operation type
    operations = {}
    for key, data in stats.items():
        parts = key.split("_")
        if len(parts) >= 2:
            op_type = "_".join(parts[1:])  # Everything after layer_X
            if op_type not in operations:
                operations[op_type] = []
            operations[op_type].append((key, data))
    
    for op_type in sorted(operations.keys()):
        print(f"\n{op_type}:")
        for key, data in sorted(operations[op_type]):
            print(f"  {key:50s} | Mean: {data['mean']:8.4f} ms | "
                  f"Total: {data['total']:10.4f} ms | Count: {data['count']:4d}")
    
    print("="*80)
    def qkv_projection_kernel(*args, **kwargs):
        raise NotImplementedError("Kernel ops not available")
    def qk_norm_perhead_kernel(*args, **kwargs):
        raise NotImplementedError("Kernel ops not available")
    def rmsnorm_gated_residual_kernel(*args, **kwargs):
        raise NotImplementedError("Kernel ops not available")
    def swiglu_kernel(*args, **kwargs):
        raise NotImplementedError("Kernel ops not available")
    def check_kernels_available():
        return False


ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )

        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        with torch.amp.autocast("cuda", enabled=False):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
            )
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        weight_dtype = self.mlp[0].weight.dtype
        compute_dtype = getattr(self.mlp[0], "compute_dtype", None)
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        elif compute_dtype is not None:
            t_freq = t_freq.to(compute_dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class ZSingleStreamAttnProcessor:
    """
    Processor for Z-Image single stream attention that adapts the existing Attention class to match the behavior of the
    original Z-ImageAttention module.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "ZSingleStreamAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        precomputed_q: Optional[torch.Tensor] = None,
        precomputed_k: Optional[torch.Tensor] = None,
        precomputed_v: Optional[torch.Tensor] = None,
        already_qk_normed: bool = False,
    ) -> torch.Tensor:
        # Optional fast-path for precomputed Q, K, V coming from custom kernels.
        if precomputed_q is not None and precomputed_k is not None and precomputed_v is not None:
            # Custom kernel path: Q, K, V are already shaped as [B, S, H, head_dim]
            query = precomputed_q
            key = precomputed_k
            value = precomputed_v
        else:
            # Standard path: project from hidden_states with separate linear layers, then
            # reshape from [B, S, D] -> [B, S, H, head_dim].
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        if query.ndim == 3:
            query = query.unflatten(-1, (attn.heads, -1))
        if key.ndim == 3:
            key = key.unflatten(-1, (attn.heads, -1))
        if value.ndim == 3:
            value = value.unflatten(-1, (attn.heads, -1))

        # Apply Norms (unless we've already applied them in a custom kernel path)
        if not already_qk_normed:
            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

        # Apply RoPE
        def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
            with torch.amp.autocast("cuda", enabled=False):
                x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
                freqs_cis = freqs_cis.unsqueeze(2)
                x_out = torch.view_as_real(x * freqs_cis).flatten(3)
                return x_out.type_as(x_in)  # todo

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        # Cast to correct dtype
        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        # From [batch, seq_len] to [batch, 1, 1, seq_len] -> broadcast to [batch, heads, seq_len, seq_len]
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        # Compute joint attention
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        # Reshape back
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:  # dropout
            output = attn.to_out[1](output)

        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


@maybe_allow_in_graph
class ZImageTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation: bool = True,
        use_custom_kernels: bool = False,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.dim = dim
        self.head_dim = dim // n_heads
        self.norm_eps = norm_eps

        # flag is checked at runtime against kernel availability
        self._use_custom_kernels_flag = use_custom_kernels

        # Refactored to use diffusers Attention with custom processor
        self.attention = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // n_heads,
            heads=n_heads,
            qk_norm="rms_norm" if qk_norm else None,
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=ZSingleStreamAttnProcessor(),
        )

        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)

        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True))

    @property
    def use_custom_kernels(self) -> bool:
        """Check if custom kernels should be used (flag + availability)."""
        return self._use_custom_kernels_flag and KERNEL_OPS_AVAILABLE and check_kernels_available()

    @use_custom_kernels.setter
    def use_custom_kernels(self, value: bool):
        self._use_custom_kernels_flag = value

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        if ENABLE_TIMING:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
        
        if self.modulation:
            assert adaln_input is not None

            # ------------------------------------------------------------------
            # 1) AdaLN modulation
            # ------------------------------------------------------------------
            if ENABLE_TIMING:
                torch.cuda.synchronize()
                adaln_start = torch.cuda.Event(enable_timing=True)
                adaln_end = torch.cuda.Event(enable_timing=True)
                adaln_start.record()
            
            if self.use_custom_kernels:
                weight = self.adaLN_modulation[0].weight  # [4*dim, emb_dim]
                bias = self.adaLN_modulation[0].bias      # [4*dim]
                scale_msa, gate_msa, scale_mlp, gate_mlp = adaln_modulation_kernel(
                    adaln_input, weight, bias
                )
            else:
                adaln_out = self.adaLN_modulation(adaln_input)          # [B, 4*dim]
                adaln_out = adaln_out.unsqueeze(1)                      # [B, 1, 4*dim]
                scale_msa, gate_msa, scale_mlp, gate_mlp = adaln_out.chunk(4, dim=2)
                gate_msa = torch.tanh(gate_msa)
                gate_mlp = torch.tanh(gate_mlp)
                scale_msa = 1.0 + scale_msa
                scale_mlp = 1.0 + scale_mlp
            
            if ENABLE_TIMING:
                adaln_end.record()
                torch.cuda.synchronize()
                adaln_time = adaln_start.elapsed_time(adaln_end)
                key = f"layer_{self.layer_id}_adaln_{'kernel' if self.use_custom_kernels else 'pytorch'}"
                _timing_stats.setdefault(key, []).append(adaln_time)

            # ------------------------------------------------------------------
            # 2) Attention block
            # ------------------------------------------------------------------
            if ENABLE_TIMING:
                attn_start = torch.cuda.Event(enable_timing=True)
                attn_end = torch.cuda.Event(enable_timing=True)
                attn_start.record()
            
            if self.use_custom_kernels:
                # RMSNorm + scale
                rmsnorm_start = torch.cuda.Event(enable_timing=True) if ENABLE_TIMING else None
                rmsnorm_end = torch.cuda.Event(enable_timing=True) if ENABLE_TIMING else None
                if ENABLE_TIMING:
                    torch.cuda.synchronize()
                    rmsnorm_start.record()
                
                attn_input = rmsnorm_scale_kernel(
                    x, self.attention_norm1.weight, scale_msa.squeeze(1), eps=self.norm_eps
                )
                
                if ENABLE_TIMING:
                    rmsnorm_end.record()
                    torch.cuda.synchronize()
                    rmsnorm_time = rmsnorm_start.elapsed_time(rmsnorm_end)
                    key = f"layer_{self.layer_id}_rmsnorm_scale_{'kernel' if self.use_custom_kernels else 'pytorch'}"
                    _timing_stats.setdefault(key, []).append(rmsnorm_time)

                # Fused QKV + per-head QK norm (custom kernels)
                qkv_start = torch.cuda.Event(enable_timing=True) if ENABLE_TIMING else None
                qkv_end = torch.cuda.Event(enable_timing=True) if ENABLE_TIMING else None
                if ENABLE_TIMING:
                    torch.cuda.synchronize()
                    qkv_start.record()
                
                q, k, v = qkv_projection_kernel(
                    attn_input,
                    self.attention.to_q.weight,
                    self.attention.to_k.weight,
                    self.attention.to_v.weight,
                )
                q = q.unflatten(-1, (self.attention.heads, -1))
                k = k.unflatten(-1, (self.attention.heads, -1))
                v = v.unflatten(-1, (self.attention.heads, -1))
                if self.attention.norm_q is not None:
                    q, k = qk_norm_perhead_kernel(
                        q,
                        k,
                        self.attention.norm_q.weight,
                        self.attention.norm_k.weight,
                        eps=self.norm_eps,
                    )
                
                if ENABLE_TIMING:
                    qkv_end.record()
                    torch.cuda.synchronize()
                    qkv_time = qkv_start.elapsed_time(qkv_end)
                    key = f"layer_{self.layer_id}_qkv_norm_{'kernel' if self.use_custom_kernels else 'pytorch'}"
                    _timing_stats.setdefault(key, []).append(qkv_time)
                
                # Attention processor now consumes the precomputed Q, K, V and skips
                # its own QKV projections and Q/K norms.
                attn_processor_start = torch.cuda.Event(enable_timing=True) if ENABLE_TIMING else None
                attn_processor_end = torch.cuda.Event(enable_timing=True) if ENABLE_TIMING else None
                if ENABLE_TIMING:
                    torch.cuda.synchronize()
                    attn_processor_start.record()
                
                attn_out = self.attention(
                    attn_input,
                    attention_mask=attn_mask,
                    freqs_cis=freqs_cis,
                    precomputed_q=q,
                    precomputed_k=k,
                    precomputed_v=v,
                    already_qk_normed=True,
                )
                
                if ENABLE_TIMING:
                    attn_processor_end.record()
                    torch.cuda.synchronize()
                    attn_processor_time = attn_processor_start.elapsed_time(attn_processor_end)
                    key = f"layer_{self.layer_id}_attention_processor"
                    _timing_stats.setdefault(key, []).append(attn_processor_time)

                # RMSNorm + gated residual
                gated_res_start = torch.cuda.Event(enable_timing=True) if ENABLE_TIMING else None
                gated_res_end = torch.cuda.Event(enable_timing=True) if ENABLE_TIMING else None
                if ENABLE_TIMING:
                    torch.cuda.synchronize()
                    gated_res_start.record()
                
                x = rmsnorm_gated_residual_kernel(
                    attn_out, x, self.attention_norm2.weight, gate_msa.squeeze(1), eps=self.norm_eps
                )
                
                if ENABLE_TIMING:
                    gated_res_end.record()
                    torch.cuda.synchronize()
                    gated_res_time = gated_res_start.elapsed_time(gated_res_end)
                    key = f"layer_{self.layer_id}_gated_residual_{'kernel' if self.use_custom_kernels else 'pytorch'}"
                    _timing_stats.setdefault(key, []).append(gated_res_time)
            else:
                attn_norm1_out = self.attention_norm1(x)
                attn_input = attn_norm1_out * scale_msa

                attn_out = self.attention(attn_input, attention_mask=attn_mask, freqs_cis=freqs_cis)

                attn_norm2_out = self.attention_norm2(attn_out)
                gated_attn = gate_msa * attn_norm2_out
                x = x + gated_attn
            
            if ENABLE_TIMING:
                attn_end.record()
                torch.cuda.synchronize()
                attn_time = attn_start.elapsed_time(attn_end)
                key = f"layer_{self.layer_id}_attention_block_{'kernel' if self.use_custom_kernels else 'pytorch'}"
                _timing_stats.setdefault(key, []).append(attn_time)

            # ------------------------------------------------------------------
            # 3) FFN block (SwiGLU)
            # ------------------------------------------------------------------
            if ENABLE_TIMING:
                ffn_start = torch.cuda.Event(enable_timing=True)
                ffn_end = torch.cuda.Event(enable_timing=True)
                ffn_start.record()
            
            if self.use_custom_kernels:
                ffn_rmsnorm_start = torch.cuda.Event(enable_timing=True) if ENABLE_TIMING else None
                ffn_rmsnorm_end = torch.cuda.Event(enable_timing=True) if ENABLE_TIMING else None
                if ENABLE_TIMING:
                    torch.cuda.synchronize()
                    ffn_rmsnorm_start.record()
                
                ffn_input = rmsnorm_scale_kernel(
                    x, self.ffn_norm1.weight, scale_mlp.squeeze(1), eps=self.norm_eps
                )
                
                if ENABLE_TIMING:
                    ffn_rmsnorm_end.record()
                    torch.cuda.synchronize()
                    ffn_rmsnorm_time = ffn_rmsnorm_start.elapsed_time(ffn_rmsnorm_end)
                    key = f"layer_{self.layer_id}_ffn_rmsnorm_{'kernel' if self.use_custom_kernels else 'pytorch'}"
                    _timing_stats.setdefault(key, []).append(ffn_rmsnorm_time)
                
                w2_out = swiglu_kernel(
                    ffn_input,
                    self.feed_forward.w1.weight,
                    self.feed_forward.w2.weight,
                    self.feed_forward.w3.weight,
                )
                
                ffn_gated_res_start = torch.cuda.Event(enable_timing=True) if ENABLE_TIMING else None
                ffn_gated_res_end = torch.cuda.Event(enable_timing=True) if ENABLE_TIMING else None
                if ENABLE_TIMING:
                    torch.cuda.synchronize()
                    ffn_gated_res_start.record()
                
                x = rmsnorm_gated_residual_kernel(
                    w2_out, x, self.ffn_norm2.weight, gate_mlp.squeeze(1), eps=self.norm_eps
                )
                
                if ENABLE_TIMING:
                    ffn_gated_res_end.record()
                    torch.cuda.synchronize()
                    ffn_gated_res_time = ffn_gated_res_start.elapsed_time(ffn_gated_res_end)
                    key = f"layer_{self.layer_id}_ffn_gated_residual_{'kernel' if self.use_custom_kernels else 'pytorch'}"
                    _timing_stats.setdefault(key, []).append(ffn_gated_res_time)
            else:
                ffn_norm1_out = self.ffn_norm1(x)
                ffn_input = ffn_norm1_out * scale_mlp

                w1_out = self.feed_forward.w1(ffn_input)
                w3_out = self.feed_forward.w3(ffn_input)
                silu_gating = self.feed_forward._forward_silu_gating(w1_out, w3_out)
                w2_out = self.feed_forward.w2(silu_gating)

                ffn_norm2_out = self.ffn_norm2(w2_out)
                gated_ffn = gate_mlp * ffn_norm2_out
                x = x + gated_ffn
            
            if ENABLE_TIMING:
                ffn_end.record()
                torch.cuda.synchronize()
                ffn_time = ffn_start.elapsed_time(ffn_end)
                key = f"layer_{self.layer_id}_ffn_block_{'kernel' if self.use_custom_kernels else 'pytorch'}"
                _timing_stats.setdefault(key, []).append(ffn_time)

        else:
            # ------------------------------------------------------------------
            # No modulation path
            # ------------------------------------------------------------------
            # Attention block
            if self.use_custom_kernels:
                attn_norm1_out = self.attention_norm1(x)
                q, k, v = qkv_projection_kernel(
                    attn_norm1_out,
                    self.attention.to_q.weight,
                    self.attention.to_k.weight,
                    self.attention.to_v.weight,
                )
                q = q.unflatten(-1, (self.attention.heads, -1))
                k = k.unflatten(-1, (self.attention.heads, -1))
                v = v.unflatten(-1, (self.attention.heads, -1))
                if self.attention.norm_q is not None:
                    q, k = qk_norm_perhead_kernel(
                        q,
                        k,
                        self.attention.norm_q.weight,
                        self.attention.norm_k.weight,
                        eps=self.norm_eps,
                    )

                attn_out = self.attention(
                    attn_norm1_out,
                    attention_mask=attn_mask,
                    freqs_cis=freqs_cis,
                    precomputed_q=q,
                    precomputed_k=k,
                    precomputed_v=v,
                    already_qk_normed=True,
                )
                attn_norm2_out = self.attention_norm2(attn_out)
                x = x + attn_norm2_out
            else:
                attn_norm1_out = self.attention_norm1(x)
                attn_out = self.attention(attn_norm1_out, attention_mask=attn_mask, freqs_cis=freqs_cis)
                attn_norm2_out = self.attention_norm2(attn_out)
                x = x + attn_norm2_out

            # FFN block
            if self.use_custom_kernels:
                ffn_norm1_out = self.ffn_norm1(x)
                w2_out = swiglu_kernel(
                    ffn_norm1_out,
                    self.feed_forward.w1.weight,
                    self.feed_forward.w2.weight,
                    self.feed_forward.w3.weight,
                )
                ffn_norm2_out = self.ffn_norm2(w2_out)
                x = x + ffn_norm2_out
            else:
                ffn_norm1_out = self.ffn_norm1(x)
                w1_out = self.feed_forward.w1(ffn_norm1_out)
                w3_out = self.feed_forward.w3(ffn_norm1_out)
                silu_gating = self.feed_forward._forward_silu_gating(w1_out, w3_out)
                w2_out = self.feed_forward.w2(silu_gating)
                ffn_norm2_out = self.ffn_norm2(w2_out)
                x = x + ffn_norm2_out

        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c):
        scale = 1.0 + self.adaLN_modulation(c)
        x = self.norm_final(x) * scale.unsqueeze(1)
        x = self.linear(x)
        return x


class RopeEmbedder:
    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: List[int] = (16, 56, 56),
        axes_lens: List[int] = (64, 128, 128),
    ):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        assert len(axes_dims) == len(axes_lens), "axes_dims and axes_lens must have the same length"
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim: List[int], end: List[int], theta: float = 256.0):
        with torch.device("cpu"):
            freqs_cis = []
            for i, (d, e) in enumerate(zip(dim, end)):
                freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
                timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)  # complex64
                freqs_cis.append(freqs_cis_i)

            return freqs_cis

    def __call__(self, ids: torch.Tensor):
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device

        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]
        else:
            # Ensure freqs_cis are on the same device as ids
            if self.freqs_cis[0].device != device:
                self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]

        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            result.append(self.freqs_cis[i][index])
        return torch.cat(result, dim=-1)


class ZImageTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["ZImageTransformerBlock"]
    _repeated_blocks = ["ZImageTransformerBlock"]
    _skip_layerwise_casting_patterns = ["t_embedder", "cap_embedder"]  # precision sensitive layers

    @register_to_config
    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[32, 48, 48],
        axes_lens=[1024, 512, 512],
        use_custom_kernels: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.dim = dim
        self.n_heads = n_heads

        self.rope_theta = rope_theta
        self.t_scale = t_scale
        self.gradient_checkpointing = False

        assert len(all_patch_size) == len(all_f_patch_size)

        all_x_embedder = {}
        all_final_layer = {}
        for patch_idx, (patch_size, f_patch_size) in enumerate(zip(all_patch_size, all_f_patch_size)):
            x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * in_channels, dim, bias=True)
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder

            final_layer = FinalLayer(dim, patch_size * patch_size * f_patch_size * self.out_channels)
            all_final_layer[f"{patch_size}-{f_patch_size}"] = final_layer

        self.all_x_embedder = nn.ModuleDict(all_x_embedder)
        self.all_final_layer = nn.ModuleDict(all_final_layer)
        self.noise_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    1000 + layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                    use_custom_kernels=use_custom_kernels,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                    use_custom_kernels=use_custom_kernels,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.cap_embedder = nn.Sequential(RMSNorm(cap_feat_dim, eps=norm_eps), nn.Linear(cap_feat_dim, dim, bias=True))

        self.x_pad_token = nn.Parameter(torch.empty((1, dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))

        self.layers = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, use_custom_kernels=use_custom_kernels
                )
                for layer_id in range(n_layers)
            ]
        )
        head_dim = dim // n_heads
        assert head_dim == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens

        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)

    def unpatchify(self, x: List[torch.Tensor], size: List[Tuple], patch_size, f_patch_size) -> List[torch.Tensor]:
        pH = pW = patch_size
        pF = f_patch_size
        bsz = len(x)
        assert len(size) == bsz
        for i in range(bsz):
            F, H, W = size[i]
            ori_len = (F // pF) * (H // pH) * (W // pW)
            # "f h w pf ph pw c -> c (f pf) (h ph) (w pw)"
            x[i] = (
                x[i][:ori_len]
                .view(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
                .permute(6, 0, 3, 1, 4, 2, 5)
                .reshape(self.out_channels, F, H, W)
            )
        return x

    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)

        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def patchify_and_embed(
        self,
        all_image: List[torch.Tensor],
        all_cap_feats: List[torch.Tensor],
        patch_size: int,
        f_patch_size: int,
    ):
        pH = pW = patch_size
        pF = f_patch_size
        device = all_image[0].device

        all_image_out = []
        all_image_size = []
        all_image_pos_ids = []
        all_image_pad_mask = []
        all_cap_pos_ids = []
        all_cap_pad_mask = []
        all_cap_feats_out = []

        for i, (image, cap_feat) in enumerate(zip(all_image, all_cap_feats)):
            ### Process Caption
            cap_ori_len = len(cap_feat)
            cap_padding_len = (-cap_ori_len) % SEQ_MULTI_OF
            # padded position ids
            cap_padded_pos_ids = self.create_coordinate_grid(
                size=(cap_ori_len + cap_padding_len, 1, 1),
                start=(1, 0, 0),
                device=device,
            ).flatten(0, 2)
            all_cap_pos_ids.append(cap_padded_pos_ids)
            # pad mask
            cap_pad_mask = torch.cat(
                [
                    torch.zeros((cap_ori_len,), dtype=torch.bool, device=device),
                    torch.ones((cap_padding_len,), dtype=torch.bool, device=device),
                ],
                dim=0,
            )
            all_cap_pad_mask.append(
                cap_pad_mask if cap_padding_len > 0 else torch.zeros((cap_ori_len,), dtype=torch.bool, device=device)
            )

            # padded feature
            cap_padded_feat = torch.cat([cap_feat, cap_feat[-1:].repeat(cap_padding_len, 1)], dim=0)
            all_cap_feats_out.append(cap_padded_feat)

            ### Process Image
            C, F, H, W = image.size()
            all_image_size.append((F, H, W))
            F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW

            image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
            # "c f pf h ph w pw -> (f h w) (pf ph pw c)"
            image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)

            image_ori_len = len(image)
            image_padding_len = (-image_ori_len) % SEQ_MULTI_OF

            image_ori_pos_ids = self.create_coordinate_grid(
                size=(F_tokens, H_tokens, W_tokens),
                start=(cap_ori_len + cap_padding_len + 1, 0, 0),
                device=device,
            ).flatten(0, 2)
            image_padded_pos_ids = torch.cat(
                [
                    image_ori_pos_ids,
                    self.create_coordinate_grid(size=(1, 1, 1), start=(0, 0, 0), device=device)
                    .flatten(0, 2)
                    .repeat(image_padding_len, 1),
                ],
                dim=0,
            )
            all_image_pos_ids.append(image_padded_pos_ids if image_padding_len > 0 else image_ori_pos_ids)
            # pad mask
            image_pad_mask = torch.cat(
                [
                    torch.zeros((image_ori_len,), dtype=torch.bool, device=device),
                    torch.ones((image_padding_len,), dtype=torch.bool, device=device),
                ],
                dim=0,
            )
            all_image_pad_mask.append(
                image_pad_mask
                if image_padding_len > 0
                else torch.zeros((image_ori_len,), dtype=torch.bool, device=device)
            )
            # padded feature
            image_padded_feat = torch.cat(
                [image, image[-1:].repeat(image_padding_len, 1)],
                dim=0,
            )
            all_image_out.append(image_padded_feat if image_padding_len > 0 else image)

        return (
            all_image_out,
            all_cap_feats_out,
            all_image_size,
            all_image_pos_ids,
            all_cap_pos_ids,
            all_image_pad_mask,
            all_cap_pad_mask,
        )

    def forward(
        self,
        x: List[torch.Tensor],
        t,
        cap_feats: List[torch.Tensor],
        patch_size=2,
        f_patch_size=1,
        return_dict: bool = True,
    ):
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size

        bsz = len(x)
        device = x[0].device
        t = t * self.t_scale
        t = self.t_embedder(t)

        (
            x,
            cap_feats,
            x_size,
            x_pos_ids,
            cap_pos_ids,
            x_inner_pad_mask,
            cap_inner_pad_mask,
        ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

        # x embed & refine
        x_item_seqlens = [len(_) for _ in x]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in x_item_seqlens)
        x_max_item_seqlen = max(x_item_seqlens)

        x = torch.cat(x, dim=0)
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)

        # Match t_embedder output dtype to x for layerwise casting compatibility
        adaln_input = t.type_as(x)
        x[torch.cat(x_inner_pad_mask)] = self.x_pad_token
        x = list(x.split(x_item_seqlens, dim=0))
        x_freqs_cis = list(self.rope_embedder(torch.cat(x_pos_ids, dim=0)).split([len(_) for _ in x_pos_ids], dim=0))

        x = pad_sequence(x, batch_first=True, padding_value=0.0)
        x_freqs_cis = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)
        # Clarify the length matches to satisfy Dynamo due to "Symbolic Shape Inference" to avoid compilation errors
        x_freqs_cis = x_freqs_cis[:, : x.shape[1]]

        x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(x_item_seqlens):
            x_attn_mask[i, :seq_len] = 1

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for layer in self.noise_refiner:
                x = self._gradient_checkpointing_func(layer, x, x_attn_mask, x_freqs_cis, adaln_input)
        else:
            for layer in self.noise_refiner:
                x = layer(x, x_attn_mask, x_freqs_cis, adaln_input)

        # cap embed & refine
        cap_item_seqlens = [len(_) for _ in cap_feats]
        cap_max_item_seqlen = max(cap_item_seqlens)

        cap_feats = torch.cat(cap_feats, dim=0)
        cap_feats = self.cap_embedder(cap_feats)
        cap_feats[torch.cat(cap_inner_pad_mask)] = self.cap_pad_token
        cap_feats = list(cap_feats.split(cap_item_seqlens, dim=0))
        cap_freqs_cis = list(
            self.rope_embedder(torch.cat(cap_pos_ids, dim=0)).split([len(_) for _ in cap_pos_ids], dim=0)
        )

        cap_feats = pad_sequence(cap_feats, batch_first=True, padding_value=0.0)
        cap_freqs_cis = pad_sequence(cap_freqs_cis, batch_first=True, padding_value=0.0)
        # Clarify the length matches to satisfy Dynamo due to "Symbolic Shape Inference" to avoid compilation errors
        cap_freqs_cis = cap_freqs_cis[:, : cap_feats.shape[1]]

        cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(cap_item_seqlens):
            cap_attn_mask[i, :seq_len] = 1

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for layer in self.context_refiner:
                cap_feats = self._gradient_checkpointing_func(layer, cap_feats, cap_attn_mask, cap_freqs_cis)
        else:
            for layer in self.context_refiner:
                cap_feats = layer(cap_feats, cap_attn_mask, cap_freqs_cis)

        # unified
        unified = []
        unified_freqs_cis = []
        for i in range(bsz):
            x_len = x_item_seqlens[i]
            cap_len = cap_item_seqlens[i]
            unified.append(torch.cat([x[i][:x_len], cap_feats[i][:cap_len]]))
            unified_freqs_cis.append(torch.cat([x_freqs_cis[i][:x_len], cap_freqs_cis[i][:cap_len]]))
        unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens)]
        assert unified_item_seqlens == [len(_) for _ in unified]
        unified_max_item_seqlen = max(unified_item_seqlens)

        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs_cis = pad_sequence(unified_freqs_cis, batch_first=True, padding_value=0.0)
        unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(unified_item_seqlens):
            unified_attn_mask[i, :seq_len] = 1

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for layer in self.layers:
                unified = self._gradient_checkpointing_func(
                    layer, unified, unified_attn_mask, unified_freqs_cis, adaln_input
                )
        else:
            for layer in self.layers:
                unified = layer(unified, unified_attn_mask, unified_freqs_cis, adaln_input)

        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)
        unified = list(unified.unbind(dim=0))
        x = self.unpatchify(unified, x_size, patch_size, f_patch_size)

        if not return_dict:
            return (x,)

        return Transformer2DModelOutput(sample=x)
