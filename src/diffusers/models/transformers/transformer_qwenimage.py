# Copyright 2025 Qwen-Image Team, The HuggingFace Team. All rights reserved.
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

import functools
import math
import os

from math import prod
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# NVTX profiling support — enabled by QWEN_NVTX=1 environment variable.
# These markers appear in Nsight Systems timelines for per-block visibility.
_QWEN_NVTX_ENABLED = os.environ.get("QWEN_NVTX", "0") == "1"
if _QWEN_NVTX_ENABLED:
    try:
        import torch.cuda.nvtx as nvtx
        _nvtx_range_push = nvtx.range_push
        _nvtx_range_pop = nvtx.range_pop
    except (ImportError, AttributeError):
        _QWEN_NVTX_ENABLED = False
        _nvtx_range_push = lambda msg: None  # noqa: E731
        _nvtx_range_pop = lambda: None  # noqa: E731
else:
    _nvtx_range_push = lambda msg: None  # noqa: E731
    _nvtx_range_pop = lambda: None  # noqa: E731

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import apply_lora_scale, deprecate, logging
from ...utils.torch_utils import maybe_allow_in_graph
from .._modeling_parallel import ContextParallelInput, ContextParallelOutput
from ..attention import AttentionMixin, FeedForward
from ..attention_dispatch import dispatch_attention_fn
from ..attention_processor import Attention
from ..cache_utils import CacheMixin
from ..embeddings import TimestepEmbedding, Timesteps
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNormContinuous, RMSNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _complex_freqs_to_real(freqs_complex: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert complex RoPE frequencies to (cos, sin) real format.

    The complex tensor has shape [S, D/2] where each element is e^(iθ) = cos(θ) + i*sin(θ).
    The real format returns (cos, sin) each of shape [S, D] with values interleaved
    (each frequency repeated for the pair), matching the layout expected by
    ``apply_rotary_emb_qwen(..., use_real=True, use_real_unbind_dim=-1)``.

    This conversion eliminates ``torch.view_as_complex``/``torch.view_as_real``
    graph breaks that prevent torch.compile from consolidating the block loop.
    """
    cos_half = freqs_complex.real   # [S, D/2]
    sin_half = freqs_complex.imag   # [S, D/2]
    # Interleave: [S, D/2] -> [S, D/2, 2] -> [S, D]
    cos_full = torch.stack([cos_half, cos_half], dim=-1).flatten(-2)
    sin_full = torch.stack([sin_half, sin_half], dim=-1).flatten(-2)
    return cos_full, sin_full


# Optional import of QWEN-specific fused kernels. This keeps the transformer
# usable even when the custom CUDA extension is not built.
try:
    from .kernel_ops_qwen import (  # type: ignore[import]
        _KERNEL_DEBUG as _QWEN_KERNEL_DEBUG,
        check_kernels_available as _qwen_kernels_available,
        qwen_qk_norm_perhead_kernel as _qwen_qk_norm_perhead_kernel,
        qwen_qk_norm_rope_3d_fused_kernel as _qwen_qk_norm_rope_3d_fused_kernel,
        qwen_layernorm_modulate_kernel as _qwen_layernorm_modulate_kernel,
        qwen_layernorm_modulate_gemm_up_kernel as _qwen_lnm_gemm_up_kernel,
        qwen_rope_3d_fused_kernel as _qwen_rope_3d_fused_kernel,
    )
except Exception:  # pragma: no cover - purely defensive
    _QWEN_KERNEL_DEBUG = False
    _qwen_kernels_available = lambda: False  # type: ignore[assignment]
    _qwen_qk_norm_perhead_kernel = None
    _qwen_qk_norm_rope_3d_fused_kernel = None
    _qwen_layernorm_modulate_kernel = None
    _qwen_lnm_gemm_up_kernel = None
    _qwen_rope_3d_fused_kernel = None

# Cache the kernel availability check once at import time instead of calling
# the function ~2,160 times per image (4 calls × 540 block forward passes).
_QWEN_KERNELS_AVAILABLE: bool = _qwen_kernels_available()

# Optional import of Triton fused GEMM + gate + residual kernel.
# Eliminates the global-memory round-trip between the output projection GEMM
# and the subsequent gate + residual epilogue.
try:
    from kernex.triton import fused_gemm_gate_residual as _fused_gemm_gate_residual
    _TRITON_FUSED_GEMM_AVAILABLE = True
except Exception:  # pragma: no cover
    _fused_gemm_gate_residual = None
    _TRITON_FUSED_GEMM_AVAILABLE = False


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent).to(timesteps.dtype)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: torch.Tensor | tuple[torch.Tensor],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, S, H, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if x.ndim == 4:
            # x: [B, S, H, D] — attention Q/K tensor with explicit head dim.
            # cos/sin: [S, D] -> [1, S, 1, D] to broadcast over B and H.
            cos = cos[None, :, None, :]
            sin = sin[None, :, None, :]
        else:
            # x: [B, S, D] — standard 3D layout (used by flux, etc.)
            cos = cos[None, None]
            sin = sin[None, None]

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit, qwen text RoPE
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [..., D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(-2)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [..., D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


def compute_text_seq_len_from_mask(
    encoder_hidden_states: torch.Tensor, encoder_hidden_states_mask: torch.Tensor | None
) -> tuple[int, torch.Tensor | None, torch.Tensor | None]:
    """
    Compute text sequence length without assuming contiguous masks. Returns length for RoPE and a normalized bool mask.
    """
    batch_size, text_seq_len = encoder_hidden_states.shape[:2]
    if encoder_hidden_states_mask is None:
        return text_seq_len, None, None

    if encoder_hidden_states_mask.shape[:2] != (batch_size, text_seq_len):
        raise ValueError(
            f"`encoder_hidden_states_mask` shape {encoder_hidden_states_mask.shape} must match "
            f"(batch_size, text_seq_len)=({batch_size}, {text_seq_len})."
        )

    if encoder_hidden_states_mask.dtype != torch.bool:
        encoder_hidden_states_mask = encoder_hidden_states_mask.to(torch.bool)

    position_ids = torch.arange(text_seq_len, device=encoder_hidden_states.device, dtype=torch.long)
    active_positions = torch.where(encoder_hidden_states_mask, position_ids, position_ids.new_zeros(()))
    has_active = encoder_hidden_states_mask.any(dim=1)
    per_sample_len = torch.where(
        has_active,
        active_positions.max(dim=1).values + 1,
        torch.as_tensor(text_seq_len, device=encoder_hidden_states.device),
    )
    return text_seq_len, per_sample_len, encoder_hidden_states_mask


class QwenTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim, use_additional_t_cond=False):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.use_additional_t_cond = use_additional_t_cond
        if use_additional_t_cond:
            self.addition_t_embedding = nn.Embedding(2, embedding_dim)

    def forward(self, timestep, hidden_states, addition_t_cond=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))  # (N, D)

        conditioning = timesteps_emb
        if self.use_additional_t_cond:
            if addition_t_cond is None:
                raise ValueError("When additional_t_cond is True, addition_t_cond must be provided.")
            addition_t_emb = self.addition_t_embedding(addition_t_cond)
            addition_t_emb = addition_t_emb.to(dtype=hidden_states.dtype)
            conditioning = conditioning + addition_t_emb

        return conditioning


class QwenEmbedRope(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        # persistent=False avoids complex-dtype serialisation issues while
        # ensuring buffers move with .to(device), eliminating per-forward
        # .to(device) calls that cause torch.compile graph breaks.
        self.register_buffer(
            "pos_freqs",
            torch.cat([
                self.rope_params(pos_index, self.axes_dim[0], self.theta),
                self.rope_params(pos_index, self.axes_dim[1], self.theta),
                self.rope_params(pos_index, self.axes_dim[2], self.theta),
            ], dim=1),
            persistent=False,
        )
        self.register_buffer(
            "neg_freqs",
            torch.cat([
                self.rope_params(neg_index, self.axes_dim[0], self.theta),
                self.rope_params(neg_index, self.axes_dim[1], self.theta),
                self.rope_params(neg_index, self.axes_dim[2], self.theta),
            ], dim=1),
            persistent=False,
        )

        self.scale_rope = scale_rope

    def rope_params(self, index, dim, theta=10000):
        """
        Args:
            index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        """
        assert dim % 2 == 0
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def forward(
        self,
        video_fhw: tuple[int, int, int, list[tuple[int, int, int]]],
        txt_seq_lens: list[int] | None = None,
        device: torch.device = None,
        max_txt_seq_len: int | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            video_fhw (`tuple[int, int, int]` or `list[tuple[int, int, int]]`):
                A list of 3 integers [frame, height, width] representing the shape of the video.
            txt_seq_lens (`list[int]`, *optional*, **Deprecated**):
                Deprecated parameter. Use `max_txt_seq_len` instead. If provided, the maximum value will be used.
            device: (`torch.device`, *optional*):
                The device on which to perform the RoPE computation.
            max_txt_seq_len (`int` or `torch.Tensor`, *optional*):
                The maximum text sequence length for RoPE computation. This should match the encoder hidden states
                sequence length. Can be either an int or a scalar tensor (for torch.compile compatibility).
        """
        # Handle deprecated txt_seq_lens parameter
        if txt_seq_lens is not None:
            deprecate(
                "txt_seq_lens",
                "0.39.0",
                "Passing `txt_seq_lens` is deprecated and will be removed in version 0.39.0. "
                "Please use `max_txt_seq_len` instead. "
                "The new parameter accepts a single int or tensor value representing the maximum text sequence length.",
                standard_warn=False,
            )
            if max_txt_seq_len is None:
                # Use max of txt_seq_lens for backward compatibility
                max_txt_seq_len = max(txt_seq_lens) if isinstance(txt_seq_lens, list) else txt_seq_lens

        if max_txt_seq_len is None:
            raise ValueError("Either `max_txt_seq_len` or `txt_seq_lens` (deprecated) must be provided.")

        # Validate batch inference with variable-sized images
        if isinstance(video_fhw, list) and len(video_fhw) > 1:
            # Check if all instances have the same size
            first_fhw = video_fhw[0]
            if not all(fhw == first_fhw for fhw in video_fhw):
                logger.warning(
                    "Batch inference with variable-sized images is not currently supported in QwenEmbedRope. "
                    "All images in the batch should have the same dimensions (frame, height, width). "
                    f"Detected sizes: {video_fhw}. Using the first image's dimensions {first_fhw} "
                    "for RoPE computation, which may lead to incorrect results for other images in the batch."
                )

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            # RoPE frequencies are cached via a lru_cache decorator on _compute_video_freqs
            video_freq = self._compute_video_freqs(frame, height, width, idx, device)
            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_txt_seq_len_int = int(max_txt_seq_len)
        txt_freqs_complex = self.pos_freqs[max_vid_index : max_vid_index + max_txt_seq_len_int, ...]
        vid_freqs_complex = torch.cat(vid_freqs, dim=0)

        # Return complex tensors directly. The attention processor auto-detects
        # the format (complex tensor vs (cos, sin) tuple) and selects the
        # appropriate RoPE path. Complex arithmetic uses zero-copy views
        # (view_as_complex/view_as_real) and works correctly under both eager
        # mode and torch.compile (including reduce-overhead / CUDA graphs).
        return vid_freqs_complex, txt_freqs_complex

    @functools.lru_cache(maxsize=128)
    def _compute_video_freqs(
        self, frame: int, height: int, width: int, idx: int = 0, device: torch.device = None
    ) -> torch.Tensor:
        seq_lens = frame * height * width
        pos_freqs = self.pos_freqs
        neg_freqs = self.neg_freqs

        freqs_pos = pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()


class QwenEmbedLayer3DRope(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        # persistent=False avoids complex-dtype serialisation issues while
        # ensuring buffers move with .to(device), eliminating per-forward
        # .to(device) calls that cause torch.compile graph breaks.
        self.register_buffer(
            "pos_freqs",
            torch.cat([
                self.rope_params(pos_index, self.axes_dim[0], self.theta),
                self.rope_params(pos_index, self.axes_dim[1], self.theta),
                self.rope_params(pos_index, self.axes_dim[2], self.theta),
            ], dim=1),
            persistent=False,
        )
        self.register_buffer(
            "neg_freqs",
            torch.cat([
                self.rope_params(neg_index, self.axes_dim[0], self.theta),
                self.rope_params(neg_index, self.axes_dim[1], self.theta),
                self.rope_params(neg_index, self.axes_dim[2], self.theta),
            ], dim=1),
            persistent=False,
        )

        self.scale_rope = scale_rope

    def rope_params(self, index, dim, theta=10000):
        """
        Args:
            index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        """
        assert dim % 2 == 0
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def forward(
        self,
        video_fhw: tuple[int, int, int, list[tuple[int, int, int]]],
        max_txt_seq_len: int | torch.Tensor,
        device: torch.device = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            video_fhw (`tuple[int, int, int]` or `list[tuple[int, int, int]]`):
                A list of 3 integers [frame, height, width] representing the shape of the video, or a list of layer
                structures.
            max_txt_seq_len (`int` or `torch.Tensor`):
                The maximum text sequence length for RoPE computation. This should match the encoder hidden states
                sequence length. Can be either an int or a scalar tensor (for torch.compile compatibility).
            device: (`torch.device`, *optional*):
                The device on which to perform the RoPE computation.
        """
        # Validate batch inference with variable-sized images
        # In Layer3DRope, the outer list represents batch, inner list/tuple represents layers
        if isinstance(video_fhw, list) and len(video_fhw) > 1:
            # Check if this is batch inference (list of layer lists/tuples)
            first_entry = video_fhw[0]
            if not all(entry == first_entry for entry in video_fhw):
                logger.warning(
                    "Batch inference with variable-sized images is not currently supported in QwenEmbedLayer3DRope. "
                    "All images in the batch should have the same layer structure. "
                    f"Detected sizes: {video_fhw}. Using the first image's layer structure {first_entry} "
                    "for RoPE computation, which may lead to incorrect results for other images in the batch."
                )

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        layer_num = len(video_fhw) - 1
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            if idx != layer_num:
                video_freq = self._compute_video_freqs(frame, height, width, idx, device)
            else:
                ### For the condition image, we set the layer index to -1
                video_freq = self._compute_condition_freqs(frame, height, width, device)
            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_vid_index = max(max_vid_index, layer_num)
        max_txt_seq_len_int = int(max_txt_seq_len)
        txt_freqs_complex = self.pos_freqs[max_vid_index : max_vid_index + max_txt_seq_len_int, ...]
        vid_freqs_complex = torch.cat(vid_freqs, dim=0)

        # Return complex tensors directly — see QwenEmbedRope.forward() comment.
        return vid_freqs_complex, txt_freqs_complex

    @functools.lru_cache(maxsize=None)
    def _compute_video_freqs(self, frame, height, width, idx=0, device: torch.device = None):
        seq_lens = frame * height * width
        pos_freqs = self.pos_freqs
        neg_freqs = self.neg_freqs

        freqs_pos = pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()

    @functools.lru_cache(maxsize=None)
    def _compute_condition_freqs(self, frame, height, width, device: torch.device = None):
        seq_lens = frame * height * width
        pos_freqs = self.pos_freqs
        neg_freqs = self.neg_freqs

        freqs_pos = pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_neg[0][-1:].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()


# Pre-allocated joint Q/K/V buffer cache for eliminating torch.cat in attention.
# Keys are (batch_size, total_seq_len, num_heads, head_dim, device) tuples.
# This eliminates 3 torch.cat calls per block (55ms total across 60 blocks × 9 steps).
_joint_qkv_buffers: dict = {}


def _get_joint_buffer(
    key: str, batch_size: int, total_seq: int, num_heads: int, head_dim: int,
    dtype: torch.dtype, device: torch.device,
) -> torch.Tensor:
    """Get or allocate a pre-allocated joint Q/K/V buffer."""
    cache_key = (key, batch_size, total_seq, num_heads, head_dim, device)
    buf = _joint_qkv_buffers.get(cache_key)
    if buf is not None and buf.dtype == dtype:
        return buf
    buf = torch.empty(batch_size, total_seq, num_heads, head_dim, dtype=dtype, device=device)
    _joint_qkv_buffers[cache_key] = buf
    return buf


# Buffer cache for flattened attention output to avoid per-block allocation.
_attn_output_buffers: dict = {}


def _get_attn_flat_buffer(
    batch_size: int, total_seq: int, inner_dim: int,
    dtype: torch.dtype, device: torch.device,
) -> torch.Tensor:
    """Get or allocate a buffer for flattened attention output."""
    cache_key = (batch_size, total_seq, inner_dim, device)
    buf = _attn_output_buffers.get(cache_key)
    if buf is not None and buf.dtype == dtype:
        return buf
    buf = torch.empty(batch_size, total_seq, inner_dim, dtype=dtype, device=device)
    _attn_output_buffers[cache_key] = buf
    return buf


def clear_buffer_caches() -> None:
    """Free all module-level CUDA tensor caches to reclaim VRAM.

    Call this between pipeline loads (e.g., in aspect-ratio sweeps) to prevent
    OOM from accumulated buffers across different resolutions.
    """
    _joint_qkv_buffers.clear()
    _attn_output_buffers.clear()


class QwenDoubleStreamAttnProcessor2_0:
    """
    Attention processor for Qwen double-stream architecture, matching DoubleStreamLayerMegatron logic. This processor
    implements joint attention computation where text and image streams are processed together.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "QwenDoubleStreamAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        rope_grid_info: Optional[Dict[str, Any]] = None,
        skip_output_proj: bool = False,
    ) -> torch.FloatTensor:
        if encoder_hidden_states is None:
            raise ValueError("QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)")

        seq_txt = encoder_hidden_states.shape[1]

        # Compute QKV projections.  When fused (via fuse_qkv_projections()),
        # a single GEMM per stream replaces 3 separate ones.
        if attn.fused_projections:
            qkv = attn.to_qkv(hidden_states)
            split_size = qkv.shape[-1] // 3
            img_query, img_key, img_value = torch.split(qkv, split_size, dim=-1)

            encoder_qkv = attn.to_added_qkv(encoder_hidden_states)
            split_size = encoder_qkv.shape[-1] // 3
            txt_query, txt_key, txt_value = torch.split(encoder_qkv, split_size, dim=-1)
        else:
            img_query = attn.to_q(hidden_states)
            img_key = attn.to_k(hidden_states)
            img_value = attn.to_v(hidden_states)

            txt_query = attn.add_q_proj(encoder_hidden_states)
            txt_key = attn.add_k_proj(encoder_hidden_states)
            txt_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # Apply QK normalization + RoPE.
        # Three paths, in order of preference:
        #   1. Fused QK-Norm+RoPE kernel (single kernel for image Q/K: norm+rotate)
        #   2. Separate QK-Norm kernel + separate RoPE kernel
        #   3. PyTorch fallback
        use_qwen_kernels = (
            _QWEN_KERNELS_AVAILABLE
            and getattr(attn, "_qwen_use_custom_kernels", False)
        )

        _used_fused_qk_norm_rope = False

        if use_qwen_kernels:
            B, _, H, head_dim = img_query.shape
            if H == 24 and head_dim == 128:
                # --- Path 1: Fused QK-Norm + 3D RoPE for image stream ---
                if (
                    rope_grid_info is not None
                    and _qwen_qk_norm_rope_3d_fused_kernel is not None
                    and attn.norm_q is not None
                    and attn.norm_k is not None
                ):
                    q_weight = attn.norm_q.weight
                    k_weight = attn.norm_k.weight
                    eps_q = getattr(attn.norm_q, "eps", 1e-6)
                    fused_result = _qwen_qk_norm_rope_3d_fused_kernel(
                        img_query, img_key, q_weight, k_weight,
                        rope_grid_info["grid_frame"],
                        rope_grid_info["grid_height"],
                        rope_grid_info["grid_width"],
                        rope_grid_info["theta"],
                        eps_q,
                        rope_grid_info["axes_dim"],
                        rope_grid_info["height_offset"],
                        rope_grid_info["width_offset"],
                    )
                    if fused_result is not None:
                        img_query, img_key = fused_result
                        _used_fused_qk_norm_rope = True

                # Text stream QK-Norm (always separate — no 3D RoPE fusion)
                if attn.norm_added_q is not None and attn.norm_added_k is not None:
                    q_weight_txt = attn.norm_added_q.weight
                    k_weight_txt = attn.norm_added_k.weight
                    eps_txt = getattr(attn.norm_added_q, "eps", 1e-6)
                    if _qwen_qk_norm_perhead_kernel is not None:
                        txt_query, txt_key = _qwen_qk_norm_perhead_kernel(
                            txt_query, txt_key, q_weight_txt, k_weight_txt, eps=eps_txt
                        )
                    else:
                        if attn.norm_added_q is not None:
                            txt_query = attn.norm_added_q(txt_query)
                        if attn.norm_added_k is not None:
                            txt_key = attn.norm_added_k(txt_key)

                if not _used_fused_qk_norm_rope:
                    # --- Path 2: Separate QK-Norm kernel (image) ---
                    if attn.norm_q is not None and attn.norm_k is not None and _qwen_qk_norm_perhead_kernel is not None:
                        q_weight = attn.norm_q.weight
                        k_weight = attn.norm_k.weight
                        eps_q = getattr(attn.norm_q, "eps", 1e-6)
                        img_query, img_key = _qwen_qk_norm_perhead_kernel(
                            img_query, img_key, q_weight, k_weight, eps=eps_q
                        )
                    else:
                        if attn.norm_q is not None:
                            img_query = attn.norm_q(img_query)
                        if attn.norm_k is not None:
                            img_key = attn.norm_k(img_key)
            else:
                use_qwen_kernels = False

        if not use_qwen_kernels:
            if attn.norm_q is not None:
                img_query = attn.norm_q(img_query)
            if attn.norm_k is not None:
                img_key = attn.norm_k(img_key)
            if attn.norm_added_q is not None:
                txt_query = attn.norm_added_q(txt_query)
            if attn.norm_added_k is not None:
                txt_key = attn.norm_added_k(txt_key)

        # Apply RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb

            if not _used_fused_qk_norm_rope:
                # Image RoPE: try separate fused kernel, then fallback to PyTorch
                _used_fused_rope = False
                if rope_grid_info is not None and _qwen_rope_3d_fused_kernel is not None and use_qwen_kernels:
                    result_q = _qwen_rope_3d_fused_kernel(
                        img_query,
                        rope_grid_info["grid_frame"],
                        rope_grid_info["grid_height"],
                        rope_grid_info["grid_width"],
                        rope_grid_info["theta"],
                        rope_grid_info["axes_dim"],
                        rope_grid_info["height_offset"],
                        rope_grid_info["width_offset"],
                    )
                    if result_q is not None:
                        result_k = _qwen_rope_3d_fused_kernel(
                            img_key,
                            rope_grid_info["grid_frame"],
                            rope_grid_info["grid_height"],
                            rope_grid_info["grid_width"],
                            rope_grid_info["theta"],
                            rope_grid_info["axes_dim"],
                            rope_grid_info["height_offset"],
                            rope_grid_info["width_offset"],
                        )
                        if result_k is not None:
                            img_query = result_q
                            img_key = result_k
                            _used_fused_rope = True

                if not _used_fused_rope:
                    # Auto-detect format: tuple → (cos, sin) real, tensor → complex.
                    # Eager mode uses complex (faster zero-copy views); torch.compile
                    # uses real (avoids view_as_complex/view_as_real graph breaks).
                    _img_real = isinstance(img_freqs, tuple)
                    img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=_img_real, use_real_unbind_dim=-1)
                    img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=_img_real, use_real_unbind_dim=-1)

            # Text RoPE: auto-detect format like image RoPE above.
            _txt_real = isinstance(txt_freqs, tuple)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=_txt_real, use_real_unbind_dim=-1)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=_txt_real, use_real_unbind_dim=-1)

        # Combine text and image Q/K/V into joint tensors.
        # Under torch.compile: use torch.cat so the graph has no input mutations
        # (in-place slice writes on cached buffers prevent CUDA graph capture).
        # In eager mode: use pre-allocated buffers with slice writes to avoid
        # 3 torch.cat allocations + copies per block (~55ms across all blocks).
        if torch.compiler.is_compiling():
            joint_query = torch.cat([txt_query, img_query], dim=1)
            joint_key = torch.cat([txt_key, img_key], dim=1)
            joint_value = torch.cat([txt_value, img_value], dim=1)
        else:
            batch_size = img_query.shape[0]
            total_seq = seq_txt + img_query.shape[1]
            num_heads = img_query.shape[2]
            head_dim = img_query.shape[3]

            joint_query = _get_joint_buffer("q", batch_size, total_seq, num_heads, head_dim, img_query.dtype, img_query.device)
            joint_key = _get_joint_buffer("k", batch_size, total_seq, num_heads, head_dim, img_key.dtype, img_key.device)
            joint_value = _get_joint_buffer("v", batch_size, total_seq, num_heads, head_dim, img_value.dtype, img_value.device)

            joint_query[:, :seq_txt] = txt_query
            joint_query[:, seq_txt:] = img_query
            joint_key[:, :seq_txt] = txt_key
            joint_key[:, seq_txt:] = img_key
            joint_value[:, :seq_txt] = txt_value
            joint_value[:, seq_txt:] = img_value

        joint_hidden_states = dispatch_attention_fn(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        # Reshape back: [B, S, H, D] -> [B, S, H*D]
        # Use reshape instead of flatten to avoid allocation when contiguous
        B_out, S_out, H_out, D_out = joint_hidden_states.shape
        joint_hidden_states = joint_hidden_states.reshape(B_out, S_out, H_out * D_out)
        if joint_hidden_states.dtype != joint_query.dtype:
            joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part

        # Apply output projections (skipped when caller will fuse them)
        if skip_output_proj:
            return img_attn_output, txt_attn_output

        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output.contiguous())

        return img_attn_output, txt_attn_output


@maybe_allow_in_graph
class QwenImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        zero_cond_t: bool = False,
        use_custom_kernels: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        # Flag is checked at runtime against kernel availability in QWEN-specific kernel_ops.
        # Wiring to actual kernels is added separately so we keep this minimal and backwards compatible.
        self._use_custom_kernels_flag = use_custom_kernels

        # Image processing modules
        self.img_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),  # For scale, shift, gate for norm1 and norm2
        )
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,  # Enable cross attention for joint computation
            added_kv_proj_dim=dim,  # Enable added KV projections for text stream
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=QwenDoubleStreamAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=eps,
        )
        # Mark this attention module so the processor knows whether it is
        # allowed to use QWEN-specific fused CUDA kernels.
        self.attn._qwen_use_custom_kernels = use_custom_kernels
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # Text processing modules
        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),  # For scale, shift, gate for norm1 and norm2
        )
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        # Text doesn't need separate attention - it's handled by img_attn joint computation
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.zero_cond_t = zero_cond_t
        # Pre-compute kernel availability flags once, updated when use_custom_kernels changes.
        self._update_kernel_availability_cache()

    def _update_kernel_availability_cache(self) -> None:
        """Pre-compute kernel availability flags to avoid per-forward boolean evaluation."""
        self._cached_use_fused_lnm = (
            self._use_custom_kernels_flag
            and _qwen_layernorm_modulate_kernel is not None
            and _QWEN_KERNELS_AVAILABLE
        )
        self._cached_use_fused_lnm_txt = self._cached_use_fused_lnm
        # DISABLED: Triton autotuned tl.dot achieves ~70-85% of cuBLAS for
        # these shapes (M=4352, K=3072/12288, N=3072).  The gate+residual
        # epilogue fusion saves ~2ms total, but the GEMM regression costs
        # ~25-40ms → net ~3% slower.  Re-enable if/when a CUTLASS epilogue
        # visitor is used so cuBLAS-class GEMM performance is preserved.
        self._cached_use_fused_gemm_gate = False
        # Fused LayerNorm + Modulate + MLP up-projection.
        # DISABLED: the hand-written WMMA GEMM cannot compete with cuBLAS for
        # the MLP up-projection (4352×12288×3072).  The HBM savings from
        # eliminating the [B,S,D] intermediate (~26 MB write+read ≈ 16 µs on
        # H100) are dwarfed by the GEMM slowdown (10-50×).  Re-enable only
        # after switching to a CUTLASS-based prologue fusion or Triton wrapper
        # that delegates the matmul to cuBLAS/cuTLASS.
        self._cached_use_fused_lnm_gemm_up = False

    @property
    def use_custom_kernels(self) -> bool:
        """
        Whether this block should use custom CUDA kernels (when available).

        The actual kernel wiring lives in QWEN-specific kernel_ops; this flag only
        controls whether the fast path is allowed to be taken.
        """
        return self._use_custom_kernels_flag

    @use_custom_kernels.setter
    def use_custom_kernels(self, value: bool) -> None:
        self._use_custom_kernels_flag = bool(value)
        self._update_kernel_availability_cache()

    def _modulate(self, x, mod_params, index=None):
        """Apply modulation to input tensor"""
        # x: b l d, shift: b d, scale: b d, gate: b d
        shift, scale, gate = mod_params.chunk(3, dim=-1)

        if index is not None:
            # Assuming mod_params batch dim is 2*actual_batch (chunked into 2 parts)
            # So shift, scale, gate have shape [2*actual_batch, d]
            actual_batch = shift.size(0) // 2
            shift_0, shift_1 = shift[:actual_batch], shift[actual_batch:]  # each: [actual_batch, d]
            scale_0, scale_1 = scale[:actual_batch], scale[actual_batch:]
            gate_0, gate_1 = gate[:actual_batch], gate[actual_batch:]

            # index: [b, l] where b is actual batch size
            # Expand to [b, l, 1] to match feature dimension
            index_expanded = index.unsqueeze(-1)  # [b, l, 1]

            # Expand chunks to [b, 1, d] then broadcast to [b, l, d]
            shift_0_exp = shift_0.unsqueeze(1)  # [b, 1, d]
            shift_1_exp = shift_1.unsqueeze(1)  # [b, 1, d]
            scale_0_exp = scale_0.unsqueeze(1)
            scale_1_exp = scale_1.unsqueeze(1)
            gate_0_exp = gate_0.unsqueeze(1)
            gate_1_exp = gate_1.unsqueeze(1)

            # Use torch.where to select based on index
            shift_result = torch.where(index_expanded == 0, shift_0_exp, shift_1_exp)
            scale_result = torch.where(index_expanded == 0, scale_0_exp, scale_1_exp)
            gate_result = torch.where(index_expanded == 0, gate_0_exp, gate_1_exp)
        else:
            shift_result = shift.unsqueeze(1)
            scale_result = scale.unsqueeze(1)
            gate_result = gate.unsqueeze(1)

        return x * (1 + scale_result) + shift_result, gate_result

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        modulate_index: Optional[List[int]] = None,
        precomputed_img_mod: Optional[torch.Tensor] = None,
        precomputed_txt_mod: Optional[torch.Tensor] = None,
        parallel_stream=None,
        parallel_event_default=None,
        parallel_event_side=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get modulation parameters for both streams.
        # If pre-computed (from batched GEMM in Phase 5), skip per-block computation.
        if _QWEN_NVTX_ENABLED:
            _nvtx_range_push("modulation")
        if precomputed_img_mod is not None:
            img_mod_params = precomputed_img_mod  # [B, 6*dim]
        else:
            img_mod_params = self.img_mod(temb)  # [B, 6*dim]

        if precomputed_txt_mod is not None:
            txt_mod_params = precomputed_txt_mod  # [B, 6*dim]
        else:
            if self.zero_cond_t:
                temb = torch.chunk(temb, 2, dim=0)[0]
            txt_mod_params = self.txt_mod(temb)  # [B, 6*dim]

        # Split modulation parameters for norm1 and norm2
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
        if _QWEN_NVTX_ENABLED:
            _nvtx_range_pop()

        # Use cached kernel availability flags from the model-level pre-computation.
        # This avoids re-evaluating 3 boolean conditions per block per step
        # (was 540 evaluations per image, now 0).
        _use_fused_lnm = (
            self._cached_use_fused_lnm
            and modulate_index is None  # fused kernel only handles non-indexed path
        )
        _use_fused_lnm_txt = self._cached_use_fused_lnm_txt
        # Process image stream - norm1 + modulation
        if _QWEN_NVTX_ENABLED:
            _nvtx_range_push("norm1_mod")
        if _use_fused_lnm:
            img_shift1, img_scale1, img_gate1_raw = img_mod1.chunk(3, dim=-1)
            img_modulated = _qwen_layernorm_modulate_kernel(
                hidden_states, img_scale1, img_shift1, eps=self.img_norm1.eps
            )
            img_gate1 = img_gate1_raw.unsqueeze(1)
        else:
            img_normed = self.img_norm1(hidden_states)
            img_modulated, img_gate1 = self._modulate(img_normed, img_mod1, modulate_index)

        # Process text stream - norm1 + modulation
        if _use_fused_lnm_txt:
            txt_shift1, txt_scale1, txt_gate1_raw = txt_mod1.chunk(3, dim=-1)
            txt_modulated = _qwen_layernorm_modulate_kernel(
                encoder_hidden_states, txt_scale1, txt_shift1, eps=self.txt_norm1.eps
            )
            txt_gate1 = txt_gate1_raw.unsqueeze(1)
        else:
            txt_normed = self.txt_norm1(encoder_hidden_states)
            txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        # Use QwenAttnProcessor2_0 for joint attention computation
        # This directly implements the DoubleStreamLayerMegatron logic:
        # 1. Computes QKV for both streams
        # 2. Applies QK normalization and RoPE
        # 3. Concatenates and runs joint attention
        # 4. Splits results back to separate streams
        if _QWEN_NVTX_ENABLED:
            _nvtx_range_pop()  # norm1_mod
            _nvtx_range_push("attention")
        joint_attention_kwargs = joint_attention_kwargs or {}
        _use_fused_gemm_gate = self._cached_use_fused_gemm_gate and modulate_index is None
        if _use_fused_gemm_gate:
            joint_attention_kwargs = {**joint_attention_kwargs, "skip_output_proj": True}
        attn_output = self.attn(
            hidden_states=img_modulated,  # Image stream (will be processed as "sample")
            encoder_hidden_states=txt_modulated,  # Text stream (will be processed as "context")
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        # QwenAttnProcessor2_0 returns (img_output, txt_output) when encoder_hidden_states is provided
        img_attn_output, txt_attn_output = attn_output

        if _QWEN_NVTX_ENABLED:
            _nvtx_range_pop()  # attention
            _nvtx_range_push("attn_gate_res")
        if _use_fused_gemm_gate:
            # Fused path: GEMM (output projection) + gate + residual in a single Triton kernel.
            # Eliminates global-memory round-trip between the Linear and addcmul.
            img_out_linear = self.attn.to_out[0]
            hidden_states = _fused_gemm_gate_residual(
                img_attn_output, img_out_linear.weight, img_out_linear.bias,
                img_gate1, hidden_states,
            )
            txt_out_linear = self.attn.to_add_out
            encoder_hidden_states = _fused_gemm_gate_residual(
                txt_attn_output, txt_out_linear.weight, txt_out_linear.bias,
                txt_gate1, encoder_hidden_states,
            )
        else:
            # Standard path: output already projected by the processor.
            # torch.addcmul is faster than both the custom kernel and separate mul+add
            hidden_states = torch.addcmul(hidden_states, img_gate1, img_attn_output)
            encoder_hidden_states = torch.addcmul(encoder_hidden_states, txt_gate1, txt_attn_output)

        # --- MLP Phase: parallel streams overlap img and txt MLP ---
        _use_parallel_mlp = parallel_stream is not None
        if _use_parallel_mlp:
            parallel_event_default.record()
            parallel_stream.wait_event(parallel_event_default)

        # Process image stream - norm2 + MLP (always on default stream)
        if _QWEN_NVTX_ENABLED:
            _nvtx_range_pop()  # attn_gate_res
            _nvtx_range_push("img_norm2_mlp")
        _use_fused_lnm_gemm = (
            self._cached_use_fused_lnm_gemm_up
            and modulate_index is None
        )

        if _use_fused_lnm_gemm:
            # === Fused path: LN + Modulate + MLP up-projection in one CUDA kernel ===
            img_shift2, img_scale2, img_gate2_raw = img_mod2.chunk(3, dim=-1)
            img_gate2 = img_gate2_raw.unsqueeze(1)
            # Single kernel: LayerNorm(hidden_states) * (1+scale) + shift → Linear(D→4D)
            img_mlp_intermediate = _qwen_lnm_gemm_up_kernel(
                hidden_states, img_scale2, img_shift2,
                self.img_mlp.net[0].proj.weight, self.img_mlp.net[0].proj.bias,
                eps=self.img_norm2.eps,
            )
            # GELU activation (only the activation, linear was fused above)
            img_mlp_intermediate = F.gelu(img_mlp_intermediate, approximate="tanh")
            # Down-projection + gate + residual
            if _use_fused_gemm_gate:
                img_down_proj = self.img_mlp.net[-1]
                hidden_states = _fused_gemm_gate_residual(
                    img_mlp_intermediate, img_down_proj.weight, img_down_proj.bias,
                    img_gate2, hidden_states,
                )
            else:
                img_down_out = self.img_mlp.net[-1](img_mlp_intermediate)
                hidden_states = torch.addcmul(hidden_states, img_gate2, img_down_out)
        elif _use_fused_lnm:
            img_shift2, img_scale2, img_gate2_raw = img_mod2.chunk(3, dim=-1)
            img_modulated2 = _qwen_layernorm_modulate_kernel(
                hidden_states, img_scale2, img_shift2, eps=self.img_norm2.eps
            )
            img_gate2 = img_gate2_raw.unsqueeze(1)

            if _use_fused_gemm_gate:
                img_mlp_intermediate = img_modulated2
                for layer in self.img_mlp.net[:-1]:
                    img_mlp_intermediate = layer(img_mlp_intermediate)
                img_down_proj = self.img_mlp.net[-1]
                hidden_states = _fused_gemm_gate_residual(
                    img_mlp_intermediate, img_down_proj.weight, img_down_proj.bias,
                    img_gate2, hidden_states,
                )
            else:
                img_mlp_output = self.img_mlp(img_modulated2)
                hidden_states = torch.addcmul(hidden_states, img_gate2, img_mlp_output)
        else:
            img_normed2 = self.img_norm2(hidden_states)
            img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2, modulate_index)

            if _use_fused_gemm_gate:
                img_mlp_intermediate = img_modulated2
                for layer in self.img_mlp.net[:-1]:
                    img_mlp_intermediate = layer(img_mlp_intermediate)
                img_down_proj = self.img_mlp.net[-1]
                hidden_states = _fused_gemm_gate_residual(
                    img_mlp_intermediate, img_down_proj.weight, img_down_proj.bias,
                    img_gate2, hidden_states,
                )
            else:
                img_mlp_output = self.img_mlp(img_modulated2)
                hidden_states = torch.addcmul(hidden_states, img_gate2, img_mlp_output)

        # Process text stream - norm2 + MLP
        # If parallel streams enabled, switch to side stream for txt MLP
        if _use_parallel_mlp:
            _parallel_ctx = torch.cuda.stream(parallel_stream)
            _parallel_ctx.__enter__()
        if _QWEN_NVTX_ENABLED:
            _nvtx_range_pop()  # img_norm2_mlp
            _nvtx_range_push("txt_norm2_mlp")
        _use_fused_lnm_gemm_txt = (
            self._cached_use_fused_lnm_gemm_up
            and True  # text never uses modulate_index
        )

        if _use_fused_lnm_gemm_txt:
            txt_shift2, txt_scale2, txt_gate2_raw = txt_mod2.chunk(3, dim=-1)
            txt_gate2 = txt_gate2_raw.unsqueeze(1)
            txt_mlp_intermediate = _qwen_lnm_gemm_up_kernel(
                encoder_hidden_states, txt_scale2, txt_shift2,
                self.txt_mlp.net[0].proj.weight, self.txt_mlp.net[0].proj.bias,
                eps=self.txt_norm2.eps,
            )
            txt_mlp_intermediate = F.gelu(txt_mlp_intermediate, approximate="tanh")
            if _use_fused_gemm_gate:
                txt_down_proj = self.txt_mlp.net[-1]
                encoder_hidden_states = _fused_gemm_gate_residual(
                    txt_mlp_intermediate, txt_down_proj.weight, txt_down_proj.bias,
                    txt_gate2, encoder_hidden_states,
                )
            else:
                txt_down_out = self.txt_mlp.net[-1](txt_mlp_intermediate)
                encoder_hidden_states = torch.addcmul(encoder_hidden_states, txt_gate2, txt_down_out)
        elif _use_fused_lnm_txt:
            txt_shift2, txt_scale2, txt_gate2_raw = txt_mod2.chunk(3, dim=-1)
            txt_modulated2 = _qwen_layernorm_modulate_kernel(
                encoder_hidden_states, txt_scale2, txt_shift2, eps=self.txt_norm2.eps
            )
            txt_gate2 = txt_gate2_raw.unsqueeze(1)

            if _use_fused_gemm_gate:
                txt_mlp_intermediate = txt_modulated2
                for layer in self.txt_mlp.net[:-1]:
                    txt_mlp_intermediate = layer(txt_mlp_intermediate)
                txt_down_proj = self.txt_mlp.net[-1]
                encoder_hidden_states = _fused_gemm_gate_residual(
                    txt_mlp_intermediate, txt_down_proj.weight, txt_down_proj.bias,
                    txt_gate2, encoder_hidden_states,
                )
            else:
                txt_mlp_output = self.txt_mlp(txt_modulated2)
                encoder_hidden_states = torch.addcmul(encoder_hidden_states, txt_gate2, txt_mlp_output)
        else:
            txt_normed2 = self.txt_norm2(encoder_hidden_states)
            txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)

            if _use_fused_gemm_gate:
                txt_mlp_intermediate = txt_modulated2
                for layer in self.txt_mlp.net[:-1]:
                    txt_mlp_intermediate = layer(txt_mlp_intermediate)
                txt_down_proj = self.txt_mlp.net[-1]
                encoder_hidden_states = _fused_gemm_gate_residual(
                    txt_mlp_intermediate, txt_down_proj.weight, txt_down_proj.bias,
                    txt_gate2, encoder_hidden_states,
                )
            else:
                txt_mlp_output = self.txt_mlp(txt_modulated2)
                encoder_hidden_states = torch.addcmul(encoder_hidden_states, txt_gate2, txt_mlp_output)

        if _QWEN_NVTX_ENABLED:
            _nvtx_range_pop()  # txt_norm2_mlp

        if _use_parallel_mlp:
            parallel_event_side.record()
            _parallel_ctx.__exit__(None, None, None)
            torch.cuda.current_stream().wait_event(parallel_event_side)

        # Clip to prevent overflow for fp16
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class QwenImageTransformer2DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    """
    The Transformer model introduced in Qwen.

    Args:
        patch_size (`int`, defaults to `2`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `64`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output. If not specified, it defaults to `in_channels`.
        num_layers (`int`, defaults to `60`):
            The number of layers of dual stream DiT blocks to use.
        attention_head_dim (`int`, defaults to `128`):
            The number of dimensions to use for each attention head.
        num_attention_heads (`int`, defaults to `24`):
            The number of attention heads to use.
        joint_attention_dim (`int`, defaults to `3584`):
            The number of dimensions to use for the joint attention (embedding/channel dimension of
            `encoder_hidden_states`).
        guidance_embeds (`bool`, defaults to `False`):
            Whether to use guidance embeddings for guidance-distilled variant of the model.
        axes_dims_rope (`tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions to use for the rotary positional embeddings.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["QwenImageTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["QwenImageTransformerBlock"]
    # Make CP plan compatible with https://github.com/huggingface/diffusers/pull/12702
    _cp_plan = {
        "transformer_blocks.0": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "transformer_blocks.*": {
            "modulate_index": ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
        },
        "pos_embed": {
            0: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
            1: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
        },
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
    }

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: int | None = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        guidance_embeds: bool = False,  # TODO: this should probably be removed
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
        zero_cond_t: bool = False,
        use_additional_t_cond: bool = False,
        use_layer3d_rope: bool = False,
        use_custom_kernels: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        if not use_layer3d_rope:
            self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)
        else:
            self.pos_embed = QwenEmbedLayer3DRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)

        self.time_text_embed = QwenTimestepProjEmbeddings(
            embedding_dim=self.inner_dim, use_additional_t_cond=use_additional_t_cond
        )

        self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6)

        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    zero_cond_t=zero_cond_t,
                    use_custom_kernels=use_custom_kernels,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False
        self.zero_cond_t = zero_cond_t

        # Cached stacked modulation weights for batched pre-computation (Phase 5).
        # Populated lazily on first use via _ensure_batched_mod_weights().
        self._batched_img_mod_weight: Optional[torch.Tensor] = None
        self._batched_img_mod_bias: Optional[torch.Tensor] = None
        self._batched_txt_mod_weight: Optional[torch.Tensor] = None
        self._batched_txt_mod_bias: Optional[torch.Tensor] = None

        # Parallel CUDA streams for overlapping image/text MLP phases.
        # Lazily initialized via _ensure_parallel_streams().
        self._parallel_stream = None
        self._parallel_event_default = None
        self._parallel_event_side = None
        self._use_parallel_streams = False

    def _ensure_batched_mod_weights(self) -> None:
        """Stack all blocks' modulation Linear weights for batched GEMM.

        Called lazily on first use. The stacked tensors are views/copies of
        the original parameters, so they stay on the correct device. They
        are invalidated if the number of blocks changes (unlikely in inference).
        """
        if self._batched_img_mod_weight is not None:
            return

        blocks = self.transformer_blocks
        n = len(blocks)
        # img_mod is nn.Sequential(SiLU, Linear(D, 6*D))
        # The Linear is at index [1]
        self._batched_img_mod_weight = torch.stack(
            [blocks[i].img_mod[1].weight for i in range(n)]
        )  # [N, 6*D, D]
        self._batched_img_mod_bias = torch.stack(
            [blocks[i].img_mod[1].bias for i in range(n)]
        )  # [N, 6*D]
        self._batched_txt_mod_weight = torch.stack(
            [blocks[i].txt_mod[1].weight for i in range(n)]
        )  # [N, 6*D, D]
        self._batched_txt_mod_bias = torch.stack(
            [blocks[i].txt_mod[1].bias for i in range(n)]
        )  # [N, 6*D]

    def _precompute_all_modulations(
        self, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch all 60 blocks' modulation projections into 2 batched GEMMs.

        Instead of 120 separate small GEMMs (60 img + 60 txt, each [B, D] x [6*D, D]),
        this computes all modulations in 2 operations:
          SiLU(temb) @ stacked_weights^T + stacked_biases

        Args:
            temb: [B, D] timestep embedding

        Returns:
            (all_img_mods, all_txt_mods): each [B, N, 6*D] where N = num_blocks
        """
        self._ensure_batched_mod_weights()

        silu_temb = F.silu(temb)  # [B, D]

        # Batched GEMM for all image modulations:
        # einsum('bd,nod->bno', silu_temb, weights) computes silu_temb @ weight^T for each block
        all_img_mods = (
            torch.einsum('bd,nod->bno', silu_temb, self._batched_img_mod_weight)
            + self._batched_img_mod_bias.unsqueeze(0)
        )  # [B, N, 6*D]

        # For txt_mod, handle zero_cond_t: use chunked temb if applicable
        if self.zero_cond_t:
            txt_silu_temb = F.silu(torch.chunk(temb, 2, dim=0)[0])
        else:
            txt_silu_temb = silu_temb

        all_txt_mods = (
            torch.einsum('bd,nod->bno', txt_silu_temb, self._batched_txt_mod_weight)
            + self._batched_txt_mod_bias.unsqueeze(0)
        )  # [B_txt, N, 6*D]

        return all_img_mods, all_txt_mods

    def _ensure_parallel_streams(self) -> None:
        """Lazily create CUDA stream and events for parallel MLP execution."""
        if self._parallel_stream is None:
            self._parallel_stream = torch.cuda.Stream()
            self._parallel_event_default = torch.cuda.Event(enable_timing=False)
            self._parallel_event_side = torch.cuda.Event(enable_timing=False)

    def enable_parallel_streams(self) -> None:
        """Enable parallel CUDA streams for overlapping image/text MLP in each block."""
        self._use_parallel_streams = True

    def disable_parallel_streams(self) -> None:
        """Disable parallel CUDA streams for image/text MLP overlap."""
        self._use_parallel_streams = False

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: list[tuple[int, int, int]] | None = None,
        txt_seq_lens: list[int] | None = None,
        guidance: torch.Tensor = None,  # TODO: this should probably be removed
        attention_kwargs: dict[str, Any] | None = None,
        controlnet_block_samples=None,
        additional_t_cond=None,
        return_dict: bool = True,
    ) -> torch.Tensor | Transformer2DModelOutput:
        """
        The [`QwenTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            encoder_hidden_states_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`, *optional*):
                Mask for the encoder hidden states. Expected to have 1.0 for valid tokens and 0.0 for padding tokens.
                Used in the attention processor to prevent attending to padding tokens. The mask can have any pattern
                (not just contiguous valid tokens followed by padding) since it's applied element-wise in attention.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            img_shapes (`list[tuple[int, int, int]]`, *optional*):
                Image shapes for RoPE computation.
            txt_seq_lens (`list[int]`, *optional*, **Deprecated**):
                Deprecated parameter. Use `encoder_hidden_states_mask` instead. If provided, the maximum value will be
                used to compute RoPE sequence length.
            guidance (`torch.Tensor`, *optional*):
                Guidance tensor for conditional generation.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            controlnet_block_samples (*optional*):
                ControlNet block samples to add to the transformer blocks.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if txt_seq_lens is not None:
            deprecate(
                "txt_seq_lens",
                "0.39.0",
                "Passing `txt_seq_lens` is deprecated and will be removed in version 0.39.0. "
                "Please use `encoder_hidden_states_mask` instead. "
                "The mask-based approach is more flexible and supports variable-length sequences.",
                standard_warn=False,
            )

        hidden_states = self.img_in(hidden_states)

        timestep = timestep.to(hidden_states.dtype)

        if self.zero_cond_t:
            timestep = torch.cat([timestep, timestep * 0], dim=0)
            modulate_index = torch.tensor(
                [[0] * prod(sample[0]) + [1] * sum([prod(s) for s in sample[1:]]) for sample in img_shapes],
                device=timestep.device,
                dtype=torch.int,
            )
        else:
            modulate_index = None

        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        # Use the encoder_hidden_states sequence length for RoPE computation and normalize mask
        text_seq_len, _, encoder_hidden_states_mask = compute_text_seq_len_from_mask(
            encoder_hidden_states, encoder_hidden_states_mask
        )

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states, additional_t_cond)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states, additional_t_cond)
        )

        image_rotary_emb = self.pos_embed(img_shapes, max_txt_seq_len=text_seq_len, device=hidden_states.device)

        # Construct joint attention mask once to avoid reconstructing in every block
        # This eliminates 60 GPU syncs during training while maintaining torch.compile compatibility
        block_attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}

        # Build rope_grid_info for fused CUDA RoPE kernel (image tokens only).
        # Only for non-Layer3D single-resolution images with custom kernels enabled.
        # Cache the check result on the model to avoid re-evaluating every forward pass.
        _any_block_uses_kernels = getattr(self, "_cached_any_block_uses_kernels", None)
        if _any_block_uses_kernels is None:
            _any_block_uses_kernels = (
                len(self.transformer_blocks) > 0
                and getattr(self.transformer_blocks[0], "_use_custom_kernels_flag", False)
            )
            self._cached_any_block_uses_kernels = _any_block_uses_kernels
        if (
            _any_block_uses_kernels
            and _qwen_rope_3d_fused_kernel is not None
            and _QWEN_KERNELS_AVAILABLE
            and img_shapes is not None
            and not isinstance(self.pos_embed, QwenEmbedLayer3DRope)
        ):
            _fhw = img_shapes[0] if isinstance(img_shapes[0], (list, tuple)) else img_shapes
            if isinstance(_fhw, (list, tuple)) and len(_fhw) == 3:
                _frame, _height, _width = _fhw
            elif isinstance(_fhw, (list, tuple)) and len(_fhw) >= 1 and isinstance(_fhw[0], (list, tuple)):
                _frame, _height, _width = _fhw[0]
            else:
                _frame, _height, _width = None, None, None

            if _frame is not None:
                block_attention_kwargs["rope_grid_info"] = {
                    "grid_frame": int(_frame),
                    "grid_height": int(_height),
                    "grid_width": int(_width),
                    "theta": float(self.pos_embed.theta),
                    "axes_dim": tuple(self.pos_embed.axes_dim),
                    "height_offset": int(_height) - int(_height) // 2,
                    "width_offset": int(_width) - int(_width) // 2,
                }

        if encoder_hidden_states_mask is not None:
            # Build joint mask: [text_mask, all_ones_for_image]
            # Cache the image_mask tensor to avoid torch.ones() allocation every step.
            batch_size, image_seq_len = hidden_states.shape[:2]
            _cached = getattr(self, "_cached_image_mask", None)
            if (
                _cached is not None
                and _cached.shape == (batch_size, image_seq_len)
                and _cached.device == hidden_states.device
            ):
                image_mask = _cached
            else:
                image_mask = torch.ones(
                    (batch_size, image_seq_len), dtype=torch.bool, device=hidden_states.device
                )
                self._cached_image_mask = image_mask
            joint_attention_mask = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)
            block_attention_kwargs["attention_mask"] = joint_attention_mask

        # Pre-compute all 60 blocks' modulation parameters in 2 batched GEMMs
        # instead of 120 separate small GEMMs. This reduces Python dispatch overhead
        # and may improve GPU utilization for the small [B, D] x [6*D, D] shapes.
        # Skip when gradient checkpointing is active (it recomputes forward anyway).
        _use_batched_mods = (
            not (torch.is_grad_enabled() and self.gradient_checkpointing)
            and not torch.compiler.is_compiling()  # Let torch.compile handle fusion
            and not getattr(self, "_disable_batched_modulations", False)
        )
        if _use_batched_mods:
            if _QWEN_NVTX_ENABLED:
                _nvtx_range_push("batched_modulation")
            all_img_mods, all_txt_mods = self._precompute_all_modulations(temb)
            if _QWEN_NVTX_ENABLED:
                _nvtx_range_pop()
        else:
            all_img_mods = all_txt_mods = None

        _use_parallel = (
            self._use_parallel_streams
            and not torch.compiler.is_compiling()
            and not (torch.is_grad_enabled() and self.gradient_checkpointing)
        )
        if _use_parallel:
            self._ensure_parallel_streams()

        for index_block, block in enumerate(self.transformer_blocks):
            if _QWEN_NVTX_ENABLED:
                _nvtx_range_push(f"block_{index_block}")
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    None,  # Don't pass encoder_hidden_states_mask (using attention_mask instead)
                    temb,
                    image_rotary_emb,
                    block_attention_kwargs,
                    modulate_index,
                )

            else:
                # Extract pre-computed modulation for this block
                _img_mod = all_img_mods[:, index_block, :] if all_img_mods is not None else None
                _txt_mod = all_txt_mods[:, index_block, :] if all_txt_mods is not None else None

                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=None,  # Don't pass (using attention_mask instead)
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=block_attention_kwargs,
                    modulate_index=modulate_index,
                    precomputed_img_mod=_img_mod,
                    precomputed_txt_mod=_txt_mod,
                    parallel_stream=self._parallel_stream if _use_parallel else None,
                    parallel_event_default=self._parallel_event_default if _use_parallel else None,
                    parallel_event_side=self._parallel_event_side if _use_parallel else None,
                )
            if _QWEN_NVTX_ENABLED:
                _nvtx_range_pop()

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        if self.zero_cond_t:
            temb = temb.chunk(2, dim=0)[0]
        # Use only the image part (hidden_states) from the dual-stream blocks
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
