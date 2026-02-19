"""
FP8 quantization utilities for QWEN-2512 inference optimization.

Provides E4M3 FP8 GEMM wrappers using torch._scaled_mm (H100 Transformer Engine)
for ~1.5-1.8x speedup on compute-bound matrix multiplications.

Key features:
  - Per-tensor E4M3 scaling with calibrated scale factors
  - Online quantization (quantize activations on-the-fly each forward pass)
  - Offline weight quantization (done once at model load)
  - FP8 Linear drop-in replacement module
  - Selective quantization (image-stream GEMMs first, text-stream stays bf16)

Usage:
  from .fp8_utils import convert_linear_to_fp8, FP8Config

  config = FP8Config(quantize_image_mlp=True, quantize_qkv=True)
  convert_linear_to_fp8(model, config)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Check for FP8 support (requires H100/sm_90 and PyTorch 2.1+)
_FP8_AVAILABLE = False
_SCALED_MM_AVAILABLE = False

try:
    if hasattr(torch, "float8_e4m3fn") and torch.cuda.is_available():
        # Check compute capability >= 8.9 (H100 = 9.0, L4 = 8.9)
        if torch.cuda.get_device_capability()[0] >= 9:
            _FP8_AVAILABLE = True
        # Check torch._scaled_mm availability
        if hasattr(torch, "_scaled_mm"):
            _SCALED_MM_AVAILABLE = True
except Exception:
    pass


def is_fp8_available() -> bool:
    """Check if FP8 inference is supported on the current hardware."""
    return _FP8_AVAILABLE and _SCALED_MM_AVAILABLE


@dataclass
class FP8Config:
    """Configuration for FP8 quantization."""
    # Which GEMM categories to quantize
    quantize_image_mlp: bool = True      # Image MLP (largest GEMMs, M=4096)
    quantize_text_mlp: bool = False      # Text MLP (small M=256, less benefit)
    quantize_qkv_proj: bool = True       # QKV projections
    quantize_output_proj: bool = True    # Output projections
    quantize_img_in: bool = False        # Initial projection (one-time cost)

    # Scaling configuration
    use_per_tensor_scaling: bool = True  # Per-tensor vs per-channel
    calibration_method: str = "absmax"   # "absmax" or "percentile"
    percentile: float = 99.99           # For percentile calibration

    # Quality safeguards
    fallback_to_bf16_on_overflow: bool = True
    max_scale_factor: float = 1e4       # Clamp scale factors

    # Pre-computed scale factors (populated by calibration)
    weight_scales: Dict[str, float] = field(default_factory=dict)
    activation_scales: Dict[str, float] = field(default_factory=dict)


def compute_scale_factor(tensor: torch.Tensor, method: str = "absmax", percentile: float = 99.99) -> float:
    """Compute per-tensor scale factor for FP8 E4M3 quantization.

    E4M3 range: [-448, 448]. Scale maps tensor range to FP8 range.
    scale = fp8_max / tensor_amax
    """
    fp8_max = 448.0  # torch.finfo(torch.float8_e4m3fn).max

    if method == "absmax":
        amax = tensor.abs().max().item()
    elif method == "percentile":
        amax = torch.quantile(tensor.abs().float(), percentile / 100.0).item()
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    if amax == 0:
        return 1.0

    scale = fp8_max / amax
    return scale


def quantize_to_fp8(tensor: torch.Tensor, scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a bf16/fp16/fp32 tensor to FP8 E4M3 with the given scale.

    Args:
        tensor: Input tensor (any float dtype)
        scale: Scale factor (computed via compute_scale_factor)

    Returns:
        (fp8_tensor, scale_inv_tensor): Quantized tensor and inverse scale for dequantization
    """
    # Scale and clamp to FP8 range
    scaled = tensor.float() * scale
    fp8_max = 448.0
    scaled = scaled.clamp(-fp8_max, fp8_max)

    # Convert to FP8
    fp8_tensor = scaled.to(torch.float8_e4m3fn)

    # Inverse scale for torch._scaled_mm
    scale_inv = torch.tensor(1.0 / scale, dtype=torch.float32, device=tensor.device)

    return fp8_tensor, scale_inv


class FP8Linear(nn.Module):
    """Drop-in replacement for nn.Linear using FP8 GEMMs via torch._scaled_mm.

    Weights are stored in FP8 E4M3 format. Activations are quantized on-the-fly
    each forward pass. The GEMM is executed in FP8 on H100 Transformer Engine.

    The output is in bf16, matching the original Linear's output dtype.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_scale: Optional[float] = None,
        activation_scale: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # FP8 weight storage
        self.register_buffer(
            "weight_fp8",
            torch.zeros(out_features, in_features, dtype=torch.float8_e4m3fn, device=device),
        )
        self.register_buffer(
            "weight_scale_inv",
            torch.tensor(1.0, dtype=torch.float32, device=device),
        )

        if bias:
            # Bias stays in bf16 — added after the FP8 GEMM
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16, device=device))
        else:
            self.register_parameter("bias", None)

        # Activation scale: either pre-calibrated or computed online
        self._activation_scale = activation_scale
        self.register_buffer(
            "activation_scale_inv",
            torch.tensor(1.0, dtype=torch.float32, device=device),
        )

        # For online activation scaling (running absmax)
        self._use_online_scaling = activation_scale is None
        self.register_buffer(
            "_running_amax",
            torch.tensor(1.0, dtype=torch.float32, device=device),
        )

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        weight_scale: Optional[float] = None,
        activation_scale: Optional[float] = None,
    ) -> "FP8Linear":
        """Convert an existing nn.Linear to FP8Linear.

        Args:
            linear: Source linear layer
            weight_scale: Pre-calibrated weight scale (computed if None)
            activation_scale: Pre-calibrated activation scale (online if None)
        """
        has_bias = linear.bias is not None
        fp8_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=has_bias,
            weight_scale=weight_scale,
            activation_scale=activation_scale,
            device=linear.weight.device,
        )

        # Quantize weights to FP8
        if weight_scale is None:
            weight_scale = compute_scale_factor(linear.weight.data)

        fp8_weight, weight_scale_inv = quantize_to_fp8(linear.weight.data, weight_scale)
        fp8_linear.weight_fp8.copy_(fp8_weight)
        fp8_linear.weight_scale_inv.copy_(weight_scale_inv)

        # Copy bias
        if has_bias:
            fp8_linear.bias.data.copy_(linear.bias.data.to(torch.bfloat16))

        # Set activation scale
        if activation_scale is not None:
            fp8_linear._use_online_scaling = False
            fp8_linear._activation_scale = activation_scale
            fp8_linear.activation_scale_inv.copy_(
                torch.tensor(1.0 / activation_scale, dtype=torch.float32)
            )

        return fp8_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FP8 GEMM forward pass.

        1. Quantize activation to FP8 (online or pre-calibrated scale)
        2. Execute FP8 GEMM via torch._scaled_mm
        3. Add bias in bf16
        """
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)

        # Quantize activation
        if self._use_online_scaling:
            # Online per-tensor absmax scaling
            amax = x_2d.abs().max()
            scale = 448.0 / amax.clamp(min=1e-12)
            x_fp8 = (x_2d.float() * scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            # Use reciprocal tensor op instead of 1.0/scale.item() to avoid
            # graph break under torch.compile (Tensor.item() is not traceable).
            act_scale_inv = scale.reciprocal().to(torch.float32)
        else:
            scale = self._activation_scale
            x_fp8 = (x_2d.float() * scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            act_scale_inv = self.activation_scale_inv

        # FP8 GEMM: output = (x_fp8 @ weight_fp8^T) * (act_scale_inv * weight_scale_inv)
        # torch._scaled_mm expects: (A, B^T, scale_a, scale_b) -> C
        out = torch._scaled_mm(
            x_fp8,
            self.weight_fp8.t(),
            scale_a=act_scale_inv,
            scale_b=self.weight_scale_inv,
            out_dtype=torch.bfloat16,
        )

        # Add bias
        if self.bias is not None:
            out = out + self.bias

        return out.reshape(*orig_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, fp8=True, "
            f"online_scaling={self._use_online_scaling}"
        )


def convert_linear_to_fp8(
    model: nn.Module,
    config: FP8Config,
    prefix: str = "",
) -> int:
    """Convert eligible nn.Linear layers in the model to FP8Linear.

    Applies selective quantization based on FP8Config — only converts layers
    that match the configured categories (image MLP, QKV projections, etc.).

    Args:
        model: The model to convert (typically pipe.transformer)
        config: FP8 quantization configuration
        prefix: Module name prefix for tracking

    Returns:
        Number of layers converted
    """
    if not is_fp8_available():
        warnings.warn(
            "FP8 is not available on this hardware. "
            "Requires H100 (sm_90+) and PyTorch with torch._scaled_mm support."
        )
        return 0

    converted = 0

    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(module, nn.Linear):
            should_convert = False

            # Determine if this linear layer should be converted based on config
            if config.quantize_image_mlp and _is_image_mlp_linear(full_name):
                should_convert = True
            elif config.quantize_text_mlp and _is_text_mlp_linear(full_name):
                should_convert = True
            elif config.quantize_qkv_proj and _is_qkv_linear(full_name):
                should_convert = True
            elif config.quantize_output_proj and _is_output_proj_linear(full_name):
                should_convert = True
            elif config.quantize_img_in and "img_in" in full_name:
                should_convert = True

            if should_convert:
                weight_scale = config.weight_scales.get(full_name)
                act_scale = config.activation_scales.get(full_name)

                fp8_linear = FP8Linear.from_linear(
                    module,
                    weight_scale=weight_scale,
                    activation_scale=act_scale,
                )
                # Replace the module
                parent = model
                parts = name.split(".")
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], fp8_linear)
                converted += 1
        else:
            # Recurse into child modules
            converted += convert_linear_to_fp8(module, config, prefix=full_name)

    return converted


def _is_image_mlp_linear(name: str) -> bool:
    """Check if this is an image-stream MLP linear layer."""
    return "img_mlp" in name and ("net.0.proj" in name or "net.2" in name)


def _is_text_mlp_linear(name: str) -> bool:
    """Check if this is a text-stream MLP linear layer."""
    return "txt_mlp" in name and ("net.0.proj" in name or "net.2" in name)


def _is_qkv_linear(name: str) -> bool:
    """Check if this is a QKV projection linear layer."""
    return any(k in name for k in ["to_qkv", "to_added_qkv", "to_q", "to_k", "to_v"])


def _is_output_proj_linear(name: str) -> bool:
    """Check if this is an output projection linear layer."""
    return any(k in name for k in ["to_out.0", "to_add_out"])
