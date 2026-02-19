"""
CUDA Graph runner for QWEN-2512 denoising loop.

Captures the transformer forward pass as a CUDA graph on the first step,
then replays it for subsequent steps. This eliminates ~18,900 kernel launches
per inference (60 blocks x ~315 kernels per block x 8 replayed steps).

Key requirements for CUDA graph compatibility:
  - Static tensor shapes (no dynamic allocation during graph execution)
  - No Python-dependent control flow in the captured region
  - All CUDA operations on the same stream
  - Pre-allocated input/output buffers that are written via copy_()

Usage:
  runner = CUDAGraphRunner(transformer)
  runner.capture(latents, timestep, prompt_embeds, ...)  # Step 1: capture
  for t in timesteps[1:]:
      output = runner.replay(latents, timestep)           # Steps 2-N: replay
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


_CUDA_GRAPHS_ENABLED = os.environ.get("QWEN_CUDA_GRAPHS", "0") == "1"


def is_cuda_graphs_available() -> bool:
    """Check if CUDA graphs can be used."""
    if not torch.cuda.is_available():
        return False
    # CUDA graphs require compute capability >= 7.0
    cap = torch.cuda.get_device_capability()
    return cap[0] >= 7


class CUDAGraphRunner:
    """Wraps a transformer model with CUDA graph capture/replay.

    The first call captures the forward pass as a CUDA graph.
    Subsequent calls replay the graph with updated input values,
    avoiding the overhead of ~18,900 kernel launches per step.

    The runner persists across pipe() calls. If the input shapes
    are the same, it reuses the captured graph (no re-capture).
    If shapes change (different resolution), it re-captures.

    Improvements over naive capture:
    - Uses a private memory pool to avoid interference with other allocations
    - Runs 2 warmup passes (warm caches + stable memory pool)
    - Persists across inference calls — capture once, replay many times
    - Gracefully falls back to eager if capture fails
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.is_captured = False

        # Shapes of captured tensors (for reuse validation)
        self._captured_hidden_shape: Optional[Tuple[int, ...]] = None
        self._captured_encoder_shape: Optional[Tuple[int, ...]] = None

        # Static input buffers (filled via copy_ before replay)
        self._static_hidden_states: Optional[torch.Tensor] = None
        self._static_timestep: Optional[torch.Tensor] = None
        self._static_encoder_hidden_states: Optional[torch.Tensor] = None
        self._static_encoder_hidden_states_mask: Optional[torch.Tensor] = None

        # Static output buffer
        self._static_output: Optional[torch.Tensor] = None

        # Fixed kwargs that don't change between steps
        self._fixed_kwargs: Dict[str, Any] = {}

        # Private CUDA memory pool for graph capture to avoid fragmentation
        self._mempool: Optional[torch.cuda.graphs.graph_pool_handle] = None

    def _build_model_kwargs(self) -> Dict[str, Any]:
        """Build the full kwargs dict for model forward, using static buffers."""
        return {
            "hidden_states": self._static_hidden_states,
            "timestep": self._static_timestep / 1000,
            "encoder_hidden_states": self._static_encoder_hidden_states,
            "encoder_hidden_states_mask": self._static_encoder_hidden_states_mask,
            **self._fixed_kwargs,
        }

    def capture(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
        img_shapes: Optional[List] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Capture the transformer forward pass as a CUDA graph.

        This should be called on the first denoising step. It:
        1. Allocates static input/output buffers
        2. Pre-computes attention prep tensors (masks, varlen indices)
        3. Runs 2 warmup passes (pool memory, then warm caches)
        4. Captures the forward pass as a graph
        5. Returns the output of the captured pass

        Args:
            All arguments match QwenImageTransformer2DModel.forward()

        Returns:
            Model output tensor (from the capture run)
        """
        if not is_cuda_graphs_available():
            warnings.warn("CUDA graphs not available, falling back to eager execution")
            return self._eager_forward(
                hidden_states, timestep, encoder_hidden_states,
                encoder_hidden_states_mask, guidance, img_shapes, attention_kwargs,
            )

        # Allocate static input buffers
        self._static_hidden_states = hidden_states.clone()
        self._static_timestep = timestep.clone()
        self._static_encoder_hidden_states = encoder_hidden_states.clone()
        if encoder_hidden_states_mask is not None:
            self._static_encoder_hidden_states_mask = encoder_hidden_states_mask.clone()
        else:
            self._static_encoder_hidden_states_mask = None

        # Store fixed kwargs
        self._fixed_kwargs = {
            "guidance": guidance,
            "img_shapes": img_shapes,
            "attention_kwargs": attention_kwargs,
            "return_dict": False,
        }

        # CUDA graphs require capture on a non-default stream.
        # Create a side stream, run warmups there, capture, then
        # sync back. Replay can happen on any stream (including default).
        capture_stream = torch.cuda.Stream()

        # Warmup pass 1: populate caches (RoPE, varlen, buffer caches)
        # This ensures all lazy initializations happen before capture.
        torch.cuda.synchronize()
        with torch.cuda.stream(capture_stream):
            _ = self.model(**self._build_model_kwargs())[0]
        torch.cuda.synchronize()

        # Warmup pass 2: second run to ensure memory pool is stable.
        # CUDA graph capture will use the allocations from this pass.
        with torch.cuda.stream(capture_stream):
            _ = self.model(**self._build_model_kwargs())[0]
        torch.cuda.synchronize()

        # Capture the graph using a private memory pool.
        # This prevents the graph's memory from being freed by
        # the PyTorch caching allocator between runs.
        self._mempool = torch.cuda.graph_pool_handle()
        self.graph = torch.cuda.CUDAGraph()

        try:
            with torch.cuda.graph(self.graph, pool=self._mempool, stream=capture_stream):
                self._static_output = self.model(**self._build_model_kwargs())[0]
        except Exception as e:
            warnings.warn(
                f"CUDA graph capture failed: {e}\n"
                "Falling back to eager execution. Common causes:\n"
                "  - CPU-GPU sync (.item(), .cpu()) inside the model\n"
                "  - Data-dependent control flow on tensor values\n"
                "  - Operations on multiple CUDA streams"
            )
            self.graph = None
            self._mempool = None
            return self._eager_forward(
                hidden_states, timestep, encoder_hidden_states,
                encoder_hidden_states_mask, guidance, img_shapes, attention_kwargs,
            )

        self.is_captured = True
        self._captured_hidden_shape = hidden_states.shape
        self._captured_encoder_shape = encoder_hidden_states.shape
        return self._static_output

    def shapes_match(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> bool:
        """Check if input shapes match the captured graph's shapes."""
        if not self.is_captured:
            return False
        return (
            hidden_states.shape == self._captured_hidden_shape
            and encoder_hidden_states.shape == self._captured_encoder_shape
        )

    def replay_full(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Replay with ALL static buffers updated.

        Used on step 0 of a new inference when reusing a captured graph.
        Unlike replay() which only updates hidden_states and timestep,
        this also updates encoder_hidden_states and mask (which may
        differ between prompts).

        Args:
            hidden_states: New latent tensor
            timestep: New timestep value
            encoder_hidden_states: New prompt embeddings
            encoder_hidden_states_mask: New prompt mask

        Returns:
            Model output tensor
        """
        if not self.is_captured:
            raise RuntimeError("Must call capture() before replay_full()")

        # Update all dynamic input buffers in-place
        self._static_hidden_states.copy_(hidden_states)
        self._static_timestep.copy_(timestep)
        self._static_encoder_hidden_states.copy_(encoder_hidden_states)
        if encoder_hidden_states_mask is not None and self._static_encoder_hidden_states_mask is not None:
            self._static_encoder_hidden_states_mask.copy_(encoder_hidden_states_mask)

        # Replay the graph
        self.graph.replay()
        return self._static_output

    def replay(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Replay the captured CUDA graph with updated inputs.

        Only hidden_states and timestep change between denoising steps.
        Other inputs (prompt_embeds, mask, img_shapes) remain the same.

        Args:
            hidden_states: Updated latent tensor
            timestep: Updated timestep value

        Returns:
            Model output tensor
        """
        if not self.is_captured:
            raise RuntimeError("Must call capture() before replay()")

        # Update static input buffers in-place
        self._static_hidden_states.copy_(hidden_states)
        self._static_timestep.copy_(timestep)

        # Replay the graph
        self.graph.replay()

        return self._static_output

    def _eager_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: Optional[torch.Tensor],
        guidance: Optional[torch.Tensor],
        img_shapes: Optional[List],
        attention_kwargs: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        """Fallback eager execution."""
        return self.model(
            hidden_states=hidden_states,
            timestep=timestep / 1000,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            guidance=guidance,
            img_shapes=img_shapes,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]

    def reset(self):
        """Reset the graph runner, freeing the captured graph."""
        if self.graph is not None:
            del self.graph
            self.graph = None
        self.is_captured = False
        self._captured_hidden_shape = None
        self._captured_encoder_shape = None
        self._static_hidden_states = None
        self._static_timestep = None
        self._static_encoder_hidden_states = None
        self._static_encoder_hidden_states_mask = None
        self._static_output = None
        self._fixed_kwargs = {}
        self._mempool = None


def create_cuda_graph_denoising_fn(
    transformer: nn.Module,
    use_cuda_graphs: bool = False,
):
    """Create a denoising function that optionally uses CUDA graphs.

    Returns a callable that wraps transformer.forward() with CUDA graph
    capture/replay when enabled.

    Args:
        transformer: The transformer model
        use_cuda_graphs: Whether to enable CUDA graph capture/replay

    Returns:
        (denoise_fn, graph_runner_or_None)
    """
    if not use_cuda_graphs or not is_cuda_graphs_available():
        # Return a simple wrapper with no CUDA graph overhead
        def eager_denoise(
            hidden_states, timestep, encoder_hidden_states,
            encoder_hidden_states_mask=None, guidance=None,
            img_shapes=None, attention_kwargs=None,
        ):
            return transformer(
                hidden_states=hidden_states,
                timestep=timestep / 1000,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                guidance=guidance,
                img_shapes=img_shapes,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]

        return eager_denoise, None

    runner = CUDAGraphRunner(transformer)

    step_count = [0]

    def graph_denoise(
        hidden_states, timestep, encoder_hidden_states,
        encoder_hidden_states_mask=None, guidance=None,
        img_shapes=None, attention_kwargs=None,
    ):
        if step_count[0] == 0:
            # First step: capture
            output = runner.capture(
                hidden_states, timestep, encoder_hidden_states,
                encoder_hidden_states_mask, guidance,
                img_shapes, attention_kwargs,
            )
        else:
            if runner.is_captured:
                # Subsequent steps: replay
                output = runner.replay(hidden_states, timestep)
            else:
                # Capture failed, use eager fallback
                output = runner._eager_forward(
                    hidden_states, timestep, encoder_hidden_states,
                    encoder_hidden_states_mask, guidance,
                    img_shapes, attention_kwargs,
                )

        step_count[0] += 1
        return output

    return graph_denoise, runner
