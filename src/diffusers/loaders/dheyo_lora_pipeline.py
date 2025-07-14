from .lora_pipeline import WanLoraLoaderMixin
import os
from typing import Callable, Dict, List, Optional, Union

import torch
from .lora_base import (  
    LORA_WEIGHT_NAME,
    LORA_WEIGHT_NAME_SAFE,
    LoraBaseMixin,
    _fetch_state_dict,
    _load_lora_into_text_encoder,
    _pack_dict_with_prefix,
)

class WanLoraLoaderMixinWrapper (WanLoraLoaderMixin):
    @classmethod
    # Copied from diffusers.loaders.lora_pipeline.CogVideoXLoraLoaderMixin.save_lora_weights
    def save_lora_weights(
        cls,
        save_directory: Union[str, os.PathLike],
        transformer_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
        text_encoder_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = True,
        transformer_lora_adapter_metadata: Optional[dict] = None,
        text_encoder_lora_adapter_metadata=None,
    ):
        r"""
        Save the LoRA parameters corresponding to the transformer.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save LoRA parameters to. Will be created if it doesn't exist.
            transformer_lora_layers (`Dict[str, torch.nn.Module]` or `Dict[str, torch.Tensor]`):
                State dict of the LoRA layers corresponding to the `transformer`.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful during distributed training and you
                need to call this function on all processes. In this case, set `is_main_process=True` only on the main
                process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful during distributed training when you need to
                replace `torch.save` with another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way with `pickle`.
            transformer_lora_adapter_metadata:
                LoRA adapter metadata associated with the transformer to be serialized with the state dict.
        """
        state_dict = {}
        lora_adapter_metadata = {}

        if not transformer_lora_layers:
            raise ValueError("You must pass `transformer_lora_layers`.")

        state_dict.update(cls.pack_weights(transformer_lora_layers, cls.transformer_name))

        if transformer_lora_adapter_metadata is not None:
            lora_adapter_metadata.update(
                _pack_dict_with_prefix(transformer_lora_adapter_metadata, cls.transformer_name)
            )

        if text_encoder_lora_layers:
            state_dict.update(cls.pack_weights(text_encoder_lora_layers, "text_encoder"))

        if text_encoder_lora_adapter_metadata:
            lora_adapter_metadata.update(
                _pack_dict_with_prefix(text_encoder_lora_adapter_metadata, cls.text_encoder_name)
            )
        # Save the model
        cls.write_lora_layers(
            state_dict=state_dict,
            save_directory=save_directory,
            is_main_process=is_main_process,
            weight_name=weight_name,
            save_function=save_function,
            safe_serialization=safe_serialization,
            lora_adapter_metadata=lora_adapter_metadata,
        )