import argparse
import gc
import itertools
import json
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path
import uuid

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from packaging import version
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params
from diffusers.utils import (
    check_min_version,
    convert_all_state_dict_to_peft,
    convert_state_dict_to_diffusers,
    convert_state_dict_to_kohya,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

if is_wandb_available():
    import wandb

check_min_version("0.34.0.dev0")

logger = get_logger(__name__)

def save_model_card(
    repo_id: str,
    use_dora: bool,
    images=None,
    base_model: str = None,
    train_text_encoder=False,
    instance_prompt=None,
    validation_prompt=None,
    repo_folder=None,
    vae_path=None,
):
    widget_dict = []
    if images is not None:
        for i, image_set in enumerate(images):
            for j, image in enumerate(image_set):
                image.save(os.path.join(repo_folder, f"image_{i}_{j}.png"))
                widget_dict.append(
                    {"text": validation_prompt[i] if validation_prompt else " ", "output": {"url": f"image_{i}_{j}.png"}}
                )

    model_description = f"""
# {"SDXL" if "playground" not in base_model else "Playground"} LoRA DreamBooth - {repo_id}

<Gallery />

## Model description

These are {repo_id} LoRA adaption weights for {base_model}.

The weights were trained using [DreamBooth](https://dreambooth.github.io/) with multiple subjects.

LoRA for the text encoder was enabled: {train_text_encoder}.

Special VAE used for training: {vae_path}.

## Trigger words

You should use the following prompts to trigger image generation: {', '.join(instance_prompt) if instance_prompt else 'None'}.

## Download model

Weights for this model are available in Safetensors format.

[Download]({repo_id}/tree/main) them in the Files & versions tab.
"""
    if "playground" in base_model:
        model_description += """\n
## License

Please adhere to the licensing terms as described [here](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic/blob/main/LICENSE.md).
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="openrail++" if "playground" not in base_model else "playground-v2dot5-community",
        base_model=base_model,
        prompt=instance_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "lora" if not use_dora else "dora",
        "template:sd-lora",
    ]
    if "playground" in base_model:
        tags.extend(["playground", "playground-diffusers"])
    else:
        tags.extend(["stable-diffusion-xl", "stable-diffusion-xl-diffusers"])

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))

def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args_list,
    epoch,
    torch_dtype,
    is_final_validation=False,
):
    logger.info(f"Running validation for {len(pipeline_args_list)} prompts.")
    images_sets = []

    scheduler_args = {}
    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type
        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"
        scheduler_args["variance_type"] = variance_type
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
    autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()

    for pipeline_args in pipeline_args_list:
        with autocast_ctx:
            images = [pipeline(**pipeline_args, generator=generator).images[0] for _ in range(pipeline_args["num_images"])]
        images_sets.append(images)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            for i, images in enumerate(images_sets):
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images(f"{phase_name}_{i}", np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            for i, images in enumerate(images_sets):
                tracker.log(
                    {
                        f"{phase_name}_{i}": [
                            wandb.Image(image, caption=f"{j}: {pipeline_args_list[i]['prompt']}") 
                            for j, image in enumerate(images)
                        ]
                    }
                )

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return images_sets

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="LoRA fine-tuning script for Stable Diffusion XL with multi-subject support.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--concepts_list",
        type=str,
        default=None,
        help="Path to JSON file containing multiple concepts.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora-dreambooth-model",
    )
    parser.add_argument(
        "--output_kohya_format",
        action="store_true",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--prodigy_decouple",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--adam_weight_decay_text_encoder",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
    )
    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--validation_negative_prompt",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--validation_inference_steps",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--validation_guidance_scale",
        type=float,
        default=7.5,
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--use_dora",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--image_interpolation_mode",
        type=str,
        default="lanczos",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if not args.concepts_list and (not args.instance_data_dir or not args.instance_prompt):
        raise ValueError(
            "You must specify either instance parameters (data directory, prompt, etc.) or use "
            "the `concepts_list` parameter and specify them within the file."
        )

    if args.concepts_list:
        if args.instance_prompt:
            raise ValueError("If using `concepts_list`, define instance prompt within the file.")
        if args.instance_data_dir:
            raise ValueError("If using `concepts_list`, define instance data directory within the file.")
        if args.validation_epochs and (args.validation_prompt or args.validation_negative_prompt):
            raise ValueError(
                "If using `concepts_list`, define validation parameters within the file."
            )

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if not args.concepts_list:
            if not args.class_data_dir:
                raise ValueError("You must specify a data directory for class images.")
            if not args.class_prompt:
                raise ValueError("You must specify prompt for class images.")
        else:
            if args.class_data_dir:
                raise ValueError("If using `concepts_list`, define class data directory within the file.")
            if args.class_prompt:
                raise ValueError("If using `concepts_list`, define class prompt within the file.")
    else:
        if args.class_data_dir:
            warnings.warn(
                "Ignoring `class_data_dir` parameter, use it with `with_prior_preservation`."
            )
        if args.class_prompt:
            warnings.warn(
                "Ignoring `class_prompt` parameter, use it with `with_prior_preservation`."
            )

    return args

class DreamBoothDataset(Dataset):
    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        dataset_names,
        tokenizer_one,
        tokenizer_two,
        class_data_root=None,
        class_prompt=None,
        class_num=None,
        size=1024,
        repeats=1,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.instance_data_root = []
        self.instance_images_path = []
        self.instance_images = []
        self.num_instance_images = []
        self.crop_top_lefts = []
        self.original_sizes = []
        self.instance_prompt = []
        self.custom_instance_prompts = []
        self.class_data_root = [] if class_data_root is not None else None
        self.class_images_path = []
        self.num_class_images = []
        self.class_prompt = []
        self._length = 0

        interpolation = getattr(transforms.InterpolationMode, args.image_interpolation_mode.upper(), None)
        if interpolation is None:
            raise ValueError(f"Unsupported interpolation mode {interpolation=}.")
        train_resize = transforms.Resize(size, interpolation=interpolation)
        train_crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        train_flip = transforms.RandomHorizontalFlip(p=1.0)

        if dataset_names and args.caption_column and args.image_column:
            for i in range(len(dataset_names)):
                try:
                    from datasets import load_dataset
                except ImportError:
                    raise ImportError(
                        "You are trying to load data using the datasets library. Install it with: `pip install datasets`."
                    )
                dataset = load_dataset(
                    dataset_names[i],
                    args.dataset_config_name,
                    cache_dir=args.cache_dir,
                )
                column_names = dataset["train"].column_names
                image_column = args.image_column
                if image_column not in column_names:
                    raise ValueError(
                        f"`--image_column` value '{args.image_column}' not found in dataset columns."
                    )
                instance_images = dataset["train"][image_column]
                self.instance_images.append(instance_images)
                self.num_instance_images.append(len(instance_images))
                self._length += self.num_instance_images[i]

                if args.caption_column not in column_names:
                    raise ValueError(
                        f"`--caption_column` value '{args.caption_column}' not found in dataset columns."
                    )
                custom_instance_prompts = dataset["train"][args.caption_column]
                temp_custom_instance_prompts = []
                for caption in custom_instance_prompts:
                    temp_custom_instance_prompts.extend(itertools.repeat(caption, repeats))
                self.custom_instance_prompts.append(temp_custom_instance_prompts)
        else:
            for i in range(len(instance_data_root)):
                self.instance_data_root.append(Path(instance_data_root[i]))
                if not self.instance_data_root[i].exists():
                    raise ValueError(f"Instance images root {instance_data_root[i]} doesn't exist.")
                instance_images = [Image.open(path) for path in list(Path(instance_data_root[i]).iterdir())]
                self.num_instance_images.append(len(instance_images))
                self.instance_prompt.append(instance_prompt[i])
                self._length += self.num_instance_images[i]
                self.instance_images.append(instance_images)

        for i in range(len(self.instance_images)):
            instance_images = self.instance_images[i]
            temp_crop_top_lefts = []
            temp_original_sizes = []
            for image in instance_images:
                image = exif_transpose(image)
                if not image.mode == "RGB":
                    image = image.convert("RGB")
                temp_original_sizes.append((image.height, image.width))
                image = train_resize(image)
                if args.random_flip and random.random() < 0.5:
                    image = train_flip(image)
                if args.center_crop:
                    y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
                    x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
                    image = train_crop(image)
                else:
                    y1, x1, h, w = train_crop.get_params(image, (args.resolution, args.resolution))
                    image = crop(image, y1, x1, h, w)
                crop_top_left = (y1, x1)
                temp_crop_top_lefts.append(crop_top_left)
            self.crop_top_lefts.append(temp_crop_top_lefts)
            self.original_sizes.append(temp_original_sizes)

            if class_data_root is not None:
                self.class_data_root.append(Path(class_data_root[i]))
                self.class_data_root[i].mkdir(parents=True, exist_ok=True)
                self.class_images_path.append(list(self.class_data_root[i].iterdir()))
                self.num_class_images.append(len(self.class_images_path[i]))
                if self.num_class_images[i] > self.num_instance_images[i]:
                    self._length -= self.num_instance_images[i]
                    self._length += self.num_class_images[i]
                self.class_prompt.append(class_prompt[i])

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        for i in range(len(self.instance_images)):
            instance_image = self.instance_images[i][index % self.num_instance_images[i]]
            original_size = self.original_sizes[i][index % self.num_instance_images[i]]
            crop_top_left = self.crop_top_lefts[i][index % self.num_instance_images[i]]
            if not instance_image.mode == "RGB":
                instance_image = instance_image.convert("RGB")
            example[f"instance_images_{i}"] = self.image_transforms(instance_image)
            example[f"original_sizes_{i}"] = original_size
            example[f"crop_top_lefts_{i}"] = crop_top_left
            example[f"instance_prompt_ids_{i}"] = self.tokenizer_one(
                self.custom_instance_prompts[i][index % self.num_instance_images[i]] if self.custom_instance_prompts else self.instance_prompt[i],
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer_one.model_max_length,
                return_tensors="pt",
            ).input_ids
            example[f"instance_prompt_ids_2_{i}"] = self.tokenizer_two(
                self.custom_instance_prompts[i][index % self.num_instance_images[i]] if self.custom_instance_prompts else self.instance_prompt[i],
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer_two.model_max_length,
                return_tensors="pt",
            ).input_ids

        if self.class_data_root:
            for i in range(len(self.class_data_root)):
                class_image = Image.open(self.class_images_path[i][index % self.num_class_images[i]])
                if not class_image.mode == "RGB":
                    class_image = class_image.convert("RGB")
                example[f"class_images_{i}"] = self.image_transforms(class_image)
                example[f"class_prompt_ids_{i}"] = self.tokenizer_one(
                    self.class_prompt[i],
                    truncation=True,
                    padding="max_length",
                    max_length=self.tokenizer_one.model_max_length,
                    return_tensors="pt",
                ).input_ids
                example[f"class_prompt_ids_2_{i}"] = self.tokenizer_two(
                    self.class_prompt[i],
                    truncation=True,
                    padding="max_length",
                    max_length=self.tokenizer_two.model_max_length,
                    return_tensors="pt",
                ).input_ids

        return example

def collate_fn(num_instances, examples, with_prior_preservation=False):
    input_ids = []
    input_ids_2 = []
    pixel_values = []
    original_sizes = []
    crop_top_lefts = []
    for i in range(num_instances):
        input_ids += [example[f"instance_prompt_ids_{i}"] for example in examples]
        input_ids_2 += [example[f"instance_prompt_ids_2_{i}"] for example in examples]
        pixel_values += [example[f"instance_images_{i}"] for example in examples]
        original_sizes += [example[f"original_sizes_{i}"] for example in examples]
        crop_top_lefts += [example[f"crop_top_lefts_{i}"] for example in examples]
    if with_prior_preservation:
        for i in range(num_instances):
            input_ids += [example[f"class_prompt_ids_{i}"] for example in examples]
            input_ids_2 += [example[f"class_prompt_ids_2_{i}"] for example in examples]
            pixel_values += [example[f"class_images_{i}"] for example in examples]
            original_sizes += [example[f"original_sizes_{i}"] for example in examples]
            crop_top_lefts += [example[f"crop_top_lefts_{i}"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.cat(input_ids, dim=0)
    input_ids_2 = torch.cat(input_ids_2, dim=0)

    batch = {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "input_ids_2": input_ids_2,
        "original_sizes": original_sizes,
        "crop_top_lefts": crop_top_lefts,
    }
    return batch

def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []
    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]
        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False
        )
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)
    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    instance_data_dir = []
    instance_prompt = []
    dataset_names = []
    class_data_dir = [] if args.with_prior_preservation else None
    class_prompt = [] if args.with_prior_preservation else None
    validation_prompts = []
    validation_number_images = []
    validation_negative_prompts = []
    validation_inference_steps = []
    validation_guidance_scales = []

    if args.concepts_list:
        with open(args.concepts_list, "r") as f:
            concepts_list = json.load(f)
        for concept in concepts_list:
            if concept.get("dataset_name"):
                dataset_names.append(concept["dataset_name"])
                instance_data_dir.append(concept["dataset_name"])
            else:
                instance_data_dir.append(concept["instance_data_dir"])
                instance_prompt.append(concept["instance_prompt"])
            if args.with_prior_preservation:
                try:
                    class_data_dir.append(concept["class_data_dir"])
                    class_prompt.append(concept["class_prompt"])
                except KeyError:
                    raise KeyError(
                        "`class_data_dir` or `class_prompt` not found in concepts_list with `with_prior_preservation`."
                    )
            if args.validation_epochs:
                validation_prompts.append(concept.get("validation_prompt", None))
                validation_number_images.append(concept.get("validation_number_images", 4))
                validation_negative_prompts.append(concept.get("validation_negative_prompt", None))
                validation_inference_steps.append(concept.get("validation_inference_steps", 25))
                validation_guidance_scales.append(concept.get("validation_guidance_scale", 7.5))
    else:
        if args.dataset_name:
            dataset_names.append(args.dataset_name)
        else:
            instance_data_dir = args.instance_data_dir.split(",")
            instance_prompt = args.instance_prompt.split(",")
            assert len(instance_data_dir) == len(instance_prompt), (
                "Instance data dir and prompt inputs must have the same length."
            )
        if args.with_prior_preservation:
            class_data_dir = args.class_data_dir.split(",")
            class_prompt = args.class_prompt.split(",")
            assert len(instance_data_dir) == len(class_data_dir) == len(class_prompt), (
                "Instance and class data dir or prompt inputs must have the same length."
            )
        if args.validation_epochs:
            validation_prompts = args.validation_prompt.split(",") if args.validation_prompt else [None]
            num_of_validation_prompts = len(validation_prompts)
            validation_number_images = [args.num_validation_images] * num_of_validation_prompts
            validation_negative_prompts = args.validation_negative_prompt.split(",") if args.validation_negative_prompt else [None] * num_of_validation_prompts
            validation_inference_steps = [args.validation_inference_steps] * num_of_validation_prompts
            validation_guidance_scales = [args.validation_guidance_scale] * num_of_validation_prompts

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if args.with_prior_preservation:
        for i in range(len(class_data_dir)):
            class_images_dir = Path(class_data_dir[i])
            if not class_images_dir.exists():
                class_images_dir.mkdir(parents=True, exist_ok=True)
            cur_class_images = len(list(class_images_dir.iterdir()))
            if cur_class_images < args.num_class_images:
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                if args.prior_generation_precision == "fp32":
                    torch_dtype = torch.float32
                elif args.prior_generation_precision == "fp16":
                    torch_dtype = torch.float16
                elif args.prior_generation_precision == "bf16":
                    torch_dtype = torch.bfloat16
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    torch_dtype=torch_dtype,
                    revision=args.revision,
                    variant=args.variant,
                )
                pipeline.set_progress_bar_config(disable=True)
                num_new_images = args.num_class_images - cur_class_images
                logger.info(f"Number of class images to sample for concept {i}: {num_new_images}.")
                sample_dataset = PromptDataset(class_prompt[i], num_new_images)
                sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)
                sample_dataloader = accelerator.prepare(sample_dataloader)
                pipeline.to(accelerator.device)
                for example in tqdm(
                    sample_dataloader, desc=f"Generating class images for concept {i}", disable=not accelerator.is_local_main_process
                ):
                    images = pipeline(example["prompt"]).images
                    for ii, image in enumerate(images):
                        hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                        image_filename = class_images_dir / f"{example['index'][ii] + cur_class_images}-{hash_image}.jpg"
                        image.save(image_filename)
                del pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae_path = args.pretrained_vae_model_name_or_path or args.pretrained_model_name_or_path
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Use fp16 or fp32 instead."
        )

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    unet_target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=unet_target_modules,
        use_dora=args.use_dora,
    )
    unet.add_adapter(unet_lora_config)

    if args.train_text_encoder:
        text_target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
        text_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            lora_dropout=args.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=text_target_modules,
            use_dora=args.use_dora,
        )
        text_encoder_one.add_adapter(text_lora_config)
        text_encoder_two.add_adapter(text_lora_config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. Update to at least 0.0.17."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Install it with: `pip install xformers`")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            unet_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            text_encoder_two_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                elif isinstance(model, type(unwrap_model(text_encoder_one))):
                    text_encoder_one_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                elif isinstance(model, type(unwrap_model(text_encoder_two))):
                    text_encoder_two_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                weights.pop()

            StableDiffusionXLPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_two))):
                text_encoder_two_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(input_dir)
        unet_state_dict = {f"{k.replace('unet.', '')}": v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(f"Unexpected keys in adapter weights: {unexpected_keys}")

        if args.train_text_encoder:
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_)
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder_2.", text_encoder=text_encoder_two_)

        if args.mixed_precision == "fp16":
            models = [unet_]
            if args.train_text_encoder:
                models.extend([text_encoder_one_, text_encoder_two_])
            cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.mixed_precision == "fp16":
        models = [unet]
        if args.train_text_encoder:
            models.extend([text_encoder_one, text_encoder_two])
        cast_training_params(models, dtype=torch.float32)

    unet_lora_parameters = list(filter(lambda p: p.requires_grad, unet.parameters()))
    if args.train_text_encoder:
        text_lora_parameters_one = list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))
        text_lora_parameters_two = list(filter(lambda p: p.requires_grad, text_encoder_two.parameters()))

    unet_lora_parameters_with_lr = {"params": unet_lora_parameters, "lr": args.learning_rate}
    if args.train_text_encoder:
        text_lora_parameters_one_with_lr = {
            "params": text_lora_parameters_one,
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        text_lora_parameters_two_with_lr = {
            "params": text_lora_parameters_two,
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        params_to_optimize = [
            unet_lora_parameters_with_lr,
            text_lora_parameters_one_with_lr,
            text_lora_parameters_two_with_lr,
        ]
    else:
        params_to_optimize = [unet_lora_parameters_with_lr]

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, install bitsandbytes: `pip install bitsandbytes`."
                )
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    elif args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, install prodigyopt: `pip install prodigyopt`")
        optimizer_class = prodigyopt.Prodigy
        if args.learning_rate <= 0.1:
            logger.warning("Learning rate is too low for Prodigy. Consider setting around 1.0.")
        if args.train_text_encoder and args.text_encoder_lr:
            logger.warning("Prodigy uses only learning_rate. Text encoder LR will be ignored.")
            params_to_optimize[1]["lr"] = args.learning_rate
            params_to_optimize[2]["lr"] = args.learning_rate
        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )
    else:
        logger.warning(f"Unsupported optimizer: {args.optimizer}. Defaulting to AdamW.")
        optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    train_dataset = DreamBoothDataset(
        instance_data_root=instance_data_dir,
        instance_prompt=instance_prompt,
        dataset_names=dataset_names,
        class_data_root=class_data_dir,
        class_prompt=class_prompt,
        class_num=args.num_class_images,
        size=args.resolution,
        repeats=args.repeats,
        center_crop=args.center_crop,
        tokenizer_one=tokenizer_one,
        tokenizer_two=tokenizer_two,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(len(instance_data_dir or dataset_names), examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )

    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * accelerator.num_processes * num_update_steps_per_epoch
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    if args.train_text_encoder:
        unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        tracker_name = (
            "dreambooth-lora-sd-xl"
            if "playground" not in args.pretrained_model_name_or_path
            else "dreambooth-lora-playground"
        )
        accelerator.init_trackers(tracker_name, config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    def compute_time_ids(original_size, crops_coords_top_left):
        target_size = (args.resolution, args.resolution)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
        return add_time_ids

    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]

    def compute_text_embeddings(prompt, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt)
            prompt_embeds = prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
        return prompt_embeds, pooled_prompt_embeds

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder_one.train()
            text_encoder_two.train()
            accelerator.unwrap_model(text_encoder_one).text_model.embeddings.requires_grad_(True)
            accelerator.unwrap_model(text_encoder_two).text_model.embeddings.requires_grad_(True)

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                model_input = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                )
                timesteps = timesteps.long()

                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
                add_time_ids = torch.cat(
                    [
                        compute_time_ids(original_size=s, crops_coords_top_left=c)
                        for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])
                    ]
                )

                if args.train_text_encoder:
                    unet_added_conditions = {"time_ids": add_time_ids}
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two],
                        tokenizers=None,
                        prompt=None,
                        text_input_ids_list=[batch["input_ids"], batch["input_ids_2"]],
                    )
                    unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})
                    prompt_embeds_input = prompt_embeds
                else:
                    prompt_embeds_list = []
                    for i in range(len(instance_prompt)):
                        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                            instance_prompt[i], text_encoders, tokenizers
                        )
                        prompt_embeds_list.append(prompt_embeds)
                    prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
                    unet_add_text_embeds = torch.cat([compute_text_embeddings(p, text_encoders, tokenizers)[1] for p in instance_prompt], dim=0)
                    if args.with_prior_preservation:
                        class_prompt_embeds_list = []
                        for i in range(len(class_prompt)):
                            class_prompt_embeds, class_pooled_prompt_embeds = compute_text_embeddings(
                                class_prompt[i], text_encoders, tokenizers
                            )
                            class_prompt_embeds_list.append(class_prompt_embeds)
                        prompt_embeds = torch.cat([prompt_embeds] + class_prompt_embeds_list, dim=0)
                        unet_add_text_embeds = torch.cat([unet_add_text_embeds, torch.cat([compute_text_embeddings(p, text_encoders, tokenizers)[1] for p in class_prompt], dim=0)], dim=0)
                    unet_added_conditions = {"time_ids": add_time_ids, "text_embeds": unet_add_text_embeds}
                    prompt_embeds_input = prompt_embeds

                model_pred = unet(
                    noisy_model_input,
                    timesteps,
                    prompt_embeds_input,
                    added_cond_kwargs=unet_added_conditions,
                    return_dict=False,
                )[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.with_prior_preservation:
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet_lora_parameters, text_lora_parameters_one, text_lora_parameters_two)
                        if args.train_text_encoder
                        else unet_lora_parameters
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                logger.info(f"Removing {len(removing_checkpoints)} checkpoints")
                                for removing_checkpoint in removing_checkpoints:
                                    shutil.rmtree(os.path.join(args.output_dir, removing_checkpoint))
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompt and global_step % args.validation_epochs == 0:
                        pipeline = StableDiffusionXLPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            vae=vae,
                            text_encoder=unwrap_model(text_encoder_one),
                            text_encoder_2=unwrap_model(text_encoder_two),
                            unet=unwrap_model(unet),
                            revision=args.revision,
                            variant=args.variant,
                            torch_dtype=weight_dtype,
                        )
                        pipeline_args_list = [
                            {
                                "prompt": vp,
                                "num_images": nvi,
                                "negative_prompt": vnp,
                                "num_inference_steps": vis,
                                "guidance_scale": vgs,
                            }
                            for vp, nvi, vnp, vis, vgs in zip(
                                validation_prompts,
                                validation_number_images,
                                validation_negative_prompts,
                                validation_inference_steps,
                                validation_guidance_scales,
                            )
                        ]
                        images = log_validation(
                            pipeline,
                            args,
                            accelerator,
                            pipeline_args_list,
                            epoch,
                            torch_dtype=weight_dtype,
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet).to(torch.float32)
        unet_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

        if args.train_text_encoder:
            text_encoder_one = unwrap_model(text_encoder_one).to(torch.float32)
            text_encoder_lora_layers = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(text_encoder_one)
            )
            text_encoder_two = unwrap_model(text_encoder_two).to(torch.float32)
            text_encoder_2_lora_layers = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(text_encoder_two)
            )
        else:
            text_encoder_lora_layers = None
            text_encoder_2_lora_layers = None

        StableDiffusionXLPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers,
            text_encoder_2_lora_layers=text_encoder_2_lora_layers,
        )
        if args.output_kohya_format:
            lora_state_dict = load_file(f"{args.output_dir}/pytorch_lora_weights.safetensors")
            peft_state_dict = convert_all_state_dict_to_peft(lora_state_dict)
            kohya_state_dict = convert_state_dict_to_kohya(peft_state_dict)
            save_file(kohya_state_dict, f"{args.output_dir}/pytorch_lora_weights_kohya.safetensors")

        vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        pipeline.load_lora_weights(args.output_dir)

        images = []
        if validation_prompts and args.num_validation_images > 0:
            pipeline_args_list = [
                {
                    "prompt": vp,
                    "num_images": nvi,
                    "negative_prompt": vnp,
                    "num_inference_steps": vis,
                    "guidance_scale": vgs,
                }
                for vp, nvi, vnp, vis, vgs in zip(
                    validation_prompts,
                    validation_number_images,
                    validation_negative_prompts,
                    validation_inference_steps,
                    validation_guidance_scales,
                )
            ]
            images = log_validation(
                pipeline,
                args,
                accelerator,
                pipeline_args_list,
                epoch,
                is_final_validation=True,
                torch_dtype=weight_dtype,
            )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                use_dora=args.use_dora,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                train_text_encoder=args.train_text_encoder,
                instance_prompt=instance_prompt,
                validation_prompt=validation_prompts,
                repo_folder=args.output_dir,
                vae_path=args.pretrained_vae_model_name_or_path,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)