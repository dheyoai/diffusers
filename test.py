import dotenv

dotenv.load_dotenv(override=True)

import argparse
import os

from PIL import Image

import torch
from torchvision.transforms.functional import to_pil_image, to_tensor

from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from omnigen.pipelines.flow_matching.pipeline_fm import FlowMatchingPipeline
from omnigen.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler as Scheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="black-forest-labs/FLUX.1-dev"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct"
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct"
    )
    parser.add_argument(
        "--num_inference_step",
        type=int,
        default=28,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=998244353
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default='bf16'
    )
    parser.add_argument(
        "--text_guidance_scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--image_guidance_scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="A dog running in the park",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar",
    )
    parser.add_argument(
        "--input_image_path",
        type=str,
        nargs='+',
        default=None,
    )
    parser.add_argument(
        "--output_image_path",
        type=str,
        default="output.png"
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    return args

def load_pipeline(args, accelerator, weight_dtype):
    transformer = OmniGen2Transformer2DModel.from_pretrained(args.model_path,
                                                             token='REMOVED_TOKENYVrtMysWgKpjKpdiquPiOMevDqhiDYkKRL')
    
    transformer = transformer.eval()
    transformer = transformer.to(accelerator.device, dtype=weight_dtype)
    
    transformer = accelerator.prepare(transformer)
    transformer = accelerator.unwrap_model(transformer)

    vae = AutoencoderKL.from_pretrained(args.vae_path, subfolder="vae")
    vae = vae.to(accelerator.device, dtype=weight_dtype)

    text_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    text_tokenizer.padding_side = "right"

    text_encoder = AutoModel.from_pretrained(args.text_encoder_path, torch_dtype=weight_dtype)
    text_encoder = text_encoder.eval()
    text_encoder = text_encoder.to(accelerator.device, dtype=weight_dtype)

    pipeline = FlowMatchingPipeline(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=text_tokenizer,
        scheduler=Scheduler(dynamic_time_shift=True),
    )
    pipeline = pipeline.to(accelerator.device, dtype=weight_dtype)
    return pipeline


def preprocess(args, pipeline):
    instruction = [{"role": "user", "content": args.instruction}]
    instruction = pipeline.tokenizer.apply_chat_template(instruction, tokenize=False, add_generation_prompt=False)
    if "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." in instruction:
        instruction = instruction.replace("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.", "You are a helpful assistant that generates high-quality images based on user instructions.")
    else:
        instruction = instruction.replace("You are a helpful assistant.", "You are a helpful assistant that generates high-quality images based on user instructions.")

    negative_prompt = [{"role": "user", "content": args.negative_prompt}]
    negative_prompt = pipeline.tokenizer.apply_chat_template(negative_prompt, tokenize=False, add_generation_prompt=False)

    if "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." in negative_prompt:
        negative_prompt = negative_prompt.replace("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.", "You are a helpful assistant.")

    if args.input_image_path is None:
        input_images = []
    else:
        if len(args.input_image_path) == 1:
            if os.path.isdir(args.input_image_path[0]):
                input_images = [Image.open(os.path.join(args.input_image_path[0], f)) for f in os.listdir(args.input_image_path[0])]
            else:
                input_images = [Image.open(args.input_image_path[0])]
        else:
            input_images = [Image.open(path) for path in args.input_image_path]

    return instruction, negative_prompt, input_images

def run(args, accelerator, pipeline, instruction, negative_prompt, input_images):
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    results = pipeline(
        prompt=instruction,
        input_images=input_images,
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_step,
        max_sequence_length=1024,
        guidance_scale=args.text_guidance_scale,
        negative_prompt=negative_prompt,
        num_images_per_prompt=args.num_images_per_prompt,
        generator=generator,
        output_type="pil",
    )
    
    vis_images = [to_tensor(image) * 2 - 1 for image in results.images]

    # Concatenate input images of different sizes horizontally
    max_height = max(img.shape[-2] for img in vis_images)
    total_width = sum(img.shape[-1] for img in vis_images)
    canvas = torch.zeros((3, max_height, total_width), device=vis_images[0].device)
    
    current_x = 0
    for i, img in enumerate(vis_images):
        h, w = img.shape[-2:]
        # Place image at the top of canvas
        canvas[:, :h, current_x:current_x+w] = img * 0.5 + 0.5
        current_x += w
    output_image = to_pil_image(canvas)
    return output_image

def main(args, root_dir):
    accelerator = Accelerator(mixed_precision=args.dtype if args.dtype != 'fp32' else 'no')

    weight_dtype = torch.float32
    if args.dtype == 'fp16':
        weight_dtype = torch.float16
    elif args.dtype == 'bf16':
        weight_dtype = torch.bfloat16

    pipeline = load_pipeline(args, accelerator, weight_dtype)

    instruction, negative_prompt, input_images = preprocess(args, pipeline)
    output_image = run(args, accelerator, pipeline, instruction, negative_prompt, input_images)
    output_image.save(args.output_image_path)

if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(__file__, os.path.pardir))
    args = parse_args()
    main(args, root_dir)