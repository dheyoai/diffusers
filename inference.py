import dotenv

dotenv.load_dotenv(override=True)

import argparse
import os
from typing import List, Tuple

from PIL import Image

import torch
from torchvision.transforms.functional import to_pil_image, to_tensor

from accelerate import Accelerator

from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OmniGen2 image generation script.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--num_inference_step",
        type=int,
        default=28,
        help="Number of inference steps."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=998244353,
        help="Random seed for generation."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Output image height."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Output image width."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default='bf16',
        choices=['fp32', 'fp16', 'bf16'],
        help="Data type for model weights."
    )
    parser.add_argument(
        "--text_guidance_scale",
        type=float,
        default=1.0,
        help="Text guidance scale."
    )
    parser.add_argument(
        "--image_guidance_scale",
        type=float,
        default=1.0,
        help="Image guidance scale."
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="A dog running in the park",
        help="Text prompt for generation."
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar",
        help="Negative prompt for generation."
    )
    parser.add_argument(
        "--input_image_path",
        type=str,
        nargs='+',
        default=None,
        help="Path(s) to input image(s)."
    )
    parser.add_argument(
        "--output_image_path",
        type=str,
        default="output.png",
        help="Path to save output image."
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate per prompt."
    )
    return parser.parse_args()

def load_pipeline(args: argparse.Namespace, accelerator: Accelerator, weight_dtype: torch.dtype) -> OmniGen2Pipeline:
    pipeline = OmniGen2Pipeline.from_pretrained(args.model_path,
                                                torch_dtype=weight_dtype,
                                                trust_remote_code=True,
                                                token=os.getenv("HF_TOKEN"))
    pipeline = pipeline.to(accelerator.device, dtype=weight_dtype)
    return pipeline

def preprocess(args: argparse.Namespace, pipeline: OmniGen2Pipeline) -> Tuple[str, str, List[Image.Image]]:
    """Preprocess the instruction, negative prompt, and input images."""
    # Process instruction
    instruction = [{"role": "user", "content": args.instruction}]
    instruction = pipeline.tokenizer.apply_chat_template(instruction, tokenize=False, add_generation_prompt=False)
    if "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." in instruction:
        instruction = instruction.replace("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.", "You are a helpful assistant that generates high-quality images based on user instructions.")
    else:
        instruction = instruction.replace("You are a helpful assistant.", "You are a helpful assistant that generates high-quality images based on user instructions.")

    # Process negative prompt
    negative_prompt = [{"role": "user", "content": args.negative_prompt}]
    negative_prompt = pipeline.tokenizer.apply_chat_template(negative_prompt, tokenize=False, add_generation_prompt=False)

    # Process input images
    input_images = []
    if args.input_image_path:
        if len(args.input_image_path) == 1 and os.path.isdir(args.input_image_path[0]):
            input_images = [Image.open(os.path.join(args.input_image_path[0], f)) 
                          for f in os.listdir(args.input_image_path[0])]
        else:
            input_images = [Image.open(path) for path in args.input_image_path]

    return instruction, negative_prompt, input_images

def run(args: argparse.Namespace, 
        accelerator: Accelerator, 
        pipeline: OmniGen2Pipeline, 
        instruction: str, 
        negative_prompt: str, 
        input_images: List[Image.Image]) -> Image.Image:
    """Run the image generation pipeline with the given parameters."""
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    results = pipeline(
        prompt=instruction,
        input_images=input_images,
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_step,
        max_sequence_length=1024,
        text_guidance_scale=args.text_guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        negative_prompt=negative_prompt,
        num_images_per_prompt=args.num_images_per_prompt,
        generator=generator,
        output_type="pil",
    )
    
    vis_images = [to_tensor(image) * 2 - 1 for image in results.images]
    output_image = create_collage(vis_images)
    return output_image

def create_collage(images: List[torch.Tensor]) -> Image.Image:
    """Create a horizontal collage from a list of images."""
    max_height = max(img.shape[-2] for img in images)
    total_width = sum(img.shape[-1] for img in images)
    canvas = torch.zeros((3, max_height, total_width), device=images[0].device)
    
    current_x = 0
    for img in images:
        h, w = img.shape[-2:]
        canvas[:, :h, current_x:current_x+w] = img * 0.5 + 0.5
        current_x += w
    
    return to_pil_image(canvas)

def main(args: argparse.Namespace, root_dir: str) -> None:
    """Main function to run the image generation process."""
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision=args.dtype if args.dtype != 'fp32' else 'no')

    # Set weight dtype
    weight_dtype = torch.float32
    if args.dtype == 'fp16':
        weight_dtype = torch.float16
    elif args.dtype == 'bf16':
        weight_dtype = torch.bfloat16

    # Load pipeline and process inputs
    pipeline = load_pipeline(args, accelerator, weight_dtype)
    instruction, negative_prompt, input_images = preprocess(args, pipeline)
    
    # Generate and save image
    output_image = run(args, accelerator, pipeline, instruction, negative_prompt, input_images)
    output_image.save(args.output_image_path)
    print(f"Image saved to {args.output_image_path}")

if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(__file__, os.path.pardir))
    args = parse_args()
    main(args, root_dir)