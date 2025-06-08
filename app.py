import os
import random
from typing import List, Optional, Tuple, Union
from datetime import datetime

import gradio as gr
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor
from accelerate import Accelerator
from transformers import AutoTokenizer, Qwen2_5_VLModel as TextEncoder
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
import dotenv
from omegaconf import OmegaConf

from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler as Scheduler

# Load environment variables
dotenv.load_dotenv(override=True)

# Configuration
class Config:
    CONFIG_PATH = os.getenv("CONFIG_PATH", "configs/default.yml")
    MODEL_PATH = os.getenv("MODEL_PATH", "pretrained_models/model.bin")
    VAE_PATH = os.getenv("VAE_PATH", "black-forest-labs/FLUX.1-dev")
    TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "Qwen/Qwen2.5-VL-3B-Instruct")
    TEXT_ENCODER_PATH = os.getenv("TEXT_ENCODER_PATH", "Qwen/Qwen2.5-VL-3B-Instruct")
    DEFAULT_NEGATIVE_PROMPT = "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"
    SAVE_IMAGES = True
    OUTPUT_DIR = "outputs"

# Initialize global variables
pipeline = None
generator = None
save_images = Config.SAVE_IMAGES

def load_pipeline(time_shift_scale: float, accelerator: Accelerator, weight_dtype: torch.dtype) -> OmniGen2Pipeline:
    """Load and initialize the OmniGen2 pipeline with all necessary components."""
    conf = OmegaConf.load(Config.CONFIG_PATH)
    transformer = OmniGen2Transformer2DModel(**conf.model.arch_opt)
    
    state_dict = torch.load(Config.MODEL_PATH, map_location='cpu')
    missing, unexpect = transformer.load_state_dict(state_dict, strict=False)
    print(f"Missing parameters: {missing}")
    print(f"Unexpected parameters: {unexpect}")
    
    transformer = transformer.eval()
    transformer = transformer.to(accelerator.device, dtype=weight_dtype)
    transformer = accelerator.prepare(transformer)
    transformer = accelerator.unwrap_model(transformer)

    vae = AutoencoderKL.from_pretrained(Config.VAE_PATH, subfolder="vae")
    vae = vae.to(accelerator.device, dtype=weight_dtype)

    text_tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_PATH)
    text_tokenizer.padding_side = "right"

    text_encoder = TextEncoder.from_pretrained(Config.TEXT_ENCODER_PATH, torch_dtype=weight_dtype)
    text_encoder = text_encoder.eval()
    text_encoder = text_encoder.to(accelerator.device, dtype=weight_dtype)

    scheduler_kwargs = {'time_shift_scale': time_shift_scale}
    pipeline = OmniGen2Pipeline(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=text_tokenizer,
        scheduler=Scheduler(**scheduler_kwargs),
    )
    return pipeline.to(accelerator.device, dtype=weight_dtype)

def preprocess(instruction: str, negative_prompt: str, pipeline: OmniGen2Pipeline) -> Tuple[str, str]:
    """Preprocess the instruction and negative prompt for the model."""
    instruction = [{"role": "user", "content": instruction}]
    instruction = pipeline.tokenizer.apply_chat_template(instruction, tokenize=False, add_generation_prompt=False)
    instruction = instruction.replace("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.", 
                                    "You are a helpful assistant that generates high-quality images based on user instructions.")
    instruction = instruction.replace("You are a helpful assistant.", 
                                    "You are a helpful assistant that generates high-quality images based on user instructions.")

    negative_prompt = [{"role": "user", "content": negative_prompt}]
    negative_prompt = pipeline.tokenizer.apply_chat_template(negative_prompt, tokenize=False, add_generation_prompt=False)
    negative_prompt = negative_prompt.replace("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.", 
                                            "You are a helpful assistant.")

    return instruction, negative_prompt

def run(instruction: str, 
        width_input: int, 
        height_input: int, 
        num_inference_steps: int, 
        image_input_1: Optional[Image.Image], 
        image_input_2: Optional[Image.Image], 
        image_input_3: Optional[Image.Image],
        negative_prompt: str, 
        guidance_scale_input: float, 
        num_images_per_prompt: int) -> Image.Image:
    """Run the image generation pipeline with the given parameters."""
    input_images = [img for img in [image_input_1, image_input_2, image_input_3] if img is not None]
    if not input_images:
        input_images = None

    instruction, negative_prompt = preprocess(instruction, negative_prompt, pipeline)

    results = pipeline(
        prompt=instruction,
        input_images=input_images,
        width=width_input,
        height=height_input,
        num_inference_steps=num_inference_steps,
        max_sequence_length=1024,
        guidance_scale=guidance_scale_input,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator,
        output_type="pil",
    )
    
    vis_images = [to_tensor(image) * 2 - 1 for image in results.images]
    output_image = create_collage(vis_images)

    if save_images:
        save_output_image(output_image)

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

def save_output_image(image: Image.Image) -> None:
    """Save the generated image with timestamp."""
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_path = os.path.join(Config.OUTPUT_DIR, f'{timestamp}.png')
    image.save(output_path)

def get_example() -> List[List]:
    """Return example prompts and parameters for the demo."""
    return [
        [
            "A curly-haired man in a red shirt is drinking tea.",
            1024, 1024, 50, None, None, None, "", 4.0, 1,
        ],
        [
            "A car toy and a bear toy are placed on the bench",
            1024, 1024, 50, Image.open("example_images/02.jpg"), None, None, "", 4.0, 1,
        ],
        [
            "The woman waves her hand happily in the crowd",
            1024, 1024, 50, Image.open("example_images/zhang.png"), None, None, "", 4.0, 1,
        ],
        [
            "Change the tea in the cup to coffee",
            1024, 1024, 50, Image.open("example_images/tea.jpg"), None, None, "", 4.0, 1,
        ],
        [
            "Put the flower in the vase, then put them on a wooden table of a living room",
            1024, 1024, 50, Image.open("example_images/rose.jpg"), Image.open("example_images/vase.jpg"), None, "", 4.0, 1,
        ],
    ]

def create_demo() -> gr.Blocks:
    """Create the Gradio interface."""
    with gr.Blocks() as demo:
        gr.Markdown("# OmniGen v2: Unified Image Generation [paper](https://arxiv.org/abs/2409.11340) [code](https://github.com/VectorSpaceLab/OmniGen)")
        gr.Markdown("""
        OmniGen2 is a unified image generation model that can handle various image generation tasks including text-to-image, image editing, and multi-image composition.
        """)
        
        with gr.Row():
            with gr.Column():
                instruction = gr.Textbox(
                    label="Enter your prompt, use <img><|image_i|></img> to represent i-th input image",
                    placeholder="Type your prompt here..."
                )

                with gr.Row(equal_height=True):
                    image_input_1 = gr.Image(label="<img><|image_1|></img>", type="pil")
                    image_input_2 = gr.Image(label="<img><|image_2|></img>", type="pil")
                    image_input_3 = gr.Image(label="<img><|image_3|></img>", type="pil")

                with gr.Row():
                    width_input = gr.Slider(512, 1024, 1024, step=64, label="Width")
                    height_input = gr.Slider(512, 1024, 1024, step=64, label="Height")

                with gr.Row():
                    num_inference_steps = gr.Slider(1, 100, 50, step=1, label="Inference Steps")
                    guidance_scale_input = gr.Slider(1.0, 20.0, 4.0, step=0.1, label="Guidance Scale")

                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value=Config.DEFAULT_NEGATIVE_PROMPT,
                    placeholder="Enter negative prompt here..."
                )

                num_images_per_prompt = gr.Slider(1, 4, 1, step=1, label="Number of Images")

                generate_btn = gr.Button("Generate")

            with gr.Column():
                output_image = gr.Image(label="Generated Image", type="pil")

        generate_btn.click(
            fn=run,
            inputs=[
                instruction, width_input, height_input, num_inference_steps,
                image_input_1, image_input_2, image_input_3,
                negative_prompt, guidance_scale_input, num_images_per_prompt
            ],
            outputs=output_image
        )

        gr.Examples(
            examples=get_example(),
            inputs=[
                instruction, width_input, height_input, num_inference_steps,
                image_input_1, image_input_2, image_input_3,
                negative_prompt, guidance_scale_input, num_images_per_prompt
            ],
            outputs=output_image,
            fn=run,
            cache_examples=True,
        )

    return demo

if __name__ == "__main__":
    # Initialize accelerator and model
    accelerator = Accelerator(mixed_precision="bf16")
    weight_dtype = torch.bfloat16
    pipeline = load_pipeline(time_shift_scale=1.0, accelerator=accelerator, weight_dtype=weight_dtype)
    generator = torch.Generator(device=accelerator.device).manual_seed(998244353)

    # Launch the demo
    demo = create_demo()
    demo.launch(share=True)