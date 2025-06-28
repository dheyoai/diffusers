#### This sole file should handle guided image-gen with controlnet and ipadapter and may be insightface (in future)

from diffusers import DiffusionPipeline, UNet2DConditionModel, PNDMScheduler, StableDiffusion3Pipeline, StableDiffusionPipeline
from transformers import CLIPTextModel
import torch
from diffusers.utils import load_image
from typing import Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def generate_image(image_path_to_save: str, 
                   prompt: str, 
                   model_path: str, ## can be a HF repo id or a local path
                   num_inference_steps: Optional[int] = 25,
                   guidance_scale: Optional[float] = 7.5,
                   negative_prompt: Optional[str] = None):
    
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_path, dtype=torch.bfloat16,
    ).to(device)

    image = pipeline(prompt=prompt, 
                    num_inference_steps=num_inference_steps, 
                    guidance_scale=guidance_scale, 
                    negative_prompt=negative_prompt).images[0]
    image.save(image_path_to_save)
    print(f"Generated image successfully saved at {image_path_to_save}")


def generate_image_with_ipadapter():
    pass


def generate_image_with_controlnet():
    pass