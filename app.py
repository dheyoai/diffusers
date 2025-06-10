import dotenv
dotenv.load_dotenv(override=True)

import gradio as gr

from typing import List, Tuple
from PIL import Image
import os
import argparse
import random
import numpy as np

from PIL import Image

import torch
from torchvision.transforms.functional import to_pil_image, to_tensor

from accelerate import Accelerator

from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.utils.img_util import resize_image

# MODEL_PATH = "OmniGen2/OmniGen2-preview"
MODEL_PATH = "/share_2/luoxin/projects/OmniGen2/pretrained_models/omnigen2_pipe_model_fuse_v1"
NEGATIVE_PROMPT = "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"
# NEGATIVE_PROMPT = "low quality, blurry, out of focus, distorted, bad anatomy, poorly drawn, pixelated, grainy, artifacts, watermark, text, signature, deformed, extra limbs, cropped, jpeg artifacts, ugly"

def load_pipeline(accelerator, weight_dtype):
    pipeline = OmniGen2Pipeline.from_pretrained(MODEL_PATH,
                                                torch_dtype=weight_dtype,
                                                trust_remote_code=True,
                                                token=os.getenv("HF_TOKEN"))
    pipeline = pipeline.to(accelerator.device, dtype=weight_dtype)
    return pipeline

def run(instruction, width_input, height_input, num_inference_steps, image_input_1, image_input_2, image_input_3,
        negative_prompt, guidance_scale_input, img_guidance_scale_input,  num_images_per_prompt, max_input_image_size, seed_input, progress=gr.Progress()):

    input_images = [image_input_1, image_input_2, image_input_3]
    input_images = [img for img in input_images if img is not None]
    if len(input_images) == 0: input_images = None

    if input_images is not None:
        # input_images = [crop_arr(x, max_input_image_size, 16) for x in input_images]
        input_images = [resize_image(x, max_input_image_size * max_input_image_size, 16) for x in input_images]

    if input_images is not None and len(input_images) == 1: width_input, height_input = input_images[0].size

    if seed_input == -1:
        seed_input = random.randint(0, 2**16-1)
    generator = torch.Generator(device=accelerator.device).manual_seed(seed_input)

    def progress_callback(cur_step, timesteps):
        frac = (cur_step + 1) / float(timesteps)
        progress(frac)

    results = pipeline(
        prompt=instruction,
        input_images=input_images,
        width=width_input,
        height=height_input,
        num_inference_steps=num_inference_steps,
        max_sequence_length=1024,
        text_guidance_scale=guidance_scale_input,
        image_guidance_scale=img_guidance_scale_input,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator,
        output_type="pil",
        step_func=progress_callback,
    )

    progress(1.0)
    
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

    if save_images:
        # Save All Generated Images
        from datetime import datetime
        # Create outputs directory if it doesn't exist
        os.makedirs('outputs_yrr', exist_ok=True)
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        output_path = os.path.join('outputs_yrr', f'{timestamp}.png')
        # Save the image
        output_image.save(output_path)

    return output_image

def get_example():
    case = [
        [
            "A winter elf kneeling in heavy snow while wearing a battered and bruised armor, holding an ice blade and a shield, designed as a study for a D&D character, with ultra detail and cinematic resolution of 8k.",
            1024,
            1024,
            50,
            None,
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            1,
            1024,
            998244353,
        ],  
        [
            "In a cozy café, the anime figure is sitting in front of a laptop, smiling confidently.",
            1024,
            1024,
            50,
            "/share_2/chenyuan/data/Omnigen/anime/man/1e5953ff5e029bfc81bb0a1d4792d26d.jpg",
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            1,
            1024,
            998244353,
        ],
        [
            "Let the girl from the first image and the man from the second image get married in the church.",
            1024,
            1024,
            50,
            "/share_2/chenyuan/data/Flux_subject/anime/init_data/imgs/1girl/74ad3e79-e44b-420c-acfd-61711dca2353.png",
            "/share/shitao/wyz/datasets/Images/folder1/images/00043/000431857.jpg",
            None,
            NEGATIVE_PROMPT,
            5.0,
            3.0,
            1,
            1024,
            998244353,
        ],
        [
            "Let the man form image1 and the woman from image2 kiss and hug",
            1024,
            1024,
            50,
            "/share/shitao/wyz/datasets/Images/folder5/images/00831/008316586.jpg",
            "/share/shitao/wyz/datasets/Images/folder1/images/00007/000077066.jpg",
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            1,
            1024,
            998244353,
        ],
        [
            "Please let the person in image 2 examine the vintage green alarm clock from the firt image.",
            1024,
            1024,
            50,
            "/share/shitao/wyz/datasets/Images/folder1/images/00029/000298127.jpg",
            "/share/shitao/wyz/datasets/Images/folder2/images/00052/000523889.jpg",
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            1,
            1024,
            998244353,
        ],
        [
            "Make the girl pray in the second image.",
            1024,
            682,
            50,
            "/share/shitao/wyz/datasets/Images/folder2/images/00044/000440817.jpg",
            "/share/shitao/wyz/datasets/Images/folder2/images/00011/000119733.jpg",
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            1,
            1024,
            998244353,
        ],
        [
            "Add the bird from image 1 to the desk in image 2",
            1024,
            682,
            50,
            "/share_2/chenyuan/data/Flux_subject/object/imgs/996e2cf6-daa5-48c4-9ad7-0719af640c17_1748848108409.png",
            "/share_2/pengfei/NEW_FOLDER3/Model_fuse/Data/3.jpg",
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            1,
            1024,
            998244353,
        ],
        [
            "Replace the apple in the first image with the cat from the second image",
            1024,
            682,
            50,
            "/share_2/chenyuan/data/Flux_subject/object/imgs/427800b2-9391-43c2-b22a-9517e5ae0893_1748846513998.png",
            "/share_2/chenyuan/data/Flux_subject/object/imgs/2afb2870-9ca1-489f-8a66-b488efb9ab64_1748850623425.png",
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            1,
            1024,
            998244353,
        ],
        [
            "Replace the woman in the second image with the man from the first image",
            1024,
            682,
            50,
            "/share/shitao/wyz/datasets/Images/folder2/images/00007/000073764.jpg",
            "/share_2/chenyuan/data/Omnigen/subject_after_filter_train/imgs_flux_0dot1_cool_adaptv1/f4546e15-d54a-448a-ae80-cb33e9f0d44b.png",
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            1,
            1024,
            998244353,
        ],
        [
            "Remove the bird",
            1024,
            1024,
            50,
            "/share_2/pengfei/NEW_FOLDER3/Model_fuse/all_image/f167fca0-f991-4f08-972f-b5819bcc424d_1748850154973.png",
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            1,
            1024,
            998244353,
        ],
        [
            "Change the man's blue jacket to a maroon one.",
            1024,
            1024,
            50,
            "/share_2/luoxin/datasets/Distilled/flux_edit_pro/RefEdit/output_images/1842.png",
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            1,
            1024,
            998244353,
        ],
        [
            "Replace the white toy airplane on the right with a wooden toy train.",
            1024,
            1024,
            50,
            "/share_2/luoxin/datasets/Distilled/flux_edit_pro/RefEdit/output_images/3220.png",
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            1,
            1024,
            998244353,
        ],
        
    ]
    return case


def run_for_examples(instruction,
                     width_input, height_input,
                     num_inference_steps,
                     image_input_1, image_input_2, image_input_3,
                     negative_prompt,
                     text_guidance_scale_input, image_guidance_scale_input,
                     num_images_per_prompt,
                     max_input_image_size,
                     seed_input):
     return run(instruction, width_input, height_input, num_inference_steps, image_input_1, image_input_2, image_input_3, negative_prompt, text_guidance_scale_input, image_guidance_scale_input, num_images_per_prompt, max_input_image_size, seed_input)


description = """
This is currently a demo of OmniGen2. Contents to be added.
"""

article = """
citation to be added
"""

# Gradio 
with gr.Blocks() as demo:
    gr.Markdown("# OmniGen2: Unified Image Generation [paper](https://arxiv.org/abs/2409.11340) [code](https://github.com/VectorSpaceLab/OmniGen2)")
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            # text prompt
            instruction = gr.Textbox(
                label='Enter your prompt. Use "first/second image" or “第一张图/第二张图” as reference.', placeholder="Type your prompt here..."
            )

            with gr.Row(equal_height=True):
                # input images
                image_input_1 = gr.Image(label="First Image", type="pil")
                image_input_2 = gr.Image(label="Second Image", type="pil")
                image_input_3 = gr.Image(label="Third Image", type="pil")
            
            generate_button = gr.Button("Generate Image")

            negative_prompt = gr.Textbox(
                label="Enter your negative prompt", placeholder="Type your negative prompt here...", value=NEGATIVE_PROMPT,
            )

            # slider
            height_input = gr.Slider(
                label="Height", minimum=256, maximum=1024, value=1024, step=128
            )
            width_input = gr.Slider(
                label="Width", minimum=256, maximum=1024, value=1024, step=128
            )

            text_guidance_scale_input = gr.Slider(
                label="Text Guidance Scale", minimum=1.0, maximum=8.0, value=5.0, step=0.1
            )

            image_guidance_scale_input = gr.Slider(
                label="Image Guidance Scale", minimum=1.0, maximum=3.0, value=2.0, step=0.1
            )

            num_inference_steps = gr.Slider(
                label="Inference Steps", minimum=20, maximum=100, value=50, step=1
            )

            num_images_per_prompt = gr.Slider(
                label="Number of images per prompt", minimum=1, maximum=4, value=1, step=1
            )

            # bf16 = gr.Checkbox(
            #     label="bf16", value=True, info="Whether to use bf16."
            # )

            seed_input = gr.Slider(
                label="Seed", minimum=-1, maximum=2147483647, value=0, step=1
            )
            # randomize_seed = gr.Checkbox(label="Randomize seed", value=False)

            max_input_image_size = gr.Slider(
                label="max_input_image_size", minimum=256, maximum=1024, value=1024, step=256
            )

            # separate_cfg_infer = gr.Checkbox(
            #     label="separate_cfg_infer", info="Whether to use separate inference process for different guidance. This will reduce the memory cost.", value=True,
            # )
            # offload_model = gr.Checkbox(
            #     label="offload_model", info="Offload model to CPU, which will significantly reduce the memory cost but slow down the generation speed. You can cancel separate_cfg_infer and set offload_model=True. If both separate_cfg_infer and offload_model are True, further reduce the memory, but slowest generation", value=False,
            # )
            # use_input_image_size_as_output = gr.Checkbox(
            #     label="use_input_image_size_as_output", info="Automatically adjust the output image size to be same as input image size. For editing and controlnet task, it can make sure the output image has the same size as input image leading to better performance", value=False,
            # )

            # generate
            
        with gr.Column():
            with gr.Column():
                # output image
                output_image = gr.Image(label="Output Image")
                save_images = gr.Checkbox(label="Save generated images", value=False)

    bf16 = True
    accelerator = Accelerator(mixed_precision="bf16" if bf16 else 'no')
    weight_dtype = torch.bfloat16 if bf16 else torch.float32

    pipeline = load_pipeline(accelerator, weight_dtype)

    # click
    generate_button.click(
        run,
        inputs=[
            instruction,
            width_input, 
            height_input, 
            num_inference_steps, 
            image_input_1, 
            image_input_2, 
            image_input_3,
            negative_prompt, 
            text_guidance_scale_input, 
            image_guidance_scale_input,
            num_images_per_prompt,
            max_input_image_size,
            seed_input,
        ],
        outputs=output_image,
    )

    gr.Examples(
        examples=get_example(),
        fn=run_for_examples,
        inputs=[
            instruction,
            width_input, 
            height_input, 
            num_inference_steps, 
            image_input_1, 
            image_input_2, 
            image_input_3,
            negative_prompt, 
            text_guidance_scale_input, 
            image_guidance_scale_input,
            num_images_per_prompt,
            max_input_image_size,
            seed_input,
        ],
        outputs=output_image,
    )

    gr.Markdown(article)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the OmniGen')
    parser.add_argument('--share', action='store_true', help='Share the Gradio app')
    parser.add_argument('--port', type=int, default=7860, help='Port to use for the Gradio app')
    args = parser.parse_args()

    # launch
    demo.launch(share=args.share,
                server_port=args.port,
                allowed_paths=["/share_2",
                               "/share"])

"""
CUDA_VISIBLE_DEVICES=0 python shitao_app.py --share

CUDA_VISIBLE_DEVICES=1 python shitao_app.py --share

"""