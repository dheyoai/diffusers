import gradio as gr
from PIL import Image
import os
import argparse
import random
import numpy as np
import dotenv

dotenv.load_dotenv(override=True)

from PIL import Image

import torch
from torchvision.transforms.functional import to_pil_image, to_tensor

from accelerate import Accelerator

from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline

MODEL_PATH = "OmniGen2/OmniGen2-preview"
NEGATIVE_PROMPT = "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"
# NEGATIVE_PROMPT = "low quality, blurry, out of focus, distorted, bad anatomy, poorly drawn, pixelated, grainy, artifacts, watermark, text, signature, deformed, extra limbs, cropped, jpeg artifacts, ugly"

def crop_arr(pil_image, max_image_size, img_scale_num):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * max_image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    if max(*pil_image.size) > max_image_size:
        scale = max_image_size / max(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
    
    if min(*pil_image.size) < img_scale_num:
        scale = img_scale_num / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
    
    arr = np.array(pil_image)
    crop_y1 = (arr.shape[0] % img_scale_num) // 2
    crop_y2 = arr.shape[0] % img_scale_num - crop_y1

    crop_x1 = (arr.shape[1] % img_scale_num) // 2
    crop_x2 = arr.shape[1] % img_scale_num - crop_x1

    arr = arr[crop_y1:arr.shape[0]-crop_y2, crop_x1:arr.shape[1]-crop_x2]
    
    return Image.fromarray(arr)


def load_pipeline(accelerator, weight_dtype):
    pipeline = OmniGen2Pipeline.from_pretrained(MODEL_PATH,
                                                torch_dtype=weight_dtype,
                                                trust_remote_code=True,
                                                token=os.getenv("HF_TOKEN"))
    pipeline = pipeline.to(accelerator.device, dtype=weight_dtype)
    return pipeline


def preprocess(instruction, negative_prompt, pipeline):
    instruction = [{"role": "user", "content": instruction}]
    instruction = pipeline.tokenizer.apply_chat_template(instruction, tokenize=False, add_generation_prompt=False)
    if "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." in instruction:
        instruction = instruction.replace("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.", "You are a helpful assistant that generates high-quality images based on user instructions.")
    else:
        instruction = instruction.replace("You are a helpful assistant.", "You are a helpful assistant that generates high-quality images based on user instructions.")

    negative_prompt = [{"role": "user", "content": negative_prompt}]
    negative_prompt = pipeline.tokenizer.apply_chat_template(negative_prompt, tokenize=False, add_generation_prompt=False)

    if "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." in negative_prompt:
        negative_prompt = negative_prompt.replace("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.", "You are a helpful assistant that generates high-quality images based on user instructions.")
    else:
        negative_prompt = negative_prompt.replace("You are a helpful assistant.", "You are a helpful assistant that generates high-quality images based on user instructions.")
    
    # if "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." in negative_prompt:
    #     negative_prompt = negative_prompt.replace("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.", "You are a helpful assistant that generates images.")
    # else:
    #     negative_prompt = negative_prompt.replace("You are a helpful assistant.", "You are a helpful assistant that generates images.")

    print("instruction:", instruction)
    print("negative_prompt:", negative_prompt)
    print('-------------------')
    return instruction, negative_prompt


def run(instruction, width_input, height_input, num_inference_steps, image_input_1, image_input_2, image_input_3,
        negative_prompt, guidance_scale_input, img_guidance_scale_input,  num_images_per_prompt, max_input_image_size, seed_input):

    input_images = [image_input_1, image_input_2, image_input_3]
    input_images = [img for img in input_images if img is not None]
    if len(input_images) == 0: input_images = None

    if input_images is not None:
        input_images = [crop_arr(x, max_input_image_size, 16) for x in input_images]

    if input_images is not None and len(input_images) == 1: width_input, height_input = input_images[0].size

    instruction, negative_prompt = preprocess(instruction, negative_prompt, pipeline)

    if seed_input == -1:
        seed_input = random.randint(0, 2**16-1)
    generator = torch.Generator(device=accelerator.device).manual_seed(seed_input)

    results = pipeline(
        prompt=instruction,
        input_images=input_images,
        width=width_input,
        height=height_input,
        num_inference_steps=num_inference_steps,
        max_sequence_length=1024,
        guidance_scale=guidance_scale_input,
        ref_guidance_scale=img_guidance_scale_input,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
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
            "A curly-haired man in a red shirt is drinking tea.",
            1024,
            1024,
            50,
            None,
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            1,
            1024,
            0,
        ],
        [
            "A curly-haired man in a red shirt is drinking tea.",
            1024,
            1024,
            50,
            None,
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            1,
            1024,
            0,
        ],
        
    ]
    return case


def run_for_examples(instruction, width_input, height_input, num_inference_steps, image_input_1, image_input_2, image_input_3,
        negative_prompt, guidance_scale_input, num_images_per_prompt):    
    
    return run(instruction, width_input, height_input, num_inference_steps, image_input_1, image_input_2, image_input_3, negative_prompt, guidance_scale_input, num_images_per_prompt)


description = """
This is currently a demo of OmniGen v2. Contents to be added.
"""

article = """
citation to be added
"""

# Gradio 
with gr.Blocks() as demo:
    gr.Markdown("# OmniGen v2: Unified Image Generation [paper](https://arxiv.org/abs/2409.11340) [code](https://github.com/VectorSpaceLab/OmniGen)")
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            # text prompt
            instruction = gr.Textbox(
                label="Enter your prompt, use <img><|image_i|></img> to represent i-th input image", placeholder="Type your prompt here..."
            )

            with gr.Row(equal_height=True):
                # input images
                image_input_1 = gr.Image(label="<img><|image_1|></img>", type="pil")
                image_input_2 = gr.Image(label="<img><|image_2|></img>", type="pil")
                image_input_3 = gr.Image(label="<img><|image_3|></img>", type="pil")
            
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

            guidance_scale_input = gr.Slider(
                label="Guidance Scale", minimum=1.0, maximum=8.0, value=5.0, step=0.1
            )

            img_guidance_scale_input = gr.Slider(
                label="img_guidance_scale", minimum=1.0, maximum=8.0, value=2.0, step=0.1
            )

            num_inference_steps = gr.Slider(
                label="Inference Steps", minimum=20, maximum=100, value=50, step=1
            )

            num_images_per_prompt = gr.Slider(
                label="Number of images per prompt", minimum=1, maximum=4, value=1, step=1
            )

            time_shift_scale = gr.Slider(
                label="time_shift_scale", minimum=1.0, maximum=5.0, value=3.0, step=0.1
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
            guidance_scale_input, 
            img_guidance_scale_input,
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
            guidance_scale_input, 
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
    args = parser.parse_args()

    # launch
    demo.launch(share=args.share)

"""
CUDA_VISIBLE_DEVICES=0 python shitao_app.py --share

CUDA_VISIBLE_DEVICES=1 python shitao_app.py --share

"""

