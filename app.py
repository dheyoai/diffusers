import gradio as gr
from PIL import Image
import os
import argparse
import random

import dotenv

dotenv.load_dotenv(override=True)

from omegaconf import OmegaConf
from tqdm import tqdm

from PIL import Image

import torch
from torchvision.transforms.functional import to_pil_image, to_tensor

from accelerate import Accelerator

from transformers import AutoTokenizer
from transformers import Qwen2_5_VLModel as TextEncoder

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from omnigen.pipelines.flow_matching.pipeline_fm import FlowMatchingPipeline
from omnigen.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler as Scheduler


CONFIG_PATH = "/share_2/luoxin/projects/Omnigenv2/experiments/ominigenv2_0519_ps1024_bs128_ft_t2i_human_ai_0519_deblur_lr4e-5/ominigenv2_0519_ps1024_bs128_ft_t2i_human_ai_0519_deblur_lr4e-5.yml"

# MODEL_PATH = "/share_2/luoxin/projects/Omnigenv2/experiments/ominigenv2_0519_ps1024_bs128_ft_t2i_human_ai_0519_deblur_lr4e-5/checkpoint-1500/pytorch_model_fsdp.bin"
MODEL_PATH = "/share_2/luoxin/projects/Ominigenv2/experiments/ominigenv2_0517_ps1024_bs96_ft_t2i_debug_new_gpt4o2_lr2e-5/checkpoint-2000/pytorch_model_fsdp.bin"
# MODEL_PATH = "/share_2/luoxin/projects/Ominigenv2/experiments/ominigenv2_0520_ps1024_bs192_ft_t2i_9_lr2e-5/checkpoint-2000/pytorch_model_fsdp.bin"

VAE_PATH = "/share_2/luoxin/modelscope/hub/models/FLUX.1-dev"
TOKENIZER_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"
TEXT_ENCODER_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"
NEGATIVE_PROMPT = "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"


def load_pipeline(time_shift_scale, accelerator, weight_dtype):
    conf = OmegaConf.load(CONFIG_PATH)
    transformer = OmniGen2Transformer2DModel(**conf.model.arch_opt)
        
    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    missing, unexpect = transformer.load_state_dict(state_dict, strict=False)
    print(f"missed parameters: {missing}")
    print(f"unexpected parameters: {unexpect}")
    
    transformer = transformer.eval()
    transformer = transformer.to(accelerator.device, dtype=weight_dtype)
    
    transformer = accelerator.prepare(transformer)
    transformer = accelerator.unwrap_model(transformer)

    vae = AutoencoderKL.from_pretrained(
        VAE_PATH,
        subfolder="vae",
    )
    vae = vae.to(accelerator.device, dtype=weight_dtype)

    text_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    text_tokenizer.padding_side = "right"

    text_encoder = TextEncoder.from_pretrained(TEXT_ENCODER_PATH, torch_dtype=weight_dtype)
    text_encoder = text_encoder.eval()
    text_encoder = text_encoder.to(accelerator.device, dtype=weight_dtype)

    scheduler_kwargs = {
        'time_shift_scale': time_shift_scale
    }

    pipeline = FlowMatchingPipeline(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=text_tokenizer,
        scheduler=Scheduler(**scheduler_kwargs),
    )
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
        negative_prompt = negative_prompt.replace("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.", "You are a helpful assistant.")

    return instruction, negative_prompt


def run(instruction, width_input, height_input, num_inference_steps, image_input_1, image_input_2, image_input_3,
        negative_prompt, guidance_scale_input, num_images_per_prompt):

    input_images = [image_input_1, image_input_2, image_input_3]
    input_images = [img for img in input_images if img is not None]
    if len(input_images) == 0: input_images = None

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
        os.makedirs('outputs', exist_ok=True)
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        output_path = os.path.join('outputs', f'{timestamp}.png')
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
            "",
            4.0,
            1,
        ],
        [
            "A car toy and a bear toy are placed on the bench",
            1024,
            1024,
            50,
            Image.open("example_images/02.jpg"),
            None,
            None,
            "",
            4.0,
            1,
        ],
        [
            "The woman waves her hand happily in the crowd",
            1024,
            1024,
            50,
            Image.open("example_images/zhang.png"),
            None,
            None,
            "",
            4.0,
            1,
        ],
        [
            "Change the tea in the cup to coffee",
            1024,
            1024,
            50,
            Image.open("example_images/tea.jpg"),
            None,
            None,
            "",
            4.0,
            1,
        ],
        [
            "Put the flower in the vase, then put them on a wooden table of a living room",
            1024,
            1024,
            50,
            Image.open("example_images/rose.jpg"),
            Image.open("example_images/vase.jpg"),
            None,
            "",
            4.0,
            1,
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

            negative_prompt = gr.Textbox(
                label="Enter your negative prompt", placeholder="Type your negative prompt here...", value=""
            )

            # slider
            height_input = gr.Slider(
                label="Height", minimum=128, maximum=2048, value=1024, step=16
            )
            width_input = gr.Slider(
                label="Width", minimum=128, maximum=2048, value=1024, step=16
            )

            guidance_scale_input = gr.Slider(
                label="Guidance Scale", minimum=1.0, maximum=5.0, value=4.0, step=0.1
            )

            img_guidance_scale_input = gr.Slider(
                label="img_guidance_scale", minimum=1.0, maximum=2.0, value=1.6, step=0.1
            )

            num_inference_steps = gr.Slider(
                label="Inference Steps", minimum=1, maximum=100, value=50, step=1
            )

            num_images_per_prompt = gr.Slider(
                label="Number of images per prompt", minimum=1, maximum=4, value=1, step=1
            )

            time_shift_scale = gr.Slider(
                label="time_shift_scale", minimum=1.0, maximum=5.0, value=3.0, step=0.1
            )

            bf16 = gr.Checkbox(
                label="bf16", value=True, info="Whether to use bf16."
            )

            seed_input = gr.Slider(
                label="Seed", minimum=0, maximum=2147483647, value=42, step=1
            )
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            max_input_image_size = gr.Slider(
                label="max_input_image_size", minimum=128, maximum=2048, value=1024, step=16
            )

            separate_cfg_infer = gr.Checkbox(
                label="separate_cfg_infer", info="Whether to use separate inference process for different guidance. This will reduce the memory cost.", value=True,
            )
            offload_model = gr.Checkbox(
                label="offload_model", info="Offload model to CPU, which will significantly reduce the memory cost but slow down the generation speed. You can cancel separate_cfg_infer and set offload_model=True. If both separate_cfg_infer and offload_model are True, further reduce the memory, but slowest generation", value=False,
            )
            # use_input_image_size_as_output = gr.Checkbox(
            #     label="use_input_image_size_as_output", info="Automatically adjust the output image size to be same as input image size. For editing and controlnet task, it can make sure the output image has the same size as input image leading to better performance", value=False,
            # )

            # generate
            generate_button = gr.Button("Generate Image")
            

        with gr.Column():
            with gr.Column():
                # output image
                output_image = gr.Image(label="Output Image")
                save_images = gr.Checkbox(label="Save generated images", value=False)


    accelerator = Accelerator(mixed_precision="bf16" if bf16 else 'no')
    weight_dtype = torch.bfloat16 if bf16 else torch.float32

    seed_input = random.randint(0, 2147483647) if randomize_seed else seed_input
    generator = torch.Generator(device=accelerator.device).manual_seed(seed_input)

    time_shift_scale = 3.0 # tobe solved
    pipeline = load_pipeline(time_shift_scale, accelerator, weight_dtype)

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
            num_images_per_prompt,
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