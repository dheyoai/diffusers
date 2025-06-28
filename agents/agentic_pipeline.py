### use command line args to run this file
import os
import torch
from together import Together
from dotenv import load_dotenv
from system_prompts import INITIAL_PROMPT_ENHANCHING_SYS_PROMPT
from prompt_enhancer import PromptEnhancer
from image_generation import generate_image
from qwenvl_evaluator import get_evaluation
load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=TOGETHER_API_KEY)

## TODO: turn the below vars into cmdline args
user_prompt = "sks man dancing"
sd3_model_id = "stabilityai/stable-diffusion-3.5-medium"
guidance_scale = 7.5
num_inference_steps = 25
negative_prompt = None
num_loops = 3 ## number of times to execute the complete agentic flow..
### If guiding_image is None, only prompt adherence might be impacted
guiding_image = None ## Guide the image using another image -> may help enhancing the facial features
### for controlnet people should provide depth map or edge detection map and so on..
image_guiding_method = None ## should choose from ["ipadapter", "controlnet"] or None


if __name__ == '__main__':
    # enchance -> generate -> evaluate based on generated questions, images -> enhance with answer list
    prompt_enhancer = PromptEnhancer() ### should I use a thinking model?
    revised_prompt = prompt_enhancer.enhance("a photo of sks man dancing in rain with a happy face")
    print(revised_prompt)
    generate_image(image_path_to_save="inferenced_images/ap_1.png", 
                   prompt=revised_prompt, 
                   model_path="examples/dreambooth/sd3_large_no_text_encoder_training_balayya",
                   num_inference_steps=50,
                   guidance_scale=12,
                   negative_prompt="extra arms, extra fingers, extra legs, mutated hands, fused fingers, long neck, cross-eyed, long head, deformed hands, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality")
