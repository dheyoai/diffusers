### use command line args to run this file
import os
import torch
from together import Together
from dotenv import load_dotenv
from system_prompts import INITIAL_PROMPT_ENHANCING_SYS_PROMPT
from prompt_enhancer import PromptEnhancer
from image_generation import generate_image
from qwenvl_evaluator import get_evaluation
load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=TOGETHER_API_KEY)

## TODO: turn the below vars into cmdline args
prompt = "a close up photorealistic photo of sks man smiling gracefully in a pink shirt and blue goggles and spiky hair"
# sd3_model_id = "../examples/dreambooth/sd3_large_no_text_encoder_training_balayya"
sd3_model_id = "../examples/dreambooth/sd3_medium_balayya_3"
guidance_scale = 7.5
num_inference_steps = 25
negative_prompt = None
num_loops = 2 ## number of times to execute the complete agentic flow..
### If guiding_image is None, only prompt adherence might be impacted
guiding_image = None ## Guide the image using another image -> may help enhancing the facial features
### for controlnet people should provide depth map or edge detection map and so on..
image_guiding_method = None ## should choose from ["ipadapter", "controlnet"] or None
image_path_to_save = "inferenced_images/ap_1.png"


if __name__ == '__main__':
    # enchance -> generate -> evaluate based on generated questions, images -> enhance with answer list
    # prompt = "a photo of sks man dancing in rain with a happy face"
    # negative_prompt = "extra arms, extra fingers, extra legs, mutated hands, fused fingers, long neck, cross-eyed, long head, deformed hands, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality"
    for _ in range(num_loops):
        prompt_enhancer = PromptEnhancer() ### should I use a thinking model?
        revised_prompt = prompt_enhancer.enhance(prompt).lower()
        print(revised_prompt)
        generate_image(image_path_to_save=image_path_to_save, 
                    prompt=revised_prompt, 
                    model_path=sd3_model_id,
                    num_inference_steps=50,
                    guidance_scale=12,
                    negative_prompt=negative_prompt)
        
        questions_list = prompt_enhancer.get_questions(prompt=revised_prompt)
        answers_list = get_evaluation(generated_image_path=image_path_to_save,
                                    evaluation_prompt=f"Answer the following question based on the image: {questions_list}",
                                    num_output_evaluation_tokens=512
                                    )
        prompt, negative_prompt = prompt_enhancer.enhance_with_answer_list(prompt=revised_prompt,
                                                negative_prompt=negative_prompt,
                                                qlist=questions_list,
                                                anslist=answers_list
                                                )
