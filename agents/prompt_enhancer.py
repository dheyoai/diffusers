### use command line args to run this file
import os
import torch
from together import Together
from dotenv import load_dotenv
from system_prompts import INITIAL_PROMPT_ENHANCHING_SYS_PROMPT, VLQ_SYS_PROMPT
load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=TOGETHER_API_KEY)



def together_response(messages):
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=messages,
    )
    text_output = response.choices[0].message.content
    return text_output


def create_message(sys_prompt, prompt):
    message = [
        {"role": "system", "content": INITIAL_PROMPT_ENHANCHING_SYS_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return message



#### should also generate questions - check vidgen
class PromptEnhancer:
    def __init__(self):
        print("Prompt Enhancer loaded...")

    def enhance(self, prompt):
        message_fn = create_message
        revised_prompt = together_response(message_fn(INITIAL_PROMPT_ENHANCHING_SYS_PROMPT, prompt))
        print(f"Original prompt : {prompt}")
        print(f"Revised prompt: {revised_prompt}")

        return revised_prompt
    
    def get_questions(self, prompt):

        questions = together_response(create_message(VLQ_SYS_PROMPT, prompt))
        questions = [ ## if this is failing I can try JSON (structured response - more robust)
            q.strip() for q in questions.strip().split("\n") if (q != "" and q != "\n")
        ]
        return questions