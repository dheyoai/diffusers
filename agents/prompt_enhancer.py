### use command line args to run this file
import os
import torch
from together import Together
from dotenv import load_dotenv
from system_prompts import (
    INITIAL_PROMPT_ENHANCING_SYS_PROMPT, 
    VLQ_SYS_PROMPT, 
    QA_SYS_PROMPT, 
    QA_STRUCTURED_PROMPT
)

import json 
from json_repair import repair_json
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
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]
    return message



#### should also generate questions - check vidgen
class PromptEnhancer:
    def __init__(self):
        print("Prompt Enhancer loaded...")


    ## The output revised prompt has to be less than or equal to 77 tokens thanks to CLIP 😒
    ## TODO: Incorporate the above into the system prompt
    def enhance(self, prompt):
        message_fn = create_message
        revised_prompt = together_response(message_fn(INITIAL_PROMPT_ENHANCING_SYS_PROMPT, prompt))
        print(f"Original prompt : {prompt}")
        print(f"Revised prompt: {revised_prompt}")

        return revised_prompt
    
    def get_questions(self, prompt):

        questions = together_response(create_message(VLQ_SYS_PROMPT, prompt))
        questions = [ ## if this is failing I can try JSON (structured response - more robust)
            q.strip() for q in questions.strip().split("\n") if (q != "" and q != "\n")
        ]
        return questions


    def enhance_with_answer_list(
        self,
        prompt,
        negative_prompt,
        qlist,
        anslist,
        structured_inputs=False,
        two_step_enhancement=False,
    ):
        # import pdb; pdb.set_trace()
        anslist = anslist[0].split("\n")
        qlist = qlist[:len(anslist)]
        assert len(qlist) == len(anslist)
        if structured_inputs:
            inp_dict = {}
            inp_dict["prompt"] = prompt
            inp_dict["negative_prompt"] = negative_prompt
            inp_dict["questions_and_answers"] = list(zip(qlist, anslist))
            inp_text = json.dumps(inp_dict)
            # print(f"Input text (structured): {inp_text}")
            sysprompt = QA_STRUCTURED_PROMPT
        else:
            inp_text = f"""The user's prompt is {prompt}. The negative prompt is {negative_prompt}. The questions and answers are as follows.\n"""
            for i, q in enumerate(qlist):
                inp_text += f"Question: {q}. Answer: {anslist[i]}\n"
            sysprompt = QA_SYS_PROMPT

        json_out = together_response(create_message(sysprompt, inp_text))

        parsed_json = repair_json(json_out, return_objects=True)
        try:
            new_prompt, new_nprompt = (
                parsed_json["prompt"],
                parsed_json["negative_prompt"],
            )
        except:
            new_prompt, new_nprompt = prompt, negative_prompt
            print("error in json")
            print(f"llm output: {json_out}")
            print(f"json: {parsed_json}")

        return new_prompt, new_nprompt