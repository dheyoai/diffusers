# Messages containing multiple images and a text query
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch 
from typing import Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", device_map="auto", torch_dtype=torch.float32
).to(device)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2", ## xformers will be scary 😭 PTSD
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)



def get_evaluation (generated_image_path: str,
                    evaluation_prompt: str,
                    num_output_evaluation_tokens: Optional[int] = 256,
                    guiding_image_path: Optional[str] = None):

    if guiding_image_path:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": generated_image_path},
                    {"type": "image", "image": guiding_image_path},
                    {"type": "text", "text": evaluation_prompt},
                ],
            }
        ]
    else:
        messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": generated_image_path},
                {"type": "text", "text": evaluation_prompt},
            ],
        }
        ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=num_output_evaluation_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text



