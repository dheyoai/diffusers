# Complete End-to-End OmniGen2 LoRA Pipeline

```
ssh gpu-60
```

## Monitor GPU Usage

```
watch -n0.1 rocm-smi
```

## Dataset Generation 

1. Navigate to Imagen WORKDIR
```
cd /shareddata/dheyo/shivanvitha/image_gen
```

2. Activate Environment 
```
source imagen_env/bin/activate
```

3. Navigate to Synthetic Data Generation Directory 
```
cd synthetic_data_generation
```

4. Modify config.yaml

Modify the prompt, negative prompt and prompt suffix (if required)
Make sure to have a common prompt prefix for generating one character. For example, if you are generating images for a 17 year old Indian boy, start all your prompts with `A photo of a 17 year old lean, Indian boy with black curly hair, very light facial hair..` and keep a **constant seed** throughout all generations for a particular character

```yaml
diffuser_params:
  num_inference_steps: 50
  guidance_scale: 5.0
  prompt: "A photo of a 17 year old lean, Indian boy with black curly hair, very light facial hair, in his senior high school uniform, standing in his classroom with a happy expression"

  prompt_suffix: ""
  negative_prompt: "animated, cartoon, bad eyes, unibrow, extra arms, extra fingers, extra legs, mutated hands, fused fingers, long neck, cross-eyed, long head, deformed hands, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality"
  width: 1024
  height: 1024
  num_images_per_prompt: 5
  seed: 42

# diffuser_ckpt: "stabilityai/stable-diffusion-xl-base-1.0" 
diffuser_ckpt: "HiDream-ai/HiDream-I1-Full"

output_dir: "../synthetic_datasets"

output_sub_dir: "character_A/images" # change "character_A" to anything, keep "images" constant so that it will be easier to prepare HF compatible dataset (not required for OmniGen2 though)

device: "cuda"
```

5. Generate Images 

Find a unused GPU device (ids in the range [0..7]) and modify the value of `HIP_VISIBLE_DEVICES` environment variable 
```
HIP_VISIBLE_DEVICES=3 python3 hidream_init_generation.py
```

This will save all the images in `/shareddata/dheyo/shivanvitha/image_gen/synthetic_datasets/character_A/images`

## Prepare Fine-Tuning Set up

6. Deactivate imagen_env 
```
deactivate
```

7. Navigate to OmniGen2 and Activate Environment
```
cd /shareddata/dheyo/shivanvitha/OmniGen2
```

```
source omni/bin/activate
```


8. Modify data_configs/train/example/t2i/jsonls/0.jsonl

Change the instruction to desired prompts while keeping the special token followed by class constant in all rows (Ex: "[E]" boy, "rnp man", "[V] girl", etc..)

Also provide paths to the generated images in "output_image" field
```json

{"task_type": "t2i", "instruction": "A side profile photo of [E] boy smiling while looking through a window", "output_image": "/shareddata/dheyo/shivanvitha/image_gen/synthetic_datasets/character_A/images/<file_name_1>"}
```
**Note:** Consult Shivanvitha before modifying `options/ft_lora.yml` if you want to use more number of GPUs, a change in batch size (per device or global based on num_processes), etc..


## Launch OmniGen2 LoRA Fine-Tuning

9. The below command uses only 1 GPU to perform LoRA Fine-tuning

```bash
HIP_VISIBLE_DEVICES=3 accelerate launch  \
--machine_rank=0 \
--num_machines=1 \
--num_processes=1 \
--use_fsdp \
--fsdp_offload_params false \
--fsdp_sharding_strategy HYBRID_SHARD_ZERO2 \
--fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
--fsdp_transformer_layer_cls_to_wrap OmniGen2TransformerBlock \
--fsdp_state_dict_type FULL_STATE_DICT \
--fsdp_forward_prefetch false \
--fsdp_use_orig_params True \
--fsdp_cpu_ram_efficient_loading false \
--fsdp_sync_module_states True \
train.py --config options/ft_lora.yml
```

## Convert FSDP LoRA weights to HF weights

10. Post training, we MUST convert the FSDP-saved checkpoint (.bin) into the standard Hugging Face format before we can use it for inference.

```bash
python convert_ckpt_to_hf_format.py \
  --config_path experiments_character_A/ft/ft.yml \
  --model_path experiments_character_A/ft/checkpoint-<num>/pytorch_model_fsdp.bin \
  --save_path experiments_character_A/ft/checkpoint-<num>/transformer
```

Where <num> is the global step number whose checkpoint was saved.

## Multi-prompt Inference

11. Create a prompts.txt or any txt file with a list of prompts for inference

Each prompt should start on a newline

Sample prompts.txt is shown below:
```
A photo of [E] boy with a serious expression, in a blue jacket and black trousers, sitting in a classroom with books on the desk.
A photo of [E] boy with a serious expression, in a blue jacket and black trousers, in a playground with children playing behind.
A photo of [E] boy with a smiling face, dressed in a yellow t-shirt and shorts, sitting in a classroom with books on the desk.
....
```

12. Generate images in bulk

```bash
HIP_VISIBLE_DEVICES=3 python3 inference_2.py \
--model_path OmniGen2/OmniGen2 \
--num_inference_step 50 \
--height 1024 \
--width 1024 \
--text_guidance_scale 5.0 \
--output_image_path inferenced_images/omnigen2_lora_tests/omnigen2.png \
--num_images_per_prompt 2 \
--transformer_lora_path experiments_character_A/ft_lora/checkpoint-<num>/transformer_lora \
--prompts_path prompts.txt
```

All the images will be saved in `omnigen2_lora_tests`. Alternatively you can change the path in the command. Each image will be stored with a timestamp. 