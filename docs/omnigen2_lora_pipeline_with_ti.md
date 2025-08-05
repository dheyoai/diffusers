## Training Settings

1. Modify `options/ft_lora_pivotal_tuning.yml`

Change the save path, initializer concepts and the token abstraction.
Use "\n" as the delimiter for the concepts and a "," for token abstractions (no spaces)

```yaml
save_path: "experiments_test_dummy_dummy" # 👈 modify here

pivotal_tuning:
  pivotal_tuning: true
  initializer_concept: "A 33 year old Indian man with black hair, black beard, strong build\nA 77 year old Indian old man with slight belly fat and white facial hair\nA 17 year old Indian boy with light facial hair going to university\nA 17 year old Indian girl with long hair going to university" # 👈 modify here
  token_abstraction: "[K],[O],[E],[Y]"  # 👈 modify here

```

Note: Modifications to jsonls and t2i.yml (TI version) are similar to what's demonstrated in [complete_omnigen2_lora_pipeline](complete_omnigen2_lora_pipeline.md)


## Launch Training with DDP

The easiest and best way to train two modules simulatenously (as in case of TI where DiT and Text Encoder are fine-tuned), DDP (Distributed Data Parallelism) is the best way. Here, only the data is sharded and a copy of params, grads and optimizer states are maintained across all GPU processes. 
Note: For DDP, the model has to fit into a single GPU.

You can change any of the settings below in [config/ddp.yaml](../config/ddp.yaml)

```yaml
{
  "compute_environment": "LOCAL_MACHINE",
  "debug": false,
  "distributed_type": "MULTI_GPU",
  "downcast_bf16": false,
  "enable_cpu_affinity": false,
  "machine_rank": 0,
  "main_training_function": "main",
  "mixed_precision": "no",
  "num_machines": 1,
  "num_processes": 1, # 👈 number of GPUs
  "rdzv_backend": "static",
  "same_network": false,
  "tpu_use_cluster": false,
  "tpu_use_sudo": false,
  "use_cpu": false
}

```

```bash
accelerate launch --config_file config/ddp.yaml train_with_pivotal_tuning.py --config options/ft_lora_pivotal_tuning.yml

```

## Convert to HF LoRA Weights

Like in FSDP we need to convert the saved DiT weights to proper HF compatible LoRA weights.

```bash
python3 convert_ckpt_to_hf_format.py \
  --config_path experiments_<name>/ft_lora/ft_lora_pivotal_tuning.yml \
  --model_path experiments_<name>/ft_lora/checkpoint-<num>/model.safetensors \ 
  --save_path experiments_<name>/ft_lora/checkpoint-<num>/transformer_lora

```

This will save a `pytorch_lora_weights.safetensors` in the specified save_path

## Single Prompt Inference

A sample command is shown below:

```bash
python3 inference.py \
--model_path OmniGen2/OmniGen2 \
--num_inference_step 50 \
--height 1024 \
--width 1024 \
--text_guidance_scale 5.0 \
--instruction "A photo of (([AA] man)) and (([AB] woman)), sharing the frame in a beautiful botanical garden, mountains in the background, both are placed in a romantic pose." \
--output_image_path inferenced_images/com_ti.png \
--num_images_per_prompt 7 \
--transformer_lora_path experiments_allu_arjun_alia_bhat/ft_lora/checkpoint-2400/transformer_lora \
--token_abstraction_json_path experiments_allu_arjun_alia_bhat/ft_lora/checkpoint-2400/tokens.json 

```

## Multi Prompt Inference

In order to perform OmniGen2 dreambooth TI bulk inference, refer to the below sample command:

```bash
python3 inference_2.py \
--model_path OmniGen2/OmniGen2 \
--num_inference_step 50 \
--height 1024 \
--width 1024 \
--text_guidance_scale 5.0 \
--output_image_path inferenced_images/allu_and_alia_ckpt_2400_new/aa_ab.png \
--num_images_per_prompt 7 \
--transformer_lora_path /shareddata/dheyo/shivanvitha/OmniGen2/experiments_allu_arjun_alia_bhat/ft_lora/checkpoint-2400/transformer_lora \
--token_abstraction_json_path  experiments_allu_arjun_alia_bhat/ft_lora/checkpoint-2400/tokens.json  \
--prompts_path prompts/allu_alia_prompts.txt

```

`--prompts_path` should be a .txt file with each prompt starting in a new line.