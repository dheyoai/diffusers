# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER

source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate py3.11+pytorch2.6+cu124

# --model_path /share_2/luoxin/projects/Omnigenv2/pretrained_models/ominigenv2_0519_ps1024_bs128_ft_t2i_human_ai_0519_deblur_lr4e-5 \
# --model_path OmniGen2/OmniGen2-preview \

python test.py \
--model_path OmniGen2/OmniGen2-preview \
--vae_path /share_2/luoxin/modelscope/hub/models/FLUX.1-dev \
--tokenizer_path Qwen/Qwen2.5-VL-3B-Instruct \
--text_encoder_path Qwen/Qwen2.5-VL-3B-Instruct \
--num_inference_step 28 \
--height 1024 \
--width 1024 \
--text_guidance_scale 4.0 \
--instruction "A dog running in the park" \
--output_image_path /share_2/luoxin/projects/OmniGen2/test_t2i.png \
--num_images_per_prompt 3