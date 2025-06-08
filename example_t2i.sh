# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER

source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate py3.11+pytorch2.6+cu124

# --model_path /share_2/luoxin/projects/Omnigenv2/pretrained_models/ominigenv2_0519_ps1024_bs128_ft_t2i_human_ai_0519_deblur_lr4e-5 \
# --model_path OmniGen2/OmniGen2-preview \

# model_path="/share_2/luoxin/projects/OmniGen2/pretrained_models/omnigen2_pipe"
model_path="OmniGen2/OmniGen2-preview"

python inference.py \
--model_path $model_path \
--num_inference_step 28 \
--height 1024 \
--width 1024 \
--text_guidance_scale 6.0 \
--instruction "A dog running in the park" \
--output_image_path /share_2/luoxin/projects/OmniGen2/output_t2i.png \
--num_images_per_prompt 3