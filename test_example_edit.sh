# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER

source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate py3.11+pytorch2.6+cu124

python test.py \
--config_path /share_2/luoxin/projects/Omnigenv2/experiments/ominigenv2_0519_ps1024_bs128_ft_t2i_human_ai_0519_deblur_lr4e-5/ominigenv2_0519_ps1024_bs128_ft_t2i_human_ai_0519_deblur_lr4e-5.yml \
--model_path /share_2/luoxin/projects/Omnigenv2/experiments/ominigenv2_0519_ps1024_bs128_ft_t2i_human_ai_0519_deblur_lr4e-5/checkpoint-1500/pytorch_model_fsdp.bin \
--vae_path /share_2/luoxin/modelscope/hub/models/FLUX.1-dev \
--tokenizer_path Qwen/Qwen2.5-VL-3B-Instruct \
--text_encoder_path Qwen/Qwen2.5-VL-3B-Instruct \
--num_inference_step 28 \
--time_shift_scale 3.0 \
--height 1024 \
--width 1024 \
--guidance_scale 4.0 \
--instruction "A car toy and a bear toy are placed on the bench" \
--input_image_path example_images \
--output_image_path /share_2/luoxin/projects/Omnigenv2/test_edit.png \
--num_images_per_prompt 3