# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER

model_path="/share_2/luoxin/projects/OmniGen2/pretrained_models/omnigen2_pipe_model_fuse_v17"
python inference.py \
--model_path $model_path \
--num_inference_step 50 \
--height 1024 \
--width 1024 \
--text_guidance_scale 6.0 \
--image_guidance_scale 1.0 \
--instruction "A curly-haired man in a red shirt is drinking tea." \
--output_image_path /share_2/luoxin/projects/OmniGen2/output_t2i.png \
--num_images_per_prompt 3