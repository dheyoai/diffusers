# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER

model_path="/share_2/luoxin/projects/OmniGen2/pretrained_models/omnigen2_pipe_model_fuse_v17"
python inference.py \
--model_path $model_path \
--num_inference_step 50 \
--height 1024 \
--width 1024 \
--text_guidance_scale 5.0 \
--image_guidance_scale 1.8 \
--instruction "Add a beautiful girl with long flowing hair seated beside the bear toy on the park bench." \
--input_image_path example_images/02.jpg \
--output_image_path /share_2/luoxin/projects/OmniGen2/output_edit.png \
--num_images_per_prompt 3