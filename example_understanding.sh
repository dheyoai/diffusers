# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER

model_path="/share_2/luoxin/projects/OmniGen2/pretrained_models/omnigen2_pipe_model_fuse_v17"

python inference_chat.py \
--model_path $model_path \
--instruction "Please describe this image briefly." \
--input_image_path example_images/02.jpg \
--num_images_per_prompt 1