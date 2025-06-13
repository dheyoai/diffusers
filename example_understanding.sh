# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER

source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate py3.11+pytorch2.6+cu124

# model_path="OmniGen2/OmniGen2-preview"
# model_path="/share_2/luoxin/projects/OmniGen2/pretrained_models/omnigen2_pipe"
# model_path="/share_2/luoxin/projects/OmniGen2/pretrained_models/omnigen2_pipe_model_fuse_v1"
model_path="/share_2/luoxin/projects/OmniGen2/pretrained_models/omnigen2_pipe_model_fuse_v17_chat"


python inference.py \
--model_path $model_path \
--instruction "Please describe this image briefly." \
--input_image_path example_images/02.jpg \
--num_images_per_prompt 1