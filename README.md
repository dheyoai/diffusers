# Diffusers set-up for Dheyo AI

## Clone the Repository
```
git clone git@github.com:dheyoai/diffusers.git
cd diffusers
```

## Create Virtual Environment
```
python3.11 -m venv diffusers_env
source diffusers_env/bin/activate
```

## Set up diffusers as a Library
```
pip install -e .
```

## Install Dependencies
```
cd examples/dreambooth
pip install -r requirements.txt
```

## For AMD GPUs Only
```
pip install --pre --upgrade torch torchvision torchao --index-url https://download.pytorch.org/whl/nightly/rocm6.4
```

## Set your 🤗 Token
```
export HF_TOKEN=<>
```

## Launch your Fine-tuning Script

Create a custom accelerate config and add it to the accelerate command

```
accelerate launch --config_file /path/to/your/custom_config.yaml train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of nbks man" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 
```