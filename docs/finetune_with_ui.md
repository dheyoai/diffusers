# Fine-Tuning OmniGen2 LoRA Dreambooth Using UI Using a Single GPU

## Install Additonal Requirements

```bash
pip install gradio==5.39.0 gradio_client==1.11.0
```

## SSH Tunneling from gpu-22/gpu-60

```bash
ssh -L 7860:localhost:7860 ubuntu@gpu-22
```

**Note:** If you want to change the port, do as shown below in [../gradio_omnigen2_dreambooth_lora_ti_fit.py](../gradio_omnigen2_dreambooth_lora_ti_fit.py) and update your above SSH command accordingly:

```python
if __name__ == "__main__":
    demo.launch(share=True, show_error=True, server_port=7860) # 👈 change here

```

## Launch the Fine-Tuning Application

```bash
python3 gradio_omnigen2_dreambooth_lora_ti_fit.py --config options/ft_lora_pivotal_tuning.yml
```


## Tips for Creating Datasets

- Use high quality images in `.png` format
- Make sure captions contain the special/trigger token followed by the subject class (eg: `A photo of [Y] girl in a pink dress standing under sakura cherry blossoms`, `[dheyo_char1] man chilling in a park with a dog`)
- Use at least 16-20 images for each subject/dataset
- If you are looking for diverse outputs during inference, make sure that your datasets also contain diverse expressions, outfits, angles, etc..