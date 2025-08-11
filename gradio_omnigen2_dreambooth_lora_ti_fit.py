import os
from huggingface_hub import whoami    
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import sys
import gradio as gr
from PIL import Image
import torch
import uuid
import shutil
import json
import yaml
import threading
import argparse
from omegaconf import OmegaConf

from transformers import AutoProcessor, AutoModelForCausalLM

sys.path.insert(0, "ai-toolkit")

MAX_IMAGES = 150

def parse_args(root_path) -> OmegaConf:
    parser = argparse.ArgumentParser(description="OmniGen2 training script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (YAML format)",
    )
    parser.add_argument(
        "--global_batch_size",
        type=int,
        default=None,
        help="Global batch size.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Data path.",
    )
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)

    # output_dir = os.path.join(root_path, 'experiments', conf.name)
    output_dir = os.path.join(root_path, conf.save_path, conf.name)
    conf.root_dir = root_path
    conf.output_dir = output_dir
    conf.config_file = args.config

    # Override config with command line arguments
    if args.global_batch_size is not None:
        conf.train.global_batch_size = args.global_batch_size
    
    if args.data_path is not None:
        conf.data.data_path = args.data_path
    return conf


def load_captioning(uploaded_files, concept_token, initializer_concept, dataset_index):
    uploaded_images = [file for file in uploaded_files if not file.endswith('.txt')]
    txt_files = [file for file in uploaded_files if file.endswith('.txt')]
    txt_files_dict = {os.path.splitext(os.path.basename(txt_file))[0]: txt_file for txt_file in txt_files}
    updates = []
    
    if len(uploaded_images) <= 1:
        raise gr.Error(
            f"Please upload at least 2 images for dataset {dataset_index + 1} (the ideal number with default settings is between 4-30)"
        )
    elif len(uploaded_images) > MAX_IMAGES:
        raise gr.Error(f"For dataset {dataset_index + 1}, only {MAX_IMAGES} or less images are allowed")
    
    # Update visibility for captioning area
    updates.append(gr.update(visible=True))
    
    # Update visibility and content for each captioning row
    for i in range(1, MAX_IMAGES + 1):
        visible = i <= len(uploaded_images)
        updates.append(gr.update(visible=visible))
        
        image_value = uploaded_images[i - 1] if visible else None
        updates.append(gr.update(value=image_value, visible=visible))
        
        corresponding_caption = False
        if image_value:
            base_name = os.path.splitext(os.path.basename(image_value))[0]
            if base_name in txt_files_dict:
                with open(txt_files_dict[base_name], 'r') as file:
                    corresponding_caption = file.read()
                    
        text_value = corresponding_caption if visible and corresponding_caption else f"[trigger{dataset_index + 1}]" if visible and concept_token else None
        updates.append(gr.update(value=text_value, visible=visible))
    
    # Update sample prompts
    updates.append(gr.update(visible=True))
    updates.append(gr.update(placeholder=f'A portrait of person in a bustling cafe [trigger{dataset_index + 1}]', value=f'A person in a bustling cafe [trigger{dataset_index + 1}]'))
    updates.append(gr.update(placeholder=f"A mountainous landscape in the style of [trigger{dataset_index + 1}]"))
    updates.append(gr.update(placeholder=f"A [trigger{dataset_index + 1}] in a mall"))
    updates.append(gr.update(visible=True))
    
    return updates

def hide_captioning():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

def create_dataset(*inputs):
    # Split inputs into three datasets
    dataset_inputs = [inputs[i * (MAX_IMAGES + 1):(i + 1) * (MAX_IMAGES + 1)] for i in range(3)]
    dataset_1 = None
    dataset_2 = None
    dataset_3 = None
    
    yaml_entries = []

    for idx, dataset_input in enumerate(dataset_inputs):
        images = dataset_input[0]
        captions = dataset_input[1:]
        if not images:
            continue
        destination_folder = f"datasets/{uuid.uuid4()}"
        os.makedirs(destination_folder, exist_ok=True)
        
        jsonl_file_path = os.path.join(destination_folder, "metadata.jsonl")
        with open(jsonl_file_path, "a") as jsonl_file:
            for index, image in enumerate(images):
                new_image_path = shutil.copy(image, destination_folder)
                original_caption = captions[index] if index < len(captions) and captions[index] else ""
                file_name = os.path.basename(new_image_path)
                data = {"task_type": "t2i", "instruction": original_caption, "output_image": os.path.join(destination_folder, file_name)}
                jsonl_file.write(json.dumps(data) + "\n")

        yaml_entries.append({
            "path": jsonl_file_path,
            "type": "t2i",
            "ratio": 1.0
        })


        if idx == 0:
            dataset_1 = destination_folder
        elif idx == 1:
            dataset_2 = destination_folder
        elif idx == 2:
            dataset_3 = destination_folder
    
    # Save YAML
    if yaml_entries:
        yml_data = {"data": yaml_entries}
        yaml_path = "datasets/datasets_config.yml"
        with open(yaml_path, "w") as yml_file:
            yaml.dump(yml_data, yml_file, default_flow_style=False)

    return dataset_1, dataset_2, dataset_3 ### return the dataset_config.yml file here to pass to start_training()
    # return yaml_path if yaml_path else None

def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and v:
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d



                # lora_name,
                # *[dc['concept_token'] for dc in dataset_components[:-1]],  # <- Fixed here
                # *[dc['initializer_concept'] for dc in dataset_components[:-1]],  # <- Fixed here
                # steps,
                # lr,
                # rank,
                # model_to_train,
                # # low_vram,
                # dataset_state_1,
                # dataset_state_2,
                # dataset_state_3,
                # dataset_components[-1]['sample_1'],
                # dataset_components[-1]['sample_2'],
                # dataset_components[-1]['sample_3'],
                # use_more_advanced_options,
                # more_advanced_options
import queue
output_queue = queue.Queue()
from contextlib import redirect_stdout, redirect_stderr
import io 
import subprocess

def start_training(
    lora_name,
    concept_token_1, concept_token_2, concept_token_3, ### need to add more
    initializer_concept_1, initializer_concept_2, initializer_concept_3, ### need to add more
    steps,
    lr,
    rank,
    batch_size,
    global_batch_size,
    checkpointing_steps,
    # low_vram,
    dataset_folder_1,
    dataset_folder_2,
    dataset_folder_3,
    sample_1,
    sample_2,
    sample_3,
    use_more_advanced_options,
    more_advanced_options,
):
    print("\n\nTraining will be initialized now...\n\n")
    from train_with_pivotal_tuning import main
    root_path = os.path.abspath(os.path.join(__file__, os.path.pardir))
    args = parse_args(root_path)

    # args.name = lora_name
    # args.save_path = f"experiments_{lora_name}"
    # args.pivotal_tuning.initializer_concept = f"{initializer_concept_1}\n{initializer_concept_2}\n{initializer_concept_3}"
    # args.pivotal_tuning.initializer_concept = args.pivotal_tuning.initializer_concept.strip()
    # args.pivotal_tuning.token_abstraction = f"{concept_token_1},{concept_token_2},{concept_token_3}"
    # args.pivotal_tuning.token_abstraction = args.pivotal_tuning.token_abstraction.strip(",") 
    # args.data.data_path = "datasets/datasets_config.yml" ## path to the created t2i yml file
    # args.train.batch_size = batch_size
    # args.train.global_batch_size = global_batch_size
    # args.train.max_train_steps = steps
    # args.train.learning_rate = lr
    # args.train.lora_rank = rank
    # args.logger.checkpointing_steps = checkpointing_steps
    # args.output_dir = f"{args.root_dir}/{args.save_path}/ft_lora"

    # Modify arguments based on Gradio inputs
    OmegaConf.update(args, "name", lora_name)
    OmegaConf.update(args, "save_path", f"experiments_{lora_name}")
    OmegaConf.update(args, "pivotal_tuning.initializer_concept",
                     f"{initializer_concept_1}\n{initializer_concept_2}\n{initializer_concept_3}".strip())
    OmegaConf.update(args, "pivotal_tuning.token_abstraction",
                     f"{concept_token_1},{concept_token_2},{concept_token_3}".strip(","))
    OmegaConf.update(args, "data.data_path", "datasets/datasets_config.yml")
    OmegaConf.update(args, "train.batch_size", batch_size)
    OmegaConf.update(args, "train.global_batch_size", global_batch_size)
    OmegaConf.update(args, "train.max_train_steps", steps)
    OmegaConf.update(args, "train.learning_rate", lr)
    OmegaConf.update(args, "train.lora_rank", rank)
    OmegaConf.update(args, "logger.checkpointing_steps", checkpointing_steps)
    OmegaConf.update(args, "output_dir", f"{args.root_dir}/{args.save_path}/ft_lora")

    # Save the updated config to a YAML file
    config_path = os.path.join(root_path, args.config_file)
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(args))
    print(f"Saved updated config to {config_path}\n")
    print(args)

    # proc = subprocess.Popen([
    #     "accelerate", "launch",
    #     "--config_file", "config/ddp.yaml",
    #     "train_with_pivotal_tuning.py",
    #     "--config", "options/ft_lora_AA_AB.yml"
    # ])

    # try:
    #     proc.wait()  # block until finished
    # except KeyboardInterrupt:
    #     print("Stopping training...")
    #     proc.terminate()
    




    # import pdb; pdb.set_trace()
    try:
        # # Clear the queue
        # while not output_queue.empty():
        #     output_queue.get()

        # # Capture stdout and stderr
        # output_buffer = io.StringIO()
        
        # def run_main():
        #     try:
        #         with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
        main(args)  # Call the main function from train.py
        #         output_queue.put(output_buffer.getvalue())
        #         output_queue.put("Training completed successfully.")
        #     except Exception as e:
        #         output_queue.put(f"Error: {str(e)}")
        # # main(args)
        # threading.Thread(target=run_main, daemon=True).start()

        # # Yield initial message
        # yield "Training started...\n"

        # # Stream output from the queue
        # while True:
        #     try:
        #         line = output_queue.get_nowait()
        #         yield line
        #     except queue.Empty:
        #         yield ""
        #         # Sleep briefly to avoid busy-waiting
        #         import time
        #         time.sleep(0.1)

        ### what to do after training is done?
        # import pdb; pdb.set_trace()
        # convert the ckpts -> take last 2 ckpts and convert
        checkpoints = os.listdir(args.output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]), reverse=True)
        ckpt_conversion_args = {
                                "config_path": args.config_file, 
                                "model_path": f"{args.output_dir}/{checkpoints[0]}/model.safetensors",
                                "save_path": f"{args.output_dir}/{checkpoints[0]}/transformer_lora"
                                }
        ckpt_conversion_args = OmegaConf.create(ckpt_conversion_args)
        from convert_ckpt_to_hf_format import main as ckpt_conversion_main
        ckpt_conversion_main(ckpt_conversion_args)
        # provide the path to the ckpt and the command to run inference or editing
    except KeyboardInterrupt:
        pass
    # finally:
    #     # Ensure all processes exit cleanly
    #     torch.distributed.destroy_process_group()

    # dataset_folders = [folder for folder in [dataset_folder_1, dataset_folder_2, dataset_folder_3] if folder]
    # return f"Training started with LoRA name: {lora_name}, datasets: {dataset_folders}"
    return f"""
    # Details of this Fine-tuning ({lora_name}): 

    ### Status: Completed ✅

    ### Path to the latest chakpoint: {ckpt_conversion_args.save_path}

    ### Commands for Inference Pipeline:
    ```bash
    ssh -L 8900:localhost:8900 ubuntu@gpu-22
    ```
    ```bash
    HIP_VISIBLE_DEVICES=7 python3 gradio_img_generation.py --transformer_lora_path {ckpt_conversion_args.save_path}
    ```

    ### Commands for Editing Pipeline:
    ```bash
    ssh -L 8899:localhost:8899 ubuntu@gpu-22
    ```
    ```bash
    HIP_VISIBLE_DEVICES=7 python3 gradio_app_multi_img.py --transformer_lora_path {ckpt_conversion_args.save_path}
    ```
    """

config_yaml = '''
name: ft_lora

seed: 4200
device_specific_seed: true
workder_specific_seed: true

save_path: "experiments_gradio_dummy_3"

pivotal_tuning:
  pivotal_tuning: true
  initializer_concept: "A 35 year old 5 feet 7 inches Indian man with wavy black hair, specific hairstyle, thick black beard, strong build, fair skin\nA 30 year old 5 feet 1 inches Indian woman, slim build, slim jawline, very fair skin, dimples on cheeks"
  token_abstraction: "[A],[AB]" # AA Allu Arjun, AB Alia Bhat


data:
  data_path: data_configs/train/example/t2i/AA_AB.yml
  use_chat_template: true
  maximum_text_tokens: 888
  prompt_dropout_prob: !!float 0.0001
  ref_img_dropout_prob: !!float 0.5
  max_output_pixels: 1048576 # 1024 * 1024
  max_input_pixels: [1048576, 1048576, 589824, 262144] # [1024 * 1024, 1024 * 1024, 768 * 768, 512 * 512]
  max_side_length: 2048
  
model:
  pretrained_vae_model_name_or_path: black-forest-labs/FLUX.1-dev
  pretrained_text_encoder_model_name_or_path: Qwen/Qwen2.5-VL-3B-Instruct
  pretrained_model_path: OmniGen2/OmniGen2
  
  arch_opt:
    patch_size: 2
    in_channels: 16
    hidden_size: 2520
    num_layers: 32
    num_refiner_layers: 2
    num_attention_heads: 21
    num_kv_heads: 7
    multiple_of: 256
    norm_eps: !!float 1e-05
    axes_dim_rope: [40, 40, 40]
    axes_lens: [10000, 10000, 10000]
    text_feat_dim: 2048
    timestep_scale: !!float 1000

transport:
  snr_type: lognorm
  do_shift: true
  dynamic_time_shift: true

train:
  # Dataloader
  global_batch_size: 2
  batch_size: 2
  gradient_accumulation_steps: 1

  # max_train_steps: 3000
  num_train_epochs: 200
  
  dataloader_num_workers: 0

  # Optimizer
  learning_rate: !!float 2e-4
  scale_lr: false
  lr_scheduler: timm_constant_with_warmup
  warmup_t: 300
  warmup_lr_init: 1e-18
  warmup_prefix: true
  t_in_epochs: false

  # resume_from_checkpoint: 

  use_8bit_adam: false
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_weight_decay: !!float 0.01
  adam_epsilon: !!float 1e-08
  max_grad_norm: 1

  gradient_checkpointing: true
  
  set_grads_to_none: true

  # Misc
  allow_tf32: false
  mixed_precision: 'bf16'

  ema_decay: 0.0

  lora_ft: true
  lora_rank: 32
  lora_dropout: 0

val:
  validation_steps: 600
  train_visualization_steps: 600

logger:
  log_with: [tensorboard]
  # log_with: ~

  checkpointing_steps: 600
  checkpoints_total_limit: ~

cache_dir: 
resume_from_checkpoint: latest

'''

theme = gr.themes.Monochrome(
    text_size=gr.themes.Size(lg="18px", md="15px", sm="13px", xl="22px", xs="12px", xxl="24px", xxs="9px"),
    font=[gr.themes.GoogleFont("Source Sans Pro"), "ui-sans-serif", "system-ui", "sans-serif"],
)
css = """
body, .gradio-container {
    background-color: #0b1e3b; /* Dark navy blue */
    color: #e0e6f0; /* Light text for readability */
}
h1, h2, h3, label, .markdown-body {
    color: #e0e6f0; /* Headings and labels in light gray */
}

input, textarea, select {
    background-color: #142b4f; /* Slightly lighter blue for inputs */
    color: #ffffff;
    border: 1px solid #2e4a78;
}

button {
    background-color: #1f3d66;
    color: white;
    border: none;
}

button:hover {
    background-color: #2d558f;
}

h1{font-size: 2em}
h3{margin-top: 0}
#component-1{text-align:center}
.main_ui_logged_out{opacity: 0.3; pointer-events: none}
.tabitem{border: 0px}
.group_padding{padding: .55em}
"""

with gr.Blocks(theme=theme, css=css) as demo:
    gr.Markdown(
        """# OmniGen2 DreamBooth LoRA with Textual Inversion
### Fine-tune a high quality OmniGen2 LoRA using [dheyoai@omnigen2](https://github.com/dheyoai/diffusers/tree/omnigen2)"""
    )
    with gr.Column() as main_ui:
        lora_name = gr.Textbox(
            label="The name of your LoRA",
            info="This has to be a unique name",
            placeholder="e.g.: Persian Miniature Painting style, Cat Toy",
        )
        
        # Create three dataset sections
        dataset_components = []
        all_caption_inputs = []
        dataset_state_1 = gr.State()
        dataset_state_2 = gr.State()
        dataset_state_3 = gr.State()
        
        for i in range(3):
            with gr.Group(visible=True) as image_upload:
                gr.Markdown(f"## Dataset {i+1}")
                concept_token = gr.Textbox(
                    label=f"Trigger token {i+1}",
                    info=f"Trigger token for dataset {i+1}",
                    placeholder=f"uncommon word like p3rs0n{i+1} or trtcrd{i+1}",
                    interactive=True,
                )
                initializer_concept = gr.Textbox(
                    label=f"Initializer concept to train TI embeddings {i+1}",
                    info=f"High level subject description without the trigger word {i+1}",
                    placeholder=f"A 33 year old Indian man with a beard, strong build, fair skin",
                    interactive=True,
                )
                with gr.Row():
                    images = gr.File(
                        file_types=["image", ".txt"],
                        label=f"Upload images for dataset {i+1}",
                        file_count="multiple",
                        interactive=True,
                        visible=True,
                        scale=1,
                    )
                    with gr.Column(scale=3, visible=False) as captioning_area:
                        with gr.Column():
                            gr.Markdown(
                                f"""# Custom captioning for dataset {i+1}
<p style="margin-top:0">Add custom captions for each image. [trigger{i+1}] will represent your concept sentence/trigger word.</p>
""", elem_classes="group_padding")
                            output_components = [captioning_area]
                            caption_list = []
                            for j in range(1, MAX_IMAGES + 1):
                                locals()[f"captioning_row_{i}_{j}"] = gr.Row(visible=False)
                                with locals()[f"captioning_row_{i}_{j}"]:
                                    locals()[f"image_{i}_{j}"] = gr.Image(
                                        type="filepath",
                                        width=111,
                                        height=111,
                                        min_width=111,
                                        interactive=False,
                                        scale=2,
                                        show_label=False,
                                        show_share_button=False,
                                        show_download_button=False,
                                    )
                                    locals()[f"caption_{i}_{j}"] = gr.Textbox(
                                        label=f"Caption {j}", scale=15, interactive=True
                                    )
                                output_components.append(locals()[f"captioning_row_{i}_{j}"])
                                output_components.append(locals()[f"image_{i}_{j}"])
                                output_components.append(locals()[f"caption_{i}_{j}"])
                                caption_list.append(locals()[f"caption_{i}_{j}"])
                
                # with gr.Accordion(f"Sample prompts for dataset {i+1} (optional)", visible=False) as sample:
                #     gr.Markdown(
                #         f"Include sample prompts to test out your trained model for dataset {i+1}. Include your trigger word/sentence."
                #     )
                #     sample_1 = gr.Textbox(label="Test prompt 1")
                #     sample_2 = gr.Textbox(label="Test prompt 2")
                #     sample_3 = gr.Textbox(label="Test prompt 3")
                
                # output_components.extend([sample, sample_1, sample_2, sample_3])
                dataset_components.append({
                    'images': images,
                    'concept_token': concept_token,
                    'initializer_concept': initializer_concept,
                    'captioning_area': captioning_area,
                    'output_components': output_components,
                    'caption_list': caption_list,
                    # 'sample': sample,
                    # 'sample_1': sample_1,
                    # 'sample_2': sample_2,
                    # 'sample_3': sample_3
                })
                all_caption_inputs.extend([images] + caption_list)
                
                images.upload(
                    load_captioning,
                    inputs=[images, concept_token, initializer_concept, gr.State(i)],
                    outputs=output_components
                ).then(
                    fn=lambda x: x,
                    inputs=[dataset_state_1, dataset_state_2, dataset_state_3][i],
                    outputs=[dataset_state_1, dataset_state_2, dataset_state_3][i]
                )
                
                images.delete(
                    load_captioning,
                    inputs=[images, concept_token, initializer_concept, gr.State(i)],
                    outputs=output_components
                )
                
                images.clear(
                    hide_captioning,
                    # outputs=[captioning_area, sample]
                    outputs=[captioning_area]
                )

        with gr.Accordion(f"Sample prompts for dataset (optional)", visible=True) as sample:
            gr.Markdown(
                f"Include sample prompts to test out your trained model. Include your trigger word/sentence."
            )
            sample_1 = gr.Textbox(placeholder="Test prompt 1")
            sample_2 = gr.Textbox(placeholder="Test prompt 2")
            sample_3 = gr.Textbox(placeholder="Test prompt 3")
        
        output_components.extend([sample, sample_1, sample_2, sample_3])
        dataset_components.append({
            'sample': sample,
            'sample_1': sample_1,
            'sample_2': sample_2,
            'sample_3': sample_3
        })
        images.clear(
            hide_captioning,
            # outputs=[captioning_area, sample]
            outputs=[sample]
        )
        with gr.Accordion("Advanced options", open=False):
            steps = gr.Number(label="Steps", value=3000, minimum=1, maximum=10000, step=1)
            lr = gr.Number(label="Learning Rate", value=2e-4, minimum=1e-6, maximum=1e-3, step=1e-6)
            rank = gr.Number(label="LoRA Rank", value=16, minimum=4, maximum=128, step=4)
            batch_size = gr.Number(label="Per Device Batch Size", value=2, minimum=1, maximum=16, step=1)
            global_batch_size = gr.Number(label="Effective Batch Size (No. of GPUs * batch_size)", value=2, minimum=1, maximum=32, step=4)
            checkpointing_steps = gr.Number(label="Checkpointing Frequency", value=400, minimum=1, maximum=1000, step=100)



            # model_to_train = gr.Radio(["omnigen2", "What's next?"], value="omnigen2", label="Model to train")
            # low_vram = gr.Checkbox(label="Low VRAM", value=True)
            # batch_size, global_batch_size, checkpointing_steps
            with gr.Accordion("Even more advanced options", open=False):
                use_more_advanced_options = gr.Checkbox(label="Use more advanced options", value=False)
                more_advanced_options = gr.Code(config_yaml, language="yaml")

        start = gr.Button("Start training", visible=True)
        progress_area = gr.Markdown("")

        # print(dataset_components)
        start.click(
            fn=create_dataset,
            inputs=all_caption_inputs,
            outputs=[dataset_state_1, dataset_state_2, dataset_state_3]
        ).then(
            fn=start_training,
            inputs=[
                lora_name,
                *[dc['concept_token'] for dc in dataset_components[:-1]],  # <- Fixed here
                *[dc['initializer_concept'] for dc in dataset_components[:-1]],  # <- Fixed here
                steps,
                lr,
                rank,
                batch_size,
                global_batch_size,
                checkpointing_steps,
                # low_vram,
                dataset_state_1,
                dataset_state_2,
                dataset_state_3,
                dataset_components[-1]['sample_1'],
                dataset_components[-1]['sample_2'],
                dataset_components[-1]['sample_3'],
                use_more_advanced_options,
                more_advanced_options
            ],
            outputs=progress_area
        )

if __name__ == "__main__":
    demo.launch(share=True, show_error=True, server_port=7860)