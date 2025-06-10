<h1 align="center">OmniGen2</h1>

<p align="center">
  <a href="https://github.com/VectorSpaceLab/OmniGen2"><img src="https://img.shields.io/badge/Project%20Page-OmniGen2-yellow" alt="project page"></a>
  <a href=""><img src="https://img.shields.io/badge/arXiv%20paper-2409.11340-b31b1b.svg" alt="arxiv"></a>
  <a href=""><img src="https://img.shields.io/badge/HF%20Demo-🤗-lightblue" alt="demo"></a>
  <a href=""><img src="https://img.shields.io/badge/HF%20Model-🤗-yellow" alt="model"></a>
</p>

<h4 align="center">
    <p>
        <a href=#1-news>News</a> |
        <a href=#3-methodology>Methodology</a> |
        <a href=#4-what-can-omnigen-do>Capabilities</a> |
        <a href=#5-quick-start>Quick Start</a> |
        <a href="#6-finetune">Finetune</a> |
        <a href="#license">License</a> |
        <a href="#citation">Citation</a>
    <p>
</h4>


## 🔥 News
- **2025-06-16**: Gradio and Jupyter demo is available.
- **2025-06-16**: We release OmniGen2-preview, a multimodal generation model. 

## 📌 TODO
- [ ] Training data and scripts.

当然可以！下面是对你提供的 GitHub Page 中 Quick Start 部分进行的**重构版排版优化**，目标是更加清晰、结构分明、排版简洁、便于用户快速理解与操作：

---

## 🚀 Quick Start

### 🛠️ Environment Setup

#### ✅ Recommended Setup

```bash
# 1. Clone the repo
git clone git@github.com:VectorSpaceLab/OmniGen2.git
cd OmniGen2

# 2. (Optional) Create a clean Python environment
conda create -n omnigen2 python=3.11
conda activate omnigen2

# 3. Install dependencies
# 3.1 Install PyTorch (choose correct CUDA version)
pip install torch==2.6.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu124

# 3.2 Install other required packages
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

#### 🌏 For users in Mainland China 🇨🇳

```bash
# Install PyTorch from a domestic mirror
pip install torch==2.6.0 torchvision --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu124

# Install other dependencies from Tsinghua mirror
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install flash-attn --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

### 🧪 Run Examples

```bash
# Text-to-image generation
bash example_t2i.sh

# Instruction-guided image editing
bash example_edit.sh

# Subject-driven image editing
bash example_subject_driven_edit.sh
```

---

### 🌐 Gradio Demo

* **Online Demo**: [Available on Hugging Face Spaces 🚀](https://huggingface.co/spaces/Shitao/OmniGen2)

* **Run Locally**:

```bash
pip install gradio
python app.py
# Optional: Share demo with public link
python app.py --share
```

---

### 📓 Jupyter Notebook

Interactive example: `example.ipynb`

<!-- ## :rocket: Quick Start

- **Set up environment**
<!-- Install via Github:
```bash
git clone https://github.com/VectorSpaceLab/OmniGen.git
cd OmniGen
pip install -e .
``` -->

<!-- You also can create a new environment to avoid conflicts: -->
    ```bash
    # 1. Download our repo
    git clone git@github.com:VectorSpaceLab/OmniGen2.git
    cd OmniGen2

    # 2. Create virtual environment (Optional)
    conda create -n omnigen2 python=3.11
    conda activate omnigen2

    # 3. Install dependencies
    # 3.1 Install pytorch with your CUDA version, e.g.
    pip install torch==2.6.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu124
    # 3.2 Install other packages
    pip install -r requirements.txt
    pip install flash-attn --no-build-isolation

    # 3. 如果你是来自中国大陆的用户，可以使用下面的命令从国内源进行安装。
    # 3.1 根据你的CUDA版本安装pytorch
    pip install torch==2.6.0 torchvision --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu124
    # 3.1 安装其他依赖
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install flash-attn --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

- **Try OmniGen2 with following examples**:
    ```shell
    # text-to-image generation
    bash example_t2i.sh
    # instruction-guided image editing
    bash example_edit.sh
    # subject-driven image editing
    bash example_subject_driven_edit.sh
    ```

- **Gradio**

    We construct an online demo in [Huggingface](https://huggingface.co/spaces/Shitao/OmniGen2).

    For the local gradio demo, you need to install `pip install gradio spaces`, and then you can run:
    ```shell
    pip install gradio
    python app.py
    # or using share flag if you want to create a share link
    python app.py --share
    ```
- **Jupyter Notebook**

    see `example.ipynb` -->


<!-- ## 2. Overview

OmniGen is a unified image generation model that can generate a wide range of images from multi-modal prompts. It is designed to be simple, flexible, and easy to use. We provide [inference code](#5-quick-start) so that everyone can explore more functionalities of OmniGen.

Existing image generation models often require loading several additional network modules (such as ControlNet, IP-Adapter, Reference-Net, etc.) and performing extra preprocessing steps (e.g., face detection, pose estimation, cropping, etc.) to generate a satisfactory image. However, **we believe that the future image generation paradigm should be more simple and flexible, that is, generating various images directly through arbitrarily multi-modal instructions without the need for additional plugins and operations, similar to how GPT works in language generation.** 

Due to the limited resources, OmniGen still has room for improvement. We will continue to optimize it, and hope it inspires more universal image-generation models. You can also easily fine-tune OmniGen without worrying about designing networks for specific tasks; you just need to prepare the corresponding data, and then run the [script](#6-finetune). Imagination is no longer limited; everyone can construct any image-generation task, and perhaps we can achieve very interesting, wonderful, and creative things.

If you have any questions, ideas, or interesting tasks you want OmniGen to accomplish, feel free to discuss with us: 2906698981@qq.com, wangyueze@tju.edu.cn, zhengliu1026@gmail.com. We welcome any feedback to help us improve the model.



## 3. Methodology

You can see details in our [paper](https://arxiv.org/abs/2409.11340). 



## 4. What Can OmniGen do?

OmniGen is a unified image generation model that you can use to perform various tasks, including but not limited to text-to-image generation, subject-driven generation, Identity-Preserving Generation, image editing, and image-conditioned generation. **OmniGen doesn't need additional plugins or operations, it can automatically identify the features (e.g., required object, human pose, depth mapping) in input images according to the text prompt.**
We showcase some examples in [inference.ipynb](inference.ipynb). And in [inference_demo.ipynb](inference_demo.ipynb), we show an interesting pipeline to generate and modify an image.

Here is the illustrations of OmniGen's capabilities: 
- You can control the image generation flexibly via OmniGen
![demo](./imgs/demo_cases.png)
- Referring Expression Generation: You can input multiple images and use simple, general language to refer to the objects within those images. OmniGen can automatically recognize the necessary objects in each image and generate new images based on them. No additional operations, such as image cropping or face detection, are required.
![demo](./imgs/referring.png)

If you are not entirely satisfied with certain functionalities or wish to add new capabilities, you can try [fine-tuning OmniGen](#6-finetune).


 -->
## :heart: Citing Us
If you find this repository or our work useful, please consider giving a star :star: and citation :t-rex:, which would be greatly appreciated:

```bibtex
@article{xiao2024omnigen,
  title={Omnigen: Unified image generation},
  author={Xiao, Shitao and Wang, Yueze and Zhou, Junjie and Yuan, Huaying and Xing, Xingrun and Yan, Ruiran and Wang, Shuting and Huang, Tiejun and Liu, Zheng},
  journal={arXiv preprint arXiv:2409.11340},
  year={2024}
}
```