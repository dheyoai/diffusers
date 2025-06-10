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

#### 🌏 For users in Mainland China

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