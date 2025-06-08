<h1 align="center">OmniGen2</h1>


<p align="center">
    <a href="https://github.com/VectorSpaceLab/OmniGen2/tree/main">
        <img alt="Build" src="https://img.shields.io/badge/Project%20Page-OmniGen-yellow">
    </a>
    <a href="">
            <img alt="Build" src="https://img.shields.io/badge/arXiv%20paper-2409.11340-b31b1b.svg">
    </a>
    <a href="">
        <img alt="License" src="https://img.shields.io/badge/HF%20Demo-🤗-lightblue">
    </a>
    <a href="">
        <img alt="Build" src="https://img.shields.io/badge/HF%20Model-🤗-yellow">
    </a>    
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

OmniGen2 is a unified image generation model that can handle various image generation tasks including text-to-image, image editing, and multi-image composition.

## 1. News
- 2025-06-09:🔥🔥 We release OmniGen2-preview, a multimodal generation model. 


## TODO
- [ ] OmniGen2 checkpoint
- [ ] OmniGen2 technical report
- [ ] Trainging data and scripts


## Functions

- Text-to-Image Generation
- Image Editing
- Multi-Image Composition
- High-quality image generation with fine-grained control
- Support for various input formats and resolutions




## Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/OmniGen2.git
cd OmniGen2

# Install dependencies
pip install -r requirements.txt
```

### Web Interface

To launch the web interface:

```bash
python app.py
```

### Command Line Interface

For batch processing or programmatic usage:

```bash
python test.py --model_path /path/to/model \
               --instruction "Your prompt here" \
               --output_image_path output.png
```

### Example Scripts

We provide example scripts for common use cases:

```bash
# Text-to-Image generation
./test_example_t2i.sh

# Image editing
./test_example_edit.sh
```



