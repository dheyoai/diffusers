# OmniGen2

OmniGen2 is a unified image generation model that can handle various image generation tasks including text-to-image, image editing, and multi-image composition.

## Features

- Text-to-Image Generation
- Image Editing
- Multi-Image Composition
- High-quality image generation with fine-grained control
- Support for various input formats and resolutions

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/OmniGen2.git
cd OmniGen2

# Install dependencies
pip install -r requirements.txt
```

## Usage

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

## Model Architecture

OmniGen2 uses a transformer-based architecture with:
- Qwen2.5-VL as the text encoder
- Custom transformer for image generation
- FlowMatch scheduler for improved generation quality

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{omnigen2,
  title={OmniGen2: Unified Image Generation},
  author={Your Name and Co-authors},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.