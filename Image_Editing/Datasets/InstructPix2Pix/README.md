# InstructPix2Pix Dataset

## Overview
InstructPix2Pix is a large-scale dataset for learning to follow image editing instructions, presented at **CVPR 2023**. It uses GPT-3 and Stable Diffusion to generate a massive dataset of image editing examples.

## Dataset Details

### Size
- **450,000+ training examples**
- Generated using GPT-3 (for text) and Stable Diffusion (for images)
- Paired training data: (input image, editing instruction, output image)

### Key Features
- Large-scale automatically generated dataset
- Covers wide variety of editing operations
- Enables training of conditional diffusion models
- Supports instruction-based editing without requiring masks
- Foundation for many subsequent image editing works

### Method Overview
The dataset was created by:
1. Using GPT-3 to generate editing instructions and paired captions
2. Using Prompt-to-Prompt with Stable Diffusion to generate edited images
3. Filtering and pairing to create training triplets

## URLs and Resources

### Official Links
- **Project Website**: https://www.timothybrooks.com/instruct-pix2pix
- **GitHub Repository**: https://github.com/timothybrooks/instruct-pix2pix
- **ArXiv Paper**: https://arxiv.org/abs/2211.09800
- **Papers with Code**: https://paperswithcode.com/dataset/instructpix2pix-image-editing-dataset
- **Replicate Demo**: https://replicate.com/timothybrooks/instruct-pix2pix

### Citation
```bibtex
@inproceedings{brooks2023instructpix2pix,
  title={InstructPix2Pix: Learning to Follow Image Editing Instructions},
  author={Brooks, Tim and Holynski, Aleksander and Efros, Alexei A},
  booktitle={CVPR},
  year={2023}
}
```

## Authors
- Tim Brooks (UC Berkeley)
- Aleksander Holynski (UC Berkeley)
- Alexei A. Efros (UC Berkeley)

## How to Use

### Model Usage
```python
# Using the pretrained model
from diffusers import StableDiffusionInstructPix2PixPipeline
import torch

model_id = "timothybrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to("cuda")

# Edit an image
images = pipe(
    "turn the sky purple",
    image=input_image,
    num_inference_steps=20,
    image_guidance_scale=1.5
).images
```

### Download Dataset
Visit the official GitHub repository for dataset download instructions.
```bash
python download_samples.py
```

## Use Cases
- Large-scale training for instruction-based editing
- Baseline for image editing research
- Pre-training for downstream editing tasks
- Studying generalization of editing models
- Developing new editing architectures

## Notes
- Dataset is synthetically generated, which may introduce some noise
- Consider using MagicBrush or other manually-annotated datasets for evaluation
- Best used for pre-training or large-scale learning scenarios
