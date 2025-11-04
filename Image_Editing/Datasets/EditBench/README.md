# EditBench Dataset

## Overview
EditBench is a systematic benchmark for text-guided image inpainting introduced by **Google Research** and presented at **CVPR 2023**. It provides fine-grained evaluation of image editing models across various attributes, objects, and scenes.

## Dataset Details

### Size
- **240 images total**
  - 120 generated images (synthesized by Parti)
  - 120 natural images (from Visual Genome and Open Images)

### Structure
Each EditBench example consists of:
1. A masked input image
2. An input text prompt
3. A high-quality output image (reference for automatic metrics)

### Evaluation Categories

#### Three Evaluation Axes:
1. **Attributes**
   - Material
   - Color
   - Shape
   - Size
   - Count

2. **Objects**
   - Common objects
   - Rare objects
   - Text rendering

3. **Scenes**
   - Indoor scenes
   - Outdoor scenes
   - Realistic photos
   - Paintings

### Key Features
- Systematic, fine-grained evaluation framework
- Goes beyond coarse "image-text matching"
- Comprehensive coverage of editing scenarios
- High-quality reference images for evaluation
- Publicly released for research community

## URLs and Resources

### Official Links
- **Google Research Blog**: https://research.google/blog/imagen-editor-and-editbench-advancing-and-evaluating-text-guided-image-inpainting/
- **ArXiv Paper**: https://arxiv.org/abs/2212.06909
- **CVPR 2023 Paper**: https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Imagen_Editor_and_EditBench_Advancing_and_Evaluating_Text-Guided_Image_Inpainting_CVPR_2023_paper.pdf

### Citation
```bibtex
@inproceedings{wang2023imagen,
  title={Imagen Editor and EditBench: Advancing and Evaluating Text-Guided Image Inpainting},
  author={Wang, Su and Saharia, Chitwan and Montgomery, Ceslee and Pont-Tuset, Jordi and Noy, Shai and Pellegrini, Stefano and Onoe, Yasumasa and Laszlo, Sarah and Fleet, David J and Soricut, Radu and others},
  booktitle={CVPR},
  year={2023}
}
```

## Authors (Google Research Team)
- Su Wang
- Chitwan Saharia
- Ceslee Montgomery
- Jordi Pont-Tuset
- And others from Google Research

## How to Use

### Evaluation Focus
EditBench is primarily designed for **evaluation** rather than training. Use it to:
- Benchmark your image editing model
- Conduct fine-grained performance analysis
- Compare across different editing categories
- Assess handling of various attributes and objects

### Access Dataset
The dataset is publicly available through Google Research. Check the official blog post and paper for download instructions.

```bash
python download_samples.py
```

## Use Cases
- **Model Evaluation**: Systematic benchmarking of image editing models
- **Fine-grained Analysis**: Understanding model strengths/weaknesses
- **Research Comparison**: Standardized evaluation across papers
- **Quality Assessment**: Automatic metrics with reference images
- **Attribute Testing**: Specific evaluation of color, material, shape changes

## Related Work
- Introduced alongside **Imagen Editor** (text-guided inpainting model)
- Complements training datasets like InstructPix2Pix and MagicBrush
- Focuses on evaluation rather than training

## Notes
- Small dataset (240 images) - intended for evaluation, not training
- High-quality curated examples
- Provides reference outputs for automatic metrics
- Covers both generated and natural images for diversity
