# MagicBrush Dataset

## Overview
MagicBrush is the first large-scale, manually-annotated instruction-guided image editing dataset presented at **NeurIPS 2023**. It covers diverse scenarios including single-turn, multi-turn, mask-provided, and mask-free editing.

## Dataset Details

### Size
- **10,000+ edit triples** (source image, instruction, target image)
- Organized across **5,000+ edit sessions**
- More than **10,000 edit turns**

### Data Distribution
- **Train**: 8,807 edit turns (4,512 edit sessions)
- **Dev**: 528 edit turns (266 edit sessions)

### Key Features
- Manually annotated for high quality
- Supports both single-turn and multi-turn editing
- Includes mask-provided and mask-free editing scenarios
- Demonstrates significant improvements when used to fine-tune InstructPix2Pix
- Human evaluation scores: 4.1/5.0 for consistency, 3.9/5.0 for image quality

## URLs and Resources

### Official Links
- **Project Website**: https://osu-nlp-group.github.io/MagicBrush/
- **GitHub Repository**: https://github.com/OSU-NLP-Group/MagicBrush
- **HuggingFace Dataset**: https://huggingface.co/datasets/osunlp/MagicBrush
- **ArXiv Paper**: https://arxiv.org/abs/2306.10012

### Citation
```bibtex
@inproceedings{zhang2023magicbrush,
  title={MagicBrush: A Manually Annotated Dataset for Instruction-Guided Image Editing},
  author={Zhang, Kai and Mo, Lingbo and Chen, Wenhu and Sun, Huan and Su, Yu},
  booktitle={NeurIPS},
  year={2023}
}
```

## Authors
- Kai Zhang
- Lingbo Mo
- Wenhu Chen
- Huan Sun
- Yu Su

## How to Use

### Download via HuggingFace
```python
from datasets import load_dataset

dataset = load_dataset("osunlp/MagicBrush")
```

### Access Sample Images
Sample images can be found in the `samples/` subdirectory. To download more:
```bash
python download_samples.py
```

## Use Cases
- Instruction-based image editing research
- Multi-turn conversation-based editing
- Fine-tuning diffusion models
- Benchmarking image editing models
- Training vision-language models
