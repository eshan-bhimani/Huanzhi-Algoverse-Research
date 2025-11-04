# PIE-Bench Dataset

## Overview
PIE-Bench (Prompt-based Image Editing Benchmark) is a comprehensive benchmark for evaluating text-driven image editing methods, particularly those based on diffusion models. An enhanced version **PIE-Bench++** is also available.

## Dataset Details

### Size
- **700 images** with diverse editing scenarios
- **10 distinct editing types**
- Evenly distributed across natural and artificial scenes

### Scene Categories
Images are distributed across four categories:
- Animal
- Human
- Indoor
- Outdoor

Each category includes both natural photos and artificial scenes (e.g., paintings).

### Editing Types (10 Categories)
0. Random editing
1. Change object
2. Add object
3. Delete object
4. Change object content
5. Change object pose
6. Change object color
7. Change object material
8. Change background
9. Change image style

### Annotations
Each image includes **five annotations**:
1. Source image prompt
2. Target image prompt
3. Editing instruction
4. Main editing body
5. Editing mask (indicates the anticipated editing region)

### Key Features
- Systematic evaluation across multiple editing types
- Precise editing masks for accurate metrics
- Support for multi-aspect editing evaluation
- Balanced distribution across scene types
- Comprehensive annotation for fine-grained analysis

## PIE-Bench++ (Enhanced Version)

### Improvements
- Builds upon original PIE-Bench
- Enhanced multi-aspect editing capabilities
- More comprehensive evaluation framework
- Evaluates simultaneous multi-aspect edits

## URLs and Resources

### Official Links
- **Papers with Code**: https://paperswithcode.com/dataset/pie-bench
- **HuggingFace (PIE-Bench++)**: https://huggingface.co/datasets/UB-CVML-Group/PIE_Bench_pp
- **Leaderboard**: https://paperswithcode.com/sota/text-based-image-editing-on-pie-bench

### Related Papers
- Direct Inversion (ICLR 2024): https://arxiv.org/abs/2310.01506
- ParallelEdits: https://arxiv.org/abs/2406.00985

### Citation
```bibtex
@article{ju2024piebench,
  title={PIE-Bench: A Comprehensive Benchmark for Prompt-based Image Editing},
  author={Ju, Chenyang and others},
  journal={arXiv preprint},
  year={2024}
}
```

## How to Use

### Download via HuggingFace
```python
from datasets import load_dataset

# Load PIE-Bench++
dataset = load_dataset("UB-CVML-Group/PIE_Bench_pp")
```

### Access Sample Images
```bash
python download_samples.py
```

## Use Cases
- **Benchmarking**: Evaluate text-driven image editing models
- **Multi-aspect Editing**: Test models on simultaneous multiple edits
- **Fine-grained Analysis**: Analyze performance across 10 editing types
- **Mask-based Evaluation**: Use precise editing masks for accurate metrics
- **Comparative Studies**: Compare different diffusion-based editing methods

## Evaluation Metrics
The dataset supports:
- Editing accuracy within masked regions
- Preservation of non-edited regions
- Multi-aspect editing capabilities
- Style transfer quality
- Object manipulation accuracy

## Notes
- Editing masks are crucial for accurate evaluation
- Covers both simple single-aspect and complex multi-aspect edits
- Balanced dataset across scene types and editing operations
- Particularly useful for diffusion-based editing methods
- Regular updates with PIE-Bench++ for enhanced capabilities

## Advantages
1. **Comprehensive Coverage**: 10 editing types cover most common operations
2. **Precise Evaluation**: Editing masks enable accurate metrics
3. **Balanced Dataset**: Equal distribution across scenes and styles
4. **Multi-aspect Support**: Evaluate complex simultaneous edits
5. **Community Standard**: Active leaderboard and benchmark results
