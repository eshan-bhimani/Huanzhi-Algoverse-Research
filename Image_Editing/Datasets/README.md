# Image Editing Datasets

This directory contains information and download scripts for high-quality image editing datasets suitable for research and experimentation in computer vision and image editing.

## Available Datasets

### 1. MagicBrush
**Type**: Training & Evaluation | **Size**: 10,000+ samples | **Year**: 2023 (NeurIPS)

- First large-scale, manually-annotated instruction-guided image editing dataset
- Supports single-turn and multi-turn editing
- High-quality human annotations
- Best for: Training instruction-based editing models

[View Details](./MagicBrush/README.md)

### 2. InstructPix2Pix
**Type**: Training | **Size**: 450,000+ samples | **Year**: 2023 (CVPR)

- Largest instruction-based editing dataset
- Automatically generated using GPT-3 and Stable Diffusion
- Covers wide variety of editing operations
- Best for: Large-scale pre-training, baseline comparisons

[View Details](./InstructPix2Pix/README.md)

### 3. EditBench
**Type**: Evaluation | **Size**: 240 samples | **Year**: 2023 (CVPR)

- Systematic benchmark from Google Research
- Fine-grained evaluation across attributes, objects, and scenes
- High-quality curated examples
- Best for: Model evaluation and benchmarking

[View Details](./EditBench/README.md)

### 4. PIE-Bench
**Type**: Evaluation | **Size**: 700 samples | **Year**: 2024

- Comprehensive benchmark with 10 editing types
- Includes precise editing masks
- Supports multi-aspect editing evaluation
- Best for: Comprehensive model testing

[View Details](./PIE-Bench/README.md)

## Quick Start

### Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

### Download Samples

Each dataset folder contains a `download_samples.py` script:

```bash
# MagicBrush
cd MagicBrush
python download_samples.py

# PIE-Bench
cd ../PIE-Bench
python download_samples.py

# InstructPix2Pix (requires GPU for generation)
cd ../InstructPix2Pix
python download_samples.py

# EditBench (provides download instructions)
cd ../EditBench
python download_samples.py
```

## Dataset Comparison

| Dataset | Size | Type | Annotation | Best For |
|---------|------|------|------------|----------|
| **MagicBrush** | 10K | Train/Eval | Manual | High-quality training |
| **InstructPix2Pix** | 450K | Train | Automatic | Large-scale pre-training |
| **EditBench** | 240 | Eval | Manual | Fine-grained evaluation |
| **PIE-Bench** | 700 | Eval | Manual | Comprehensive testing |

## Recommended Usage

### For Training
1. **Pre-training**: Start with InstructPix2Pix (450K samples)
2. **Fine-tuning**: Use MagicBrush (10K high-quality samples)

### For Evaluation
1. **Comprehensive Testing**: PIE-Bench (10 editing types)
2. **Fine-grained Analysis**: EditBench (attributes, objects, scenes)
3. **Instruction Following**: MagicBrush dev set

### For Research Papers
Use all datasets for comprehensive evaluation:
- Train on InstructPix2Pix + MagicBrush
- Evaluate on EditBench + PIE-Bench + MagicBrush test set

## Dataset Statistics

### Editing Types Coverage
- **Object Manipulation**: All datasets
- **Attribute Changes**: EditBench, PIE-Bench
- **Style Transfer**: PIE-Bench, MagicBrush
- **Multi-turn Editing**: MagicBrush
- **Mask-based Editing**: EditBench, PIE-Bench

### Annotation Quality
```
MagicBrush:      ████████████ (Manual, highest quality)
EditBench:       ███████████  (Curated, high quality)
PIE-Bench:       ███████████  (Manual annotations)
InstructPix2Pix: ██████       (Automatic, large scale)
```

## Citation

If you use these datasets in your research, please cite the respective papers:

**MagicBrush:**
```bibtex
@inproceedings{zhang2023magicbrush,
  title={MagicBrush: A Manually Annotated Dataset for Instruction-Guided Image Editing},
  author={Zhang, Kai and Mo, Lingbo and Chen, Wenhu and Sun, Huan and Su, Yu},
  booktitle={NeurIPS},
  year={2023}
}
```

**InstructPix2Pix:**
```bibtex
@inproceedings{brooks2023instructpix2pix,
  title={InstructPix2Pix: Learning to Follow Image Editing Instructions},
  author={Brooks, Tim and Holynski, Aleksander and Efros, Alexei A},
  booktitle={CVPR},
  year={2023}
}
```

**EditBench:**
```bibtex
@inproceedings{wang2023imagen,
  title={Imagen Editor and EditBench: Advancing and Evaluating Text-Guided Image Inpainting},
  author={Wang, Su and Saharia, Chitwan and others},
  booktitle={CVPR},
  year={2023}
}
```

**PIE-Bench:**
```bibtex
@article{ju2024piebench,
  title={PIE-Bench: A Comprehensive Benchmark for Prompt-based Image Editing},
  author={Ju, Chenyang and others},
  year={2024}
}
```

## Additional Resources

### Official Links
- MagicBrush: https://osu-nlp-group.github.io/MagicBrush/
- InstructPix2Pix: https://www.timothybrooks.com/instruct-pix2pix
- EditBench: https://research.google/blog/imagen-editor-and-editbench-advancing-and-evaluating-text-guided-image-inpainting/
- PIE-Bench: https://paperswithcode.com/dataset/pie-bench

### Community Resources
- Papers with Code: Track latest benchmarks and leaderboards
- HuggingFace: Access datasets and models
- GitHub: View implementation code and examples

## Contributing

To add more datasets or improve documentation:
1. Create a new folder for the dataset
2. Add a comprehensive README.md
3. Include a download_samples.py script
4. Update this main README.md

## License

Each dataset has its own license. Please check individual dataset README files for licensing information.

## Support

For issues or questions:
- Check individual dataset documentation
- Visit official dataset repositories
- Consult papers for detailed methodology

---

**Last Updated**: November 2024
