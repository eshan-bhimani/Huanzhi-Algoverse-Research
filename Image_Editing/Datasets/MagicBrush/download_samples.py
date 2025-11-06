"""
Download sample images from MagicBrush dataset
"""

from datasets import load_dataset
from PIL import Image
import os

def download_magicbrush_samples(num_samples=5, output_dir="samples"):
    """
    Download sample images from MagicBrush dataset

    Args:
        num_samples: Number of samples to download
        output_dir: Directory to save samples
    """
    print("Loading MagicBrush dataset from HuggingFace...")

    # Load dataset
    dataset = load_dataset("osunlp/MagicBrush", split="train")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading {num_samples} samples...")

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]

        # Save source image
        if "source_img" in sample:
            source_img = sample["source_img"]
            source_img.save(os.path.join(output_dir, f"sample_{i:03d}_source.png"))

        # Save target image
        if "target_img" in sample:
            target_img = sample["target_img"]
            target_img.save(os.path.join(output_dir, f"sample_{i:03d}_target.png"))

        # Save instruction as text file
        if "instruction" in sample:
            with open(os.path.join(output_dir, f"sample_{i:03d}_instruction.txt"), "w") as f:
                f.write(sample["instruction"])

        print(f"  Sample {i+1}/{num_samples} saved")

    print(f"\nDone! Samples saved to '{output_dir}/' directory")
    print(f"Each sample includes:")
    print("  - source image (before editing)")
    print("  - target image (after editing)")
    print("  - instruction text file")

if __name__ == "__main__":
    # Download 5 samples by default
    download_magicbrush_samples(num_samples=5)

    # You can also specify more samples:
    # download_magicbrush_samples(num_samples=10, output_dir="samples")
