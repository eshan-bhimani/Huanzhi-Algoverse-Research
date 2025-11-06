"""
Download sample images from EditBench dataset
Note: EditBench is released by Google Research
"""

import os
import json

def download_editbench_info():
    """
    Provide information on downloading EditBench dataset
    """
    print("EditBench Dataset Download Instructions")
    print("=" * 60)
    print()
    print("EditBench is a benchmark dataset released by Google Research.")
    print()
    print("To access the dataset:")
    print()
    print("1. Visit the official paper:")
    print("   https://arxiv.org/abs/2212.06909")
    print()
    print("2. Check the Google Research blog:")
    print("   https://research.google/blog/imagen-editor-and-editbench-advancing-and-evaluating-text-guided-image-inpainting/")
    print()
    print("3. The dataset is publicly available. Look for:")
    print("   - Download links in the paper")
    print("   - Supplementary material")
    print("   - Google Cloud Storage links")
    print()
    print("Dataset Structure:")
    print("  - 240 images total")
    print("  - Each example contains:")
    print("    * Masked input image")
    print("    * Text prompt")
    print("    * Reference output image")
    print()
    print("Evaluation Categories:")
    print("  - Attributes: material, color, shape, size, count")
    print("  - Objects: common, rare, text rendering")
    print("  - Scenes: indoor, outdoor, realistic, paintings")
    print()

    # Create a sample metadata structure
    metadata = {
        "dataset_name": "EditBench",
        "total_images": 240,
        "generated_images": 120,
        "natural_images": 120,
        "categories": {
            "attributes": ["material", "color", "shape", "size", "count"],
            "objects": ["common", "rare", "text_rendering"],
            "scenes": ["indoor", "outdoor", "realistic", "paintings"]
        },
        "source": "Google Research",
        "paper": "https://arxiv.org/abs/2212.06909",
        "year": 2023,
        "conference": "CVPR"
    }

    # Save metadata
    os.makedirs("metadata", exist_ok=True)
    with open("metadata/dataset_info.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Metadata saved to: metadata/dataset_info.json")
    print()
    print("Note: This is a curated evaluation benchmark.")
    print("      It's designed for testing rather than training.")

def create_sample_structure():
    """Create directory structure for organizing downloaded samples"""
    categories = ["attributes", "objects", "scenes"]

    for category in categories:
        os.makedirs(f"samples/{category}", exist_ok=True)

    print("\nCreated sample directory structure:")
    print("  samples/")
    print("    ├── attributes/")
    print("    ├── objects/")
    print("    └── scenes/")
    print()
    print("You can organize downloaded images into these directories")
    print("based on their evaluation category.")

if __name__ == "__main__":
    download_editbench_info()
    create_sample_structure()

    print("\n" + "=" * 60)
    print("Next Steps:")
    print("1. Follow the instructions above to access the dataset")
    print("2. Download the images and organize them in the 'samples/' directories")
    print("3. Use the metadata file for reference")
