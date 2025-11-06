"""
Download sample images from PIE-Bench dataset
"""

from datasets import load_dataset
from PIL import Image
import os
import json

def download_piebench_samples(num_samples=5, output_dir="samples"):
    """
    Download sample images from PIE-Bench++ dataset

    Args:
        num_samples: Number of samples to download (default: 5)
        output_dir: Directory to save samples
    """
    print("Loading PIE-Bench++ dataset from HuggingFace...")

    try:
        # Load dataset from HuggingFace
        dataset = load_dataset("UB-CVML-Group/PIE_Bench_pp")

        # Get the appropriate split (usually 'test' or 'train')
        if 'test' in dataset:
            data = dataset['test']
        elif 'train' in dataset:
            data = dataset['train']
        else:
            data = dataset[list(dataset.keys())[0]]

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Create subdirectories for each editing type
        editing_types = [
            "random_editing", "change_object", "add_object", "delete_object",
            "change_content", "change_pose", "change_color", "change_material",
            "change_background", "change_style"
        ]

        for edit_type in editing_types:
            os.makedirs(os.path.join(output_dir, edit_type), exist_ok=True)

        print(f"Downloading up to {num_samples} samples...")

        downloaded = 0
        for i in range(min(num_samples, len(data))):
            sample = data[i]

            # Determine editing type (if available in dataset)
            edit_type = sample.get("edit_type", "unknown")
            if isinstance(edit_type, int) and 0 <= edit_type < len(editing_types):
                edit_dir = editing_types[edit_type]
            else:
                edit_dir = "unknown"

            sample_dir = os.path.join(output_dir, edit_dir)

            # Save source image
            if "source_image" in sample or "input_image" in sample:
                source_key = "source_image" if "source_image" in sample else "input_image"
                source_img = sample[source_key]
                if isinstance(source_img, Image.Image):
                    source_img.save(os.path.join(sample_dir, f"sample_{i:03d}_source.png"))

            # Save target/edited image
            if "target_image" in sample or "edited_image" in sample:
                target_key = "target_image" if "target_image" in sample else "edited_image"
                target_img = sample[target_key]
                if isinstance(target_img, Image.Image):
                    target_img.save(os.path.join(sample_dir, f"sample_{i:03d}_target.png"))

            # Save editing mask if available
            if "mask" in sample or "editing_mask" in sample:
                mask_key = "mask" if "mask" in sample else "editing_mask"
                mask_img = sample[mask_key]
                if isinstance(mask_img, Image.Image):
                    mask_img.save(os.path.join(sample_dir, f"sample_{i:03d}_mask.png"))

            # Save metadata
            metadata = {
                "sample_id": i,
                "editing_type": edit_type,
                "source_prompt": sample.get("source_prompt", ""),
                "target_prompt": sample.get("target_prompt", ""),
                "editing_instruction": sample.get("instruction", sample.get("edit_instruction", "")),
                "main_editing_body": sample.get("main_editing_body", "")
            }

            with open(os.path.join(sample_dir, f"sample_{i:03d}_metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            downloaded += 1
            print(f"  Sample {downloaded}/{num_samples} saved to {edit_dir}/")

        print(f"\nDone! {downloaded} samples saved to '{output_dir}/' directory")
        print(f"\nSamples are organized by editing type:")
        for edit_type in editing_types:
            type_dir = os.path.join(output_dir, edit_type)
            if os.path.exists(type_dir) and os.listdir(type_dir):
                count = len([f for f in os.listdir(type_dir) if f.endswith('_source.png')])
                if count > 0:
                    print(f"  - {edit_type}: {count} samples")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nAlternative: Manual download instructions")
        print_manual_instructions()

def print_manual_instructions():
    """Print manual download instructions"""
    print("\nManual Download Instructions:")
    print("=" * 60)
    print()
    print("1. Visit HuggingFace:")
    print("   https://huggingface.co/datasets/UB-CVML-Group/PIE_Bench_pp")
    print()
    print("2. You can download the dataset using:")
    print("   - HuggingFace datasets library (pip install datasets)")
    print("   - Direct download from the Files tab")
    print("   - Git LFS clone")
    print()
    print("3. Dataset includes 10 editing types:")
    editing_types = [
        "0. Random editing",
        "1. Change object",
        "2. Add object",
        "3. Delete object",
        "4. Change object content",
        "5. Change object pose",
        "6. Change object color",
        "7. Change object material",
        "8. Change background",
        "9. Change image style"
    ]
    for et in editing_types:
        print(f"   {et}")

if __name__ == "__main__":
    import sys

    # Parse command line arguments
    num_samples = 5
    if len(sys.argv) > 1:
        try:
            num_samples = int(sys.argv[1])
        except ValueError:
            print("Usage: python download_samples.py [num_samples]")
            print("Example: python download_samples.py 10")
            sys.exit(1)

    # Download samples
    download_piebench_samples(num_samples=num_samples)
