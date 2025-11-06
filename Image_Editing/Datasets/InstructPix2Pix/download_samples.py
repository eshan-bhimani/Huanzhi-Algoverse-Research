"""
Download sample images from InstructPix2Pix
This script demonstrates how to use the InstructPix2Pix model to generate edited images
"""

from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image
import torch
import os
import requests
from io import BytesIO

def download_sample_image(url, save_path):
    """Download a sample image from URL"""
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img.save(save_path)
    return img

def generate_edited_samples(output_dir="samples"):
    """
    Generate sample edited images using InstructPix2Pix model

    Args:
        output_dir: Directory to save samples
    """
    print("Loading InstructPix2Pix model...")

    # Load the model
    model_id = "timothybrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None
    )

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Sample editing instructions
    sample_instructions = [
        "turn the sky purple",
        "make it a sunset",
        "add snow on the ground",
        "make it autumn",
        "turn it into a watercolor painting",
    ]

    # You can use your own images or download sample images
    print("\nNote: Please provide input images in the 'input_images/' directory")
    print("or modify this script to download sample images.\n")

    # Example: If you have sample images
    input_dir = "input_images"
    if os.path.exists(input_dir):
        input_images = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        for idx, img_file in enumerate(input_images[:3]):  # Process first 3 images
            input_path = os.path.join(input_dir, img_file)
            input_image = Image.open(input_path).convert("RGB")

            for inst_idx, instruction in enumerate(sample_instructions[:2]):  # 2 edits per image
                print(f"Editing {img_file} with instruction: '{instruction}'")

                # Generate edited image
                edited_image = pipe(
                    instruction,
                    image=input_image,
                    num_inference_steps=20,
                    image_guidance_scale=1.5
                ).images[0]

                # Save results
                output_name = f"sample_{idx:02d}_{inst_idx:02d}"
                input_image.save(os.path.join(output_dir, f"{output_name}_input.png"))
                edited_image.save(os.path.join(output_dir, f"{output_name}_output.png"))

                # Save instruction
                with open(os.path.join(output_dir, f"{output_name}_instruction.txt"), "w") as f:
                    f.write(instruction)

                print(f"  Saved to {output_name}_*.png")

        print(f"\nDone! Samples saved to '{output_dir}/' directory")
    else:
        print(f"Please create '{input_dir}/' directory and add some images,")
        print("then run this script again to generate edited samples.")

def download_demo_images():
    """
    Alternative: Download the official demo images from the project
    Note: Update URLs with actual demo image URLs from the project
    """
    print("To download official demo images:")
    print("1. Visit: https://www.timothybrooks.com/instruct-pix2pix")
    print("2. Check the GitHub repo: https://github.com/timothybrooks/instruct-pix2pix")
    print("3. Look for example images in the repository")

if __name__ == "__main__":
    print("InstructPix2Pix Sample Generator")
    print("=" * 50)
    print("\nOptions:")
    print("1. Generate edited samples (requires GPU and input images)")
    print("2. Instructions for downloading demo images")

    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == "1":
        generate_edited_samples()
    else:
        download_demo_images()
