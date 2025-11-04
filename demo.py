"""
Demonstration of the image editing pipeline with evaluation
Shows how to use the API and evaluate results
"""

import cv2
import numpy as np
import requests
from evaluation import evaluate_grayscale_conversion, EvaluationResult
import json
from typing import Tuple
import os


def create_test_image(save_path: str = "test_image.jpg") -> str:
    """
    Create a simple test image with colors for grayscale testing
    
    Returns:
        Path to saved test image
    """
    # Create a 400x400 image with color gradients
    height, width = 400, 400
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Red gradient on left
    image[:, :width//3, 2] = np.linspace(0, 255, width//3, dtype=np.uint8)
    
    # Green gradient in middle
    image[:, width//3:2*width//3, 1] = np.linspace(0, 255, width//3, dtype=np.uint8)
    
    # Blue gradient on right
    image[:, 2*width//3:, 0] = np.linspace(0, 255, width//3, dtype=np.uint8)
    
    # Save image
    cv2.imwrite(save_path, image)
    print(f"✓ Created test image: {save_path}")
    return save_path


def test_api_with_upload(image_path: str, prompt: str, api_url: str = "http://localhost:8000") -> bytes:
    """
    Test the API with an uploaded image
    
    Args:
        image_path: Path to image file
        prompt: Editing instruction
        api_url: Base URL of the API
    
    Returns:
        Edited image bytes
    """
    url = f"{api_url}/edit"
    
    with open(image_path, 'rb') as f:
        files = {'image_file': f}
        data = {'prompt': prompt}
        
        response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        print(f"✓ API request successful (upload)")
        return response.content
    else:
        print(f"✗ API request failed: {response.status_code}")
        print(f"  Error: {response.text}")
        return None


def test_api_with_url(image_url: str, prompt: str, api_url: str = "http://localhost:8000") -> bytes:
    """
    Test the API with an image URL
    
    Args:
        image_url: URL to image
        prompt: Editing instruction
        api_url: Base URL of the API
    
    Returns:
        Edited image bytes
    """
    url = f"{api_url}/edit"
    data = {
        'prompt': prompt,
        'image_url': image_url
    }
    
    response = requests.post(url, data=data)
    
    if response.status_code == 200:
        print(f"✓ API request successful (URL)")
        return response.content
    else:
        print(f"✗ API request failed: {response.status_code}")
        print(f"  Error: {response.text}")
        return None


def run_evaluation(original_image_path: str, edited_image_bytes: bytes) -> EvaluationResult:
    """
    Evaluate the edited image against expected output
    
    Args:
        original_image_path: Path to original image
        edited_image_bytes: Edited image as bytes
    
    Returns:
        EvaluationResult with metrics
    """
    # Load original image
    original = cv2.imread(original_image_path)
    
    # Decode edited image
    edited_array = np.frombuffer(edited_image_bytes, dtype=np.uint8)
    edited = cv2.imdecode(edited_array, cv2.IMREAD_GRAYSCALE)
    
    # Run evaluation
    result = evaluate_grayscale_conversion(original, edited)
    
    return result


def print_evaluation_results(result: EvaluationResult):
    """Pretty print evaluation results"""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"MSE (Mean Squared Error):    {result.mse:.4f}")
    print(f"PSNR (Peak Signal-to-Noise): {result.psnr:.2f} dB")
    print(f"SSIM (Structural Similarity): {result.ssim:.4f}")
    print(f"Pixel Accuracy:              {result.pixel_accuracy:.2f}%")
    print("-"*60)
    
    if result.passed:
        print("✓ PASSED - Image meets quality thresholds")
    else:
        print("✗ FAILED - Image does not meet quality thresholds")
    print("="*60 + "\n")


def save_comparison_image(original_path: str, edited_bytes: bytes, output_path: str = "comparison.jpg"):
    """
    Save a side-by-side comparison of original and edited images
    
    Args:
        original_path: Path to original image
        edited_bytes: Edited image bytes
        output_path: Where to save comparison
    """
    # Load images
    original = cv2.imread(original_path)
    edited_array = np.frombuffer(edited_bytes, dtype=np.uint8)
    edited = cv2.imdecode(edited_array, cv2.IMREAD_GRAYSCALE)
    
    # Convert grayscale to 3-channel for concatenation
    edited_bgr = cv2.cvtColor(edited, cv2.COLOR_GRAY2BGR)
    
    # Concatenate horizontally
    comparison = np.hstack([original, edited_bgr])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Original", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "Grayscale", (original.shape[1] + 10, 30), font, 1, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, comparison)
    print(f"✓ Saved comparison image: {output_path}")


def check_api_health(api_url: str = "http://localhost:8000") -> bool:
    """Check if API is running"""
    try:
        response = requests.get(f"{api_url}/", timeout=2)
        if response.status_code == 200:
            print(f"✓ API is operational at {api_url}")
            return True
        else:
            print(f"✗ API returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Cannot connect to API at {api_url}")
        print(f"  Make sure the server is running: uvicorn main:app --reload")
        return False


def main():
    """Run the complete demonstration"""
    print("\n" + "="*60)
    print("IMAGE EDITING API DEMONSTRATION")
    print("="*60 + "\n")
    
    # Configuration
    API_URL = "http://localhost:8000"
    TEST_IMAGE_PATH = "test_image.jpg"
    PROMPT = "Convert this image to grayscale"
    
    # Step 1: Check API health
    print("Step 1: Checking API health...")
    if not check_api_health(API_URL):
        print("\n⚠ Please start the API server first:")
        print("   uvicorn main:app --reload")
        return
    print()
    
    # Step 2: Create test image
    print("Step 2: Creating test image...")
    create_test_image(TEST_IMAGE_PATH)
    print()
    
    # Step 3: Test with file upload
    print("Step 3: Testing API with file upload...")
    print(f"Prompt: '{PROMPT}'")
    edited_bytes = test_api_with_upload(TEST_IMAGE_PATH, PROMPT, API_URL)
    
    if edited_bytes is None:
        print("✗ Demo failed at API request step")
        return
    print()
    
    # Step 4: Run evaluation
    print("Step 4: Evaluating edited image...")
    result = run_evaluation(TEST_IMAGE_PATH, edited_bytes)
    print_evaluation_results(result)
    
    # Step 5: Save comparison
    print("Step 5: Saving comparison image...")
    save_comparison_image(TEST_IMAGE_PATH, edited_bytes)
    print()
    
    # Step 6: Test with URL (optional example)
    print("Step 6: Testing with public image URL...")
    PUBLIC_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    print(f"URL: {PUBLIC_IMAGE_URL}")
    url_edited_bytes = test_api_with_url(PUBLIC_IMAGE_URL, PROMPT, API_URL)
    
    if url_edited_bytes:
        # Save the URL result
        array = np.frombuffer(url_edited_bytes, dtype=np.uint8)
        img = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite("url_result.jpg", img)
        print("✓ Saved URL result: url_result.jpg")
    print()
    
    # Step 7: Check logs
    print("Step 7: Retrieving request logs...")
    logs_response = requests.get(f"{API_URL}/logs?limit=10")
    if logs_response.status_code == 200:
        logs_data = logs_response.json()
        print(f"✓ Retrieved {logs_data['count']} log entries")
        if logs_data['logs']:
            print("\nMost recent log entry:")
            print(json.dumps(logs_data['logs'][-1], indent=2))
    print()
    
    print("="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  - test_image.jpg (original test image)")
    print("  - comparison.jpg (side-by-side comparison)")
    print("  - url_result.jpg (result from URL test)")
    print("  - logs/ (request logs directory)")
    print("\nNext steps:")
    print("  1. Review the comparison.jpg to see results")
    print("  2. Check logs/ directory for request history")
    print("  3. Expand main.py with more CV2 operations")
    print("  4. Add more test cases to evaluation.py")


if __name__ == "__main__":
    main()
