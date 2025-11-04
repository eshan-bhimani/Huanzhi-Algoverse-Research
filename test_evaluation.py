"""
Unit tests for evaluation module
"""

import numpy as np
import cv2
from evaluation import (
    calculate_mse,
    calculate_psnr,
    calculate_ssim,
    calculate_pixel_accuracy,
    evaluate_grayscale_conversion
)


def test_identical_images():
    """Test that identical images score perfectly"""
    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    mse = calculate_mse(img, img)
    psnr = calculate_psnr(img, img)
    ssim = calculate_ssim(img, img)
    pixel_acc = calculate_pixel_accuracy(img, img)
    
    assert mse == 0.0, "MSE should be 0 for identical images"
    assert psnr == float('inf'), "PSNR should be inf for identical images"
    assert ssim == 1.0, "SSIM should be 1.0 for identical images"
    assert pixel_acc == 100.0, "Pixel accuracy should be 100% for identical images"
    print("✓ Identical images test passed")


def test_different_images():
    """Test that different images have non-zero scores"""
    img1 = np.zeros((100, 100), dtype=np.uint8)
    img2 = np.ones((100, 100), dtype=np.uint8) * 255
    
    mse = calculate_mse(img1, img2)
    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)
    pixel_acc = calculate_pixel_accuracy(img1, img2)
    
    assert mse > 0, "MSE should be > 0 for different images"
    assert psnr < float('inf'), "PSNR should be finite for different images"
    assert ssim < 1.0, "SSIM should be < 1.0 for different images"
    assert pixel_acc < 100.0, "Pixel accuracy should be < 100% for different images"
    print("✓ Different images test passed")


def test_grayscale_conversion():
    """Test grayscale conversion evaluation"""
    # Create a color image
    color_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Convert using cv2
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    
    # Evaluate
    result = evaluate_grayscale_conversion(color_img, gray_img)
    
    assert result.mse < 0.1, "Perfect grayscale conversion should have very low MSE"
    assert result.passed, "Perfect grayscale conversion should pass"
    print("✓ Grayscale conversion test passed")


def test_shape_mismatch():
    """Test that shape mismatch raises error"""
    img1 = np.zeros((100, 100), dtype=np.uint8)
    img2 = np.zeros((50, 50), dtype=np.uint8)
    
    try:
        calculate_mse(img1, img2)
        assert False, "Should have raised ValueError for shape mismatch"
    except ValueError as e:
        assert "shape" in str(e).lower(), "Error should mention shape mismatch"
        print("✓ Shape mismatch test passed")


def test_pixel_accuracy_with_threshold():
    """Test pixel accuracy with threshold"""
    img1 = np.zeros((100, 100), dtype=np.uint8)
    img2 = np.ones((100, 100), dtype=np.uint8)  # All pixels differ by 1
    
    # With threshold 0, should be 0% match
    acc_strict = calculate_pixel_accuracy(img1, img2, threshold=0)
    assert acc_strict == 0.0, "Should be 0% with strict threshold"
    
    # With threshold 1, should be 100% match
    acc_loose = calculate_pixel_accuracy(img1, img2, threshold=1)
    assert acc_loose == 100.0, "Should be 100% with threshold >= difference"
    
    print("✓ Pixel accuracy threshold test passed")


if __name__ == "__main__":
    print("\nRunning evaluation module tests...\n")
    test_identical_images()
    test_different_images()
    test_grayscale_conversion()
    test_shape_mismatch()
    test_pixel_accuracy_with_threshold()
    print("\n✓ All tests passed!\n")
