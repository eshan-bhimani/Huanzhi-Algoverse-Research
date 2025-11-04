"""
Evaluation module for image editing pipeline
Provides metrics for comparing edited images to ground truth
"""

import cv2
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Results from image evaluation"""
    mse: float
    psnr: float
    ssim: float
    pixel_accuracy: float
    passed: bool
    
    def to_dict(self) -> Dict:
        return {
            "mse": self.mse,
            "psnr": self.psnr,
            "ssim": self.ssim,
            "pixel_accuracy": self.pixel_accuracy,
            "passed": self.passed
        }


def calculate_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Mean Squared Error between two images
    
    Args:
        img1: First image
        img2: Second image
    
    Returns:
        MSE value (lower is better, 0 is perfect)
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes don't match: {img1.shape} vs {img2.shape}")
    
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    return float(mse)


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio
    
    Args:
        img1: First image
        img2: Second image
    
    Returns:
        PSNR in dB (higher is better, inf is perfect)
    """
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return float(psnr)


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Structural Similarity Index (simplified version)
    
    Args:
        img1: First image
        img2: Second image
    
    Returns:
        SSIM value between -1 and 1 (1 is perfect match)
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes don't match: {img1.shape} vs {img2.shape}")
    
    # Constants for stability
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Calculate means
    mu1 = img1.mean()
    mu2 = img2.mean()
    
    # Calculate variances and covariance
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    
    # SSIM formula
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim = numerator / denominator
    return float(ssim)


def calculate_pixel_accuracy(img1: np.ndarray, img2: np.ndarray, threshold: int = 0) -> float:
    """
    Calculate percentage of pixels that match exactly (or within threshold)
    
    Args:
        img1: First image
        img2: Second image
        threshold: Acceptable difference per pixel (default 0 for exact match)
    
    Returns:
        Accuracy as percentage (0-100)
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes don't match: {img1.shape} vs {img2.shape}")
    
    diff = np.abs(img1.astype(int) - img2.astype(int))
    matching_pixels = np.sum(diff <= threshold)
    total_pixels = img1.size
    
    accuracy = (matching_pixels / total_pixels) * 100
    return float(accuracy)


def evaluate_image(
    edited_image: np.ndarray, 
    ground_truth: np.ndarray,
    mse_threshold: float = 1.0,
    psnr_threshold: float = 40.0,
    ssim_threshold: float = 0.95,
    pixel_accuracy_threshold: float = 99.0
) -> EvaluationResult:
    """
    Comprehensive evaluation of edited image against ground truth
    
    Args:
        edited_image: The image produced by the editing pipeline
        ground_truth: The expected result
        mse_threshold: Maximum acceptable MSE (default 1.0)
        psnr_threshold: Minimum acceptable PSNR in dB (default 40.0)
        ssim_threshold: Minimum acceptable SSIM (default 0.95)
        pixel_accuracy_threshold: Minimum acceptable pixel accuracy % (default 99.0)
    
    Returns:
        EvaluationResult with all metrics and pass/fail status
    """
    # Calculate all metrics
    mse = calculate_mse(edited_image, ground_truth)
    psnr = calculate_psnr(edited_image, ground_truth)
    ssim = calculate_ssim(edited_image, ground_truth)
    pixel_accuracy = calculate_pixel_accuracy(edited_image, ground_truth)
    
    # Determine if all thresholds are met
    passed = (
        mse <= mse_threshold and
        psnr >= psnr_threshold and
        ssim >= ssim_threshold and
        pixel_accuracy >= pixel_accuracy_threshold
    )
    
    return EvaluationResult(
        mse=mse,
        psnr=psnr,
        ssim=ssim,
        pixel_accuracy=pixel_accuracy,
        passed=passed
    )


def evaluate_grayscale_conversion(
    original_image: np.ndarray,
    edited_image: np.ndarray
) -> EvaluationResult:
    """
    Specialized evaluation for grayscale conversion
    Compares against OpenCV's standard grayscale conversion
    
    Args:
        original_image: Original color image
        edited_image: Grayscale converted image
    
    Returns:
        EvaluationResult comparing to expected grayscale conversion
    """
    # Generate ground truth using cv2.cvtColor
    if len(original_image.shape) == 3:
        ground_truth = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    else:
        ground_truth = original_image
    
    # For grayscale, we expect very high accuracy
    return evaluate_image(
        edited_image,
        ground_truth,
        mse_threshold=0.1,
        psnr_threshold=50.0,
        ssim_threshold=0.99,
        pixel_accuracy_threshold=99.9
    )
