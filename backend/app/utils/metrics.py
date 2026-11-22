"""
Quantitative comparison metrics utilities.
Provides stubs and helper calculations for MSE, PSNR, SSIM, and morphological measures.
"""

import numpy as np


def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Mean Squared Error between two images."""
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions for MSE calculation.")
    return float(np.mean((img1.astype("float") - img2.astype("float")) ** 2))


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in decibels."""
    mse = compute_mse(img1, img2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    return float(20 * np.log10(max_pixel / np.sqrt(mse)))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Placeholder/stub for Structural Similarity Index (SSIM).
    Will be fully populated in the algorithmic implementation phase.
    """
    # Placeholder return
    return 1.0 if np.array_equal(img1, img2) else 0.0
