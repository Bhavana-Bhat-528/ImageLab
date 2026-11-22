"""
Quantitative comparison metrics utilities.
Computes Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR),
and Structural Similarity Index (SSIM) using NumPy and scikit-image.

Strictly distinguishes relative metrics (Enhanced vs. Input) from
fidelity metrics against known clean ground-truth reference images.
"""

from typing import Dict, Any, Optional
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn


def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Mean Squared Error between two images."""
    if len(img1.shape) == 3 and len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    elif len(img1.shape) == 2 and len(img2.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

    if img1.shape[:2] != img2.shape[:2]:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in decibels."""
    mse = compute_mse(img1, img2)
    if mse <= 1e-10:
        return 99.99  # Represents numerical infinity for identical images
    max_pixel = 255.0
    return float(20.0 * np.log10(max_pixel / np.sqrt(mse)))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Structural Similarity Index (SSIM) in range [-1, 1]."""
    if len(img1.shape) == 3 and len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    elif len(img1.shape) == 2 and len(img2.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

    if img1.shape[:2] != img2.shape[:2]:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    if len(img1.shape) == 3 and img1.shape[2] in (3, 4):
        val = ssim_fn(img1, img2, channel_axis=2, data_range=255)
    else:
        val = ssim_fn(img1, img2, data_range=255)

    return float(val)



def evaluate_enhancement_metrics(
    input_img: np.ndarray,
    enhanced_img: np.ndarray,
    ground_truth_img: Optional[np.ndarray] = None,
    ground_truth_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Computes and structures all quantitative metrics.
    Clearly distinguishes:
      1. Transformation magnitude (Enhanced vs. Input):
         Does NOT indicate quality; high PSNR / low MSE simply means minimal change.
      2. True reconstruction fidelity (Enhanced vs. Clean Ground Truth):
         Real improvement measure when ground-truth reference is available.
    """
    rel_mse = round(compute_mse(input_img, enhanced_img), 2)
    rel_psnr = round(compute_psnr(input_img, enhanced_img), 2)
    rel_ssim = round(compute_ssim(input_img, enhanced_img), 4)

    metrics_result = {
        "relative_to_input": {
            "mse": rel_mse,
            "psnr": rel_psnr,
            "ssim": rel_ssim,
            "interpretation": (
                "Measures distance from degraded input. A low MSE or high PSNR here "
                "merely reflects conservative transformation, NOT restoration quality."
            )
        },
        "ground_truth_comparison": {
            "has_ground_truth": False,
            "reference_name": None,
            "input_vs_ground_truth": None,
            "enhanced_vs_ground_truth": None,
            "psnr_gain_db": None,
            "ssim_gain": None,
            "interpretation": "No clean ground-truth reference provided (common for user uploads)."
        }
    }

    if ground_truth_img is not None:
        # Match dimensions if needed
        if ground_truth_img.shape != input_img.shape:
            ground_truth_img = cv2.resize(
                ground_truth_img,
                (input_img.shape[1], input_img.shape[0])
            )

        # Baseline: Degraded Input vs Ground Truth
        base_mse = round(compute_mse(input_img, ground_truth_img), 2)
        base_psnr = round(compute_psnr(input_img, ground_truth_img), 2)
        base_ssim = round(compute_ssim(input_img, ground_truth_img), 4)

        # Restored: Enhanced vs Ground Truth
        gt_mse = round(compute_mse(enhanced_img, ground_truth_img), 2)
        gt_psnr = round(compute_psnr(enhanced_img, ground_truth_img), 2)
        gt_ssim = round(compute_ssim(enhanced_img, ground_truth_img), 4)

        psnr_gain = round(gt_psnr - base_psnr, 2)
        ssim_gain = round(gt_ssim - base_ssim, 4)

        metrics_result["ground_truth_comparison"] = {
            "has_ground_truth": True,
            "reference_name": ground_truth_name or "Pristine Benchmark",
            "input_vs_ground_truth": {
                "mse": base_mse,
                "psnr": base_psnr,
                "ssim": base_ssim
            },
            "enhanced_vs_ground_truth": {
                "mse": gt_mse,
                "psnr": gt_psnr,
                "ssim": gt_ssim
            },
            "psnr_gain_db": psnr_gain,
            "ssim_gain": ssim_gain,
            "interpretation": (
                f"Fidelity evaluated against clean reference. PSNR improvement: {psnr_gain:+} dB, "
                f"SSIM change: {ssim_gain:+}. Positive gains confirm authentic signal recovery."
            )
        }

    return metrics_result
