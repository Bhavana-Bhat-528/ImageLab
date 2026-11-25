"""
Digital Image Processing - Chapter 3: Intensity Transformations and Spatial Filtering.
Authoritative implementation following Gonzalez and Woods (DIP 4th/3rd Edition).

Includes:
1. Intensity Transformations:
   - Image Negative: s = L - 1 - r
   - Log Transformation: s = c * log(1 + r)
   - Power-Law (Gamma) Transformation: s = c * r^gamma (gamma < 1 brightens, gamma > 1 darkens)
   - Piecewise-Linear Transformations:
     * Contrast Stretching: (r1, s1) -> (r2, s2)
     * Intensity-Level Slicing: [A, B] range with background preservation option
     * Bit-Plane Slicing: Bit planes 0 to 7 (8-bit grayscale)

2. Histogram Processing:
   - Histogram Equalization (Global)
   - Histogram Specification (Matching) with reference image
   - Local Histogram Processing (Neighborhood / Tile-based)
   - Standard 256-bin histogram computation

3. Smoothing Spatial Filters:
   - Linear Filters:
     * Averaging (Box) Filter: 3x3, 5x5, 7x7
     * Weighted Averaging Filter: Standard integer-weighted kernels
   - Order-Statistic (Nonlinear) Filters:
     * Median Filter: 3x3, 5x5, 7x7
     * Max Filter: 3x3, 5x5, 7x7 (Morphological Dilation)
     * Min Filter: 3x3, 5x5, 7x7 (Morphological Erosion)
     * Midpoint Filter: [Max + Min] / 2
     * Alpha-Trimmed Mean Filter: Trims 2*alpha extremes and averages the rest

4. Sharpening Spatial Filters:
   - Laplacian Sharpening: Standard 4-neighbor and Diagonal 8-neighbor masks
   - Unsharp Masking & Highboost Filtering: g = f + k * g_mask (k=1 unsharp, k>1 highboost)
   - Gradient-Based Sharpening: Roberts, Prewitt, and Sobel Operators (Magnitude & Sharpened)

5. Automatic Enhancement Pipeline:
   - Pure Gonzalez & Woods Chapter 3 adaptive pipeline
   - Denoise (Median / Weighted Average) -> Gamma Tone Mapping -> Local / Global Equalization -> Unsharp Masking
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from app.utils.metrics import evaluate_enhancement_metrics
from app.utils.image_io import encode_image_to_base64
from app.config import settings


# ---------------------------------------------------------------------------
# 1. Characteristic Analysis
# ---------------------------------------------------------------------------

def estimate_noise_level(gray: np.ndarray) -> Tuple[float, bool]:
    """
    Estimates additive Gaussian noise using Immerkaer's Laplacian-based estimator:
    sigma = (sqrt(pi/2) / (6 * (W-2) * (H-2))) * sum(|I * Mask|)
    Also checks for impulse (salt-and-pepper) noise via local median residual.
    """
    h, w = gray.shape
    if h < 5 or w < 5:
        return 0.0, False

    mask = np.array([[1, -2, 1],
                     [-2, 4, -2],
                     [1, -2, 1]], dtype=np.float32)
    filtered = cv2.filter2D(gray.astype(np.float32), -1, mask)
    sigma = float(np.sum(np.abs(filtered)) * np.sqrt(0.5 * np.pi) / (6.0 * (w - 2) * (h - 2)))

    med = cv2.medianBlur(gray, 3)
    diff = np.abs(gray.astype(np.int32) - med.astype(np.int32))
    extreme_pixels = ((gray <= 3) | (gray >= 252)) & (diff > 45)
    sp_ratio = float(np.mean(extreme_pixels))
    has_salt_pepper = sp_ratio > 0.010

    return round(sigma, 2), has_salt_pepper


def estimate_sharpness(gray: np.ndarray) -> Tuple[float, float]:
    """
    Estimates spatial image sharpness:
    1. Raw variance of the Laplacian (ksize=3): Var(Delta I)
    2. Dynamic-range normalized sharpness index: Var(Delta I) / (std^2 + 1e-4)
    """
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    var = float(lap.var())
    std_val = float(np.std(gray))
    norm_sharp = float(var / (std_val ** 2 + 1e-4))
    return round(var, 2), round(norm_sharp, 4)


def analyze_image_characteristics(image: np.ndarray) -> Dict[str, Any]:
    """
    Analyzes input image characteristics:
    - Mean luminance and standard deviation
    - Contrast and dynamic range
    - Histogram distribution, skewness, entropy
    - Shadow and highlight clipping
    - Noise level and impulse classification
    - Spatial sharpness and normalized edge acuity
    - Spatial illumination uniformity across a 4x4 grid
    """
    is_3channel = len(image.shape) == 3 and image.shape[2] == 3
    if is_3channel:
        ch_diff1 = np.max(np.abs(image[:, :, 0].astype(np.int32) - image[:, :, 1].astype(np.int32)))
        ch_diff2 = np.max(np.abs(image[:, :, 1].astype(np.int32) - image[:, :, 2].astype(np.int32)))
        is_color = bool(ch_diff1 > 3 or ch_diff2 > 3)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        is_color = False
        gray = image.copy()

    h, w = gray.shape
    total_pixels = h * w

    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    min_val = int(np.min(gray))
    max_val = int(np.max(gray))
    dynamic_range = max_val - min_val
    rms_contrast = round(std_val / 255.0, 3)

    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    prob_dist = hist / float(total_pixels)
    nonzero_probs = prob_dist[prob_dist > 0]
    entropy = float(-np.sum(nonzero_probs * np.log2(nonzero_probs)))

    shadow_clipping_pct = round(float(np.sum(hist[:16]) / total_pixels) * 100.0, 2)
    highlight_clipping_pct = round(float(np.sum(hist[240:]) / total_pixels) * 100.0, 2)

    if std_val > 1e-3:
        skewness = float(np.mean(((gray - mean_val) / std_val) ** 3))
    else:
        skewness = 0.0

    noise_sigma, has_salt_pepper = estimate_noise_level(gray)
    sharpness_score, norm_sharpness = estimate_sharpness(gray)

    blocks = []
    bh, bw = max(h // 4, 1), max(w // 4, 1)
    for r in range(4):
        for c in range(4):
            blocks.append(float(np.mean(gray[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw])))
    block_range = round(float(np.max(blocks) - np.min(blocks)), 2)

    is_dark = mean_val < 90.0 or (mean_val < 105.0 and shadow_clipping_pct > 12.0)
    is_bright = mean_val > 165.0 or (mean_val > 155.0 and highlight_clipping_pct > 12.0)
    is_low_contrast = std_val < 38.0 and dynamic_range < 130
    is_noisy = (noise_sigma > 13.0 and not is_low_contrast) or has_salt_pepper
    
    is_blurry = (norm_sharpness < 0.20 or sharpness_score < 600.0) and not is_noisy and not is_low_contrast
    is_severely_blurry = is_blurry and (norm_sharpness < 0.02 or sharpness_score < 100.0)

    is_uneven = (block_range > 75.0 and std_val > 45.0 and not is_dark and not is_bright and not is_noisy and not is_low_contrast)

    primary_defects = []
    if is_dark:
        primary_defects.append("Severe Underexposure / Shadow Compression")
    elif is_bright:
        primary_defects.append("Severe Overexposure / Saturated Highlights")
    elif is_uneven:
        primary_defects.append(f"Non-Uniform Spatial Illumination (Block Range={block_range})")
        
    if is_low_contrast:
        primary_defects.append(f"Narrow Dynamic Range / Low Contrast (std={std_val:.1f})")
    if has_salt_pepper:
        primary_defects.append("High-Impulse Salt & Pepper Noise")
    elif is_noisy:
        primary_defects.append(f"Significant Sensor Noise (Sigma={noise_sigma})")
    if is_severely_blurry:
        primary_defects.append(f"Severe Optical / Spatial Blur (Norm Sharpness={norm_sharpness})")
    elif is_blurry:
        primary_defects.append(f"Mild Spatial Blur / Low Edge Acutance (Laplacian Var={sharpness_score})")

    if not primary_defects:
        primary_defects.append("Well-Balanced Intensity & Clear Spatial Details")

    return {
        "mean_intensity": round(mean_val, 2),
        "std_intensity": round(std_val, 2),
        "min_intensity": min_val,
        "max_intensity": max_val,
        "dynamic_range": dynamic_range,
        "rms_contrast": rms_contrast,
        "histogram_entropy": round(entropy, 3),
        "histogram_skewness": round(skewness, 3),
        "shadow_clipping_pct": shadow_clipping_pct,
        "highlight_clipping_pct": highlight_clipping_pct,
        "noise_sigma": noise_sigma,
        "has_salt_pepper_noise": has_salt_pepper,
        "sharpness_score": sharpness_score,
        "norm_sharpness": norm_sharpness,
        "spatial_block_range": block_range,
        "is_color": is_color,
        "dimensions": {
            "width": w,
            "height": h,
            "channels": 3 if is_3channel else 1
        },
        "diagnostics": {
            "is_dark": is_dark,
            "is_bright": is_bright,
            "is_low_contrast": is_low_contrast,
            "is_noisy": is_noisy,
            "is_blurry": is_blurry,
            "is_severely_blurry": is_severely_blurry,
            "is_uneven": is_uneven,
            "primary_defects": primary_defects
        }
    }


# ---------------------------------------------------------------------------
# 2. Section 2: Intensity Transformations (Gonzalez & Woods Section 3.2)
# ---------------------------------------------------------------------------

def apply_image_negative(image: np.ndarray) -> np.ndarray:
    """
    Image Negative: s = L - 1 - r = 255 - r.
    Reverses pixel intensities. Highly effective for enhancing white or gray details
    embedded in large dark regions (e.g. digital mammograms / medical imagery).
    """
    return 255 - image


def apply_log_transform(image: np.ndarray, c: Optional[float] = None) -> np.ndarray:
    """
    Log Transformation: s = c * log(1 + r).
    Expands the values of dark pixels while compressing the higher-level values.
    Standard constant c = 255 / ln(1 + 255) ~ 45.986.
    """
    if c is None or c <= 0.0:
        c = 255.0 / np.log(1.0 + 255.0)

    is_color = len(image.shape) == 3 and image.shape[2] == 3
    if is_color:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_float = l.astype(np.float32)
        l_log = c * np.log(1.0 + l_float)
        l_out = np.clip(l_log, 0, 255).astype(np.uint8)
        return cv2.cvtColor(cv2.merge((l_out, a, b)), cv2.COLOR_LAB2BGR)
    else:
        img_float = image.astype(np.float32)
        res = c * np.log(1.0 + img_float)
        return np.clip(res, 0, 255).astype(np.uint8)


def apply_power_law(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Power-Law (Gamma) Transformation: s = c * (r / 255)^gamma * 255.
    - gamma < 1.0 brightens underexposed images (concave-down curve).
    - gamma > 1.0 darkens overexposed images (concave-up curve).
    - gamma == 1.0 leaves intensities unchanged (identity).
    """
    if abs(gamma - 1.0) < 1e-3:
        return image.copy()
    
    gamma_val = max(gamma, 0.05)
    table = np.array([
        ((i / 255.0) ** gamma_val) * 255.0
        for i in range(256)
    ], dtype=np.uint8)
    return cv2.LUT(image, table)


def apply_contrast_stretching(
    image: np.ndarray,
    r1: Optional[int] = None,
    s1: int = 0,
    r2: Optional[int] = None,
    s2: int = 255
) -> np.ndarray:
    """
    Piecewise-Linear Contrast Stretching:
    Expands the dynamic range of intensity levels across (r1, s1) and (r2, s2).
    Linear interpolation:
      r < r1: s = (s1 / r1) * r
      r1 <= r <= r2: s = s1 + ((s2 - s1) / (r2 - r1)) * (r - r1)
      r > r2: s = s2 + ((255 - s2) / (255 - r2)) * (r - r2)
    """
    is_color = len(image.shape) == 3 and image.shape[2] == 3
    ref_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if is_color else image

    if r1 is None:
        r1 = int(np.percentile(ref_gray, 2))
    if r2 is None:
        r2 = int(np.percentile(ref_gray, 98))

    r1 = max(0, min(254, r1))
    r2 = max(r1 + 1, min(255, r2))
    s1 = max(0, min(255, s1))
    s2 = max(0, min(255, s2))

    # Precompute 256-entry lookup table
    table = np.zeros(256, dtype=np.uint8)
    for r in range(256):
        if r < r1:
            val = (s1 / max(r1, 1)) * r
        elif r <= r2:
            val = s1 + ((s2 - s1) / max(r2 - r1, 1)) * (r - r1)
        else:
            val = s2 + ((255 - s2) / max(255 - r2, 1)) * (r - r2)
        table[r] = np.clip(round(val), 0, 255)

    if is_color:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_stretched = cv2.LUT(l, table)
        return cv2.cvtColor(cv2.merge((l_stretched, a, b)), cv2.COLOR_LAB2BGR)
    else:
        return cv2.LUT(image, table)


def apply_intensity_slicing(
    image: np.ndarray,
    low: int = 100,
    high: int = 180,
    preserve_background: bool = True,
    highlight_value: int = 255
) -> np.ndarray:
    """
    Intensity-Level Slicing (Gonzalez & Woods Fig 3.11):
    Highlights a specific band of intensity [low, high].
    - preserve_background = True: Enhances the range of interest to highlight_value
      while leaving all other intensity levels unchanged.
    - preserve_background = False: Sets the range of interest to highlight_value
      and darkens all other pixels to 0.
    """
    is_color = len(image.shape) == 3 and image.shape[2] == 3
    low = max(0, min(254, low))
    high = max(low, min(255, high))

    if is_color:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = (gray >= low) & (gray <= high)
        if preserve_background:
            out = image.copy()
            out[mask] = [highlight_value, highlight_value, highlight_value]
            return out
        else:
            out = np.zeros_like(image)
            out[mask] = [highlight_value, highlight_value, highlight_value]
            return out
    else:
        mask = (image >= low) & (image <= high)
        if preserve_background:
            out = image.copy()
            out[mask] = highlight_value
            return out
        else:
            out = np.zeros_like(image)
            out[mask] = highlight_value
            return out


def apply_bit_plane_slicing(image: np.ndarray, bit_plane: int = 7) -> np.ndarray:
    """
    Bit-Plane Slicing (Gonzalez & Woods Section 3.2.4):
    Deconstructs an 8-bit image into its 8 constituent binary planes (bits 0 to 7).
    - Bit 0 is the Least Significant Bit (LSB) containing fine random noise / microtexture.
    - Bit 7 is the Most Significant Bit (MSB) containing the dominant structural shape.
    Output is displayed as a binary image (0 or 255).
    """
    bit = max(0, min(7, int(bit_plane)))
    is_color = len(image.shape) == 3 and image.shape[2] == 3
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if is_color else image

    plane = ((gray >> bit) & 1) * 255
    plane_uint8 = plane.astype(np.uint8)

    if is_color:
        return cv2.cvtColor(plane_uint8, cv2.COLOR_GRAY2BGR)
    return plane_uint8


# ---------------------------------------------------------------------------
# 3. Section 3: Histogram Processing (Gonzalez & Woods Section 3.3)
# ---------------------------------------------------------------------------

def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Global Histogram Equalization:
    Transforms intensity distribution to yield a flat, uniformly distributed histogram.
    For color images: converts to LAB, equalizes luminance L channel, and converts back.
    """
    is_color = len(image.shape) == 3 and image.shape[2] == 3
    if is_color:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = cv2.equalizeHist(l)
        return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)
    else:
        return cv2.equalizeHist(image)


def apply_histogram_matching(image: np.ndarray, reference_image: np.ndarray) -> np.ndarray:
    """
    Histogram Specification / Matching (Gonzalez & Woods Section 3.3.2):
    Transforms the input image so that its intensity histogram closely matches
    the histogram of a specified reference image.
    Algorithm:
    1. Compute cumulative distribution functions CDF_in and CDF_ref.
    2. Build mapping LUT: for each intensity r in [0, 255], find z such that
       CDF_ref(z) is closest to CDF_in(r).
    3. Map input pixels through LUT.
    """
    is_color = len(image.shape) == 3 and image.shape[2] == 3
    gray_in = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if is_color else image

    is_ref_color = len(reference_image.shape) == 3 and reference_image.shape[2] == 3
    gray_ref = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY) if is_ref_color else reference_image

    # Compute CDFs
    hist_in, _ = np.histogram(gray_in, bins=256, range=(0, 256))
    cdf_in = hist_in.cumsum().astype(np.float64)
    if cdf_in[-1] > 0:
        cdf_in /= cdf_in[-1]

    hist_ref, _ = np.histogram(gray_ref, bins=256, range=(0, 256))
    cdf_ref = hist_ref.cumsum().astype(np.float64)
    if cdf_ref[-1] > 0:
        cdf_ref /= cdf_ref[-1]

    # Nearest neighbor matching in CDF space
    lut = np.zeros(256, dtype=np.uint8)
    for r in range(256):
        target = cdf_in[r]
        diffs = np.abs(cdf_ref - target)
        lut[r] = np.argmin(diffs)

    if is_color:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_matched = cv2.LUT(l, lut)
        return cv2.cvtColor(cv2.merge((l_matched, a, b)), cv2.COLOR_LAB2BGR)
    else:
        return cv2.LUT(image, lut)


def apply_local_histogram_processing(image: np.ndarray, window_size: int = 7) -> np.ndarray:
    """
    Local Histogram Processing (Gonzalez & Woods Section 3.3.3):
    Enhances local details in small neighborhoods without over-amplifying global noise.
    Implemented using CLAHE with a tile grid corresponding to the local neighborhood scale.
    """
    tile_size = max(4, min(32, int(window_size)))
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(tile_size, tile_size))

    is_color = len(image.shape) == 3 and image.shape[2] == 3
    if is_color:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_local = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l_local, a, b)), cv2.COLOR_LAB2BGR)
    else:
        return clahe.apply(image)


def compute_histogram_data(image: np.ndarray) -> Dict[str, Any]:
    """
    Computes 256-bin histogram distributions for plotting.
    Returns intensity bins and channel-specific data for color images.
    """
    is_color = len(image.shape) == 3 and image.shape[2] == 3
    bins = list(range(256))

    if is_color:
        b_hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        g_hist = cv2.calcHist([image], [1], None, [256], [0, 256]).flatten()
        r_hist = cv2.calcHist([image], [2], None, [256], [0, 256]).flatten()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        l_hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

        return {
            "bins": bins,
            "is_color": True,
            "luminance": [int(x) for x in l_hist],
            "red": [int(x) for x in r_hist],
            "green": [int(x) for x in g_hist],
            "blue": [int(x) for x in b_hist]
        }
    else:
        l_hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        return {
            "bins": bins,
            "is_color": False,
            "luminance": [int(x) for x in l_hist]
        }


# ---------------------------------------------------------------------------
# 4. Section 4: Smoothing Spatial Filters (Gonzalez & Woods Section 3.5)
# ---------------------------------------------------------------------------

def apply_box_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Averaging (Box) Filter:
    Linear filter replacing each pixel by the unweighted average of its neighborhood.
    Kernel size: 3x3, 5x5, 7x7.
    """
    k = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    return cv2.blur(image, (k, k))


def apply_weighted_average_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Weighted Averaging Filter (Gonzalez & Woods Section 3.5.1, Fig 3.32b):
    Linear filter giving higher weight to the center pixel and diminishing weights to outer pixels.
    - 3x3: 1/16 * [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
    - 5x5: 1/256 * [1, 4, 6, 4, 1]^T * [1, 4, 6, 4, 1]
    - 7x7: 1/4096 * [1, 6, 15, 20, 15, 6, 1]^T * [1, 6, 15, 20, 15, 6, 1]
    """
    k = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    if k == 3:
        w_1d = np.array([1, 2, 1], dtype=np.float32)
    elif k == 5:
        w_1d = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    else:
        w_1d = np.array([1, 6, 15, 20, 15, 6, 1], dtype=np.float32)

    kernel_2d = np.outer(w_1d, w_1d)
    kernel_2d /= np.sum(kernel_2d)

    return cv2.filter2D(image, -1, kernel_2d)


def apply_median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Order-Statistic Median Filter:
    Replaces pixel value with the median of gray levels in its neighborhood.
    Provides excellent noise reduction for impulse (salt-and-pepper) noise with minimal blurring.
    """
    k = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    return cv2.medianBlur(image, k)


def apply_max_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Order-Statistic Max Filter (100th percentile):
    Replaces center pixel with the maximum value in the neighborhood.
    Useful for finding the brightest points and eliminating negative pepper noise spikes.
    """
    k = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.dilate(image, kernel)


def apply_min_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Order-Statistic Min Filter (0th percentile):
    Replaces center pixel with the minimum value in the neighborhood.
    Useful for finding the darkest points and eliminating positive salt noise spikes.
    """
    k = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.erode(image, kernel)


def apply_midpoint_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Order-Statistic Midpoint Filter:
    Computes the midpoint between the maximum and minimum values in the neighborhood:
    R = [max(z) + min(z)] / 2.
    Combines order statistics with averaging; works well for randomly distributed noise
    (Gaussian or uniform).
    """
    k = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    max_f = apply_max_filter(image, k)
    min_f = apply_min_filter(image, k)
    midpoint = (max_f.astype(np.float32) + min_f.astype(np.float32)) / 2.0
    return np.clip(midpoint, 0, 255).astype(np.uint8)


def apply_alpha_trimmed_mean_filter(
    image: np.ndarray,
    kernel_size: int = 3,
    alpha: float = 0.2
) -> np.ndarray:
    """
    Alpha-Trimmed Mean Filter (Gonzalez & Woods Section 5.3.2):
    Order-statistic filter that deletes the d/2 lowest and d/2 highest values in the
    sorted neighborhood (where d = floor(2 * alpha * m * n)) and computes the mean of the rest.
    - alpha = 0: Reducible to Box Averaging filter.
    - alpha -> 0.5: Reducible to Median filter.
    Useful in situations containing multiple noise types (e.g., combination of salt-and-pepper and Gaussian noise).
    """
    k = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    pad = k // 2
    n_pix = k * k
    safe_alpha = max(0.0, min(0.49, float(alpha)))
    d = int(2.0 * safe_alpha * n_pix)
    d2 = d // 2

    is_3ch = len(image.shape) == 3 and image.shape[2] == 3

    def _trim_plane(plane: np.ndarray) -> np.ndarray:
        padded = np.pad(plane, pad, mode='reflect')
        windows = sliding_window_view(padded, (k, k))  # (H, W, k, k)
        flat = windows.reshape(windows.shape[0], windows.shape[1], -1)  # (H, W, n_pix)
        sorted_win = np.sort(flat, axis=-1)
        if d2 > 0:
            trimmed = sorted_win[..., d2:n_pix - d2]
        else:
            trimmed = sorted_win
        return np.mean(trimmed, axis=-1).astype(np.uint8)

    if is_3ch:
        chans = [_trim_plane(image[:, :, i]) for i in range(3)]
        return cv2.merge(chans)
    else:
        return _trim_plane(image)


# ---------------------------------------------------------------------------
# 5. Section 5: Sharpening Spatial Filters (Gonzalez & Woods Section 3.6)
# ---------------------------------------------------------------------------

def apply_laplacian_filter(
    image: np.ndarray,
    mask_type: str = "4_neighbor"
) -> np.ndarray:
    """
    Laplacian Sharpening (Gonzalez & Woods Section 3.6.2):
    Second-derivative spatial sharpening: g(x,y) = f(x,y) - laplacian(f)
    Mask choices:
    - '4_neighbor': Standard cross mask [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    - '8_neighbor': Diagonal omnidirectional mask [[1, 1, 1], [1, -8, 1], [1, 1, 1]]
    """
    if mask_type == "8_neighbor":
        mask = np.array([[1, 1, 1],
                         [1, -8, 1],
                         [1, 1, 1]], dtype=np.float32)
    else:
        mask = np.array([[0, 1, 0],
                         [1, -4, 1],
                         [0, 1, 0]], dtype=np.float32)

    is_color = len(image.shape) == 3 and image.shape[2] == 3
    if is_color:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_f = l.astype(np.float32)
        lap = cv2.filter2D(l_f, cv2.CV_32F, mask)
        # Center is negative (-4 or -8), so formula is g = f - lap
        sharpened_l = np.clip(l_f - lap, 0, 255).astype(np.uint8)
        return cv2.cvtColor(cv2.merge((sharpened_l, a, b)), cv2.COLOR_LAB2BGR)
    else:
        img_f = image.astype(np.float32)
        lap = cv2.filter2D(img_f, cv2.CV_32F, mask)
        sharpened = np.clip(img_f - lap, 0, 255).astype(np.uint8)
        return sharpened


def apply_unsharp_highboost_filter(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: Optional[float] = None,
    k_factor: float = 1.0
) -> np.ndarray:
    """
    Unsharp Masking and Highboost Filtering (Gonzalez & Woods Section 3.6.3):
    1. Blur original image: f_bar = GaussianBlur(f)
    2. Obtain mask: g_mask = f - f_bar
    3. Add weighted mask: g = f + k * g_mask
       - k = 1.0: Standard Unsharp Masking
       - k > 1.0: Highboost Filtering (factor A = 1 + k, or k >= 1.2)
    """
    k_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    if sigma is None or sigma <= 0.0:
        sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8

    blurred = cv2.GaussianBlur(image, (k_size, k_size), sigma)
    mask = image.astype(np.float32) - blurred.astype(np.float32)
    sharpened = image.astype(np.float32) + (float(k_factor) * mask)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_gradient_filter(
    image: np.ndarray,
    operator: str = "sobel",
    calc_method: str = "abs_sum",
    output_mode: str = "sharpened"
) -> np.ndarray:
    """
    Gradient-Based Sharpening & Edge Detection (Gonzalez & Woods Section 3.6.4):
    First-order derivative operators:
    - 'roberts': Roberts Cross-Gradient (2x2)
        Gx = [[-1, 0], [0, 1]], Gy = [[0, -1], [1, 0]]
    - 'prewitt': Prewitt Operator (3x3)
        Gx = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], Gy = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
    - 'sobel': Sobel Operator (3x3)
        Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    Magnitude Calculation:
    - 'abs_sum': M = |Gx| + |Gy|
    - 'euclidean': M = sqrt(Gx^2 + Gy^2)

    Output Mode:
    - 'magnitude': Displays the gradient edge magnitude map
    - 'sharpened': Adds the gradient magnitude to the original image: g = f + M
    """
    is_color = len(image.shape) == 3 and image.shape[2] == 3
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if is_color else image
    gray_f = gray.astype(np.float32)

    op = operator.lower()
    if op == "roberts":
        kx = np.array([[-1, 0], [0, 1]], dtype=np.float32)
        ky = np.array([[0, -1], [1, 0]], dtype=np.float32)
        gx = cv2.filter2D(gray_f, cv2.CV_32F, kx)
        gy = cv2.filter2D(gray_f, cv2.CV_32F, ky)
    elif op == "prewitt":
        kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        gx = cv2.filter2D(gray_f, cv2.CV_32F, kx)
        gy = cv2.filter2D(gray_f, cv2.CV_32F, ky)
    else:  # Sobel (default)
        gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)

    if calc_method == "euclidean":
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
    else:
        magnitude = np.abs(gx) + np.abs(gy)

    if output_mode == "magnitude":
        norm_mag = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if is_color:
            return cv2.cvtColor(norm_mag, cv2.COLOR_GRAY2BGR)
        return norm_mag
    else:  # 'sharpened'
        if is_color:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_sharp = np.clip(l.astype(np.float32) + 0.5 * magnitude, 0, 255).astype(np.uint8)
            return cv2.cvtColor(cv2.merge((l_sharp, a, b)), cv2.COLOR_LAB2BGR)
        else:
            sharpened = np.clip(gray_f + 0.5 * magnitude, 0, 255).astype(np.uint8)
            return sharpened


# ---------------------------------------------------------------------------
# 6. Section 8: Automatic Enhancement Decision Engine (Gonzalez & Woods Ch 3)
# ---------------------------------------------------------------------------

def decide_enhancement_pipeline(characteristics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decides appropriate transformations and parameters based on characteristics:
    - Step 1: Denoise first (Median for impulse noise, Weighted Averaging for Gaussian/sensor noise).
    - Step 2: Power-law gamma transformation (gamma < 1 expands shadows, gamma > 1 compresses highlights).
    - Step 3: Contrast equalization (Local Histogram Processing / CLAHE for narrow dynamic range).
    - Step 4: Spatial sharpening (Unsharp Masking with tuned factor k for blurry edges).
    """
    diag = characteristics["diagnostics"]
    mean_val = characteristics["mean_intensity"]
    noise_sigma = characteristics["noise_sigma"]
    has_sp = characteristics["has_salt_pepper_noise"]
    sharpness = characteristics["sharpness_score"]
    norm_sharp = characteristics.get("norm_sharpness", 1.0)

    selected_ops = []
    params = {
        "gamma": 1.0,
        "equalization_mode": "none",
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid_size": 8,
        "spatial_filter": "none",
        "kernel_size": 3,
        "sharpen_strength": 0.0
    }
    justifications = []
    expected_improvements = []
    potential_risks = []

    # -------------------------------------------------------------------------
    # Stage A: Spatial Smoothing (Must be applied FIRST to eliminate noise)
    # -------------------------------------------------------------------------
    if has_sp:
        params["spatial_filter"] = "median"
        params["kernel_size"] = 3
        selected_ops.append("Order-Statistic Median Filter (3x3)")
        justifications.append("Detected high-impulse salt-and-pepper noise; non-linear median filtering eliminates extreme outlier spikes without smearing genuine object edges (Gonzalez & Woods Section 3.5.2).")
        expected_improvements.append("Elimination of salt-and-pepper impulse artifacts while preserving edge transitions.")
        potential_risks.append("Slight rounding of acute corners if kernel size exceeds 3.")
    elif diag["is_noisy"]:
        params["spatial_filter"] = "weighted_average"
        params["kernel_size"] = 7 if noise_sigma >= 20.0 else 5
        selected_ops.append(f"Weighted Averaging Linear Filter ({params['kernel_size']}x{params['kernel_size']})")
        justifications.append(f"Detected additive sensor noise (sigma={noise_sigma}); standard weighted averaging smoothing reduces variance while preserving photometric edge gradients (Gonzalez & Woods Section 3.5.1).")
        expected_improvements.append("Significant noise reduction and cleaner uniform regions.")
        potential_risks.append("Mild reduction in low-contrast fine micro-texture.")

    # -------------------------------------------------------------------------
    # Stage B: Intensity Correction (Power-Law Gamma Transformation)
    # -------------------------------------------------------------------------
    if diag["is_dark"]:
        safe_mean = max(mean_val, 15.0)
        computed_gamma = float(np.clip(np.log(125.0 / 255.0) / np.log(safe_mean / 255.0), 0.40, 0.75))
        params["gamma"] = round(computed_gamma, 2)
        selected_ops.append(f"Power-Law Gamma Expansion (gamma={params['gamma']})")
        justifications.append(f"Low mean luminance ({mean_val}) with shadow compression; power-law gamma < 1.0 elevates dark midtones while keeping maximum white anchored (Gonzalez & Woods Section 3.2.3).")
        expected_improvements.append("Visibly brighter shadow regions and recovered textural visibility.")
        potential_risks.append("May reveal latent sensor noise in extreme deep shadows.")
    elif diag["is_bright"]:
        safe_mean = min(mean_val, 245.0)
        computed_gamma = float(np.clip(np.log(145.0 / 255.0) / np.log(safe_mean / 255.0), 1.40, 2.20))
        params["gamma"] = round(computed_gamma, 2)
        selected_ops.append(f"Power-Law Gamma Compression (gamma={params['gamma']})")
        justifications.append(f"Excessive mean luminance ({mean_val}) with highlight saturation; power-law gamma > 1.0 compresses washed-out highlights into the dynamic range (Gonzalez & Woods Section 3.2.3).")
        expected_improvements.append("Restoration of highlight detail and elimination of overexposed blowout.")
        potential_risks.append("Very dark shadow patches may become slightly deeper.")

    # -------------------------------------------------------------------------
    # Stage C: Contrast Equalization (Histogram Processing)
    # -------------------------------------------------------------------------
    if diag["is_low_contrast"]:
        params["equalization_mode"] = "clahe"
        params["clahe_clip_limit"] = 3.0
        params["clahe_tile_grid_size"] = 8
        selected_ops.append("Local Histogram Processing (CLAHE)")
        justifications.append("Narrow histogram dispersion (std < 38, dynamic range < 130); local histogram processing redistributes gradients across neighborhoods without over-amplifying background noise (Gonzalez & Woods Section 3.3.3).")
        expected_improvements.append("Enhanced local edge contrast and feature visibility across all image regions.")
        potential_risks.append("Local slope gradients may feel slightly unnatural if clip limit is set too high.")
    elif diag["is_dark"]:
        params["equalization_mode"] = "clahe"
        params["clahe_clip_limit"] = 1.8
        params["clahe_tile_grid_size"] = 8
        selected_ops.append("Adaptive Shadow Contrast Enhancement (Local Histogram)")
        justifications.append("Local histogram processing following gamma expansion reveals latent textural details within recovered shadow zones.")
        expected_improvements.append("Crisp local contrast in newly illuminated shadow regions.")
        potential_risks.append("Mild local gradient artifacts in flat shadows.")
    elif diag.get("is_uneven", False):
        params["equalization_mode"] = "clahe"
        params["clahe_clip_limit"] = 2.5
        params["clahe_tile_grid_size"] = 8
        selected_ops.append("Local Histogram Equalization (Uneven Lighting Compensation)")
        justifications.append("Detected significant spatial illumination variation across the image; local histogram processing normalizes contrast across tiles.")
        expected_improvements.append("Uniform legibility and balanced contrast across non-uniformly illuminated regions.")
        potential_risks.append("Slight variation in background shade.")

    # -------------------------------------------------------------------------
    # Stage D: Spatial Sharpening (Executed AFTER denoising and tone-mapping)
    # -------------------------------------------------------------------------
    if diag.get("is_severely_blurry", False):
        params["sharpen_strength"] = 1.3
        params["kernel_size"] = 5
        selected_ops.append("High-Boost Spatial Sharpening (factor k=1.3, kernel=5x5)")
        justifications.append(f"Severe spatial/optical blur detected (sharpness={sharpness}, norm={norm_sharp}); high-boost unsharp masking steepens surviving edge gradients (Gonzalez & Woods Section 3.6.3).")
        expected_improvements.append("Improved edge definition and perceived clarity on surviving contour transitions.")
        potential_risks.append("Severe blur cannot be completely reversed; excessive sharpening causes overshoot halos.")
    elif diag["is_blurry"]:
        params["sharpen_strength"] = 1.4
        params["kernel_size"] = 5
        selected_ops.append("Unsharp Masking Spatial Sharpening (factor k=1.4, kernel=5x5)")
        justifications.append(f"Sub-optimal edge acuity (sharpness={sharpness}, norm={norm_sharp}) under low-noise conditions; unsharp masking subtracts a 5x5 low-pass Gaussian envelope to amplify edge transitions (Gonzalez & Woods Section 3.6.3).")
        expected_improvements.append("Crisper object outlines, heightened visual sharpness, and elevated acutance.")
        potential_risks.append("Overshoot halos around high-contrast step edges if strength exceeds 1.8.")
    elif diag["is_noisy"]:
        params["sharpen_strength"] = 0.0

    if not selected_ops:
        selected_ops.append("Conservative Neutral Pass (Image already well-balanced)")
        justifications.append("Image exhibits healthy contrast, dynamic range, and low noise; aggressive transformation would introduce artificial artifacts.")
        expected_improvements.append("Preservation of authentic ground-truth fidelity.")

    return {
        "mode": "automatic",
        "selected_operations": selected_ops,
        "parameters": params,
        "justifications": justifications,
        "expected_improvements": expected_improvements,
        "potential_risks": potential_risks
    }


# ---------------------------------------------------------------------------
# 7. Core Pipeline Orchestrator
# ---------------------------------------------------------------------------

def process_image_enhancement(
    image: np.ndarray,
    mode: str = "auto",
    category: str = "auto",
    operation: str = "auto",
    # Legacy / Auto parameters (for backward compatibility)
    gamma: float = 1.0,
    equalization_mode: str = "clahe",
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: int = 8,
    spatial_filter: str = "none",
    kernel_size: int = 3,
    sharpen_strength: float = 0.0,
    # Section 2: Intensity Transformations
    log_c: Optional[float] = None,
    contrast_r1: Optional[int] = None,
    contrast_s1: int = 0,
    contrast_r2: Optional[int] = None,
    contrast_s2: int = 255,
    slice_low: int = 100,
    slice_high: int = 180,
    slice_preserve_bg: bool = True,
    bit_plane: int = 7,
    # Section 3: Histogram Processing
    reference_image: Optional[np.ndarray] = None,
    local_window_size: int = 8,
    # Section 4: Smoothing Filters
    smoothing_type: str = "box",
    alpha_trim: float = 0.2,
    # Section 5: Sharpening Filters
    sharpening_type: str = "laplacian",
    laplacian_mask: str = "4_neighbor",
    unsharp_sigma: Optional[float] = None,
    highboost_k: float = 1.0,
    gradient_operator: str = "sobel",
    gradient_calc: str = "abs_sum",
    gradient_mode: str = "sharpened",
    filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the Automatic or Manual Image Enhancement workflow following Gonzalez and Woods Chapter 3.
    """
    characteristics = analyze_image_characteristics(image)

    # Determine execution route: Automatic vs Chapter 3 Categories vs Legacy Manual
    cat = category.lower() if category != "auto" else ("auto" if mode.lower() == "auto" else "manual_legacy")

    if cat == "auto":
        decision = decide_enhancement_pipeline(characteristics)
        p = decision["parameters"]
        active_gamma = p["gamma"]
        active_eq_mode = p["equalization_mode"]
        active_clip_limit = p["clahe_clip_limit"]
        active_tile_size = p["clahe_tile_grid_size"]
        active_filter = p["spatial_filter"]
        active_ksize = p["kernel_size"]
        active_sharpen = p["sharpen_strength"]
        pipeline_info = decision

        # Execute automatic pipeline strictly using Gonzalez & Woods Ch 3 operations:
        current = image.copy()

        # Step 1: Denoise first
        if active_filter == "median":
            current = apply_median_filter(current, active_ksize)
        elif active_filter == "weighted_average":
            current = apply_weighted_average_filter(current, active_ksize)
        elif active_filter in ("gaussian", "bilateral"):
            # Map legacy bilateral/gaussian to weighted average filter
            current = apply_weighted_average_filter(current, active_ksize)

        # Step 2: Gamma correction
        if abs(active_gamma - 1.0) >= 0.01:
            current = apply_power_law(current, active_gamma)

        # Step 3: Equalization / Local Histogram
        if active_eq_mode in ("global", "equalization"):
            current = apply_histogram_equalization(current)
        elif active_eq_mode == "clahe":
            current = apply_local_histogram_processing(current, active_tile_size)

        # Step 4: Unsharp masking / highboost
        if active_sharpen > 0.0:
            current = apply_unsharp_highboost_filter(
                current,
                kernel_size=active_ksize,
                k_factor=active_sharpen
            )

        enhanced = current

    elif cat == "intensity_transform":
        op = operation.lower()
        ops_applied = []
        if op == "negative":
            enhanced = apply_image_negative(image)
            ops_applied.append("Image Negative Transformation: s = 255 - r")
        elif op == "log":
            enhanced = apply_log_transform(image, c=log_c)
            ops_applied.append(f"Logarithmic Transformation: s = c * log(1 + r) (c={log_c or 45.98:.2f})")
        elif op == "gamma":
            enhanced = apply_power_law(image, gamma=gamma)
            ops_applied.append(f"Power-Law (Gamma) Transformation: s = 255 * (r/255)^{gamma:.2f}")
        elif op == "contrast_stretch":
            enhanced = apply_contrast_stretching(image, r1=contrast_r1, s1=contrast_s1, r2=contrast_r2, s2=contrast_s2)
            ops_applied.append(f"Piecewise-Linear Contrast Stretching: [r1, r2] -> [s1, s2]")
        elif op == "intensity_slicing":
            enhanced = apply_intensity_slicing(image, low=slice_low, high=slice_high, preserve_background=slice_preserve_bg)
            ops_applied.append(f"Intensity-Level Slicing: [{slice_low}, {slice_high}] (Preserve BG: {slice_preserve_bg})")
        elif op == "bit_plane":
            enhanced = apply_bit_plane_slicing(image, bit_plane=bit_plane)
            ops_applied.append(f"Bit-Plane Slicing: Bit Plane {bit_plane} (MSB=7, LSB=0)")
        else:
            enhanced = image.copy()
            ops_applied.append("Identity Pass")

        pipeline_info = {
            "mode": "manual",
            "category": "Intensity Transformations",
            "selected_operations": ops_applied,
            "parameters": {
                "operation": op,
                "gamma": gamma,
                "log_c": log_c,
                "contrast_r1": contrast_r1,
                "contrast_s1": contrast_s1,
                "contrast_r2": contrast_r2,
                "contrast_s2": contrast_s2,
                "slice_low": slice_low,
                "slice_high": slice_high,
                "slice_preserve_bg": slice_preserve_bg,
                "bit_plane": bit_plane
            },
            "justifications": ["User-selected intensity transformation (Gonzalez & Woods Section 3.2)."],
            "expected_improvements": ["Applies targeted point-processing transformation directly to intensity values."],
            "potential_risks": ["Extreme parameters may cause clipping or loss of dynamic range."]
        }

    elif cat == "histogram_processing":
        op = operation.lower()
        ops_applied = []
        if op in ("equalization", "global"):
            enhanced = apply_histogram_equalization(image)
            ops_applied.append("Histogram Equalization (Global Flat Distribution)")
        elif op in ("matching", "specification"):
            if reference_image is not None:
                enhanced = apply_histogram_matching(image, reference_image)
                ops_applied.append("Histogram Specification (Matching Target Reference Image)")
            else:
                # Default reference fallback
                enhanced = apply_histogram_equalization(image)
                ops_applied.append("Histogram Equalization (No reference image provided)")
        elif op in ("local", "clahe"):
            enhanced = apply_local_histogram_processing(image, window_size=local_window_size)
            ops_applied.append(f"Local Histogram Processing (Window / Tile Size={local_window_size}x{local_window_size})")
        else:
            enhanced = image.copy()
            ops_applied.append("Original Histogram Preserved")

        pipeline_info = {
            "mode": "manual",
            "category": "Histogram Processing",
            "selected_operations": ops_applied,
            "parameters": {
                "operation": op,
                "local_window_size": local_window_size
            },
            "justifications": ["User-selected histogram processing operation (Gonzalez & Woods Section 3.3)."],
            "expected_improvements": ["Redistributes intensity probabilities to maximize feature visibility."],
            "potential_risks": ["Global equalization may over-enhance background noise in flat regions."]
        }

    elif cat == "smoothing":
        st = smoothing_type.lower()
        k = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
        ops_applied = []
        if st == "box":
            enhanced = apply_box_filter(image, kernel_size=k)
            ops_applied.append(f"Linear Averaging (Box) Filter ({k}x{k})")
        elif st == "weighted_average":
            enhanced = apply_weighted_average_filter(image, kernel_size=k)
            ops_applied.append(f"Linear Weighted Averaging Filter ({k}x{k})")
        elif st == "median":
            enhanced = apply_median_filter(image, kernel_size=k)
            ops_applied.append(f"Order-Statistic Median Filter ({k}x{k})")
        elif st == "max":
            enhanced = apply_max_filter(image, kernel_size=k)
            ops_applied.append(f"Order-Statistic Max Filter ({k}x{k})")
        elif st == "min":
            enhanced = apply_min_filter(image, kernel_size=k)
            ops_applied.append(f"Order-Statistic Min Filter ({k}x{k})")
        elif st == "midpoint":
            enhanced = apply_midpoint_filter(image, kernel_size=k)
            ops_applied.append(f"Order-Statistic Midpoint Filter ({k}x{k})")
        elif st == "alpha_trimmed":
            enhanced = apply_alpha_trimmed_mean_filter(image, kernel_size=k, alpha=alpha_trim)
            ops_applied.append(f"Order-Statistic Alpha-Trimmed Mean Filter ({k}x{k}, alpha={alpha_trim})")
        else:
            enhanced = apply_box_filter(image, kernel_size=k)
            ops_applied.append(f"Box Filter ({k}x{k})")

        pipeline_info = {
            "mode": "manual",
            "category": "Smoothing Filters",
            "selected_operations": ops_applied,
            "parameters": {
                "smoothing_type": st,
                "kernel_size": k,
                "alpha_trim": alpha_trim
            },
            "justifications": ["User-selected smoothing spatial filter (Gonzalez & Woods Section 3.5)."],
            "expected_improvements": ["Attenuates high-frequency noise spikes and smooths sharp fluctuations."],
            "potential_risks": ["Blurs sharp edge contours proportionally to kernel size."]
        }

    elif cat == "sharpening":
        sh = sharpening_type.lower()
        ops_applied = []
        if sh == "laplacian":
            enhanced = apply_laplacian_filter(image, mask_type=laplacian_mask)
            ops_applied.append(f"Laplacian Spatial Sharpening ({laplacian_mask.replace('_', ' ').capitalize()} Mask)")
        elif sh in ("unsharp", "highboost", "unsharp_highboost"):
            k_val = highboost_k if highboost_k > 0.0 else (sharpen_strength if sharpen_strength > 0.0 else 1.0)
            enhanced = apply_unsharp_highboost_filter(
                image,
                kernel_size=kernel_size,
                sigma=unsharp_sigma,
                k_factor=k_val
            )
            mode_label = "Highboost Filtering" if k_val > 1.0 else "Unsharp Masking"
            ops_applied.append(f"{mode_label} (k={k_val:.2f}, kernel={kernel_size}x{kernel_size})")
        elif sh in ("gradient", "roberts", "prewitt", "sobel"):
            op_name = gradient_operator if gradient_operator in ("roberts", "prewitt", "sobel") else "sobel"
            enhanced = apply_gradient_filter(
                image,
                operator=op_name,
                calc_method=gradient_calc,
                output_mode=gradient_mode
            )
            ops_applied.append(f"Gradient-Based Sharpening: {op_name.capitalize()} Operator ({gradient_mode.capitalize()})")
        else:
            enhanced = apply_laplacian_filter(image, mask_type="4_neighbor")
            ops_applied.append("Laplacian Sharpening (4-neighbor)")

        pipeline_info = {
            "mode": "manual",
            "category": "Sharpening Filters",
            "selected_operations": ops_applied,
            "parameters": {
                "sharpening_type": sh,
                "laplacian_mask": laplacian_mask,
                "highboost_k": highboost_k,
                "gradient_operator": gradient_operator,
                "gradient_calc": gradient_calc,
                "gradient_mode": gradient_mode
            },
            "justifications": ["User-selected sharpening spatial filter (Gonzalez & Woods Section 3.6)."],
            "expected_improvements": ["Enhances edge transitions and elevates fine structural acutance."],
            "potential_risks": ["Will amplify high-frequency noise if applied to un-smoothed noisy images."]
        }

    else:
        # Legacy manual fallback
        current = image.copy()
        ops_applied = []
        if spatial_filter == "median":
            current = apply_median_filter(current, kernel_size)
            ops_applied.append(f"Median Filter ({kernel_size}x{kernel_size})")
        elif spatial_filter in ("gaussian", "weighted_average", "box", "bilateral"):
            current = apply_weighted_average_filter(current, kernel_size)
            ops_applied.append(f"Smoothing Filter ({kernel_size}x{kernel_size})")

        if abs(gamma - 1.0) >= 0.05:
            current = apply_power_law(current, gamma)
            ops_applied.append(f"Gamma Correction (gamma={gamma:.2f})")

        if equalization_mode in ("global", "equalization"):
            current = apply_histogram_equalization(current)
            ops_applied.append("Global Histogram Equalization")
        elif equalization_mode in ("clahe", "local"):
            current = apply_local_histogram_processing(current, clahe_tile_grid_size)
            ops_applied.append(f"Local Histogram Equalization ({clahe_tile_grid_size}x{clahe_tile_grid_size})")

        if sharpen_strength > 0.0 or spatial_filter == "unsharp":
            str_val = sharpen_strength if sharpen_strength > 0.0 else 1.0
            current = apply_unsharp_highboost_filter(current, kernel_size=kernel_size, k_factor=str_val)
            ops_applied.append(f"Sharpening (strength={str_val:.1f})")

        enhanced = current
        pipeline_info = {
            "mode": "manual",
            "selected_operations": ops_applied or ["Identity / No Modifications"],
            "parameters": {
                "gamma": gamma,
                "equalization_mode": equalization_mode,
                "clahe_clip_limit": clahe_clip_limit,
                "clahe_tile_grid_size": clahe_tile_grid_size,
                "spatial_filter": spatial_filter,
                "kernel_size": kernel_size,
                "sharpen_strength": sharpen_strength
            },
            "justifications": ["User-defined manual parameter exploration."],
            "expected_improvements": ["Reflects user-specified parameter selection."],
            "potential_risks": ["Manual parameter combinations may over-amplify noise if unbalanced."]
        }

    # Analyze enhanced characteristics & compute histograms
    enhanced_characteristics = analyze_image_characteristics(enhanced)
    original_hist = compute_histogram_data(image)
    enhanced_hist = compute_histogram_data(enhanced)

    # Reference metrics
    ground_truth_img = None
    ground_truth_name = None
    if filename and any(filename.startswith(p) for p in ["01_", "02_", "03_", "04_", "05_", "07_"]):
        clean_ref_path = settings.SAMPLES_DIR / "enhancement" / "ground_truth" / "clean_reference.png"
        if not clean_ref_path.exists():
            clean_ref_path = settings.SAMPLES_DIR / "enhancement" / "clean_reference.png"
        if clean_ref_path.exists():
            ground_truth_img = cv2.imread(str(clean_ref_path), cv2.IMREAD_GRAYSCALE)
            ground_truth_name = "clean_reference.png (Pristine Cameraman)"

    quantitative_metrics = evaluate_enhancement_metrics(
        input_img=image,
        enhanced_img=enhanced,
        ground_truth_img=ground_truth_img,
        ground_truth_name=ground_truth_name
    )

    orig_b64 = encode_image_to_base64(image)
    enh_b64 = encode_image_to_base64(enhanced)

    return {
        "status": "success",
        "module": "enhancement",
        "mode": pipeline_info["mode"],
        "pipeline": pipeline_info,
        "input_characteristics": characteristics,
        "enhanced_characteristics": enhanced_characteristics,
        "metrics": quantitative_metrics,
        "histograms": {
            "original": original_hist,
            "enhanced": enhanced_hist
        },
        "images": {
            "original_base64": orig_b64,
            "enhanced_base64": enh_b64
        }
    }
