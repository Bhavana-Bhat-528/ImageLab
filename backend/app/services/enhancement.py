"""
Automatic Image Enhancement Service.
Implements:
- Quantitative image characteristic analysis (mean, std, contrast, noise, sharpness, clipping, illumination uniformity).
- Multi-factor adaptive decision mechanism that selects and tunes operations based on detected features.
- Accurate power-law gamma transformation (gamma < 1 brightens, gamma > 1 compresses highlights).
- Disciplined noise reduction (median for impulse, edge-preserving bilateral for Gaussian/sensor noise).
- Multi-scale unsharp masking for optical/spatial blur with conservative limits.
- Order-controlled pipeline: Denoising -> Gamma Correction -> Equalization / CLAHE -> Sharpening.
- Manual parameter execution mode.
- Non-blind operational pipeline with rationale, expected improvements, and artifact warnings.
- Histogram computation for before/after visual comparison.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

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

    # Mask for Gaussian noise estimation
    mask = np.array([[1, -2, 1],
                     [-2, 4, -2],
                     [1, -2, 1]], dtype=np.float32)
    filtered = cv2.filter2D(gray.astype(np.float32), -1, mask)
    sigma = float(np.sum(np.abs(filtered)) * np.sqrt(0.5 * np.pi) / (6.0 * (w - 2) * (h - 2)))

    # Impulse noise check: count extreme pixels that differ heavily from 3x3 median
    med = cv2.medianBlur(gray, 3)
    diff = np.abs(gray.astype(np.int32) - med.astype(np.int32))
    extreme_pixels = ((gray <= 3) | (gray >= 252)) & (diff > 45)
    sp_ratio = float(np.mean(extreme_pixels))
    has_salt_pepper = sp_ratio > 0.010

    return round(sigma, 2), has_salt_pepper


def estimate_sharpness(gray: np.ndarray) -> Tuple[float, float]:
    """
    Estimates spatial image sharpness using:
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
    Analyzes input image characteristics using multi-factor statistical measures:
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
        # Check if 3-channel image is true color or grayscale duplicated across channels
        ch_diff1 = np.max(np.abs(image[:, :, 0].astype(np.int32) - image[:, :, 1].astype(np.int32)))
        ch_diff2 = np.max(np.abs(image[:, :, 1].astype(np.int32) - image[:, :, 2].astype(np.int32)))
        is_color = bool(ch_diff1 > 3 or ch_diff2 > 3)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        is_color = False
        gray = image.copy()

    h, w = gray.shape
    total_pixels = h * w

    # Mean & Standard Deviation
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    # Min, Max, Dynamic Range
    min_val = int(np.min(gray))
    max_val = int(np.max(gray))
    dynamic_range = max_val - min_val
    rms_contrast = round(std_val / 255.0, 3)

    # Histogram & Clipping
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    prob_dist = hist / float(total_pixels)
    nonzero_probs = prob_dist[prob_dist > 0]
    entropy = float(-np.sum(nonzero_probs * np.log2(nonzero_probs)))

    shadow_clipping_pct = round(float(np.sum(hist[:16]) / total_pixels) * 100.0, 2)
    highlight_clipping_pct = round(float(np.sum(hist[240:]) / total_pixels) * 100.0, 2)

    # Skewness: measures asymmetric shift of intensity mass
    if std_val > 1e-3:
        skewness = float(np.mean(((gray - mean_val) / std_val) ** 3))
    else:
        skewness = 0.0

    # Noise & Sharpness
    noise_sigma, has_salt_pepper = estimate_noise_level(gray)
    sharpness_score, norm_sharpness = estimate_sharpness(gray)

    # Spatial Illumination Uniformity (4x4 block grid)
    blocks = []
    bh, bw = max(h // 4, 1), max(w // 4, 1)
    for r in range(4):
        for c in range(4):
            blocks.append(float(np.mean(gray[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw])))
    block_range = round(float(np.max(blocks) - np.min(blocks)), 2)

    # Multi-factor Diagnostics
    is_dark = mean_val < 90.0 or (mean_val < 105.0 and shadow_clipping_pct > 12.0)
    is_bright = mean_val > 165.0 or (mean_val > 155.0 and highlight_clipping_pct > 12.0)
    is_low_contrast = std_val < 38.0 and dynamic_range < 130
    is_noisy = (noise_sigma > 13.0 and not is_low_contrast) or has_salt_pepper
    
    # Blur detection: low normalized sharpness and low raw sharpness in low-noise conditions
    is_blurry = (norm_sharpness < 0.20 or sharpness_score < 600.0) and not is_noisy and not is_low_contrast
    is_severely_blurry = is_blurry and (norm_sharpness < 0.02 or sharpness_score < 100.0)

    # Uneven illumination: strong spatial luminance gradient across blocks
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
        primary_defects.append(f"Low Spatial Acuity / Edge Softness (Sharpness={sharpness_score})")

    if not primary_defects:
        primary_defects.append("Balanced Baseline Lighting and Contrast")

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
        "illumination_block_range": block_range,
        "is_color": is_color,
        "diagnostics": {
            "is_dark": is_dark,
            "is_bright": is_bright,
            "is_low_contrast": is_low_contrast,
            "is_noisy": is_noisy,
            "has_salt_pepper": has_salt_pepper,
            "is_blurry": is_blurry,
            "is_severely_blurry": is_severely_blurry,
            "is_uneven": is_uneven,
            "primary_defects": primary_defects
        }
    }


# ---------------------------------------------------------------------------
# 2. Adaptive Decision Mechanism
# ---------------------------------------------------------------------------

def decide_enhancement_pipeline(characteristics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decides appropriate transformations and parameters based on characteristics.
    Never applies every operation blindly:
    - Step 1: Denoise first (median for impulse, bilateral for Gaussian/sensor noise).
    - Step 2: Power-law gamma transformation (gamma < 1 expands shadows, gamma > 1 compresses highlights).
    - Step 3: Contrast equalization (CLAHE for narrow dynamic range or uneven illumination).
    - Step 4: Spatial sharpening (unsharp masking with tuned kernel only for soft/blurry edges).
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
    # Stage A: Noise Filtering (must be applied FIRST before contrast or sharpening)
    # -------------------------------------------------------------------------
    if has_sp:
        params["spatial_filter"] = "median"
        params["kernel_size"] = 3
        selected_ops.append("Non-Linear Median Filtering (3x3)")
        justifications.append("Detected high-impulse salt-and-pepper noise; non-linear median filtering eliminates extreme outlier spikes without smearing genuine object edges.")
        expected_improvements.append("Complete elimination of salt-and-pepper impulse artifacts while preserving edge transitions.")
        potential_risks.append("Slight rounding of acute corners if kernel size exceeds 3.")
    elif diag["is_noisy"]:
        params["spatial_filter"] = "bilateral"
        params["kernel_size"] = 7 if noise_sigma >= 20.0 else 5
        selected_ops.append(f"Edge-Preserving Bilateral Smoothing ({params['kernel_size']}x{params['kernel_size']})")
        justifications.append(f"Detected additive sensor noise (sigma={noise_sigma}); bilateral filtering smooths planar intensity variance while preserving photometric edge gradients.")
        expected_improvements.append("Significant SNR elevation and clean uniform regions without blurring salient boundaries.")
        potential_risks.append("Mild reduction in low-contrast fine micro-texture.")

    # -------------------------------------------------------------------------
    # Stage B: Intensity Correction (Power-Law Gamma Transformation)
    # In standard DIP: s = c * (r / 255)^gamma
    # gamma < 1.0 brightens underexposed shadows (concave down)
    # gamma > 1.0 darkens / compresses overexposed highlights (concave up)
    # -------------------------------------------------------------------------
    if diag["is_dark"]:
        # Adaptive formula: target mean ~ 125
        safe_mean = max(mean_val, 15.0)
        computed_gamma = float(np.clip(np.log(125.0 / 255.0) / np.log(safe_mean / 255.0), 0.40, 0.75))
        params["gamma"] = round(computed_gamma, 2)
        selected_ops.append(f"Power-Law Gamma Expansion (gamma={params['gamma']})")
        justifications.append(f"Low mean luminance ({mean_val}) with shadow compression; power-law gamma < 1.0 elevates dark midtones while keeping maximum white anchored.")
        expected_improvements.append("Visibly brighter shadow regions and recovered textural visibility.")
        potential_risks.append("May reveal latent sensor noise in extreme deep shadows.")
    elif diag["is_bright"]:
        # Adaptive formula: target mean ~ 145
        safe_mean = min(mean_val, 245.0)
        computed_gamma = float(np.clip(np.log(145.0 / 255.0) / np.log(safe_mean / 255.0), 1.40, 2.20))
        params["gamma"] = round(computed_gamma, 2)
        selected_ops.append(f"Power-Law Gamma Compression (gamma={params['gamma']})")
        justifications.append(f"Excessive mean luminance ({mean_val}) with highlight saturation; power-law gamma > 1.0 compresses washed-out highlights and pulls tones back into the dynamic range.")
        expected_improvements.append("Restoration of highlight detail and elimination of overexposed blowout.")
        potential_risks.append("Very dark shadow patches may become slightly deeper.")

    # -------------------------------------------------------------------------
    # Stage C: Contrast Equalization (CLAHE / Equalization)
    # -------------------------------------------------------------------------
    if diag["is_low_contrast"]:
        params["equalization_mode"] = "clahe"
        params["clahe_clip_limit"] = 3.0
        params["clahe_tile_grid_size"] = 8
        selected_ops.append("Contrast Limited Adaptive Histogram Equalization (CLAHE)")
        justifications.append("Narrow histogram dispersion (std < 38, dynamic range < 130); CLAHE redistributes local gradients across 8x8 tiles without over-amplifying background noise.")
        expected_improvements.append("Enhanced local edge contrast and feature visibility across all image regions.")
        potential_risks.append("Local slope gradients may feel slightly unnatural if clip limit is set too high.")
    elif diag["is_dark"]:
        # Underexposed images benefit from mild CLAHE after gamma expansion to reveal deep shadow textures
        params["equalization_mode"] = "clahe"
        params["clahe_clip_limit"] = 1.8
        params["clahe_tile_grid_size"] = 8
        selected_ops.append("Adaptive Shadow Contrast Enhancement (CLAHE)")
        justifications.append("Mild CLAHE following gamma expansion reveals latent textural details within recovered shadow zones.")
        expected_improvements.append("Crisp local contrast in newly illuminated shadow regions.")
        potential_risks.append("Mild local gradient artifacts in flat shadows.")
    elif diag.get("is_uneven", False):
        params["equalization_mode"] = "clahe"
        params["clahe_clip_limit"] = 2.5
        params["clahe_tile_grid_size"] = 8
        selected_ops.append("Adaptive Illumination Gradient Equalization (CLAHE)")
        justifications.append("Detected significant spatial illumination variation across the image; CLAHE locally normalizes contrast across 8x8 tiles.")
        expected_improvements.append("Uniform legibility and balanced contrast across non-uniformly illuminated regions.")
        potential_risks.append("Slight variation in background shade.")

    # -------------------------------------------------------------------------
    # Stage D: Spatial Sharpening (Executed AFTER denoising and tone-mapping)
    # -------------------------------------------------------------------------
    if diag.get("is_severely_blurry", False):
        params["sharpen_strength"] = 1.3
        params["kernel_size"] = 5
        selected_ops.append("High-Boost Spatial Sharpening (strength=1.3, kernel=5x5)")
        justifications.append(f"Severe spatial/optical blur detected (sharpness={sharpness}, norm={norm_sharp}); high-boost unsharp masking steepens surviving edge gradients. Note: High-frequency details lost beneath the optical diffraction limit cannot be fully synthesized.")
        expected_improvements.append("Improved edge definition and perceived clarity on surviving contour transitions.")
        potential_risks.append("Severe blur cannot be completely reversed; excessive sharpening on severe blur causes overshoot halos.")
    elif diag["is_blurry"]:
        params["sharpen_strength"] = 1.4
        params["kernel_size"] = 5
        selected_ops.append("Unsharp Masking Spatial Sharpening (strength=1.4, kernel=5x5)")
        justifications.append(f"Sub-optimal edge acuity (sharpness={sharpness}, norm={norm_sharp}) under low-noise conditions; unsharp masking subtracts a 5x5 low-pass Gaussian envelope to amplify edge transitions.")
        expected_improvements.append("Crisper object outlines, heightened visual sharpness, and elevated acutance.")
        potential_risks.append("Overshoot halos around high-contrast step edges if strength exceeds 1.8.")
    elif diag["is_noisy"]:
        # Do not apply sharpening when significant noise is present, as sharpening amplifies noise remnants
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
# 3. Image Transformation Operations
# ---------------------------------------------------------------------------

def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """
    Applies standard power-law intensity transformation: s = 255 * (r / 255)^gamma.
    - gamma < 1.0 (e.g. 0.40 - 0.75): maps darker values up -> BRIGHTENS underexposed images.
    - gamma > 1.0 (e.g. 1.30 - 2.20): maps midtones down -> COMPRESSES overexposed highlights.
    """
    if abs(gamma - 1.0) < 1e-3:
        return image.copy()
    
    # Precompute lookup table directly with gamma
    gamma_val = max(gamma, 0.05)
    table = np.array([
        ((i / 255.0) ** gamma_val) * 255.0
        for i in range(256)
    ], dtype=np.uint8)
    return cv2.LUT(image, table)


def apply_equalization(
    image: np.ndarray,
    mode: str = "clahe",
    clip_limit: float = 2.0,
    tile_size: int = 8
) -> np.ndarray:
    """
    Applies Global Histogram Equalization or CLAHE.
    For color images: converts to LAB color space, equalizes luminance L channel only,
    and converts back to BGR, preserving authentic color balance without hue shifting.
    """
    if mode == "none":
        return image.copy()

    is_color = len(image.shape) == 3 and image.shape[2] == 3

    if mode == "global":
        if is_color:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_eq = cv2.equalizeHist(l)
            return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)
        else:
            return cv2.equalizeHist(image)

    elif mode == "clahe":
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        if is_color:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_clahe = clahe.apply(l)
            return cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)
        else:
            return clahe.apply(image)

    return image.copy()


def apply_spatial_filtering(
    image: np.ndarray,
    filter_type: str = "none",
    kernel_size: int = 3,
    sharpen_strength: float = 0.0
) -> np.ndarray:
    """
    Applies spatial filtering:
    - 'gaussian': Gaussian smoothing with scale-appropriate sigma
    - 'median': Non-linear median filtering (removes impulse spikes)
    - 'bilateral': Edge-preserving bilateral filter
    - 'unsharp': Multi-scale unsharp masking [I + alpha * (I - I_blur)]
    """
    k = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    out = image.copy()

    if filter_type == "gaussian":
        sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
        out = cv2.GaussianBlur(out, (k, k), sigma)
    elif filter_type == "median":
        out = cv2.medianBlur(out, k)
    elif filter_type == "bilateral":
        # Bilateral filter with parameters tuned to kernel size
        sigma_color = float(k) * 11.0
        sigma_space = float(k) * 1.2
        out = cv2.bilateralFilter(out, k, sigma_color, sigma_space)

    # Apply sharpening if requested or if filter_type is unsharp
    if sharpen_strength > 0.0 or filter_type == "unsharp":
        k_sharp = max(3, k if k % 2 == 1 else k + 1)
        sigma_sharp = 0.3 * ((k_sharp - 1) * 0.5 - 1) + 0.8
        blurred = cv2.GaussianBlur(out, (k_sharp, k_sharp), sigma_sharp)
        diff = out.astype(np.float32) - blurred.astype(np.float32)
        sharpened = out.astype(np.float32) + (float(sharpen_strength) * diff)
        out = np.clip(sharpened, 0, 255).astype(np.uint8)

    return out


# ---------------------------------------------------------------------------
# 4. Histogram Generation
# ---------------------------------------------------------------------------

def compute_histogram_data(image: np.ndarray) -> Dict[str, Any]:
    """
    Computes normalized histogram distributions (256 bins) for plotting.
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
# 5. Core Pipeline Orchestrator
# ---------------------------------------------------------------------------

def process_image_enhancement(
    image: np.ndarray,
    mode: str = "auto",
    gamma: float = 1.0,
    equalization_mode: str = "clahe",
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: int = 8,
    spatial_filter: str = "none",
    kernel_size: int = 3,
    sharpen_strength: float = 0.0,
    filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the Automatic or Manual Image Enhancement workflow.
    - Analyzes image characteristics using multi-factor statistical measures.
    - Decides or accepts parameters.
    - Executes order-controlled transformations:
        1. Denoise
        2. Tone mapping (Gamma)
        3. Local contrast equalization (CLAHE)
        4. Sharpening
    - Computes before/after histograms.
    - Evaluates relative and ground-truth reference metrics.
    """
    # 1. Analyze input characteristics
    characteristics = analyze_image_characteristics(image)

    # 2. Determine execution parameters
    if mode == "auto":
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
    else:
        # Manual mode
        active_gamma = float(gamma)
        active_eq_mode = str(equalization_mode).lower()
        active_clip_limit = float(clahe_clip_limit)
        active_tile_size = int(clahe_tile_grid_size)
        active_filter = str(spatial_filter).lower()
        active_ksize = int(kernel_size)
        active_sharpen = float(sharpen_strength)

        ops = []
        if active_filter != "none":
            ops.append(f"Spatial Filter: {active_filter.capitalize()} ({active_ksize}x{active_ksize})")
        if abs(active_gamma - 1.0) >= 0.05:
            ops.append(f"Gamma Correction (gamma={active_gamma:.2f})")
        if active_eq_mode != "none":
            ops.append(f"Equalization: {active_eq_mode.upper()}")
        if active_sharpen > 0.0:
            ops.append(f"Sharpening (strength={active_sharpen:.1f})")

        pipeline_info = {
            "mode": "manual",
            "selected_operations": ops or ["Identity / No Modifications"],
            "parameters": {
                "gamma": active_gamma,
                "equalization_mode": active_eq_mode,
                "clahe_clip_limit": active_clip_limit,
                "clahe_tile_grid_size": active_tile_size,
                "spatial_filter": active_filter,
                "kernel_size": active_ksize,
                "sharpen_strength": active_sharpen
            },
            "justifications": ["User-defined manual parameter exploration."],
            "expected_improvements": ["Reflects user-specified parameter selection."],
            "potential_risks": ["Manual parameter combinations may over-amplify noise or clip dynamic range if unbalanced."]
        }

    # 3. Execute processing in disciplined mathematical order:
    #    Step 1: Noise reduction first (prevents noise from being amplified)
    #    Step 2: Gamma correction (global tone-mapping)
    #    Step 3: Equalization / CLAHE (local contrast redistribution)
    #    Step 4: Sharpening (final edge enhancement)
    current = image.copy()

    # Step 1: Spatial noise filtering
    if active_filter in ("gaussian", "median", "bilateral"):
        current = apply_spatial_filtering(current, filter_type=active_filter, kernel_size=active_ksize)

    # Step 2: Gamma intensity transformation
    if abs(active_gamma - 1.0) >= 0.01:
        current = apply_gamma(current, active_gamma)

    # Step 3: Histogram Equalization / CLAHE
    if active_eq_mode in ("global", "clahe"):
        current = apply_equalization(
            current,
            mode=active_eq_mode,
            clip_limit=active_clip_limit,
            tile_size=active_tile_size
        )

    # Step 4: Spatial sharpening
    if active_sharpen > 0.0 or active_filter == "unsharp":
        current = apply_spatial_filtering(
            current,
            filter_type="unsharp",
            kernel_size=active_ksize,
            sharpen_strength=active_sharpen if active_sharpen > 0.0 else 1.0
        )

    enhanced = current

    # 4. Analyze enhanced image characteristics
    enhanced_characteristics = analyze_image_characteristics(enhanced)

    # 5. Compute histograms
    original_hist = compute_histogram_data(image)
    enhanced_hist = compute_histogram_data(enhanced)

    # 6. Check for clean ground-truth reference
    ground_truth_img = None
    ground_truth_name = None
    
    # Check if this input matches one of our controlled benchmark degraded samples
    if filename and any(filename.startswith(p) for p in ["01_", "02_", "03_", "04_", "05_", "07_"]):
        clean_ref_path = settings.SAMPLES_DIR / "enhancement" / "ground_truth" / "clean_reference.png"
        if not clean_ref_path.exists():
            clean_ref_path = settings.SAMPLES_DIR / "enhancement" / "clean_reference.png"
        if clean_ref_path.exists():
            ground_truth_img = cv2.imread(str(clean_ref_path), cv2.IMREAD_GRAYSCALE)
            ground_truth_name = "clean_reference.png (Pristine Cameraman)"

    # Compute MSE, PSNR, SSIM
    quantitative_metrics = evaluate_enhancement_metrics(
        input_img=image,
        enhanced_img=enhanced,
        ground_truth_img=ground_truth_img,
        ground_truth_name=ground_truth_name
    )

    # Encode images to Base64 data URLs for in-memory frontend display
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
