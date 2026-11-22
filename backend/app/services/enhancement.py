"""
Automatic Image Enhancement Service.
Implements:
- Quantitative image characteristic analysis (mean, std, contrast, noise, sharpness, clipping).
- Adaptive decision mechanism that selects and tunes operations based on detected features.
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

    # Impulse noise check: count pixels that differ heavily from 3x3 median
    med = cv2.medianBlur(gray, 3)
    diff = np.abs(gray.astype(np.int32) - med.astype(np.int32))
    extreme_pixels = ((gray <= 2) | (gray >= 253)) & (diff > 50)
    sp_ratio = float(np.mean(extreme_pixels))
    has_salt_pepper = sp_ratio > 0.015

    return round(sigma, 2), has_salt_pepper


def estimate_sharpness(gray: np.ndarray) -> float:
    """
    Estimates spatial image sharpness using the variance of the Laplacian:
    Var(Delta I) = mean((Delta I - mean(Delta I))^2)
    """
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    var = float(lap.var())
    return round(var, 2)


def analyze_image_characteristics(image: np.ndarray) -> Dict[str, Any]:
    """
    Analyzes input image characteristics:
    - Mean luminance and standard deviation
    - Contrast and dynamic range
    - Histogram distribution, skewness, entropy
    - Shadow and highlight clipping
    - Noise level and sharpness
    """
    is_color = len(image.shape) == 3 and image.shape[2] == 3
    if is_color:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
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

    shadow_clipping_pct = round(float(np.sum(hist[:8]) / total_pixels) * 100.0, 2)
    highlight_clipping_pct = round(float(np.sum(hist[248:]) / total_pixels) * 100.0, 2)

    # Skewness: measures asymmetric shift of intensity mass
    if std_val > 1e-3:
        skewness = float(np.mean(((gray - mean_val) / std_val) ** 3))
    else:
        skewness = 0.0

    # Noise & Sharpness
    noise_sigma, has_salt_pepper = estimate_noise_level(gray)
    sharpness_score = estimate_sharpness(gray)

    # Diagnostics categorization
    is_dark = mean_val < 85.0 or shadow_clipping_pct > 18.0
    is_bright = mean_val > 175.0 or highlight_clipping_pct > 18.0
    is_low_contrast = std_val < 42.0 and dynamic_range < 120
    is_noisy = noise_sigma > 14.0 or has_salt_pepper
    is_blurry = sharpness_score < 110.0 and not is_noisy

    primary_defects = []
    if is_dark:
        primary_defects.append("Severe Underexposure / Shadow Compression")
    elif is_bright:
        primary_defects.append("Severe Overexposure / Saturated Highlights")
    if is_low_contrast:
        primary_defects.append("Narrow Dynamic Range / Low Contrast")
    if has_salt_pepper:
        primary_defects.append("High-Impulse Salt & Pepper Noise")
    elif is_noisy:
        primary_defects.append(f"Significant Sensor Noise (Sigma={noise_sigma})")
    if is_blurry:
        primary_defects.append(f"Low Spatial Acuity / Edge Softness")

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
        "is_color": is_color,
        "diagnostics": {
            "is_dark": is_dark,
            "is_bright": is_bright,
            "is_low_contrast": is_low_contrast,
            "is_noisy": is_noisy,
            "has_salt_pepper": has_salt_pepper,
            "is_blurry": is_blurry,
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
    - Gamma expansion for dark images
    - Compressive gamma for overexposed images
    - CLAHE / Equalization for narrow dynamic range
    - Median / Gaussian filter only when noise is present
    - Unsharp masking only when sharpness is low AND noise is low/suppressed
    """
    diag = characteristics["diagnostics"]
    mean_val = characteristics["mean_intensity"]
    noise_sigma = characteristics["noise_sigma"]
    has_sp = characteristics["has_salt_pepper_noise"]
    sharpness = characteristics["sharpness_score"]

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

    # Stage A: Noise Filtering (must be applied BEFORE contrast enhancement or sharpening)
    if has_sp:
        params["spatial_filter"] = "median"
        params["kernel_size"] = 3
        selected_ops.append("Non-Linear Median Filtering (3x3)")
        justifications.append("Detected salt-and-pepper impulse noise; median filtering eliminates isolated outliers without blurring step edges.")
        expected_improvements.append("Elimination of salt-and-pepper pixel spikes.")
        potential_risks.append("Slight rounding of sharp 90-degree corners if kernel size exceeds 3.")
    elif diag["is_noisy"]:
        params["spatial_filter"] = "bilateral" if characteristics["is_color"] else "gaussian"
        params["kernel_size"] = 5 if noise_sigma > 25.0 else 3
        selected_ops.append(f"{params['spatial_filter'].capitalize()} Smoothing ({params['kernel_size']}x{params['kernel_size']})")
        justifications.append(f"Detected additive noise (sigma={noise_sigma}); smoothing suppresses high-frequency noise floor before contrast adjustment.")
        expected_improvements.append("Clean uniform regions and elevated SNR.")
        potential_risks.append("Minor reduction in fine texture fidelity.")

    # Stage B: Intensity Transformation (Gamma Correction)
    if diag["is_dark"]:
        # Gamma < 1.0 expands compressed shadow regions
        # Adaptive formula: lower mean -> stronger expansion (lower gamma)
        computed_gamma = float(np.clip(mean_val / 140.0, 0.40, 0.85))
        params["gamma"] = round(computed_gamma, 2)
        selected_ops.append(f"Power-Law Gamma Expansion (gamma={params['gamma']})")
        justifications.append(f"Low mean luminance ({mean_val}) and shadow clipping; gamma expansion brightens dark regions while keeping max white anchored.")
        expected_improvements.append("Recovered shadow visibility and dynamic range expansion.")
        potential_risks.append("May amplify noise in extreme deep shadows if noise suppression is not applied.")
    elif diag["is_bright"]:
        # Gamma > 1.0 compresses blown-out highlights
        computed_gamma = float(np.clip(255.0 / (255.0 - mean_val + 30.0), 1.30, 2.20))
        params["gamma"] = round(computed_gamma, 2)
        selected_ops.append(f"Power-Law Gamma Compression (gamma={params['gamma']})")
        justifications.append(f"High mean luminance ({mean_val}); compressive gamma recovers midtone distinction from washed-out highlights.")
        expected_improvements.append("Highlight tone recovery and contrast restoration in bright regions.")
        potential_risks.append("Dark midtones may become slightly dimmer.")

    # Stage C: Contrast Equalization (CLAHE / Equalization)
    if diag["is_low_contrast"]:
        params["equalization_mode"] = "clahe"
        params["clahe_clip_limit"] = 3.0
        params["clahe_tile_grid_size"] = 8
        selected_ops.append("Contrast Limited Adaptive Histogram Equalization (CLAHE)")
        justifications.append("Narrow histogram dispersion (std < 42); CLAHE operates on localized 8x8 tiles to redistribute local gradients without over-amplifying noise.")
        expected_improvements.append("Enhanced local edge contrast and feature visibility across all image regions.")
        potential_risks.append("Excessive clip limits could cause unnatural local intensity gradients.")
    elif not diag["is_dark"] and not diag["is_bright"] and characteristics["std_intensity"] < 50.0:
        # Mild contrast enhancement
        params["equalization_mode"] = "clahe"
        params["clahe_clip_limit"] = 1.8
        params["clahe_tile_grid_size"] = 8
        selected_ops.append("Mild Adaptive Contrast Equalization (CLAHE)")
        justifications.append("Slight contrast deficit; mild CLAHE optimizes dynamic range utilization.")
        expected_improvements.append("Subtle improvement in textural depth.")

    # Stage D: Spatial Sharpening (Only if not noisy or if noise was cleanly handled)
    if diag["is_blurry"] and not has_sp and noise_sigma < 18.0:
        params["sharpen_strength"] = 1.2
        selected_ops.append("Unsharp Masking Spatial Sharpening (strength=1.2)")
        justifications.append(f"Low edge acuity (sharpness={sharpness}) and low noise floor; high-boost unsharp masking restores edge gradient slopes.")
        expected_improvements.append("Crisper object outlines and heightened structural acuity.")
        potential_risks.append("Overshoot halos around high-contrast step boundaries if strength is set too high.")

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
    """Applies power-law intensity transformation: s = 255 * (r / 255)^gamma."""
    if abs(gamma - 1.0) < 1e-3:
        return image.copy()
    
    # Precompute lookup table for high performance
    inv_gamma = 1.0 / max(gamma, 0.05)
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255.0
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
    - 'gaussian': Gaussian smoothing
    - 'median': Median filtering (removes impulse noise)
    - 'bilateral': Edge-preserving bilateral filter
    - 'unsharp': Unsharp masking [I + alpha * (I - I_blur)]
    - 'laplacian': Laplacian high-pass sharpening
    """
    # Ensure kernel_size is odd and at least 3
    k = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    
    out = image.copy()

    if filter_type == "gaussian":
        out = cv2.GaussianBlur(out, (k, k), 0)
    elif filter_type == "median":
        out = cv2.medianBlur(out, k)
    elif filter_type == "bilateral":
        sigma_color = 50.0
        sigma_space = float(k) * 2.0
        out = cv2.bilateralFilter(out, k, sigma_color, sigma_space)

    # Apply sharpening if requested
    if sharpen_strength > 0.0:
        if filter_type == "unsharp" or sharpen_strength > 0.0:
            blurred = cv2.GaussianBlur(out, (k, k), 0)
            # Unsharp mask formula: I_sharp = I + strength * (I - I_blurred)
            diff = out.astype(np.float32) - blurred.astype(np.float32)
            sharpened = out.astype(np.float32) + (sharpen_strength * diff)
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
    - Analyzes image characteristics.
    - Decides or accepts parameters.
    - Executes order-controlled transformations.
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
            sharpen_strength=max(active_sharpen, 0.5)
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
    if filename and any(filename.startswith(p) for p in ["01_", "02_", "03_", "04_", "05_"]):
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
