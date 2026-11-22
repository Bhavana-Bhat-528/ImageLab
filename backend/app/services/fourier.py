"""
Object Recognition using Fourier Descriptors Service (Module 4).
Implements:
1. Planar contour extraction and uniform arc-length resampling (N=128 points).
2. Complex coordinate representation: z(n) = x(n) + j * y(n).
3. Discrete Fourier Transform (DFT) computation via FFT.
4. Comprehensive invariance normalizations:
   - Translation invariance (zeroing DC component Z(0) = 0).
   - Scale invariance (dividing harmonics by fundamental magnitude max(|Z(1)|, |Z(-1)|)).
   - Rotation and starting-point invariance (taking harmonic magnitude spectrum).
   - Frequency-order weighting (sqrt(k)) to balance high-order polygonal corner frequencies.
5. Low-pass contour reconstruction from truncated Fourier coefficients (IDFT)
   demonstrating harmonic shape smoothing.
6. Nearest-Neighbor classification against a 7-class shape library
   (Circle, Square, Rectangle, Triangle, Star, Hexagon, Trapezium).
7. Validation-based acceptance thresholding for open-set recognition:
   Rejects foreign or irregular shapes as 'Unknown / Unrecognized Shape'.
8. Independent test set evaluation with Confusion Matrix, Accuracy, Precision, Recall, F1-Score,
   and out-of-distribution unknown shape rejection testing.
9. Educational discussion on the effect of coefficient count and boundary-based limitations.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import io
import os
import tempfile
import cv2
import numpy as np

mpl_cache_dir = Path(tempfile.gettempdir()) / "imagelab_mpl"
mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from app.utils.image_io import encode_image_to_base64
from app.config import settings

SHAPE_CATEGORIES = ["circle", "square", "rectangle", "triangle", "star", "hexagon", "trapezium"]

_TRAINING_CACHE: Optional[Dict[str, Any]] = None
_EVALUATION_CACHE: Optional[Dict[str, Any]] = None


def resample_contour(cnt: np.ndarray, num_points: int = 128) -> np.ndarray:
    """
    Resamples a 2D contour to uniformly spaced points along its perimeter.
    Ensures consistent counter-clockwise orientation using signed polygon area.
    """
    if cnt.ndim == 3:
        cnt = cnt.squeeze()
    if cnt.ndim != 2 or len(cnt) < 3:
        raise ValueError("Contour must have at least 3 points.")

    # 1. Enforce consistent counter-clockwise orientation
    # Shoelace signed area: A = 0.5 * sum(x_i * y_{i+1} - x_{i+1} * y_i)
    signed_area = 0.5 * np.sum(cnt[:, 0] * np.roll(cnt[:, 1], -1) - np.roll(cnt[:, 0], -1) * cnt[:, 1])
    if signed_area < 0:
        cnt = cnt[::-1]

    # 2. Close contour loop
    closed_cnt = np.vstack([cnt, cnt[0]])

    # 3. Compute cumulative arc-length distances
    diffs = np.diff(closed_cnt, axis=0)
    segment_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_dists = np.insert(np.cumsum(segment_lengths), 0, 0.0)
    total_perimeter = cum_dists[-1]

    if total_perimeter <= 0:
        raise ValueError("Contour has zero perimeter length.")

    # 4. Interpolate uniformly spaced sample points
    target_dists = np.linspace(0.0, total_perimeter, num_points, endpoint=False)
    rx = np.interp(target_dists, cum_dists, closed_cnt[:, 0])
    ry = np.interp(target_dists, cum_dists, closed_cnt[:, 1])

    return np.column_stack([rx, ry])


def compute_fourier_coefficients(resampled_pts: np.ndarray) -> np.ndarray:
    """
    Computes the Discrete Fourier Transform on 1D complex coordinates:
    z(n) = x(n) + j * y(n)
    Returns Z(k) for k = 0, 1, ..., N-1.
    """
    N = len(resampled_pts)
    z = resampled_pts[:, 0] + 1j * resampled_pts[:, 1]
    # Normalized DFT
    Z = np.fft.fft(z) / float(N)
    return Z


def extract_invariant_descriptors(
    Z: np.ndarray,
    num_descriptors: int = 16,
    normalize_scale: bool = True,
    normalize_rotation: bool = True
) -> np.ndarray:
    """
    Constructs an invariant Fourier Descriptor feature vector:
    - Translation invariance: sets DC component Z(0) = 0.
    - Scale invariance: normalizes by the fundamental harmonic magnitude max(|Z(1)|, |Z(N-1)|).
    - Rotation and starting point invariance: uses magnitude spectrum |Z(k)|.
    - Harmonic weighting: weights harmonics by sqrt(k) to prevent higher-order corner
      frequencies (e.g. k=5, 6, 7 for hexagons and trapezoids) from decaying away.
    """
    N = len(Z)
    Z_work = Z.copy()
    Z_work[0] = 0.0  # Translation invariance: centroid at origin

    # Fundamental harmonic magnitude (semi-major axis of fundamental ellipse)
    f_fundamental = max(abs(Z_work[1]), abs(Z_work[N - 1]))
    if f_fundamental < 1e-7:
        f_fundamental = np.max(np.abs(Z_work)) or 1.0

    if normalize_scale:
        Z_scaled = Z_work / f_fundamental
    else:
        Z_scaled = Z_work

    if normalize_rotation:
        # Phase cancellation: magnitude spectrum
        harmonics = np.abs(Z_scaled)
    else:
        harmonics = np.abs(Z_scaled)

    # Index 0: ratio of minor to major fundamental axis (aspect ratio of fundamental ellipse)
    minor_axis_ratio = min(abs(Z_scaled[1]), abs(Z_scaled[N - 1]))
    desc = [float(minor_axis_ratio)]

    # Symmetric harmonics for k = 2, 3, ...
    k = 2
    while len(desc) < num_descriptors and k < N // 2:
        pos_idx = k % N
        neg_idx = (N - k) % N
        # Weighting by sqrt(k) preserves high-order polygonal corner frequencies
        w = np.sqrt(k)
        desc.append(float(harmonics[pos_idx] * w))
        if len(desc) < num_descriptors:
            desc.append(float(harmonics[neg_idx] * w))
        k += 1

    return np.array(desc[:num_descriptors], dtype=np.float64)


def reconstruct_contour_from_coefficients(
    Z: np.ndarray,
    num_coefficients: int = 16,
    target_points: int = 128
) -> np.ndarray:
    """
    Reconstructs the spatial contour boundary using only the first K Fourier coefficients.
    Demonstrates low-frequency shape smoothing and high-frequency corner restoration.
    """
    N = len(Z)
    Z_truncated = np.zeros_like(Z, dtype=np.complex128)

    # Retain DC offset (centroid)
    Z_truncated[0] = Z[0]

    # Retain lowest K/2 positive and negative harmonic pairs
    half = max(1, min(num_coefficients // 2, N // 2 - 1))
    for k in range(1, half + 1):
        Z_truncated[k] = Z[k]
        Z_truncated[N - k] = Z[N - k]

    # Inverse Fourier Transform
    z_reconstructed = np.fft.ifft(Z_truncated) * float(N)
    pts_rec = np.column_stack([np.real(z_reconstructed), np.imag(z_reconstructed)])
    return pts_rec


def plot_spectrum_chart_base64(descriptors: np.ndarray, num_descriptors: int) -> str:
    """
    Renders a bar chart of the Fourier Descriptor harmonic magnitudes.
    Returns the visualization as an in-memory Base64 PNG image.
    """
    fig, ax = plt.subplots(figsize=(6.5, 3.0), dpi=100)
    fig.patch.set_facecolor('#06090e')
    ax.set_facecolor('#0b0f19')

    indices = np.arange(len(descriptors))
    bars = ax.bar(
        indices,
        descriptors,
        color='#38bdf8',
        edgecolor='#0284c7',
        width=0.65,
        alpha=0.9
    )

    # Highlight the first descriptor (fundamental minor/major ratio)
    if len(bars) > 0:
        bars[0].set_color('#34d399')
        bars[0].set_edgecolor('#10b981')

    ax.set_title(
        f"Weighted Invariant Fourier Descriptors (K = {len(descriptors)} Harmonics)",
        color='#f1f5f9',
        fontsize=10,
        fontweight='bold',
        pad=10
    )
    ax.set_xlabel("Descriptor Index (Harmonic Pair)", color='#94a3b8', fontsize=8)
    ax.set_ylabel("Weighted Amplitude |Z(k)| · √k", color='#94a3b8', fontsize=8)

    ax.tick_params(colors='#94a3b8', labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.15, color='#94a3b8')

    for spine in ax.spines.values():
        spine.set_color('#334155')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buf.seek(0)

    import base64
    b64_str = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"


def get_training_library(num_descriptors: int = 16) -> Dict[str, Any]:
    """
    Loads and caches normalized Fourier Descriptors for the 7-class training dataset.
    Also calculates per-category empirical acceptance thresholds to support
    open-set rejection of unfamiliar or foreign shapes.
    """
    global _TRAINING_CACHE
    if _TRAINING_CACHE is not None and _TRAINING_CACHE.get("num_descriptors") == num_descriptors:
        return _TRAINING_CACHE

    recog_dir = settings.SAMPLES_DIR / "recognition"
    train_samples = []

    if recog_dir.exists():
        for file_path in sorted(recog_dir.glob("*.png")):
            name = file_path.name
            if name.startswith("test_") or "mask" in name or "_gt" in name:
                continue

            category = name.split("_")[0].lower()
            if category not in SHAPE_CATEGORIES:
                continue

            img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            try:
                cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if not cnts:
                    continue
                cnt = max(cnts, key=cv2.contourArea).squeeze()
                pts_resampled = resample_contour(cnt, num_points=128)
                Z = compute_fourier_coefficients(pts_resampled)
                desc = extract_invariant_descriptors(Z, num_descriptors=num_descriptors)

                train_samples.append({
                    "filename": name,
                    "category": category,
                    "descriptor": desc
                })
            except Exception:
                continue

    X_train = np.array([s["descriptor"] for s in train_samples]) if train_samples else np.empty((0, num_descriptors))
    y_train = np.array([s["category"] for s in train_samples]) if train_samples else np.empty(0)

    # Compute validation-based empirical acceptance threshold for each category
    # Calibrated envelope: max(0.065, min(0.38, 1.3 * max_intra_dist))
    # Ensures organic variation (such as natural starfish arm curves and varying lengths)
    # is accepted while keeping tight bounds for regular polygons and rejecting foreign outliers.
    acceptance_thresholds = {}
    for cat in SHAPE_CATEGORIES:
        cat_vecs = [s["descriptor"] for s in train_samples if s["category"] == cat]
        max_intra_dist = 0.0
        if len(cat_vecs) > 1:
            for i in range(len(cat_vecs)):
                for j in range(i + 1, len(cat_vecs)):
                    d = float(np.linalg.norm(cat_vecs[i] - cat_vecs[j]))
                    if d > max_intra_dist:
                        max_intra_dist = d
        acceptance_thresholds[cat] = round(max(0.065, min(0.38, 1.3 * max_intra_dist)), 4)

    _TRAINING_CACHE = {
        "num_descriptors": num_descriptors,
        "samples": train_samples,
        "X_train": X_train,
        "y_train": y_train,
        "acceptance_thresholds": acceptance_thresholds
    }
    return _TRAINING_CACHE


def evaluate_shape_model(num_descriptors: int = 16) -> Dict[str, Any]:
    """
    Evaluates the 1-NN shape classifier on the independent test split across all 7 categories.
    Also tests open-set rejection on out-of-distribution unknown test specimens.
    Computes Confusion Matrix, Precision, Recall, F1-Score, and Rejection Performance.
    """
    global _EVALUATION_CACHE
    if _EVALUATION_CACHE is not None and _EVALUATION_CACHE.get("num_descriptors") == num_descriptors:
        return _EVALUATION_CACHE

    train_data = get_training_library(num_descriptors=num_descriptors)
    X_train = train_data["X_train"]
    y_train = train_data["y_train"]
    thresholds = train_data["acceptance_thresholds"]

    if len(X_train) == 0:
        return {
            "status": "error",
            "message": "Training dataset is empty."
        }

    recog_dir = settings.SAMPLES_DIR / "recognition"
    test_samples = []
    unknown_samples = []

    if recog_dir.exists():
        # 1. Independent in-distribution test specimens (test_<category>_*.png)
        for file_path in sorted(recog_dir.glob("test_*.png")):
            name = file_path.name
            if "unknown" in name:
                continue
            parts = name.split("_")
            if len(parts) < 2:
                continue
            category = parts[1].lower()
            if category not in SHAPE_CATEGORIES:
                continue

            img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            try:
                cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if not cnts:
                    continue
                cnt = max(cnts, key=cv2.contourArea).squeeze()
                pts_resampled = resample_contour(cnt, num_points=128)
                Z = compute_fourier_coefficients(pts_resampled)
                desc = extract_invariant_descriptors(Z, num_descriptors=num_descriptors)

                test_samples.append({
                    "filename": name,
                    "category": category,
                    "descriptor": desc
                })
            except Exception:
                continue

        # 2. Out-of-distribution / unfamiliar shapes (test_unknown_*.png)
        for file_path in sorted(recog_dir.glob("test_unknown_*.png")):
            img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            try:
                cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if not cnts:
                    continue
                cnt = max(cnts, key=cv2.contourArea).squeeze()
                pts_resampled = resample_contour(cnt, num_points=128)
                Z = compute_fourier_coefficients(pts_resampled)
                desc = extract_invariant_descriptors(Z, num_descriptors=num_descriptors)
                unknown_samples.append({
                    "filename": file_path.name,
                    "descriptor": desc
                })
            except Exception:
                continue

    if not test_samples:
        return {
            "status": "warning",
            "message": "Independent test set not found."
        }

    X_test = np.array([s["descriptor"] for s in test_samples])
    y_test = np.array([s["category"] for s in test_samples])

    # Fit 1-NN classifier
    clf = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, labels=SHAPE_CATEGORIES, output_dict=True, zero_division=0)
    conf_mat = confusion_matrix(y_test, y_pred, labels=SHAPE_CATEGORIES)

    per_class = {}
    for cat in SHAPE_CATEGORIES:
        cat_stats = report.get(cat, {})
        per_class[cat] = {
            "precision": round(float(cat_stats.get("precision", 0.0)), 3),
            "recall": round(float(cat_stats.get("recall", 0.0)), 3),
            "f1_score": round(float(cat_stats.get("f1-score", 0.0)), 3),
            "support": int(cat_stats.get("support", 0)),
            "acceptance_threshold": thresholds.get(cat, 0.065)
        }

    macro_avg = report.get("macro avg", {})
    weighted_avg = report.get("weighted avg", {})

    # Evaluate unknown rejection performance
    rejected_unknowns = 0
    for u in unknown_samples:
        dists = [float(np.linalg.norm(u["descriptor"] - train_s["descriptor"])) for train_s in train_data["samples"]]
        min_idx = int(np.argmin(dists))
        min_d = dists[min_idx]
        nearest_cat = train_data["samples"][min_idx]["category"]
        thresh = thresholds.get(nearest_cat, 0.065)
        if min_d > thresh:
            rejected_unknowns += 1

    unknown_rejection_pct = round(
        (rejected_unknowns / len(unknown_samples) * 100.0) if unknown_samples else 100.0, 1
    )

    _EVALUATION_CACHE = {
        "num_descriptors": num_descriptors,
        "train_samples_count": len(X_train),
        "test_samples_count": len(X_test),
        "unknown_samples_tested": len(unknown_samples),
        "unknown_rejection_rate_pct": unknown_rejection_pct,
        "categories": SHAPE_CATEGORIES,
        "overall_accuracy_pct": round(acc * 100.0, 2),
        "macro_precision": round(float(macro_avg.get("precision", 0.0)), 3),
        "macro_recall": round(float(macro_avg.get("recall", 0.0)), 3),
        "macro_f1": round(float(macro_avg.get("f1-score", 0.0)), 3),
        "weighted_f1": round(float(weighted_avg.get("f1-score", 0.0)), 3),
        "per_class_metrics": per_class,
        "confusion_matrix": {
            "labels": SHAPE_CATEGORIES,
            "matrix": conf_mat.tolist()
        }
    }
    return _EVALUATION_CACHE


def recognize_shape_fourier(
    image: np.ndarray,
    num_descriptors: int = 16,
    distance_metric: str = "euclidean",
    normalize_scale: bool = True,
    normalize_rotation: bool = True,
    filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes complete Fourier Descriptor Shape Recognition with Open-Set Rejection:
    1. Preprocesses image: converts to grayscale and Otsu binarization with polarity correction.
    2. Extracts dominant object contour.
    3. Resamples contour uniformly to N=128 points with standardized counter-clockwise orientation.
    4. Computes DFT coefficients z(n) -> Z(k).
    5. Normalizes for translation, scale, and rotation invariance with sqrt(k) harmonic weighting.
    6. Reconstructs contour from truncated coefficients K to visualize smoothing.
    7. Evaluates distance against the 7-class prototype library.
    8. Validates nearest match against the category acceptance threshold:
       Returns 'Unknown / Unrecognized Shape' if distance exceeds threshold.
    9. Returns quantitative distances, top matches, confusion matrix, and educational notes.
    """
    h, w = image.shape[:2]
    is_color = len(image.shape) == 3 and image.shape[2] == 3

    # 1. Grayscale conversion
    if is_color:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        color_src = image.copy()
    else:
        gray = image.copy()
        color_src = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 2. Binarization & polarity detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Auto-detect polarity: background should be black (0), object white (255)
    border_pixels = np.concatenate([binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]])
    if np.mean(border_pixels) > 127:
        binary = cv2.bitwise_not(binary)

    # Morphological noise clean-up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 3. Contour extraction (with safety border padding to prevent accidental edge cropping)
    pad = 4
    padded_binary = cv2.copyMakeBorder(cleaned_binary, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    contours, _ = cv2.findContours(padded_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No distinct object boundary detected in image. Ensure foreground object is visible.")

    dominant_contour = max(contours, key=cv2.contourArea)
    # Shift contour coordinates back to source image coordinate system
    dominant_contour[:, :, 0] = np.clip(dominant_contour[:, :, 0] - pad, 0, w - 1)
    dominant_contour[:, :, 1] = np.clip(dominant_contour[:, :, 1] - pad, 0, h - 1)
    contour_area = float(cv2.contourArea(dominant_contour))
    contour_perimeter = float(cv2.arcLength(dominant_contour, True))

    if len(dominant_contour) < 15 or contour_area < 25.0:
        raise ValueError("Detected boundary has insufficient resolution or area for Fourier descriptor computation.")

    # 4. Uniform arc-length resampling (N = 128)
    resampled_pts = resample_contour(dominant_contour.squeeze(), num_points=128)

    # 5. DFT computation
    Z = compute_fourier_coefficients(resampled_pts)

    # 6. Invariance normalization
    num_desc = max(4, min(int(num_descriptors), 64))
    query_descriptors = extract_invariant_descriptors(
        Z,
        num_descriptors=num_desc,
        normalize_scale=normalize_scale,
        normalize_rotation=normalize_rotation
    )

    # 7. Contour reconstruction from truncated coefficients (K)
    reconstructed_pts = reconstruct_contour_from_coefficients(
        Z,
        num_coefficients=num_desc,
        target_points=128
    )

    # Calculate reconstruction fidelity / energy retention
    mse_reconstruction = float(np.mean(np.sum((resampled_pts - reconstructed_pts) ** 2, axis=1)))
    orig_energy = float(np.sum(np.abs(Z[1:]) ** 2))
    trunc_energy = float(np.sum(np.abs(Z[1:num_desc // 2 + 1]) ** 2) + np.sum(np.abs(Z[-(num_desc // 2):]) ** 2))
    energy_retention_pct = round(min(100.0, (trunc_energy / orig_energy) * 100.0) if orig_energy > 0 else 100.0, 2)

    # 8. Visualizations
    # Overlay extracted original contour (Cyan)
    overlay_contour = color_src.copy()
    cv2.drawContours(overlay_contour, [dominant_contour], -1, (255, 235, 0), 2)  # Cyan

    # Centroid marker
    M = cv2.moments(dominant_contour)
    if M["m00"] > 0:
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        cv2.circle(overlay_contour, (cx, cy), 4, (0, 0, 255), -1)

    # Reconstructed contour comparison (Original in Cyan, Reconstructed in Neon Magenta)
    reconstruction_vis = color_src.copy()
    cv2.drawContours(reconstruction_vis, [dominant_contour], -1, (255, 235, 0), 1)
    pts_int = np.round(reconstructed_pts).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(reconstruction_vis, [pts_int], isClosed=True, color=(255, 0, 255), thickness=2)

    # Spectrum bar chart
    spectrum_chart_b64 = plot_spectrum_chart_base64(query_descriptors, num_desc)

    # 9. Nearest-Neighbor Matching against 7-class training library
    train_data = get_training_library(num_descriptors=num_desc)
    train_samples = train_data["samples"]
    thresholds = train_data["acceptance_thresholds"]

    matches = []
    metric_lower = distance_metric.lower()

    for item in train_samples:
        train_vec = item["descriptor"]
        if metric_lower == "manhattan":
            dist = float(np.sum(np.abs(query_descriptors - train_vec)))
        elif metric_lower == "cosine":
            norm_q = np.linalg.norm(query_descriptors)
            norm_t = np.linalg.norm(train_vec)
            if norm_q > 0 and norm_t > 0:
                cosine_sim = np.dot(query_descriptors, train_vec) / (norm_q * norm_t)
                dist = float(max(0.0, 1.0 - cosine_sim))
            else:
                dist = 1.0
        else:
            # Default Euclidean (L2)
            dist = float(np.linalg.norm(query_descriptors - train_vec))

        matches.append({
            "filename": item["filename"],
            "category": item["category"],
            "distance": round(dist, 4)
        })

    matches.sort(key=lambda m: m["distance"])
    top_matches = matches[:5]

    nearest_candidate_cat = top_matches[0]["category"] if top_matches else "unknown"
    nearest_dist = top_matches[0]["distance"] if top_matches else 999.0
    cand_threshold = thresholds.get(nearest_candidate_cat, 0.065)

    # 10. Open-Set Validation & Rejection Logic
    # Reject as Unknown if distance to closest prototype exceeds calibrated threshold
    is_unknown = nearest_dist > cand_threshold

    if is_unknown:
        recognition_status = "Unknown / Unrecognized Shape"
        predicted_category = "Unknown / Unrecognized Shape"
        calibrated_similarity_pct = 0.0
        status_note = (
            f"Descriptor distance (d = {nearest_dist:.4f}) exceeds the empirical recognition threshold "
            f"(τ = {cand_threshold:.4f}) for nearest category '{nearest_candidate_cat}'. "
            f"The shape is rejected as an unfamiliar or irregular contour."
        )
    else:
        recognition_status = "Recognized"
        predicted_category = nearest_candidate_cat
        # Mathematically calibrated similarity score: linear decay from 100% at d=0 to 0% at d=tau
        calibrated_similarity_pct = round(
            max(0.0, min(100.0, 100.0 * (1.0 - nearest_dist / cand_threshold))), 1
        )
        status_note = (
            f"Successfully recognized as '{predicted_category}'. Descriptor distance (d = {nearest_dist:.4f}) "
            f"is well within the acceptance envelope (τ = {cand_threshold:.4f})."
        )

    # Add similarity percentage to top matches
    for rank, m in enumerate(top_matches):
        m["rank"] = rank + 1
        t = thresholds.get(m["category"], 0.065)
        m["similarity_pct"] = round(max(0.0, min(100.0, 100.0 * (1.0 - m["distance"] / t))), 1)

    # 11. Pre-computed model evaluation report
    evaluation_report = evaluate_shape_model(num_descriptors=num_desc)

    # 12. Educational Limitations Analysis
    limitations = [
        {
            "topic": "Higher-Order Regular Polygon Convergence (Hexagons & Octagons)",
            "observation": "As the number of sides S increases, a regular polygon converges towards a circular silhouette.",
            "impact": "A hexagon's fundamental harmonic carries >96% of its energy. Without harmonic weighting, distance to a circle is small.",
            "mitigation": "Harmonic weighting by sqrt(k) boosts high-frequency corner sensitivity (k=5, 6, 7) for sharp polygonal separation."
        },
        {
            "topic": "Open-Set Unfamiliar Shape Rejection",
            "observation": "Closed-set 1-NN classifiers naively assign arbitrary inputs to whichever training prototype is closest.",
            "impact": "Irregular blobs, crescents, or unfamiliar polygons were previously forced into circle or triangle categories.",
            "mitigation": "Calibrated acceptance thresholds (tau_c) based on training intra-class variance ensure unfamiliar shapes are returned as Unknown."
        },
        {
            "topic": "Internal Markings & Topological Holes",
            "observation": "Fourier descriptors operate strictly on the outer planar boundary curve.",
            "impact": "A solid disk and a hollow ring (washer) of identical outer radius yield identical Fourier descriptors.",
            "mitigation": "Combine external boundary descriptors with regional moment invariants (Hu moments) or topological Euler numbers."
        },
        {
            "topic": "Harmonic Truncation Trade-Off (K)",
            "observation": "Low K (4-8) captures gross aspect ratios; high K (16-32) captures sharp vertices; very high K (>64) admits discretization noise.",
            "impact": "Setting K too low merges hexagons into circles; setting K too high overfits to pixel quantization steps.",
            "mitigation": "Select a balanced coefficient cutoff (K = 16 to 32) matching the specimen boundary complexity."
        }
    ]

    return {
        "status": "success",
        "module": "fourier_descriptors",
        "parameters": {
            "num_descriptors": num_desc,
            "distance_metric": distance_metric,
            "normalize_scale": normalize_scale,
            "normalize_rotation": normalize_rotation
        },
        "shape_properties": {
            "contour_points_count": len(dominant_contour),
            "resampled_points_count": 128,
            "contour_area_px": contour_area,
            "contour_perimeter_px": round(contour_perimeter, 1),
            "circularity": round(float(4.0 * np.pi * contour_area / (contour_perimeter ** 2)), 3) if contour_perimeter > 0 else 0.0,
            "reconstruction_mse": round(mse_reconstruction, 2),
            "energy_retention_pct": energy_retention_pct
        },
        "classification": {
            "recognition_status": recognition_status,
            "is_recognized": not is_unknown,
            "predicted_category": predicted_category,
            "nearest_candidate_category": nearest_candidate_cat,
            "confidence_percentage": calibrated_similarity_pct,
            "best_match_filename": top_matches[0]["filename"] if top_matches else "",
            "nearest_distance": round(nearest_dist, 4),
            "acceptance_threshold": round(cand_threshold, 4),
            "status_note": status_note,
            "top_matches": top_matches
        },
        "evaluation_dataset": evaluation_report,
        "descriptors": [round(float(v), 5) for v in query_descriptors],
        "limitations": limitations,
        "images": {
            "original_base64": encode_image_to_base64(color_src),
            "binary_mask_base64": encode_image_to_base64(cleaned_binary),
            "contour_overlay_base64": encode_image_to_base64(overlay_contour),
            "reconstructed_contour_base64": encode_image_to_base64(reconstruction_vis),
            "spectrum_chart_base64": spectrum_chart_b64
        }
    }
