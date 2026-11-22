"""
Morphological Watershed Segmentation Service.
Implements:
- Full mathematical watershed pipeline:
    1. Grayscale conversion & Gaussian noise suppression
    2. Thresholding (Otsu, Manual, Adaptive) with auto-polarity check
    3. Morphological opening for noise/debris removal
    4. Morphological dilation for sure background
    5. Euclidean distance transformation (cv2.DIST_L2)
    6. Adjustable sure foreground thresholding (ratio * dist.max())
    7. Unknown region calculation: Unknown = Sure_BG - Sure_FG
    8. Connected-component marker generation (cv2.connectedComponentsWithStats)
    9. Morphological watershed execution (cv2.watershed)
   10. Region extraction, area/centroid statistics, and distinct color labeling
- Estimated vs. reference cell count comparison
- Spatial overlap IoU and Dice evaluation against reference masks
- Multi-stage visual pipeline (8 synchronized stages)
- Structured scientific discussion of microscopic watershed limitations
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import colorsys
import cv2
import numpy as np

from app.utils.metrics import compute_iou_dice
from app.utils.image_io import encode_image_to_base64
from app.config import settings

COLOR_PALETTE_BGR = {
    "red": (0, 0, 255),
    "yellow": (0, 235, 255),
    "cyan": (255, 255, 0),
    "green": (0, 255, 0),
    "white": (255, 255, 255),
    "magenta": (255, 0, 255)
}

BENCHMARK_REFERENCE_COUNTS: Dict[str, Dict[str, Any]] = {
    "01_human_mitosis.png": {
        "name": "Human Mitosis Cancer Cells",
        "reference_count": 24,
        "modality": "Phase-Contrast / Fluorescence Microscopy",
        "optimal_dist_ratio": 0.35,
        "description": "Dividing human cancer cells undergoing mitosis with touching nuclear clusters."
    },
    "02_cell_culture.png": {
        "name": "Fluorescent Cell Culture",
        "reference_count": 26,
        "modality": "Confluent Fluorescent Cell Membrane Assay",
        "optimal_dist_ratio": 0.25,
        "description": "Adjacent cell bodies in confluent in-vitro culture requiring distance peak separation."
    },
    "03_protein_transport_nuclei.png": {
        "name": "Nuclear Translocation Assay",
        "reference_count": 22,
        "modality": "Confocal Fluorescence Microscopy",
        "optimal_dist_ratio": 0.35,
        "description": "Compact spatial clusters of fluorescent cell nuclei with localized fluorescence intensities."
    },
    "04_overlapping_coins.png": {
        "name": "Overlapping Particles Benchmark",
        "reference_count": 24,
        "modality": "Macroscopic Optical Imaging Benchmark",
        "optimal_dist_ratio": 0.28,
        "description": "Standard digital image processing benchmark containing exactly 24 touching/overlapping convex disks."
    }
}


def generate_distinct_bgr_colors(n: int) -> List[Tuple[int, int, int]]:
    """
    Generates n visually distinct colors using golden ratio hue spacing.
    """
    colors = []
    golden_ratio_conjugate = 0.618033988749895
    h = 0.1
    for _ in range(n):
        h = (h + golden_ratio_conjugate) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))
    return colors


def segment_cells_watershed(
    image: np.ndarray,
    threshold_method: str = "otsu",
    threshold_val: int = 128,
    blur_ksize: int = 5,
    morph_ksize: int = 3,
    dist_ratio: float = 0.35,
    min_cell_area: int = 25,
    invert_polarity: str = "auto",
    boundary_color: str = "red",
    boundary_thickness: int = 2,
    manual_reference_count: Optional[int] = None,
    filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the complete Morphological Watershed Segmentation pipeline:
    1. Preprocesses image: grayscale conversion and Gaussian smoothing.
    2. Binarizes specimen using Otsu, manual, or adaptive thresholding.
    3. Cleans noise via morphological opening (A o B).
    4. Computes sure background via morphological dilation (A + B).
    5. Computes Euclidean distance transformation on the binary foreground.
    6. Extracts sure foreground by thresholding the distance map at dist_ratio * dist.max().
    7. Computes unknown transition zone: Unknown = Sure_BG - Sure_FG.
    8. Generates connected-component markers on sure foreground seeds.
    9. Applies cv2.watershed to flood the topological surface and establish dividing ridges.
    10. Labels and measures individual segmented cell regions.
    11. Compares estimated cell count against reference counts & ground-truth masks.
    """
    h, w = image.shape[:2]
    total_pixels = h * w
    is_color = len(image.shape) == 3 and image.shape[2] == 3

    # -------------------------------------------------------------------------
    # Stage 1: Preprocessing & Grayscale Noise Reduction
    # -------------------------------------------------------------------------
    if is_color:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Enforce odd blur kernel size
    bk = max(3, blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1)
    blurred = cv2.GaussianBlur(gray, (bk, bk), 0)

    # -------------------------------------------------------------------------
    # Stage 2: Binary Thresholding & Polarity Check
    # -------------------------------------------------------------------------
    method_lower = str(threshold_method).lower()
    applied_threshold_val = float(threshold_val)

    if method_lower == "otsu":
        applied_threshold_val, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    elif method_lower == "adaptive":
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 4
        )
        applied_threshold_val = -1.0  # Locally computed
    else:
        # Manual thresholding
        manual_t = max(0, min(int(threshold_val), 255))
        _, binary = cv2.threshold(blurred, manual_t, 255, cv2.THRESH_BINARY)
        applied_threshold_val = float(manual_t)

    # Determine polarity: cells should be white (255) and background black (0)
    should_invert = False
    if invert_polarity.lower() == "auto":
        border_pixels = np.concatenate([binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]])
        if np.mean(border_pixels) > 127:  # Background is bright white
            should_invert = True
    elif invert_polarity.lower() == "always":
        should_invert = True

    if should_invert:
        binary = cv2.bitwise_not(binary)

    # -------------------------------------------------------------------------
    # Stage 3: Morphological Opening (Noise Removal)
    # -------------------------------------------------------------------------
    mk = max(3, morph_ksize if morph_ksize % 2 == 1 else morph_ksize + 1)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # -------------------------------------------------------------------------
    # Stage 4: Sure Background (Dilation)
    # -------------------------------------------------------------------------
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
    sure_bg = cv2.dilate(opening, kernel_bg, iterations=3)

    # -------------------------------------------------------------------------
    # Stage 5: Euclidean Distance Transformation
    # -------------------------------------------------------------------------
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    max_dist = float(dist_transform.max())

    # Distance map visualization (normalized 0-255 JET heatmap)
    if max_dist > 1e-4:
        norm_dist = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        norm_dist = np.zeros((h, w), dtype=np.uint8)
    dist_heatmap = cv2.applyColorMap(norm_dist, cv2.COLORMAP_JET)

    # -------------------------------------------------------------------------
    # Stage 6: Sure Foreground Extraction (Distance Threshold)
    # -------------------------------------------------------------------------
    active_dist_ratio = max(0.05, min(float(dist_ratio), 0.95))
    fg_thresh_val = active_dist_ratio * max_dist
    _, sure_fg = cv2.threshold(dist_transform, fg_thresh_val, 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)

    # -------------------------------------------------------------------------
    # Stage 7: Unknown Transition Region
    # -------------------------------------------------------------------------
    unknown = cv2.subtract(sure_bg, sure_fg)

    # -------------------------------------------------------------------------
    # Stage 8: Connected Components & Marker Generation
    # -------------------------------------------------------------------------
    num_labels, markers_raw, stats, centroids = cv2.connectedComponentsWithStats(sure_fg)

    # Filter out tiny spurious markers smaller than min_cell_area
    valid_marker_indices = []
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= max(1, min_cell_area // 2):
            valid_marker_indices.append(lbl)

    # Re-index markers sequentially: background is 1, valid cell seeds are 2, 3, ...
    markers = np.zeros((h, w), dtype=np.int32)
    markers.fill(1)  # Default all pixels to background (1)

    for new_id, old_id in enumerate(valid_marker_indices, start=2):
        markers[markers_raw == old_id] = new_id

    # Mark unknown boundaries as 0 (unlabeled watershed basins)
    markers[unknown == 255] = 0

    # -------------------------------------------------------------------------
    # Stage 9: Morphological Watershed Algorithm Execution
    # -------------------------------------------------------------------------
    color_src = image.copy() if is_color else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    markers_after_ws = cv2.watershed(color_src, markers.copy())

    # -------------------------------------------------------------------------
    # Stage 10: Region Analysis, Cell Counting & Visual Overlay
    # -------------------------------------------------------------------------
    # Identified cells have markers > 1
    unique_cells = [int(lbl) for lbl in np.unique(markers_after_ws) if lbl > 1]
    estimated_cell_count = len(unique_cells)

    # Generate distinct colors for each cell
    distinct_colors = generate_distinct_bgr_colors(max(estimated_cell_count, 1))

    # Construct colored labeled regions overlay
    colored_cells_mask = np.zeros_like(color_src)
    cell_details = []

    for idx, cell_lbl in enumerate(unique_cells):
        color = distinct_colors[idx % len(distinct_colors)]
        region_pixels = (markers_after_ws == cell_lbl)
        colored_cells_mask[region_pixels] = color

        area_px = int(np.count_nonzero(region_pixels))
        
        # Calculate cell centroid and contours
        reg_u8 = np.uint8(region_pixels) * 255
        contours, _ = cv2.findContours(reg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        perimeter = 0.0
        circularity = 0.0
        if contours:
            perimeter = float(cv2.arcLength(contours[0], True))
            if perimeter > 0:
                circularity = round(float(4.0 * np.pi * area_px / (perimeter ** 2)), 3)
                circularity = min(1.0, circularity)

        # Centroid calculation
        M = cv2.moments(reg_u8)
        if M["m00"] > 0:
            cx = round(float(M["m10"] / M["m00"]), 1)
            cy = round(float(M["m01"] / M["m00"]), 1)
        else:
            cx, cy = 0.0, 0.0

        eq_diameter = round(2.0 * np.sqrt(area_px / np.pi), 1)

        hex_color = f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"
        cell_details.append({
            "cell_id": idx + 1,
            "id": idx + 1,
            "area_pixels": area_px,
            "area_px": area_px,
            "centroid": {"x": cx, "y": cy},
            "centroid_x": cx,
            "centroid_y": cy,
            "equivalent_diameter_px": eq_diameter,
            "circularity": circularity,
            "color_hex": hex_color
        })

    # Filter out tiny cell fragments if requested
    if min_cell_area > 0:
        cell_details = [c for c in cell_details if c["area_pixels"] >= min_cell_area]
        estimated_cell_count = len(cell_details)

    # Blend colored mask onto original image (alpha blending)
    alpha = 0.50
    overlay = cv2.addWeighted(color_src, 1.0 - alpha, colored_cells_mask, alpha, 0)

    # Highlight watershed boundary lines (markers == -1)
    bnd_color = COLOR_PALETTE_BGR.get(boundary_color.lower(), (0, 0, 255))
    thick = max(1, min(int(boundary_thickness), 5))

    bnd_mask = (markers_after_ws == -1).astype(np.uint8) * 255
    if thick > 1:
        th_k = thick if thick % 2 == 1 else thick + 1
        disp_bnd = cv2.dilate(bnd_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (th_k, th_k)))
    else:
        disp_bnd = bnd_mask

    overlay[disp_bnd > 0] = bnd_color

    # Add numeric labels at cell centroids on the overlay
    for cell in cell_details:
        cx = int(cell["centroid"]["x"])
        cy = int(cell["centroid"]["y"])
        if 0 <= cx < w and 0 <= cy < h:
            cv2.putText(
                overlay,
                str(cell["cell_id"]),
                (cx - 6, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                overlay,
                str(cell["cell_id"]),
                (cx - 6, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

    # Markers map visualization: seeds in distinct colors, unknown in dark slate
    markers_vis = np.zeros_like(color_src)
    markers_vis[unknown == 255] = (40, 40, 50)  # Dark slate for unknown
    for idx, cell_lbl in enumerate(unique_cells):
        color = distinct_colors[idx % len(distinct_colors)]
        markers_vis[markers == cell_lbl] = color

    # Statistical Area Distribution
    areas = [c["area_pixels"] for c in cell_details]
    mean_area = round(float(np.mean(areas)), 1) if areas else 0.0
    std_area = round(float(np.std(areas)), 1) if areas else 0.0
    min_area = int(np.min(areas)) if areas else 0
    max_area = int(np.max(areas)) if areas else 0
    total_fg_pixels = int(np.sum(areas))
    fg_area_pct = round((total_fg_pixels / total_pixels) * 100.0, 2)

    # -------------------------------------------------------------------------
    # Stage 11: Reference Count & Ground-Truth Comparison
    # -------------------------------------------------------------------------
    reference_count = None
    ref_source = "None"
    
    # 1. Check manual reference override
    if manual_reference_count is not None and manual_reference_count > 0:
        reference_count = int(manual_reference_count)
        ref_source = "User Manually Annotated Count"
    # 2. Check pre-calibrated benchmark catalog
    elif filename and filename in BENCHMARK_REFERENCE_COUNTS:
        reference_count = BENCHMARK_REFERENCE_COUNTS[filename]["reference_count"]
        ref_source = f"Benchmark Reference ({BENCHMARK_REFERENCE_COUNTS[filename]['name']})"

    count_comparison = {
        "has_reference": reference_count is not None,
        "reference_count": reference_count,
        "estimated_count": estimated_cell_count,
        "reference_source": ref_source,
        "count_difference": None,
        "difference": None,
        "absolute_error": None,
        "percentage_error": None,
        "status": "No Reference Specified",
        "status_color": "blue",
        "notes": None
    }

    if reference_count is not None and reference_count > 0:
        diff = estimated_cell_count - reference_count
        abs_err = abs(diff)
        pct_err = round((abs_err / reference_count) * 100.0, 2)

        if diff == 0:
            status = "Exact Match (100% Count Accuracy)"
            status_color = "green"
            notes = "Segmented cell count matches the ground-truth benchmark exactly."
        elif abs_err <= 2 or pct_err <= 10.0:
            status = "High Accuracy (Minor Variation)"
            status_color = "blue"
            notes = "Count is within standard inter-observer manual hemocytometer counting tolerance."
        elif diff > 0:
            status = f"Over-Segmentation (+{diff} extra regions)"
            status_color = "amber"
            notes = "Internal cell organelle intensity variations or low distance threshold generated extra markers. Increase distance ratio or min cell area."
        else:
            status = f"Under-Segmentation ({diff} missing cells)"
            status_color = "amber"
            notes = "Adjacent cells merged into single basins or threshold ratio excluded smaller cells. Lower distance ratio or blur kernel."

        count_comparison.update({
            "count_difference": diff,
            "difference": diff,
            "absolute_error": abs_err,
            "percentage_error": pct_err,
            "status": status,
            "status_color": status_color,
            "notes": notes
        })

    # Ground-truth segmentation mask comparison (IoU & Dice)
    gt_comparison = {
        "has_ground_truth": False,
        "reference_mask_name": None,
        "ground_truth_name": None,
        "iou": 0.0,
        "dice": 0.0,
        "overlap_pixels": 0,
        "ground_truth_pixels": 0,
        "interpretation": "No reference mask available for custom upload. Region morphology reported above."
    }

    if filename:
        gt_filename = f"{Path(filename).stem}_gt_mask.png"
        gt_path = settings.SAMPLES_DIR / "watershed" / "ground_truth" / gt_filename
        if not gt_path.exists():
            gt_path = settings.SAMPLES_DIR / "watershed" / gt_filename

        if gt_path.exists():
            gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None:
                # Segmented foreground mask: all regions where markers_after_ws > 1
                seg_fg_mask = np.where(markers_after_ws > 1, 255, 0).astype(np.uint8)
                if gt_mask.shape[:2] != (h, w):
                    gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                iou_dice = compute_iou_dice(seg_fg_mask, gt_mask)
                gt_comparison = {
                    "has_ground_truth": True,
                    "reference_mask_name": gt_filename,
                    "ground_truth_name": gt_filename,
                    "iou": iou_dice["iou"],
                    "dice": iou_dice["dice"],
                    "overlap_pixels": iou_dice["intersection_pixels"],
                    "ground_truth_pixels": iou_dice["ground_truth_pixels"],
                    "interpretation": (
                        f"Evaluated against reference segmentation mask. IoU: {iou_dice['iou']:.4f}, "
                        f"Dice: {iou_dice['dice']:.4f}. Measures spatial boundary agreement with annotated cell bodies."
                    )
                }

    # -------------------------------------------------------------------------
    # Stage 12: Base64 Encoding for In-Memory Web Display
    # -------------------------------------------------------------------------
    orig_b64 = encode_image_to_base64(image)
    prep_b64 = encode_image_to_base64(blurred)
    binary_b64 = encode_image_to_base64(opening)
    dist_b64 = encode_image_to_base64(dist_heatmap)
    sure_fg_b64 = encode_image_to_base64(sure_fg)
    sure_bg_b64 = encode_image_to_base64(sure_bg)
    markers_b64 = encode_image_to_base64(markers_vis)
    overlay_b64 = encode_image_to_base64(overlay)

    # -------------------------------------------------------------------------
    # Stage 13: Educational Limitation Analysis
    # -------------------------------------------------------------------------
    limitations = [
        {
            "category": "Lighting & Field Inhomogeneity",
            "observation": "Microscope condenser vignetting causes radial brightness drop towards image borders.",
            "impact": "Global Otsu binarization may miss dim cells near peripheral borders or falsely segment dark corners.",
            "mitigation": "Use adaptive Gaussian thresholding or background shading normalization (rolling-ball filter)."
        },
        {
            "category": "Translucent Membranes & Low Contrast",
            "observation": "Thin intercellular membranes have faint photometric contrast with adjacent cytoplasm.",
            "impact": "Weak membranes blur away during smoothing, merging adjacent cell bodies prior to distance mapping.",
            "mitigation": "Apply gentle contrast-limited adaptive equalization (CLAHE) or reduce Gaussian blur kernel."
        },
        {
            "category": "Heavy 3D Overlap vs 2D Projection",
            "observation": "Cells overlapping along the optical z-axis project onto a single 2D plane.",
            "impact": "Lacking geometric curvature necking, two overlapping cells generate only a single local distance peak.",
            "mitigation": "Increase distance threshold ratio (e.g. 0.45 - 0.60) to isolate deeper interior peaks."
        },
        {
            "category": "Over-Segmentation vs. Under-Segmentation",
            "observation": "Marker extraction is extremely sensitive to distance transform threshold ratio.",
            "impact": "Low ratios (< 0.20) cause over-segmentation; high ratios (> 0.55) eliminate smaller valid cells.",
            "mitigation": "Calibrate threshold ratio against expected specimen cell diameter."
        }
    ]

    return {
        "status": "success",
        "module": "watershed_segmentation",
        "parameters": {
            "threshold_method": threshold_method,
            "threshold_value": round(applied_threshold_val, 1),
            "blur_kernel_size": bk,
            "morphological_kernel_size": mk,
            "distance_threshold_ratio": active_dist_ratio,
            "min_cell_area": min_cell_area,
            "invert_polarity": invert_polarity,
            "boundary_color": boundary_color,
            "boundary_thickness": thick
        },
        "results": {
            "estimated_cell_count": estimated_cell_count,
            "image_dimensions": f"{w}x{h}",
            "total_pixels": total_pixels,
            "total_foreground_pixels": total_fg_pixels,
            "foreground_area_percentage": fg_area_pct,
            "max_distance_transform_px": round(max_dist, 1),
            "sure_fg_threshold_value": round(fg_thresh_val, 2),
            "statistics": {
                "mean_cell_area_px": mean_area,
                "std_cell_area_px": std_area,
                "min_cell_area_px": min_area,
                "max_cell_area_px": max_area
            },
            "cell_details": cell_details[:100]  # Cap at top 100 for network payload
        },
        "evaluation": {
            "count_comparison": count_comparison,
            "ground_truth_segmentation": gt_comparison
        },
        "limitations": limitations,
        "images": {
            "original_base64": orig_b64,
            "preprocessed_base64": prep_b64,
            "binary_mask_base64": binary_b64,
            "distance_transform_base64": dist_b64,
            "sure_fg_base64": sure_fg_b64,
            "sure_bg_base64": sure_bg_b64,
            "markers_map_base64": markers_b64,
            "watershed_overlay_base64": overlay_b64
        }
    }
