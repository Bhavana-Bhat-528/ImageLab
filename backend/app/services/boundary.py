"""
Morphological Boundary Extraction Service.
Implements:
- Foreground animal segmentation using OpenCV GrabCut with user-adjustable bounding boxes (and Otsu fallback).
- Morphological boundary extraction: beta(A) = A - (A (-) B) using cv2.erode and cv2.subtract.
- Structuring element shapes (Rectangle, Ellipse, Cross), sizes (3x3 to 15x15+), and erosion iterations.
- Visual overlays with configurable boundary color and display thickness.
- Quantitative evaluation: foreground pixels, boundary pixels, connectivity, and IoU/Dice against ground-truth.
- Cross-configuration comparison across structuring elements.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

from app.utils.metrics import compute_iou_dice
from app.utils.image_io import encode_image_to_base64
from app.config import settings

COLOR_MAP_BGR = {
    "cyan": (255, 229, 0),       # High-visibility Neon Cyan
    "yellow": (0, 234, 255),     # High-visibility Yellow
    "magenta": (255, 0, 255),    # Vibrant Magenta
    "green": (0, 230, 118),      # Lime Green
    "red": (0, 0, 255),          # Pure Red
    "white": (255, 255, 255)     # White
}

SE_SHAPE_MAP = {
    "rect": cv2.MORPH_RECT,
    "rectangle": cv2.MORPH_RECT,
    "ellipse": cv2.MORPH_ELLIPSE,
    "cross": cv2.MORPH_CROSS
}


# ---------------------------------------------------------------------------
# 1. Foreground Animal Segmentation
# ---------------------------------------------------------------------------

def segment_animal_foreground(
    image: np.ndarray,
    method: str = "grabcut",
    box_coords: Optional[Tuple[int, int, int, int]] = None,
    margin_pct: float = 6.0,
    iterations: int = 5
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Segments animal foreground from complex natural background.
    Uses GrabCut with bounding-box initialization to isolate non-trivial silhouettes.
    
    Returns:
        (binary_mask, preprocessed_visualization, applied_rect)
    """
    h, w = image.shape[:2]

    # Resolve bounding box
    if box_coords and len(box_coords) == 4 and box_coords[2] > 10 and box_coords[3] > 10:
        bx, by, bw, bh = box_coords
        # Clamp to image dimensions
        bx = max(1, min(bx, w - 2))
        by = max(1, min(by, h - 2))
        bw = max(10, min(bw, w - bx - 1))
        bh = max(10, min(bh, h - by - 1))
        rect = (bx, by, bw, bh)
    else:
        # Default margin percentage around image boundary
        mx = int(w * (margin_pct / 100.0))
        my = int(h * (margin_pct / 100.0))
        rect = (mx, my, w - 2 * mx, h - 2 * my)

    # Preprocessed visualization: image with bounding box annotation
    prep_vis = image.copy()
    cv2.rectangle(
        prep_vis,
        (rect[0], rect[1]),
        (rect[0] + rect[2], rect[1] + rect[3]),
        (0, 255, 0),
        2
    )

    if method.lower() == "grabcut":
        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        iters = max(1, min(int(iterations), 10))
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iters, cv2.GC_INIT_WITH_RECT)

        # Foreground is definite foreground (1) or probable foreground (3)
        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

        # Morphological post-cleanup to fill minor fur pinholes and remove single-pixel specks
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, clean_kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, clean_kernel)

    elif method.lower() == "otsu":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, fg_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # If border is predominantly foreground, invert mask
        border_mean = (np.mean(fg_mask[0, :]) + np.mean(fg_mask[-1, :]) + np.mean(fg_mask[:, 0]) + np.mean(fg_mask[:, -1])) / 4.0
        if border_mean > 127:
            fg_mask = cv2.bitwise_not(fg_mask)

    else:
        # Simple adaptive Gaussian thresholding
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        fg_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 4)

    return fg_mask, prep_vis, rect


# ---------------------------------------------------------------------------
# 2. Morphological Boundary Extraction: beta(A) = A - (A (-) B)
# ---------------------------------------------------------------------------

def extract_boundary(
    binary_mask: np.ndarray,
    se_shape: str = "rect",
    se_size: int = 3,
    iterations: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts outer boundary using morphological set difference:
        beta(A) = A - (A (-) B)
    
    Returns:
        (eroded_mask, boundary_mask, structuring_element)
    """
    # Enforce odd size at least 3
    k = max(3, se_size if se_size % 2 == 1 else se_size + 1)
    iters = max(1, min(int(iterations), 5))

    shape_flag = SE_SHAPE_MAP.get(se_shape.lower(), cv2.MORPH_RECT)
    se = cv2.getStructuringElement(shape_flag, (k, k))

    # Morphological Erosion: A (-) B
    eroded = cv2.erode(binary_mask, se, iterations=iters)

    # Set Subtraction: beta(A) = A - (A (-) B)
    boundary = cv2.subtract(binary_mask, eroded)

    return eroded, boundary, se


# ---------------------------------------------------------------------------
# 3. Boundary Visual Overlay
# ---------------------------------------------------------------------------

def create_boundary_overlay(
    original_image: np.ndarray,
    boundary_mask: np.ndarray,
    color_name: str = "cyan",
    display_thickness: int = 1
) -> np.ndarray:
    """
    Overlays the extracted boundary on the original photograph.
    Allows boundary color selection and display thickness dilation.
    """
    color = COLOR_MAP_BGR.get(color_name.lower(), (255, 229, 0))
    overlay = original_image.copy()

    # Dilation for enhanced visual inspection thickness if requested
    if display_thickness > 1:
        th_k = display_thickness if display_thickness % 2 == 1 else display_thickness + 1
        disp_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (th_k, th_k))
        disp_boundary = cv2.dilate(boundary_mask, disp_se)
    else:
        disp_boundary = boundary_mask

    # Overlay colored pixels where boundary is active
    overlay[disp_boundary > 0] = color
    return overlay


# ---------------------------------------------------------------------------
# 4. Structuring Element Cross-Benchmarking
# ---------------------------------------------------------------------------

def benchmark_structuring_elements(binary_mask: np.ndarray) -> List[Dict[str, Any]]:
    """
    Evaluates multiple structuring element shapes and sizes on the same mask
    to quantify the geometric effect on boundary thickness and pixel counts.
    """
    configs = [
        {"shape": "rect", "shape_label": "Rectangle", "size": 3},
        {"shape": "ellipse", "shape_label": "Ellipse / Disk", "size": 3},
        {"shape": "cross", "shape_label": "Cross", "size": 3},
        {"shape": "rect", "shape_label": "Rectangle", "size": 5},
        {"shape": "ellipse", "shape_label": "Ellipse / Disk", "size": 5},
        {"shape": "cross", "shape_label": "Cross", "size": 5},
        {"shape": "rect", "shape_label": "Rectangle", "size": 7},
        {"shape": "ellipse", "shape_label": "Ellipse / Disk", "size": 7},
        {"shape": "cross", "shape_label": "Cross", "size": 7},
    ]

    results = []
    total_fg = int(np.count_nonzero(binary_mask))

    for cfg in configs:
        k = cfg["size"]
        shape_flag = SE_SHAPE_MAP.get(cfg["shape"], cv2.MORPH_RECT)
        se = cv2.getStructuringElement(shape_flag, (k, k))
        eroded = cv2.erode(binary_mask, se, iterations=1)
        bnd = cv2.subtract(binary_mask, eroded)
        bnd_pixels = int(np.count_nonzero(bnd))
        ratio = round((bnd_pixels / total_fg * 100.0), 2) if total_fg > 0 else 0.0

        # Number of connected components in the boundary
        num_labels, _ = cv2.connectedComponents(bnd)
        boundary_components = max(0, num_labels - 1)  # subtract background

        results.append({
            "shape": cfg["shape"],
            "shape_label": cfg["shape_label"],
            "size": f"{k}x{k}",
            "boundary_pixels": bnd_pixels,
            "boundary_to_fg_ratio_pct": ratio,
            "connected_components": boundary_components
        })

    return results


# ---------------------------------------------------------------------------
# 5. Core Pipeline Orchestrator
# ---------------------------------------------------------------------------

def extract_animal_boundary(
    image: np.ndarray,
    se_shape: str = "rect",
    se_size: int = 3,
    iterations: int = 1,
    segmentation_method: str = "grabcut",
    box_x: Optional[int] = None,
    box_y: Optional[int] = None,
    box_w: Optional[int] = None,
    box_h: Optional[int] = None,
    margin_pct: float = 6.0,
    grabcut_iters: int = 5,
    overlay_color: str = "cyan",
    display_thickness: int = 1,
    filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes complete Morphological Boundary Extraction pipeline:
    1. Segments animal foreground (GrabCut with adjustable bounding box).
    2. Applies mathematical erosion: A (-) B.
    3. Performs set subtraction: beta(A) = A - (A (-) B).
    4. Generates high-visibility photograph overlay.
    5. Calculates topological metrics and IoU/Dice against ground-truth.
    6. Benchmarks SE shape and size variations.
    """
    h, w = image.shape[:2]
    total_pixels = h * w

    # Assemble user-provided bounding box if valid
    custom_box = None
    if box_x is not None and box_y is not None and box_w is not None and box_h is not None:
        if box_w > 10 and box_h > 10:
            custom_box = (int(box_x), int(box_y), int(box_w), int(box_h))

    # 1. Segment foreground
    fg_mask, prep_vis, applied_rect = segment_animal_foreground(
        image=image,
        method=segmentation_method,
        box_coords=custom_box,
        margin_pct=margin_pct,
        iterations=grabcut_iters
    )

    # 2. Extract boundary using morphological operation
    eroded_mask, boundary_mask, se = extract_boundary(
        binary_mask=fg_mask,
        se_shape=se_shape,
        se_size=se_size,
        iterations=iterations
    )

    # 3. Create visual overlay on original photograph
    overlay = create_boundary_overlay(
        original_image=image,
        boundary_mask=boundary_mask,
        color_name=overlay_color,
        display_thickness=display_thickness
    )

    # 4. Calculate quantitative measurements
    fg_pixels = int(np.count_nonzero(fg_mask))
    bnd_pixels = int(np.count_nonzero(boundary_mask))
    eroded_pixels = int(np.count_nonzero(eroded_mask))
    fg_percentage = round((fg_pixels / total_pixels) * 100.0, 2)
    bnd_ratio = round((bnd_pixels / fg_pixels * 100.0), 2) if fg_pixels > 0 else 0.0

    # Connected components
    num_fg_labels, _, _, _ = cv2.connectedComponentsWithStats(fg_mask)
    num_bnd_labels, _, _, _ = cv2.connectedComponentsWithStats(boundary_mask)
    fg_components = max(0, num_fg_labels - 1)
    bnd_components = max(0, num_bnd_labels - 1)

    # 5. Ground-truth evaluation (if matching GT mask exists on disk)
    gt_comparison = {
        "has_ground_truth": False,
        "reference_name": None,
        "iou": None,
        "dice": None,
        "interpretation": "No reference mask available for custom upload. Component and pixel counts reported below."
    }

    if filename:
        gt_filename = f"{Path(filename).stem}_gt_mask.png"
        gt_path = settings.SAMPLES_DIR / "boundary" / gt_filename
        if gt_path.exists():
            gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None:
                iou_dice = compute_iou_dice(fg_mask, gt_mask)
                gt_comparison = {
                    "has_ground_truth": True,
                    "reference_name": gt_filename,
                    "iou": iou_dice["iou"],
                    "dice": iou_dice["dice"],
                    "overlap_pixels": iou_dice["intersection_pixels"],
                    "ground_truth_pixels": iou_dice["ground_truth_pixels"],
                    "interpretation": (
                        f"Segmentation evaluated against reference mask. IoU: {iou_dice['iou']:.4f}, "
                        f"Dice: {iou_dice['dice']:.4f}. Measures boundary agreement with true animal silhouette."
                    )
                }

    # 6. Benchmark structuring element combinations
    se_benchmarks = benchmark_structuring_elements(fg_mask)

    # 7. Encode images to Base64 data URLs for in-memory frontend display
    orig_b64 = encode_image_to_base64(image)
    prep_b64 = encode_image_to_base64(prep_vis)
    fg_b64 = encode_image_to_base64(fg_mask)
    eroded_b64 = encode_image_to_base64(eroded_mask)
    boundary_b64 = encode_image_to_base64(boundary_mask)
    overlay_b64 = encode_image_to_base64(overlay)

    # 8. Analytical explanation of structuring element impacts
    se_shape_effect = {
        "rect": "Rectangle (8-connected): Produces thicker diagonal boundaries and square-like corners; aggressive on fine acute features.",
        "ellipse": "Ellipse/Disk: Produces rotationally uniform (isotropic) boundary contours; smooths angular vertices evenly.",
        "cross": "Cross (4-connected): Preserves acute diagonal corners and fine protruding details (ears, whiskers, legs); produces thinner boundaries."
    }.get(se_shape.lower(), "Standard geometric structuring element.")

    return {
        "status": "success",
        "module": "boundary_extraction",
        "parameters": {
            "se_shape": se_shape,
            "se_size": se_size,
            "iterations": iterations,
            "segmentation_method": segmentation_method,
            "bounding_box": {
                "x": applied_rect[0],
                "y": applied_rect[1],
                "width": applied_rect[2],
                "height": applied_rect[3]
            },
            "overlay_color": overlay_color,
            "display_thickness": display_thickness
        },
        "metrics": {
            "image_dimensions": f"{w}x{h}",
            "total_pixels": total_pixels,
            "foreground_pixels": fg_pixels,
            "foreground_percentage": fg_percentage,
            "eroded_pixels": eroded_pixels,
            "boundary_pixels": bnd_pixels,
            "boundary_to_foreground_ratio_pct": bnd_ratio,
            "foreground_connected_components": fg_components,
            "boundary_connected_components": bnd_components,
            "ground_truth_evaluation": gt_comparison
        },
        "se_benchmarks": se_benchmarks,
        "explanation": {
            "mathematical_formula": "beta(A) = A - (A (-) B)",
            "structuring_element_effect": se_shape_effect,
            "size_effect": f"Size {se_size}x{se_size} with {iterations} iteration(s) removes an outer band of {se_size // 2 * iterations} pixel(s).",
            "metric_accuracy_caveat": (
                "IMPORTANT: Component count and pixel count alone DO NOT prove segmentation accuracy. "
                "A large foreground count can contain erroneous background regions (grass, shadows), and a single "
                "connected component may falsely fuse separate objects. True segmentation accuracy requires "
                "spatial overlap metrics such as IoU (Jaccard Index) or Dice coefficient against a reference ground truth."
            )
        },
        "images": {
            "original_base64": orig_b64,
            "preprocessed_base64": prep_b64,
            "binary_mask_base64": fg_b64,
            "eroded_mask_base64": eroded_b64,
            "boundary_mask_base64": boundary_b64,
            "overlay_base64": overlay_b64
        }
    }
