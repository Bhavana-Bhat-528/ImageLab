"""
Comprehensive Verification Suite for Gonzalez & Woods Chapter 3:
Intensity Transformations and Spatial Filtering.
"""

import numpy as np
import cv2
from pathlib import Path

from app.services.enhancement import (
    process_image_enhancement,
    apply_image_negative,
    apply_log_transform,
    apply_power_law,
    apply_contrast_stretching,
    apply_intensity_slicing,
    apply_bit_plane_slicing,
    apply_histogram_equalization,
    apply_histogram_matching,
    apply_local_histogram_processing,
    apply_box_filter,
    apply_weighted_average_filter,
    apply_median_filter,
    apply_max_filter,
    apply_min_filter,
    apply_midpoint_filter,
    apply_alpha_trimmed_mean_filter,
    apply_laplacian_filter,
    apply_unsharp_highboost_filter,
    apply_gradient_filter
)
from app.config import settings


def run_all_chapter3_tests():
    print("=" * 80)
    print("TESTING GONZALEZ & WOODS CHAPTER 3 ENHANCEMENT SUITE")
    print("=" * 80)

    # 1. Synthetic test images
    img_gray = np.linspace(0, 255, 256 * 256, dtype=np.uint8).reshape((256, 256))
    img_color = cv2.merge([img_gray, 255 - img_gray, img_gray // 2])

    # Sample reference image
    ref_path = settings.SAMPLES_DIR / "enhancement" / "01_low_contrast.png"
    ref_img = cv2.imread(str(ref_path)) if ref_path.exists() else img_color

    # --- SECTION 2: INTENSITY TRANSFORMATIONS ---
    print("\n--- 1. Testing Intensity Transformations ---")
    
    # 1.1 Negative
    neg_gray = apply_image_negative(img_gray)
    assert neg_gray[0, 0] == 255, "Negative failed at 0"
    assert neg_gray[255, 255] == 0, "Negative failed at 255"
    print("  ✓ Image Negative verified")

    # 1.2 Log
    log_out = apply_log_transform(img_gray, c=45.98)
    assert log_out.shape == img_gray.shape
    assert log_out[0, 0] == 0
    assert log_out[255, 255] > 200
    print("  ✓ Log Transformation verified")

    # 1.3 Power-Law (Gamma)
    gamma_bright = apply_power_law(img_gray, gamma=0.5)
    gamma_dark = apply_power_law(img_gray, gamma=2.0)
    gamma_id = apply_power_law(img_gray, gamma=1.0)
    assert gamma_bright[128, 128] > img_gray[128, 128], "Gamma < 1 should brighten"
    assert gamma_dark[128, 128] < img_gray[128, 128], "Gamma > 1 should darken"
    assert np.all(gamma_id == img_gray), "Gamma == 1 should be identity"
    print("  ✓ Power-Law (Gamma) Transformation verified (<1, >1, ==1)")

    # 1.4 Piecewise-Linear: Contrast Stretching
    cs = apply_contrast_stretching(img_gray, r1=50, s1=10, r2=200, s2=240)
    assert cs.shape == img_gray.shape
    print("  ✓ Contrast Stretching verified")

    # 1.5 Piecewise-Linear: Intensity Slicing
    slice_bg = apply_intensity_slicing(img_gray, low=100, high=150, preserve_background=True)
    slice_no_bg = apply_intensity_slicing(img_gray, low=100, high=150, preserve_background=False)
    assert slice_bg[125, 125] == 255
    assert slice_no_bg[10, 10] == 0
    print("  ✓ Intensity-Level Slicing verified (preserve BG & zero BG)")

    # 1.6 Piecewise-Linear: Bit-Plane Slicing
    for bit in range(8):
        bp = apply_bit_plane_slicing(img_gray, bit_plane=bit)
        assert set(np.unique(bp)).issubset({0, 255}), f"Bit plane {bit} must be binary"
    print("  ✓ Bit-Plane Slicing verified (all 8 planes 0..7)")

    # --- SECTION 3: HISTOGRAM PROCESSING ---
    print("\n--- 2. Testing Histogram Processing ---")
    he = apply_histogram_equalization(img_gray)
    assert he.shape == img_gray.shape
    print("  ✓ Histogram Equalization verified")

    hm = apply_histogram_matching(img_gray, ref_img)
    assert hm.shape == img_gray.shape
    print("  ✓ Histogram Matching (Specification) verified")

    lhp = apply_local_histogram_processing(img_gray, window_size=8)
    assert lhp.shape == img_gray.shape
    print("  ✓ Local Histogram Processing verified")

    # --- SECTION 4: SMOOTHING FILTERS ---
    print("\n--- 3. Testing Smoothing Spatial Filters ---")
    for k in [3, 5, 7]:
        box = apply_box_filter(img_gray, kernel_size=k)
        w_avg = apply_weighted_average_filter(img_gray, kernel_size=k)
        med = apply_median_filter(img_gray, kernel_size=k)
        max_f = apply_max_filter(img_gray, kernel_size=k)
        min_f = apply_min_filter(img_gray, kernel_size=k)
        mid_f = apply_midpoint_filter(img_gray, kernel_size=k)
        alpha_f = apply_alpha_trimmed_mean_filter(img_gray, kernel_size=k, alpha=0.2)
        assert box.shape == img_gray.shape
        assert w_avg.shape == img_gray.shape
        assert med.shape == img_gray.shape
        assert max_f.shape == img_gray.shape
        assert min_f.shape == img_gray.shape
        assert mid_f.shape == img_gray.shape
        assert alpha_f.shape == img_gray.shape
    print("  ✓ All 7 Smoothing Filters verified across 3x3, 5x5, 7x7 (Box, Weighted Avg, Median, Max, Min, Midpoint, Alpha-Trimmed)")

    # --- SECTION 5: SHARPENING FILTERS ---
    print("\n--- 4. Testing Sharpening Spatial Filters ---")
    lap4 = apply_laplacian_filter(img_gray, mask_type="4_neighbor")
    lap8 = apply_laplacian_filter(img_gray, mask_type="8_neighbor")
    assert lap4.shape == img_gray.shape
    assert lap8.shape == img_gray.shape
    print("  ✓ Laplacian Filter verified (4-neighbor & 8-neighbor)")

    unsharp = apply_unsharp_highboost_filter(img_gray, kernel_size=5, k_factor=1.0)
    highboost = apply_unsharp_highboost_filter(img_gray, kernel_size=5, k_factor=1.8)
    assert unsharp.shape == img_gray.shape
    assert highboost.shape == img_gray.shape
    print("  ✓ Unsharp Masking (k=1) & Highboost Filtering (k=1.8) verified")

    for op in ["roberts", "prewitt", "sobel"]:
        for calc in ["abs_sum", "euclidean"]:
            for mode in ["magnitude", "sharpened"]:
                g_out = apply_gradient_filter(img_gray, operator=op, calc_method=calc, output_mode=mode)
                assert g_out.shape == img_gray.shape
    print("  ✓ Gradient Sharpening verified (Roberts, Prewitt, Sobel; abs_sum & euclidean; magnitude & sharpened)")

    # --- PIPELINE DISPATCH VERIFICATION ---
    print("\n--- 5. Testing Process Pipeline Dispatch Across Categories ---")
    categories = [
        ("intensity_transform", "negative", {}),
        ("intensity_transform", "log", {"log_c": 50.0}),
        ("intensity_transform", "gamma", {"gamma": 0.6}),
        ("intensity_transform", "contrast_stretch", {"contrast_r1": 40, "contrast_r2": 210}),
        ("intensity_transform", "intensity_slicing", {"slice_low": 80, "slice_high": 160}),
        ("intensity_transform", "bit_plane", {"bit_plane": 6}),
        ("histogram_processing", "equalization", {}),
        ("histogram_processing", "matching", {"reference_image": ref_img}),
        ("histogram_processing", "local", {"local_window_size": 16}),
        ("smoothing", "smoothing", {"smoothing_type": "weighted_average", "kernel_size": 5}),
        ("smoothing", "smoothing", {"smoothing_type": "midpoint", "kernel_size": 3}),
        ("smoothing", "smoothing", {"smoothing_type": "alpha_trimmed", "kernel_size": 5, "alpha_trim": 0.25}),
        ("sharpening", "sharpening", {"sharpening_type": "laplacian", "laplacian_mask": "8_neighbor"}),
        ("sharpening", "sharpening", {"sharpening_type": "highboost", "highboost_k": 1.7, "kernel_size": 5}),
        ("sharpening", "sharpening", {"sharpening_type": "gradient", "gradient_operator": "prewitt", "gradient_mode": "sharpened"}),
        ("auto", "auto", {})
    ]

    for cat, op, kwargs in categories:
        res = process_image_enhancement(img_color, category=cat, operation=op, **kwargs)
        assert res["status"] == "success", f"Failed on category {cat}, op {op}"
        assert res["images"]["enhanced_base64"].startswith("data:image/")
        assert "original" in res["histograms"]
        assert "enhanced" in res["histograms"]
        print(f"  ✓ [{cat:22s} : {op:16s}] -> {res['pipeline']['selected_operations'][0]}")

    print("\n" + "=" * 80)
    print("ALL GONZALEZ & WOODS CHAPTER 3 TESTS PASSED PERFECTLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_chapter3_tests()
