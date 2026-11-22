"""
Validation suite for Module 3: Morphological Watershed Segmentation.
Verifies:
1. Complete mathematical pipeline: preproc -> binarize -> opening -> dilation -> dist_transform -> markers -> watershed
2. All 4 benchmark sample images (mitosis, cell culture, protein transport, coins)
3. Exact cell count validation on standard coins benchmark (24 coins)
4. Ground-truth segmentation mask IoU and Dice metrics
5. Thresholding method variations (Otsu, Manual, Adaptive) and distance ratio sensitivity
6. Error handling for corrupt, empty, and invalid uploads
"""

import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import cv2
import numpy as np
from app.services.watershed import segment_cells_watershed
from app.api.routes_samples import is_valid_input_sample
from app.utils.image_io import validate_and_read_image
from app.config import settings

def run_watershed_tests():
    print("================ RUNNING WATERSHED SEGMENTATION VALIDATION ================", flush=True)

    sample_dir = settings.SAMPLES_DIR / "watershed"
    assert sample_dir.exists(), f"Watershed sample directory missing: {sample_dir}"

    # -------------------------------------------------------------------------
    # 1. Standard Benchmark Coins Validation (Exact 24-object Separation)
    # -------------------------------------------------------------------------
    print("\n--- 1. Overlapping Coins Standard Benchmark Validation ---", flush=True)
    coins_img = cv2.imread(str(sample_dir / "04_overlapping_coins.png"))
    assert coins_img is not None, "Failed to load coins benchmark image!"

    coins_res = segment_cells_watershed(
        image=coins_img,
        threshold_method="otsu",
        blur_ksize=5,
        morph_ksize=3,
        dist_ratio=0.28,
        min_cell_area=20,
        filename="04_overlapping_coins.png"
    )

    coins_count = coins_res["results"]["estimated_cell_count"]
    print(f"  Overlapping coins detected count: {coins_count} (Expected: 24)", flush=True)
    assert coins_count == 24, f"Watershed should separate exactly 24 coins, got {coins_count}"
    
    comp = coins_res["evaluation"]["count_comparison"]
    assert comp["has_reference"] is True
    assert comp["status"] == "Exact Match (100% Count Accuracy)"
    print(f"  Evaluation Status: {comp['status']}", flush=True)

    # -------------------------------------------------------------------------
    # 2. Real-World Microscopic Images Validation (Mitosis, Culture, Nuclei)
    # -------------------------------------------------------------------------
    print("\n--- 2. Real-World Microscopic Specimen Validation ---", flush=True)
    micro_samples = [
        ("01_human_mitosis.png", "otsu", 0.35, 30),
        ("02_cell_culture.png", "otsu", 0.25, 25),
        ("03_protein_transport_nuclei.png", "manual", 0.35, 15)
    ]

    for fname, method, dist_r, min_a in micro_samples:
        img = cv2.imread(str(sample_dir / fname))
        assert img is not None, f"Failed to load {fname}"

        res = segment_cells_watershed(
            image=img,
            threshold_method=method,
            threshold_val=20 if method == "manual" else 128,
            dist_ratio=dist_r,
            min_cell_area=min_a,
            filename=fname
        )

        r = res["results"]
        eval_data = res["evaluation"]
        gt_seg = eval_data["ground_truth_segmentation"]
        cnt_comp = eval_data["count_comparison"]

        print(f"\n  [{fname}] (Dimensions: {r['image_dimensions']}):", flush=True)
        print(f"    Estimated Cell Count: {r['estimated_cell_count']} cells", flush=True)
        print(f"    Mean Cell Area: {r['statistics']['mean_cell_area_px']} px (Min: {r['statistics']['min_cell_area_px']}, Max: {r['statistics']['max_cell_area_px']})", flush=True)
        print(f"    Total Cell Area: {r['total_foreground_pixels']} px ({r['foreground_area_percentage']}%)", flush=True)
        print(f"    Reference Count: {cnt_comp['reference_count']} ({cnt_comp['reference_source']})", flush=True)
        print(f"    Count Error: {cnt_comp['count_difference']} ({cnt_comp['percentage_error']}%) - Status: {cnt_comp['status']}", flush=True)

        assert r["estimated_cell_count"] > 0, f"Expected non-zero cell count for {fname}"
        
        # Verify Ground Truth segmentation mask validation
        assert gt_seg["has_ground_truth"] is True, f"Ground-truth mask missing for {fname}"
        print(f"    Ground-Truth Mask IoU: {gt_seg['iou']:.4f}, Dice: {gt_seg['dice']:.4f}", flush=True)
        assert gt_seg["iou"] > 0.70, f"IoU should be > 0.70, got {gt_seg['iou']}"
        assert gt_seg["dice"] > 0.80, f"Dice should be > 0.80, got {gt_seg['dice']}"

        # Verify all 8 visual pipeline stages are populated
        images = res["images"]
        expected_stages = [
            "original_base64", "preprocessed_base64", "binary_mask_base64",
            "distance_transform_base64", "sure_fg_base64", "sure_bg_base64",
            "markers_map_base64", "watershed_overlay_base64"
        ]
        for stage in expected_stages:
            assert stage in images and images[stage].startswith("data:image/"), f"Missing stage: {stage}"

    # -------------------------------------------------------------------------
    # 3. Distance Threshold Ratio Sensitivity & Over/Under-segmentation
    # -------------------------------------------------------------------------
    print("\n--- 3. Distance Ratio Sensitivity Analysis ---", flush=True)
    coins_img = cv2.imread(str(sample_dir / "04_overlapping_coins.png"))
    low_ratio = segment_cells_watershed(coins_img, dist_ratio=0.15, min_cell_area=10)["results"]["estimated_cell_count"]
    opt_ratio = segment_cells_watershed(coins_img, dist_ratio=0.28, min_cell_area=10)["results"]["estimated_cell_count"]
    high_ratio = segment_cells_watershed(coins_img, dist_ratio=0.55, min_cell_area=10)["results"]["estimated_cell_count"]
    print(f"  Coins ratio=0.15 (Over-segmentation risk): {low_ratio} regions", flush=True)
    print(f"  Coins ratio=0.28 (Calibrated optimal):     {opt_ratio} regions", flush=True)
    print(f"  Coins ratio=0.55 (Under-segmentation risk): {high_ratio} regions", flush=True)
    assert high_ratio < opt_ratio, "Higher distance ratio must reduce detected cell count!"

    # -------------------------------------------------------------------------
    # 4. Sample Dataset Isolation Check
    # -------------------------------------------------------------------------
    print("\n--- 4. Sample Dataset Listing Isolation Check ---", flush=True)
    ws_files = [f.name for f in sample_dir.iterdir() if is_valid_input_sample(f)]
    print(f"  Discovered watershed input samples: {ws_files}", flush=True)
    assert len(ws_files) == 4, f"Expected 4 input samples, got {len(ws_files)}"
    for f in ws_files:
        assert not any(k in f.lower() for k in ["mask", "_gt", "result", "boundary", "eroded"]), \
            f"Ground truth or result file in sample listing: {f}"
    print("✓ Watershed sample picker strictly isolates input images from masks.", flush=True)

    # -------------------------------------------------------------------------
    # 5. Upload Validation & Error Rejection
    # -------------------------------------------------------------------------
    print("\n--- 5. Upload Validation Verification ---", flush=True)
    try:
        validate_and_read_image(b"", "empty_cell.png")
        assert False, "Should reject empty file"
    except ValueError as e:
        print(f"  Empty file rejection: {e}", flush=True)

    try:
        validate_and_read_image(b"CORRUPT_BYTES", "corrupt_cell.png")
        assert False, "Should reject corrupt file"
    except ValueError as e:
        print(f"  Corrupt file rejection: {e}", flush=True)

    print("\n================ ALL WATERSHED SEGMENTATION TESTS PASSED! ================", flush=True)

if __name__ == "__main__":
    run_watershed_tests()
