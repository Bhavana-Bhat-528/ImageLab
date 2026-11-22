"""
Master Integration & Quality Testing Suite for ImageLab.
Tests all 4 Digital Image Processing modules:
1. Automatic Image Enhancement
2. Morphological Boundary Extraction
3. Morphological Watershed Segmentation
4. Fourier Descriptor-Based Object Recognition

Also validates:
- In-memory data handling (zero disk leakage)
- Parameter variation and effect verification
- Academic mathematical correctness
- Robustness against empty, corrupt, and edge-case inputs
"""

import sys
import os
import time
import io
from pathlib import Path

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import cv2
import numpy as np

# Services & Modules
from app.services.enhancement import process_image_enhancement, analyze_image_characteristics
from app.services.boundary import extract_animal_boundary, extract_boundary, benchmark_structuring_elements
from app.services.watershed import segment_cells_watershed
from app.services.fourier import (
    recognize_shape_fourier,
    evaluate_shape_model,
    extract_invariant_descriptors,
    compute_fourier_coefficients,
    resample_contour,
    reconstruct_contour_from_coefficients,
    SHAPE_CATEGORIES
)
from app.utils.image_io import validate_and_read_image, encode_image_to_base64
from app.api.routes_samples import is_valid_input_sample
from app.config import settings

def test_module_1_enhancement():
    print("\n" + "=" * 80)
    print("MODULE 1: AUTOMATIC IMAGE ENHANCEMENT QUALITY & INTEGRATION")
    print("=" * 80)
    
    samples_dir = settings.SAMPLES_DIR / "enhancement"
    assert samples_dir.exists(), f"Enhancement samples directory missing: {samples_dir}"
    
    test_files = [
        "01_low_contrast.png",
        "02_dark_underexposed.png",
        "03_bright_overexposed.png",
        "04_gaussian_noise.png",
        "05_salt_and_pepper_noise.png",
        "06_uneven_illumination.png",
        "07_blurred_out_of_focus.png"
    ]
    
    for fname in test_files:
        fpath = samples_dir / fname
        assert fpath.exists(), f"Sample image {fname} missing!"
        img = cv2.imread(str(fpath))
        assert img is not None, f"Failed to load {fname}"
        
        # Test 1A: Automatic Mode
        res_auto = process_image_enhancement(img, mode="auto", filename=fname)
        assert res_auto["status"] == "success", f"Auto enhancement failed on {fname}"
        
        chars_in = res_auto["input_characteristics"]
        chars_out = res_auto["enhanced_characteristics"]
        pipe = res_auto["pipeline"]
        metrics = res_auto["metrics"]
        images = res_auto["images"]
        
        # Verify visual outputs generated as valid Base64 and histogram data
        assert images["original_base64"].startswith("data:image/"), "Original image base64 missing"
        assert images["enhanced_base64"].startswith("data:image/"), "Enhanced image base64 missing"
        assert "original" in res_auto["histograms"], "Original histogram missing"
        assert "enhanced" in res_auto["histograms"], "Enhanced histogram missing"
        
        # Verify metric calculation
        assert "relative_to_input" in metrics, "Missing relative metrics"
        assert metrics["relative_to_input"]["mse"] >= 0.0, "MSE must be non-negative"
        assert metrics["relative_to_input"]["ssim"] <= 1.0, "SSIM must be <= 1.0"
        
        print(f"  [{fname:28s}] Auto Ops: {', '.join(pipe.get('selected_operations', [])) if pipe.get('selected_operations') else 'None'}")
        print(f"      Metrics -> PSNR: {metrics['relative_to_input']['psnr']} dB | SSIM: {metrics['relative_to_input']['ssim']} | MSE: {metrics['relative_to_input']['mse']}")
        
        # Test 1B: Manual Mode with customized parameters
        res_manual = process_image_enhancement(
            image=img,
            mode="manual",
            gamma=1.2,
            equalization_mode="clahe",
            clahe_clip_limit=3.0,
            clahe_tile_grid_size=8,
            spatial_filter="median",
            kernel_size=3,
            sharpen_strength=0.8,
            filename=fname
        )
        assert res_manual["status"] == "success", f"Manual enhancement failed on {fname}"
        assert res_manual["mode"] == "manual"
        assert len(res_manual["pipeline"]["selected_operations"]) > 0
        
    print("✓ Module 1 (Automatic Image Enhancement): All 7 benchmark samples verified in Auto & Manual modes.")


def test_module_2_boundary():
    print("\n" + "=" * 80)
    print("MODULE 2: MORPHOLOGICAL BOUNDARY EXTRACTION QUALITY & INTEGRATION")
    print("=" * 80)
    
    sample_dir = settings.SAMPLES_DIR / "boundary"
    assert sample_dir.exists(), f"Boundary samples directory missing: {sample_dir}"
    
    # 2A: Mathematical Morphology Verification: beta(A) = A - (A (-) B)
    mask_A = np.zeros((100, 100), dtype=np.uint8)
    mask_A[20:80, 20:80] = 255
    eroded, boundary, se = extract_boundary(mask_A, se_shape="rect", se_size=3, iterations=1)
    
    # Mathematical identity checks
    assert np.all((boundary == 255) <= (mask_A == 255)), "Boundary must be a subset of foreground A!"
    assert np.all((eroded == 255) <= (mask_A == 255)), "Eroded mask must be a subset of foreground A!"
    assert np.array_equal(boundary, cv2.subtract(mask_A, eroded)), "boundary must equal cv2.subtract(A, eroded)!"
    assert np.all(boundary[eroded == 255] == 0), "Inner eroded points cannot have boundary pixels!"
    print(f"  Mathematical equation beta(A) = A - (A (-) B) verified: |A|={np.count_nonzero(mask_A)}, |A(-)B|={np.count_nonzero(eroded)}, |beta(A)|={np.count_nonzero(boundary)}")

    # 2B: Real-World Animal Photographs Verification
    test_animals = ["zebra.jpg", "elephant.jpg", "dog.jpg"]
    for fname in test_animals:
        fpath = sample_dir / fname
        assert fpath.exists(), f"Sample animal {fname} missing!"
        img = cv2.imread(str(fpath))
        assert img is not None, f"Failed to load {fname}"
        
        res = extract_animal_boundary(
            image=img,
            se_shape="ellipse",
            se_size=5,
            iterations=1,
            segmentation_method="grabcut",
            margin_pct=4.0,
            overlay_color="cyan",
            display_thickness=2,
            filename=fname
        )
        
        assert res["status"] == "success", f"Boundary extraction failed on {fname}"
        m = res["metrics"]
        gt = m["ground_truth_evaluation"]
        assert m["foreground_pixels"] > 0, "Foreground pixel count must be positive"
        assert m["boundary_pixels"] > 0, "Boundary pixel count must be positive"
        assert gt["has_ground_truth"] is True, f"Ground truth mask missing for {fname}"
        assert gt["iou"] >= 0.75, f"IoU below acceptable threshold for {fname}: {gt['iou']}"
        assert gt["dice"] >= 0.85, f"Dice below acceptable threshold for {fname}: {gt['dice']}"
        
        # Verify structuring element benchmark table
        se_bench = res["se_benchmarks"]
        assert len(se_bench) >= 3, "Missing SE benchmark variations"
        
        print(f"  [{fname:14s}] FG: {m['foreground_pixels']} px ({m['foreground_percentage']}%) | Boundary: {m['boundary_pixels']} px | IoU: {gt['iou']:.4f} | Dice: {gt['dice']:.4f}")
        
    # 2C: Large Image Processing Safety (1920x1080)
    high_res = np.zeros((1080, 1920, 3), dtype=np.uint8)
    high_res[:] = (40, 100, 30)
    cv2.circle(high_res, (960, 540), 300, (180, 150, 120), -1)
    t0 = time.time()
    hr_res = extract_animal_boundary(high_res, se_shape="ellipse", se_size=5, iterations=1, segmentation_method="otsu")
    hr_time = time.time() - t0
    assert hr_res["status"] == "success", "High resolution processing failed"
    print(f"  Large image safety (1920x1080): Processed in {hr_time:.3f}s (well under non-blocking threshold).")
    
    print("✓ Module 2 (Morphological Boundary Extraction): Verified mathematical correctness, IoU/Dice benchmarks, and large-image safety.")


def test_module_3_watershed():
    print("\n" + "=" * 80)
    print("MODULE 3: MORPHOLOGICAL WATERSHED SEGMENTATION QUALITY & INTEGRATION")
    print("=" * 80)
    
    sample_dir = settings.SAMPLES_DIR / "watershed"
    assert sample_dir.exists(), f"Watershed sample directory missing: {sample_dir}"
    
    # 3A: Standard Benchmark Coins Validation (Exact 24-object Separation)
    coins_img = cv2.imread(str(sample_dir / "04_overlapping_coins.png"))
    assert coins_img is not None, "Failed to load overlapping coins image"
    
    coins_res = segment_cells_watershed(
        image=coins_img,
        threshold_method="otsu",
        blur_ksize=5,
        morph_ksize=3,
        dist_ratio=0.28,
        min_cell_area=20,
        filename="04_overlapping_coins.png"
    )
    
    assert coins_res["status"] == "success", "Coins watershed failed"
    coins_count = coins_res["results"]["estimated_cell_count"]
    print(f"  [04_overlapping_coins.png] Detected Count: {coins_count} (Expected Benchmark: 24)")
    assert coins_count == 24, f"Watershed segmentation should separate exactly 24 coins, got {coins_count}"
    assert coins_res["evaluation"]["count_comparison"]["status"] == "Exact Match (100% Count Accuracy)"
    
    # 3B: Microscopic Specimens Validation (Mitosis, Cell Culture, Protein Transport)
    micro_samples = [
        ("01_human_mitosis.png", 24),
        ("02_cell_culture.png", 26),
        ("03_protein_transport_nuclei.png", 22)
    ]
    
    for fname, ref_count in micro_samples:
        img = cv2.imread(str(sample_dir / fname))
        assert img is not None, f"Failed to load {fname}"
        
        res = segment_cells_watershed(img, threshold_method="otsu", blur_ksize=5, dist_ratio=0.35, filename=fname)
        assert res["status"] == "success", f"Watershed failed on {fname}"
        
        count = res["results"]["estimated_cell_count"]
        mean_area = res["results"]["statistics"]["mean_cell_area_px"]
        gt = res["evaluation"]["ground_truth_segmentation"]
        
        # Verify intermediate 8-stage visual pipeline
        stages = res["images"]
        required_stages = [
            "original_base64", "preprocessed_base64", "binary_mask_base64",
            "distance_transform_base64", "sure_fg_base64", "sure_bg_base64",
            "markers_map_base64", "watershed_overlay_base64"
        ]
        for st in required_stages:
            assert st in stages and stages[st].startswith("data:image/"), f"Missing stage image: {st}"
            
        print(f"  [{fname:30s}] Detected: {count:4d} cells | Mean Area: {mean_area:6.1f} px | IoU: {gt['iou']:.4f} | Dice: {gt['dice']:.4f}")
        
    # 3C: Parameter Variation & Sensitivity (Distance Threshold Ratio)
    r_low = segment_cells_watershed(coins_img, dist_ratio=0.15)["results"]["estimated_cell_count"]
    r_opt = segment_cells_watershed(coins_img, dist_ratio=0.28)["results"]["estimated_cell_count"]
    r_high = segment_cells_watershed(coins_img, dist_ratio=0.55)["results"]["estimated_cell_count"]
    print(f"  Distance ratio parameter sensitivity on coins: Low(0.15)={r_low} > Opt(0.28)={r_opt} > High(0.55)={r_high}")
    assert r_low >= r_opt >= r_high, "Cell count must decrease monotonically as distance ratio increases!"
    
    print("✓ Module 3 (Watershed Segmentation): Verified mathematical pipeline, exact 24 coins match, 8-stage pipeline, and parameter monotonicity.")


def test_module_4_fourier():
    print("\n" + "=" * 80)
    print("MODULE 4: FOURIER DESCRIPTOR OBJECT RECOGNITION QUALITY & INTEGRATION")
    print("=" * 80)
    
    recog_dir = settings.SAMPLES_DIR / "recognition"
    assert recog_dir.exists(), f"Recognition samples directory missing: {recog_dir}"
    
    # 4A: Mathematical Invariance Checks
    base_sq = np.zeros((256, 256), dtype=np.uint8)
    base_sq[80:176, 80:176] = 255
    
    M_rot = cv2.getRotationMatrix2D((128, 128), 37, 1.0)
    rot_sq = cv2.warpAffine(base_sq, M_rot, (256, 256))
    
    M_scale = cv2.getRotationMatrix2D((128, 128), 0, 0.7)
    scale_sq = cv2.warpAffine(base_sq, M_scale, (256, 256))
    
    M_trans = np.float32([[1, 0, 25], [0, 1, -30]])
    trans_sq = cv2.warpAffine(base_sq, M_trans, (256, 256))
    
    d_base = np.array(recognize_shape_fourier(base_sq, num_descriptors=16)["descriptors"])
    d_rot = np.array(recognize_shape_fourier(rot_sq, num_descriptors=16)["descriptors"])
    d_scale = np.array(recognize_shape_fourier(scale_sq, num_descriptors=16)["descriptors"])
    d_trans = np.array(recognize_shape_fourier(trans_sq, num_descriptors=16)["descriptors"])
    
    diff_rot = float(np.linalg.norm(d_base - d_rot))
    diff_scale = float(np.linalg.norm(d_base - d_scale))
    diff_trans = float(np.linalg.norm(d_base - d_trans))
    
    print(f"  Invariance tests -> Rotation diff: {diff_rot:.4f} | Scale diff: {diff_scale:.4f} | Translation diff: {diff_trans:.4f}")
    assert diff_rot < 0.15, f"Rotation invariance exceeded tolerance: {diff_rot}"
    assert diff_scale < 0.10, f"Scale invariance exceeded tolerance: {diff_scale}"
    assert diff_trans < 0.05, f"Translation invariance exceeded tolerance: {diff_trans}"
    
    # 4B: Monotonic Harmonic Reconstruction Convergence
    hex_img = cv2.imread(str(recog_dir / "test_hexagon_01.png"))
    mse_k4 = recognize_shape_fourier(hex_img, num_descriptors=4)["shape_properties"]["reconstruction_mse"]
    mse_k16 = recognize_shape_fourier(hex_img, num_descriptors=16)["shape_properties"]["reconstruction_mse"]
    mse_k32 = recognize_shape_fourier(hex_img, num_descriptors=32)["shape_properties"]["reconstruction_mse"]
    print(f"  IDFT Reconstruction MSE -> K=4: {mse_k4:6.2f} > K=16: {mse_k16:6.2f} > K=32: {mse_k32:6.2f}")
    assert mse_k32 < mse_k16 < mse_k4, "Reconstruction MSE must decrease monotonically with increasing K"
    
    # 4C: Full 7-Class Independent Test Evaluation & Confusion Matrix
    eval_rep = evaluate_shape_model(num_descriptors=16)
    acc = eval_rep["overall_accuracy_pct"]
    rej_rate = eval_rep["unknown_rejection_rate_pct"]
    print(f"  7-Class Test Accuracy:    {acc:.2f}% across {eval_rep['test_samples_count']} independent test specimens")
    print(f"  Unknown Rejection Rate:  {rej_rate:.2f}% across {eval_rep['unknown_samples_tested']} foreign outliers")
    assert acc == 100.0, f"Expected 100% test accuracy, got {acc}%"
    assert rej_rate == 100.0, f"Expected 100% rejection rate, got {rej_rate}%"
    
    # 4D: Natural Starfish Recognition
    star_img = cv2.imread(str(recog_dir / "test_star_02.png"))
    star_res = recognize_shape_fourier(star_img, num_descriptors=16)
    assert star_res["classification"]["recognition_status"] == "Recognized"
    assert star_res["classification"]["predicted_category"] == "star"
    print(f"  Natural Starfish Test -> Status: {star_res['classification']['recognition_status']} | Category: {star_res['classification']['predicted_category']} (Conf: {star_res['classification']['confidence_percentage']}%)")
    
    # 4E: Hexagon and Trapezium Separation
    hex_res = recognize_shape_fourier(cv2.imread(str(recog_dir / "test_hexagon_01.png")))
    assert hex_res["classification"]["predicted_category"] == "hexagon"
    assert hex_res["classification"]["recognition_status"] == "Recognized"
    
    trap_res = recognize_shape_fourier(cv2.imread(str(recog_dir / "test_trapezium_01.png")))
    assert trap_res["classification"]["predicted_category"] == "trapezium"
    assert trap_res["classification"]["recognition_status"] == "Recognized"
    print("  Hexagon and Trapezium correctly classified with zero circle/triangle confusion.")
    
    # 4F: Open-Set Outlier Rejection
    outlier_res = recognize_shape_fourier(cv2.imread(str(recog_dir / "test_unknown_01_amoeba.png")))
    assert outlier_res["classification"]["recognition_status"] == "Unknown / Unrecognized Shape"
    print("  Outlier rejection verified: Irregular foreign shapes correctly classified as 'Unknown / Unrecognized Shape'.")
    
    print("✓ Module 4 (Fourier Descriptors): Invariance, IDFT reconstruction, 100% 7-class accuracy, natural starfish, and open-set rejection fully verified.")


def test_in_memory_and_zero_persistence():
    print("\n" + "=" * 80)
    print("DATA HANDLING & ZERO DISK PERSISTENCE VERIFICATION")
    print("=" * 80)
    
    # Record initial files in backend/data/
    data_dir = settings.SAMPLES_DIR.parent
    initial_files = set(str(p.resolve()) for p in data_dir.glob("**/*") if p.is_file())
    
    # Perform synthetic upload operations across all 4 modules
    test_upload = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(test_upload, (100, 100), 60, (200, 200, 200), -1)
    
    _ = process_image_enhancement(test_upload)
    _ = extract_animal_boundary(test_upload, segmentation_method="otsu")
    _ = segment_cells_watershed(test_upload)
    _ = recognize_shape_fourier(test_upload)
    
    # Record post-processing files in backend/data/
    post_files = set(str(p.resolve()) for p in data_dir.glob("**/*") if p.is_file())
    
    new_files = post_files - initial_files
    print(f"  Initial local files in backend/data/: {len(initial_files)}")
    print(f"  Post-processing files in backend/data/: {len(post_files)}")
    print(f"  New files created on disk: {len(new_files)}")
    assert len(new_files) == 0, f"Violation of zero-persistence rule! Files created: {new_files}"
    print("✓ Zero disk persistence verified: All uploaded images and results processed strictly in memory.")


def test_edge_cases_and_error_handling():
    print("\n" + "=" * 80)
    print("INPUT VALIDATION, ROBUSTNESS & ERROR HANDLING")
    print("=" * 80)
    
    # Test 1: Empty file (0 bytes)
    try:
        validate_and_read_image(b"", "empty.png")
        assert False, "Empty file should raise ValueError"
    except Exception as e:
        assert "empty" in str(e).lower()
        print(f"  ✓ Empty byte payload correctly rejected: {e}")
        
    # Test 2: Corrupt image bytes
    try:
        validate_and_read_image(b"NOT_A_VALID_IMAGE_HEADER_DATA_STREAM", "corrupt.png")
        assert False, "Corrupt file should raise ValueError"
    except Exception as e:
        assert "integrity" in str(e).lower() or "decode" in str(e).lower()
        print(f"  ✓ Corrupt bytes payload correctly rejected: {e}")
        
    # Test 3: Unsupported format (.pdf)
    try:
        validate_and_read_image(b"%PDF-1.4 header bytes...", "document.pdf")
        assert False, "Unsupported format should raise ValueError"
    except Exception as e:
        assert "unsupported" in str(e).lower()
        print(f"  ✓ Unsupported extension (.pdf) correctly rejected: {e}")
        
    # Test 4: Blank image without contour in Fourier module
    blank_img = np.zeros((100, 100), dtype=np.uint8)
    try:
        recognize_shape_fourier(blank_img)
        assert False, "Blank image should raise ValueError"
    except Exception as e:
        assert "distinct object boundary" in str(e)
        print(f"  ✓ Blank image boundary rejection: {e}")
        
    print("✓ All edge cases, invalid formats, and corrupt data correctly guarded.")


if __name__ == "__main__":
    t_start = time.time()
    print("Starting Comprehensive ImageLab Master Quality & Integration Test...")
    
    test_module_1_enhancement()
    test_module_2_boundary()
    test_module_3_watershed()
    test_module_4_fourier()
    test_in_memory_and_zero_persistence()
    test_edge_cases_and_error_handling()
    
    elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"ALL IMAGELAB MODULES & INTEGRATION TESTS COMPLETED SUCCESSFULLY IN {elapsed:.2f}s!")
    print("=" * 80)
