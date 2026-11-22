"""
Direct validation suite for Module 2: Morphological Boundary Extraction.
Runs synchronous validation without anyio/transport hangs.
"""

import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import cv2
import numpy as np
from app.services.boundary import (
    extract_animal_boundary,
    extract_boundary,
    segment_animal_foreground,
    create_boundary_overlay,
    benchmark_structuring_elements
)
from app.api.routes_samples import is_valid_input_sample
from app.utils.image_io import validate_and_read_image
from app.config import settings

def run_boundary_tests():
    print("================ RUNNING DIRECT BOUNDARY VALIDATION ================", flush=True)

    sample_dir = settings.SAMPLES_DIR / "boundary"
    assert sample_dir.exists(), f"Sample directory missing: {sample_dir}"

    # -------------------------------------------------------------------------
    # 1. Mathematical Morphology Verification: beta(A) = A - (A (-) B)
    # -------------------------------------------------------------------------
    print("\n--- 1. Mathematical Equation Verification: beta(A) = A - (A (-) B) ---", flush=True)
    # Create a synthetic binary mask A with a square
    mask_A = np.zeros((100, 100), dtype=np.uint8)
    mask_A[20:80, 20:80] = 255
    eroded, boundary, se = extract_boundary(mask_A, se_shape="rect", se_size=3, iterations=1)

    # Mathematical identity checks
    # (a) beta(A) must be subset of A
    assert np.all((boundary == 255) <= (mask_A == 255)), "Boundary must be a subset of foreground A!"
    # (b) eroded must be subset of A
    assert np.all((eroded == 255) <= (mask_A == 255)), "Eroded mask must be a subset of foreground A!"
    # (c) A - (A (-) B) exactly equals boundary
    expected_diff = cv2.subtract(mask_A, eroded)
    assert np.array_equal(boundary, expected_diff), "boundary must exactly equal cv2.subtract(A, eroded)!"
    # (d) Inner points of eroded mask must have zero boundary
    assert np.all(boundary[eroded == 255] == 0), "Inner eroded points cannot have boundary pixels!"
    print(f"✓ Mathematical equation verified: |A|={np.count_nonzero(mask_A)}, |A(-)B|={np.count_nonzero(eroded)}, |beta(A)|={np.count_nonzero(boundary)}", flush=True)

    # -------------------------------------------------------------------------
    # 2. Benchmark Animal Samples Verification (Zebra, Elephant, Dog)
    # -------------------------------------------------------------------------
    print("\n--- 2. Benchmark Animal Photographs Verification ---", flush=True)
    test_animals = ["zebra.jpg", "elephant.jpg", "dog.jpg"]

    for fname in test_animals:
        fpath = sample_dir / fname
        assert fpath.exists(), f"Missing animal image: {fname}"
        img = cv2.imread(str(fpath))
        assert img is not None, f"Failed to load image: {fname}"

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

        metrics = res["metrics"]
        gt = metrics["ground_truth_evaluation"]
        images = res["images"]
        se_bench = res["se_benchmarks"]

        print(f"\n  [{fname}] Resolution: {metrics['image_dimensions']}:", flush=True)
        print(f"    Foreground Pixels: {metrics['foreground_pixels']} ({metrics['foreground_percentage']}%)", flush=True)
        print(f"    Boundary Pixels: {metrics['boundary_pixels']} ({metrics['boundary_to_foreground_ratio_pct']}%)", flush=True)
        print(f"    Components: FG={metrics['foreground_connected_components']}, Boundary Loops={metrics['boundary_connected_components']}", flush=True)

        # Verify Ground Truth accuracy
        assert gt["has_ground_truth"] is True, f"Ground truth evaluation missing for {fname}"
        print(f"    Ground-Truth IoU: {gt['iou']:.4f}, Dice: {gt['dice']:.4f} (Reference: {gt['reference_name']})", flush=True)
        assert gt["iou"] >= 0.75, f"IoU on {fname} should exceed 0.75, got {gt['iou']}"
        assert gt["dice"] >= 0.85, f"Dice on {fname} should exceed 0.85, got {gt['dice']}"

        # Verify all 6 viewports
        for key in ["original_base64", "preprocessed_base64", "binary_mask_base64", "eroded_mask_base64", "boundary_mask_base64", "overlay_base64"]:
            assert key in images and images[key].startswith("data:image/"), f"Missing viewport: {key}"

        # Verify 9-configuration cross-SE benchmark table
        assert len(se_bench) == 9, "Expected 9 SE benchmark entries"

    print("✓ All 3 real-world animal benchmarks passed with high IoU/Dice fidelity!", flush=True)

    # -------------------------------------------------------------------------
    # 3. Structuring Element Variations (Cross vs. Ellipse vs. Rect)
    # -------------------------------------------------------------------------
    print("\n--- 3. Structuring Element Geometry Verification ---", flush=True)
    dog_img = cv2.imread(str(sample_dir / "dog.jpg"))
    shape_results = {}
    for shape in ["cross", "ellipse", "rect"]:
        res = extract_animal_boundary(dog_img, se_shape=shape, se_size=5, iterations=1, margin_pct=4.0)
        bnd_px = res["metrics"]["boundary_pixels"]
        shape_results[shape] = bnd_px
        print(f"  Dog 5x5 {shape.upper():7s}: {bnd_px} boundary px ({res['metrics']['boundary_to_foreground_ratio_pct']}%)", flush=True)

    assert shape_results["rect"] >= max(shape_results["cross"], shape_results["ellipse"]), \
        "Rectangle SE (8-connected) must produce the thickest boundary!"
    assert min(shape_results["cross"], shape_results["ellipse"]) > 1000, \
        "Cross and Ellipse must produce clear boundaries!"
    print(f"✓ Structuring element geometry validated: Rect ({shape_results['rect']}) >= Ellipse/Cross.", flush=True)

    # -------------------------------------------------------------------------
    # 4. Multi-Scale Speed & High-Res Upload Validation
    # -------------------------------------------------------------------------
    print("\n--- 4. High-Resolution Upload & Performance Verification ---", flush=True)
    import time
    hr_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    hr_img[:] = (60, 120, 40)
    cv2.ellipse(hr_img, (960, 540), (500, 300), 10, 0, 360, (180, 160, 140), -1)

    t0 = time.time()
    hr_res = extract_animal_boundary(hr_img, se_shape="ellipse", se_size=5, iterations=1, margin_pct=4.0)
    elapsed = time.time() - t0
    print(f"  1920x1080 image processed in {elapsed:.3f}s (well under 2.0s limit)!", flush=True)
    print(f"  Boundary pixels: {hr_res['metrics']['boundary_pixels']}", flush=True)
    assert elapsed < 2.0, "High-resolution multi-scale processing took too long!"

    # -------------------------------------------------------------------------
    # 5. Sample Dataset Filtering Verification (No masks or results in selector)
    # -------------------------------------------------------------------------
    print("\n--- 5. Sample Dataset Filtering Verification ---", flush=True)
    boundary_files = [f.name for f in sample_dir.iterdir() if is_valid_input_sample(f)]
    print(f"  Discovered boundary input samples: {boundary_files}", flush=True)
    assert set(boundary_files) == {"dog.jpg", "elephant.jpg", "zebra.jpg"}, \
        f"Sample selector should ONLY see the 3 input images, got {boundary_files}"
    for bf in boundary_files:
        assert not any(k in bf.lower() for k in ["mask", "_gt", "result", "boundary", "eroded"]), \
            f"Result/mask file erroneously returned: {bf}"
    print("✓ Sample selector strictly isolates original inputs from masks and results.", flush=True)

    # -------------------------------------------------------------------------
    # 6. Upload Integrity & Error Rejection Verification
    # -------------------------------------------------------------------------
    print("\n--- 6. Upload Validation & Error Rejection Verification ---", flush=True)
    # Test empty file
    try:
        validate_and_read_image(b"", "empty.jpg")
        assert False, "Should have rejected empty file!"
    except ValueError as e:
        print(f"  Empty file correctly rejected: {e}", flush=True)

    # Test corrupt file
    try:
        validate_and_read_image(b"NOT_A_VALID_IMAGE_DATA", "corrupt.jpg")
        assert False, "Should have rejected corrupt file!"
    except ValueError as e:
        print(f"  Corrupt file correctly rejected: {e}", flush=True)

    # Test unsupported extension
    try:
        validate_and_read_image(b"header", "test.pdf")
        assert False, "Should have rejected unsupported extension!"
    except ValueError as e:
        print(f"  Unsupported format correctly rejected: {e}", flush=True)

    print("\n================ ALL MORPHOLOGICAL BOUNDARY TESTS PASSED! ================", flush=True)

if __name__ == "__main__":
    run_boundary_tests()
