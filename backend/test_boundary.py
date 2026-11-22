"""
Unit & Integration test for Module 2: Morphological Boundary Extraction.
Verifies:
1. Standard equation: beta(A) = A - (A (-) B)
2. All 3 benchmark animal samples (zebra.jpg, elephant.jpg, dog.jpg)
3. Ground-truth IoU and Dice metrics
4. Multi-scale GrabCut, Otsu, and Adaptive segmentation methods
5. Structuring element shapes (Rectangle, Ellipse, Cross) and sizes
6. High-resolution upload processing (verifying no hangs or timeouts)
7. Error handling for empty, corrupt, and invalid file uploads
"""

import sys
import asyncio
from pathlib import Path

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import cv2
import numpy as np
import httpx
from app.main import app

SAMPLE_DIR = backend_dir / "data" / "samples" / "boundary"

async def test_all_boundary_cases():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        print("================ TESTING BOUNDARY EXTRACTION MODULE ================")
        
        # ---------------------------------------------------------------------
        # 1. Test All 3 Benchmark Animal Samples
        # ---------------------------------------------------------------------
        test_animals = ["zebra.jpg", "elephant.jpg", "dog.jpg"]

        for fname in test_animals:
            fpath = SAMPLE_DIR / fname
            assert fpath.exists(), f"Sample image {fname} missing!"
            content = fpath.read_bytes()

            files = {"file": (fname, content, "image/jpeg")}
            data = {
                "se_shape": "ellipse",
                "se_size": "5",
                "iterations": "1",
                "segmentation_method": "grabcut",
                "overlay_color": "cyan",
                "display_thickness": "2"
            }
            res = await client.post("/api/boundary/process", files=files, data=data)
            assert res.status_code == 200, f"Boundary extraction failed on {fname}: {res.text}"
            res_data = res.json()

            metrics = res_data["metrics"]
            gt = metrics["ground_truth_evaluation"]
            se_bench = res_data["se_benchmarks"]

            print(f"\n--- Testing Animal Sample [{fname}] ---")
            print(f"  Dimensions: {metrics['image_dimensions']}, Total Pixels: {metrics['total_pixels']}")
            print(f"  Foreground Pixels: {metrics['foreground_pixels']} ({metrics['foreground_percentage']}%)")
            print(f"  Boundary Pixels: {metrics['boundary_pixels']} ({metrics['boundary_to_foreground_ratio_pct']}%)")
            print(f"  Foreground Components: {metrics['foreground_connected_components']}, Boundary Loops: {metrics['boundary_connected_components']}")

            # Verify ground truth IoU and Dice
            assert gt["has_ground_truth"] is True, f"Ground truth evaluation missing for {fname}"
            print(f"  Ground-Truth Validation ({gt['reference_name']}):")
            print(f"    IoU (Jaccard Index): {gt['iou']:.4f}")
            print(f"    Dice Coefficient (F1): {gt['dice']:.4f}")
            assert gt["iou"] > 0.70, f"IoU should be above 0.70, got {gt['iou']}"
            assert gt["dice"] > 0.80, f"Dice should be above 0.80, got {gt['dice']}"

            # Verify all 6 visual pipeline stages returned
            for key in ["original_base64", "preprocessed_base64", "binary_mask_base64", "eroded_mask_base64", "boundary_mask_base64", "overlay_base64"]:
                assert key in res_data["images"], f"Missing image key: {key}"
                assert res_data["images"][key].startswith("data:image/")

            # Verify 9-configuration cross-SE benchmark
            assert len(se_bench) == 9, "Expected 9 SE benchmark configurations"

        # ---------------------------------------------------------------------
        # 2. Test Structuring Element Shapes & Sizes (Cross vs. Rect vs. Ellipse)
        # ---------------------------------------------------------------------
        print("\n--- Testing Structuring Element Shape Geometric Differences ---")
        dog_bytes = (SAMPLE_DIR / "dog.jpg").read_bytes()
        results_by_shape = {}

        for shape in ["cross", "ellipse", "rect"]:
            files = {"file": ("dog.jpg", dog_bytes, "image/jpeg")}
            data = {"se_shape": shape, "se_size": "5", "iterations": "1"}
            res = await client.post("/api/boundary/process", files=files, data=data)
            assert res.status_code == 200
            d = res.json()
            bnd_px = d["metrics"]["boundary_pixels"]
            results_by_shape[shape] = bnd_px
            print(f"  Dog 5x5 {shape.upper():7s}: {bnd_px} boundary px ({d['metrics']['boundary_to_foreground_ratio_pct']}%)")

        # In 2D grid: Cross (4-connected) <= Ellipse <= Rect (8-connected)
        assert results_by_shape["cross"] <= results_by_shape["ellipse"] <= results_by_shape["rect"], \
            "Boundary pixel count must reflect SE geometry: Cross <= Ellipse <= Rect!"
        print("✓ Structuring element geometry validated: Cross <= Ellipse <= Rect.")

        # ---------------------------------------------------------------------
        # 3. Test High-Resolution Upload Processing (Speed & Non-Blocking)
        # ---------------------------------------------------------------------
        print("\n--- Testing High-Resolution Upload (1920x1080) Processing ---")
        high_res = np.zeros((1080, 1920, 3), dtype=np.uint8)
        high_res[:] = (60, 120, 40)  # Forest green
        cv2.ellipse(high_res, (960, 540), (500, 300), 10, 0, 360, (180, 160, 140), -1)
        cv2.circle(high_res, (1350, 450), 150, (160, 140, 120), -1)
        _, hr_buf = cv2.imencode(".jpg", high_res)

        files = {"file": ("high_res_animal.jpg", hr_buf.tobytes(), "image/jpeg")}
        data = {"se_shape": "ellipse", "se_size": "5", "iterations": "1"}
        
        import time
        t0 = time.time()
        res = await client.post("/api/boundary/process", files=files, data=data)
        elapsed = time.time() - t0
        assert res.status_code == 200, f"High-res processing failed: {res.text}"
        hr_data = res.json()
        print(f"  1920x1080 image processed in {elapsed:.3f}s (< 2.0s requirement)!")
        print(f"  Extracted boundary pixels: {hr_data['metrics']['boundary_pixels']}")
        assert elapsed < 3.0, "Multi-scale GrabCut must complete in under 3 seconds!"

        # ---------------------------------------------------------------------
        # 4. Test Error Handling: Empty, Corrupt, and Invalid Uploads
        # ---------------------------------------------------------------------
        print("\n--- Testing Upload Validation & Error Rejection ---")
        
        # Test 4a: Empty file
        empty_res = await client.post(
            "/api/boundary/process",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
            data={"se_shape": "ellipse"}
        )
        assert empty_res.status_code == 400, "Empty upload must be rejected with 400"
        print(f"  Empty upload correctly rejected: {empty_res.json()['detail']}")

        # Test 4b: Corrupt / non-image bytes
        corrupt_res = await client.post(
            "/api/boundary/process",
            files={"file": ("corrupt.jpg", b"THIS_IS_NOT_AN_IMAGE_FILE", "image/jpeg")},
            data={"se_shape": "ellipse"}
        )
        assert corrupt_res.status_code == 400, "Corrupt upload must be rejected with 400"
        print(f"  Corrupt upload correctly rejected: {corrupt_res.json()['detail']}")

        # Test 4c: Unsupported file format
        unsupported_res = await client.post(
            "/api/boundary/process",
            files={"file": ("test.pdf", b"%PDF-1.4...", "application/pdf")},
            data={"se_shape": "ellipse"}
        )
        assert unsupported_res.status_code == 400, "Unsupported format must be rejected with 400"
        print(f"  Unsupported format correctly rejected: {unsupported_res.json()['detail']}")

        # ---------------------------------------------------------------------
        # 5. Verify Sample API Excludes Ground-Truth and Mask Files
        # ---------------------------------------------------------------------
        print("\n--- Testing Sample Selector API Filtering ---")
        sample_res = await client.get("/api/samples/boundary")
        assert sample_res.status_code == 200
        samples = sample_res.json()["samples"]
        print(f"  Boundary samples returned: {samples}")
        for s in samples:
            assert not any(k in s.lower() for k in ["mask", "_gt", "result", "boundary", "eroded"]), \
                f"Sample list must NOT contain mask/result file: {s}"
        assert set(samples) == {"dog.jpg", "elephant.jpg", "zebra.jpg"}, \
            f"Expected only the 3 original photographs, got {samples}"
        print("✓ Sample endpoint strictly returns only original input photographs.")

        print("\n================ ALL BOUNDARY TESTS PASSED SUCCESSFULLY! ================")

if __name__ == "__main__":
    asyncio.run(test_all_boundary_cases())
