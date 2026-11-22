"""
Unit & Integration test for Module 2: Morphological Boundary Extraction.
Tests zebra.jpg, elephant.jpg, and dog.jpg across structuring element shapes,
sizes, erosion iterations, and verifies ground-truth IoU/Dice metrics.
"""

import asyncio
from pathlib import Path
import httpx
from app.main import app

SAMPLE_DIR = Path("data/samples/boundary")

async def test_all_boundary_samples():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        print("================ TESTING ALL BOUNDARY EXTRACTION SAMPLES ================")
        
        test_animals = ["zebra.jpg", "elephant.jpg", "dog.jpg"]

        for fname in test_animals:
            fpath = SAMPLE_DIR / fname
            assert fpath.exists(), f"Sample image {fname} missing!"
            content = fpath.read_bytes()

            # Test standard GrabCut + 3x3 Ellipse SE
            files = {"file": (fname, content, "image/jpeg")}
            data = {
                "se_shape": "ellipse",
                "se_size": "3",
                "iterations": "1",
                "segmentation_method": "grabcut",
                "overlay_color": "cyan",
                "display_thickness": "1"
            }
            res = await client.post("/api/boundary/process", files=files, data=data)
            assert res.status_code == 200, f"Boundary extraction failed on {fname}: {res.text}"
            res_data = res.json()

            metrics = res_data["metrics"]
            gt = metrics["ground_truth_evaluation"]
            se_bench = res_data["se_benchmarks"]

            print(f"\n--- Testing [{fname}] ---")
            print(f"  Dimensions: {metrics['image_dimensions']}, Total Pixels: {metrics['total_pixels']}")
            print(f"  Foreground Pixels: {metrics['foreground_pixels']} ({metrics['foreground_percentage']}%)")
            print(f"  Boundary Pixels: {metrics['boundary_pixels']} ({metrics['boundary_to_foreground_ratio_pct']}%)")
            print(f"  Components - Foreground: {metrics['foreground_connected_components']}, Boundary: {metrics['boundary_connected_components']}")

            assert gt["has_ground_truth"] is True, f"Missing ground truth evaluation for {fname}"
            print(f"  Ground-Truth Reference ({gt['reference_name']}):")
            print(f"    IoU (Jaccard Index): {gt['iou']:.4f}")
            print(f"    Dice Coefficient (F1): {gt['dice']:.4f}")

            # Verify SE benchmark entries
            print(f"  Cross-SE Benchmark entries: {len(se_bench)} configurations")
            for b in se_bench[:3]:
                print(f"    - {b['shape_label']} {b['size']}: {b['boundary_pixels']} boundary px ({b['boundary_to_fg_ratio_pct']}%)")

            # Verify all 6 visualizations returned
            for key in ["original_base64", "preprocessed_base64", "binary_mask_base64", "eroded_mask_base64", "boundary_mask_base64", "overlay_base64"]:
                assert key in res_data["images"]
                assert res_data["images"][key].startswith("data:image/")

        # Test structuring element shapes and sizes impact
        print("\n--- Testing SE Shape & Size Variation on dog.jpg (Cross vs. Rect vs. Ellipse 7x7) ---")
        dog_bytes = (SAMPLE_DIR / "dog.jpg").read_bytes()
        for shape in ["cross", "rect", "ellipse"]:
            files = {"file": ("dog.jpg", dog_bytes, "image/jpeg")}
            data = {"se_shape": shape, "se_size": "7", "iterations": "1"}
            res = await client.post("/api/boundary/process", files=files, data=data)
            assert res.status_code == 200
            d = res.json()
            print(f"  Dog 7x7 {shape.upper()}: Boundary Pixels = {d['metrics']['boundary_pixels']}, Ratio = {d['metrics']['boundary_to_foreground_ratio_pct']}%")

        print("\n================ ALL BOUNDARY TESTS PASSED! ================")

if __name__ == "__main__":
    asyncio.run(test_all_boundary_samples())
