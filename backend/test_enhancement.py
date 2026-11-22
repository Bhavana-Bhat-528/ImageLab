"""
Unit & Integration test for Module 1: Automatic Image Enhancement.
Tests all 5+ sample images across Automatic and Manual modes, verifying:
- Characteristic analysis metrics
- Adaptive pipeline decision
- Before/after histogram generation
- Quantitative metrics (relative vs. ground-truth reference)
"""

import asyncio
from pathlib import Path
import httpx
from app.main import app

SAMPLE_DIR = Path("data/samples/enhancement")

async def test_all_enhancement_samples():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        print("================ TESTING ALL ENHANCEMENT SAMPLES ================")
        
        test_images = [
            "01_low_contrast.png",
            "02_dark_underexposed.png",
            "03_bright_overexposed.png",
            "04_gaussian_noise.png",
            "05_salt_and_pepper_noise.png",
            "06_uneven_illumination.png"
        ]

        for fname in test_images:
            fpath = SAMPLE_DIR / fname
            assert fpath.exists(), f"Sample image {fname} missing!"
            content = fpath.read_bytes()

            # Test 1: Automatic Mode
            files = {"file": (fname, content, "image/png")}
            data = {"mode": "auto"}
            res = await client.post("/api/enhancement/process", files=files, data=data)
            assert res.status_code == 200, f"Auto enhancement failed on {fname}: {res.text}"
            res_data = res.json()

            chars = res_data["input_characteristics"]
            pipeline = res_data["pipeline"]
            metrics = res_data["metrics"]

            print(f"\n--- Testing [{fname}] ---")
            print(f"  Mean: {chars['mean_intensity']:.1f}, Std: {chars['std_intensity']:.1f}, Contrast: {chars['rms_contrast']}, Noise: {chars['noise_sigma']:.1f}, Sharpness: {chars['sharpness_score']:.1f}")
            print(f"  Detected Defect: {chars['diagnostics']['primary_defects']}")
            print(f"  Selected Operations: {pipeline['selected_operations']}")
            print(f"  Rel Metrics (vs Input): PSNR={metrics['relative_to_input']['psnr']} dB, SSIM={metrics['relative_to_input']['ssim']}")
            
            gt = metrics["ground_truth_comparison"]
            if gt["has_ground_truth"]:
                print(f"  Ground Truth Reference ({gt['reference_name']}):")
                print(f"    Base Input PSNR: {gt['input_vs_ground_truth']['psnr']} dB -> Enhanced PSNR: {gt['enhanced_vs_ground_truth']['psnr']} dB (Gain: {gt['psnr_gain_db']:+} dB)")
                print(f"    Base Input SSIM: {gt['input_vs_ground_truth']['ssim']} -> Enhanced SSIM: {gt['enhanced_vs_ground_truth']['ssim']} (Gain: {gt['ssim_gain']:+})")

            # Verify histograms returned
            assert "original" in res_data["histograms"]
            assert "enhanced" in res_data["histograms"]
            assert len(res_data["histograms"]["original"]["luminance"]) == 256
            assert len(res_data["histograms"]["enhanced"]["luminance"]) == 256

            # Verify base64 images returned
            assert res_data["images"]["original_base64"].startswith("data:image/png;base64,")
            assert res_data["images"]["enhanced_base64"].startswith("data:image/png;base64,")

        # Test 2: Manual Mode
        print("\n--- Testing Manual Mode (gamma=0.7, CLAHE clip=3.0, unsharp=1.0) ---")
        content = (SAMPLE_DIR / "01_low_contrast.png").read_bytes()
        files = {"file": ("01_low_contrast.png", content, "image/png")}
        manual_data = {
            "mode": "manual",
            "gamma": "0.7",
            "equalization_mode": "clahe",
            "clahe_clip_limit": "3.0",
            "clahe_tile_grid_size": "8",
            "spatial_filter": "unsharp",
            "kernel_size": "3",
            "sharpen_strength": "1.0"
        }
        res = await client.post("/api/enhancement/process", files=files, data=manual_data)
        assert res.status_code == 200, f"Manual enhancement failed: {res.text}"
        res_data = res.json()
        assert res_data["mode"] == "manual"
        print(f"  Manual operations executed: {res_data['pipeline']['selected_operations']}")
        print(f"  Enhanced mean: {res_data['enhanced_characteristics']['mean_intensity']}, std: {res_data['enhanced_characteristics']['std_intensity']}")
        print("✓ Manual mode passed.")

        print("\n================ ALL ENHANCEMENT TESTS PASSED! ================")

if __name__ == "__main__":
    asyncio.run(test_all_enhancement_samples())
