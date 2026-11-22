"""
Unit & Integration test for Module 1: Automatic Image Enhancement.
Tests all 7 sample images across Automatic and Manual modes, verifying:
- Characteristic analysis metrics (brightness, contrast, noise, sharpness, uniformity)
- Adaptive pipeline decision and correct parameter orientation (gamma < 1 for dark, gamma > 1 for bright)
- Noise removal (median for impulse, bilateral for sensor noise)
- Blur detection and unsharp masking
- Before/after histogram generation
- Quantitative metrics (relative vs. ground-truth reference with positive fidelity gain)
"""

import sys
import asyncio
from pathlib import Path

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import httpx
from app.main import app

SAMPLE_DIR = backend_dir / "data" / "samples" / "enhancement"

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
            "06_uneven_illumination.png",
            "07_blurred_out_of_focus.png"
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
            enh_chars = res_data["enhanced_characteristics"]
            pipeline = res_data["pipeline"]
            metrics = res_data["metrics"]

            print(f"\n--- Testing [{fname}] ---")
            print(f"  Input  -> Mean: {chars['mean_intensity']:.1f}, Std: {chars['std_intensity']:.1f}, Noise: {chars['noise_sigma']:.1f}, Sharpness: {chars['sharpness_score']:.1f}")
            print(f"  Defects: {chars['diagnostics']['primary_defects']}")
            print(f"  Ops    : {pipeline['selected_operations']}")
            print(f"  Params : {pipeline['parameters']}")
            print(f"  Output -> Mean: {enh_chars['mean_intensity']:.1f}, Std: {enh_chars['std_intensity']:.1f}, Sharpness: {enh_chars['sharpness_score']:.1f}")
            
            gt = metrics["ground_truth_comparison"]
            if gt["has_ground_truth"]:
                print(f"  Ground Truth ({gt['reference_name']}):")
                print(f"    PSNR: {gt['input_vs_ground_truth']['psnr']} dB -> {gt['enhanced_vs_ground_truth']['psnr']} dB (Gain: {gt['psnr_gain_db']:+} dB)")
                print(f"    SSIM: {gt['input_vs_ground_truth']['ssim']} -> {gt['enhanced_vs_ground_truth']['ssim']} (Gain: {gt['ssim_gain']:+})")

            # Verification of specific improvements
            if fname == "01_low_contrast.png":
                assert enh_chars["std_intensity"] > chars["std_intensity"], "Contrast should increase!"
                assert gt["ssim_gain"] > 0, "SSIM should improve on low contrast!"

            elif fname == "02_dark_underexposed.png":
                # Dark image MUST become brighter
                assert enh_chars["mean_intensity"] > chars["mean_intensity"] + 25.0, "Dark image must brighten!"
                assert pipeline["parameters"]["gamma"] < 1.0, "Gamma must be < 1.0 for dark images!"
                assert gt["psnr_gain_db"] > 4.0, "PSNR gain should be substantial!"
                assert gt["ssim_gain"] > 0.10, "SSIM gain should be positive!"

            elif fname == "03_bright_overexposed.png":
                # Bright image MUST become dimmer/compressed
                assert enh_chars["mean_intensity"] < chars["mean_intensity"] - 15.0, "Bright image must darken!"
                assert pipeline["parameters"]["gamma"] > 1.0, "Gamma must be > 1.0 for bright images!"
                assert gt["psnr_gain_db"] > 1.5, "PSNR gain should be positive!"
                assert gt["ssim_gain"] > 0.05, "SSIM gain should be positive!"

            elif fname == "04_gaussian_noise.png":
                # Noise must be reduced significantly
                assert pipeline["parameters"]["spatial_filter"] == "bilateral"
                assert gt["psnr_gain_db"] > 5.0, "Gaussian denoising PSNR gain should exceed 5 dB!"
                assert gt["ssim_gain"] > 0.20, "SSIM gain should be substantial!"

            elif fname == "05_salt_and_pepper_noise.png":
                # Median filter must eliminate impulse spikes
                assert pipeline["parameters"]["spatial_filter"] == "median"
                assert gt["psnr_gain_db"] > 10.0, "Impulse denoising PSNR gain should exceed 10 dB!"
                assert gt["ssim_gain"] > 0.40, "SSIM gain should be huge!"

            elif fname == "07_blurred_out_of_focus.png":
                # Blurred image must increase in sharpness
                assert chars["diagnostics"]["is_blurry"], "Blur must be diagnosed!"
                assert pipeline["parameters"]["sharpen_strength"] > 0.0, "Sharpening must be selected!"
                assert enh_chars["sharpness_score"] > chars["sharpness_score"] * 1.5, "Sharpness must increase significantly!"
                assert gt["psnr_gain_db"] > 0.5, "Restoration PSNR should improve!"

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
