"""
Verification script for Module 4: Fourier Descriptor-Based Shape Recognition.
Tests:
1. End-to-end shape classification across all 5 categories.
2. Invariance properties: Translation, Scale, and Rotation.
3. Harmonic truncation and contour reconstruction.
4. Independent test set evaluation and Confusion Matrix integrity.
5. Error handling on empty and corrupt inputs.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure backend app is in import path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.fourier import (
    recognize_shape_fourier,
    evaluate_shape_model,
    resample_contour,
    compute_fourier_coefficients,
    extract_invariant_descriptors,
    reconstruct_contour_from_coefficients
)


def run_fourier_verification():
    print("================ RUNNING FOURIER RECOGNITION VALIDATION ================\n")

    samples_dir = backend_dir / "data" / "samples" / "recognition"
    assert samples_dir.exists(), f"Samples directory not found: {samples_dir}"

    # -------------------------------------------------------------------------
    # 1. Independent Test Set Classification
    # -------------------------------------------------------------------------
    print("--- 1. Independent Test Set Classification (Zero Data Leakage) ---")
    test_files = [
        ("test_circle_01.png", "circle"),
        ("test_square_01.png", "square"),
        ("test_rectangle_01.png", "rectangle"),
        ("test_triangle_01.png", "triangle"),
        ("test_star_01.png", "star"),
    ]

    for fname, expected_cat in test_files:
        fpath = samples_dir / fname
        assert fpath.exists(), f"Test sample not found: {fpath}"
        img = cv2.imread(str(fpath))
        assert img is not None, f"Failed to load image: {fname}"

        res = recognize_shape_fourier(
            image=img,
            num_descriptors=16,
            distance_metric="euclidean",
            normalize_scale=True,
            normalize_rotation=True,
            filename=fname
        )

        assert res["status"] == "success", f"Processing failed for {fname}"
        pred = res["classification"]["predicted_category"]
        conf = res["classification"]["confidence_percentage"]
        dist = res["classification"]["nearest_distance"]
        top_matches = res["classification"]["top_matches"]

        print(f"  [{fname:20s}] Expected: {expected_cat:10s} -> Predicted: {pred:10s} (Conf: {conf:5.1f}%, Dist: {dist:.4f})")
        assert pred == expected_cat, f"Classification mismatch for {fname}: got {pred}, expected {expected_cat}"
        assert len(top_matches) > 0, "Top matches list is empty"
        assert len(res["descriptors"]) == 16, f"Expected 16 descriptors, got {len(res['descriptors'])}"
        assert res["images"]["spectrum_chart_base64"].startswith("data:image/png;base64,"), "Invalid spectrum chart"
        assert res["images"]["reconstructed_contour_base64"].startswith("data:image/png;base64,"), "Invalid reconstructed contour"

    print("✓ All 5 independent test categories classified with 100% precision.\n")

    # -------------------------------------------------------------------------
    # 2. Invariance Verification (Rotation, Scale, Translation)
    # -------------------------------------------------------------------------
    print("--- 2. Mathematical Invariance Verification ---")
    # Base square
    base_sq = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(base_sq, (78, 78), (178, 178), 255, -1)

    # Rotated square (37 degrees)
    rot_sq = np.zeros((256, 256), dtype=np.uint8)
    M_rot = cv2.getRotationMatrix2D((128, 128), 37, 1.0)
    rot_sq = cv2.warpAffine(base_sq, M_rot, (256, 256))

    # Scaled square (0.7x)
    scale_sq = np.zeros((256, 256), dtype=np.uint8)
    M_scale = cv2.getRotationMatrix2D((128, 128), 0, 0.7)
    scale_sq = cv2.warpAffine(base_sq, M_scale, (256, 256))

    # Translated square (+25px x, -30px y)
    trans_sq = np.zeros((256, 256), dtype=np.uint8)
    M_trans = np.float32([[1, 0, 25], [0, 1, -30]])
    trans_sq = cv2.warpAffine(base_sq, M_trans, (256, 256))

    res_base = recognize_shape_fourier(base_sq, num_descriptors=16)
    res_rot = recognize_shape_fourier(rot_sq, num_descriptors=16)
    res_scale = recognize_shape_fourier(scale_sq, num_descriptors=16)
    res_trans = recognize_shape_fourier(trans_sq, num_descriptors=16)

    d_base = np.array(res_base["descriptors"])
    d_rot = np.array(res_rot["descriptors"])
    d_scale = np.array(res_scale["descriptors"])
    d_trans = np.array(res_trans["descriptors"])

    dist_rot = np.linalg.norm(d_base - d_rot)
    dist_scale = np.linalg.norm(d_base - d_scale)
    dist_trans = np.linalg.norm(d_base - d_trans)

    print(f"  Rotation (37°) Descriptor Difference:     {dist_rot:.4f}")
    print(f"  Scale (0.7×) Descriptor Difference:       {dist_scale:.4f}")
    print(f"  Translation (+25, -30) Difference:        {dist_trans:.4f}")

    assert dist_rot < 0.15, f"Rotation invariance exceeded tolerance: {dist_rot}"
    assert dist_scale < 0.10, f"Scale invariance exceeded tolerance: {dist_scale}"
    assert dist_trans < 0.05, f"Translation invariance exceeded tolerance: {dist_trans}"
    print("✓ Translation, scale, and rotational invariance verified.\n")

    # -------------------------------------------------------------------------
    # 3. Harmonic Truncation & Reconstruction Analysis
    # -------------------------------------------------------------------------
    print("--- 3. Harmonic Truncation & Reconstruction Analysis ---")
    star_img = cv2.imread(str(samples_dir / "test_star_01.png"))
    res_k4 = recognize_shape_fourier(star_img, num_descriptors=4)
    res_k16 = recognize_shape_fourier(star_img, num_descriptors=16)
    res_k32 = recognize_shape_fourier(star_img, num_descriptors=32)

    mse_4 = res_k4["shape_properties"]["reconstruction_mse"]
    mse_16 = res_k16["shape_properties"]["reconstruction_mse"]
    mse_32 = res_k32["shape_properties"]["reconstruction_mse"]

    energy_4 = res_k4["shape_properties"]["energy_retention_pct"]
    energy_16 = res_k16["shape_properties"]["energy_retention_pct"]
    energy_32 = res_k32["shape_properties"]["energy_retention_pct"]

    print(f"  K = 4  Harmonics: Reconstruction MSE = {mse_4:6.2f}, Energy Retained = {energy_4:5.1f}%")
    print(f"  K = 16 Harmonics: Reconstruction MSE = {mse_16:6.2f}, Energy Retained = {energy_16:5.1f}%")
    print(f"  K = 32 Harmonics: Reconstruction MSE = {mse_32:6.2f}, Energy Retained = {energy_32:5.1f}%")

    assert mse_32 < mse_16 < mse_4, "Reconstruction error must monotonically decrease with increasing K"
    assert energy_32 >= energy_16 >= energy_4, "Retained spectral energy must increase with increasing K"
    print("✓ Monotonic harmonic convergence verified.\n")

    # -------------------------------------------------------------------------
    # 4. Independent Test Split Evaluation & Confusion Matrix
    # -------------------------------------------------------------------------
    print("--- 4. Full Independent Test Set Evaluation & Confusion Matrix ---")
    eval_rep = evaluate_shape_model(num_descriptors=16)
    acc = eval_rep["overall_accuracy_pct"]
    macro_f1 = eval_rep["macro_f1"]
    conf_mat = eval_rep["confusion_matrix"]["matrix"]
    labels = eval_rep["confusion_matrix"]["labels"]

    print(f"  Total Test Samples: {eval_rep['test_samples_count']}")
    print(f"  Overall Accuracy:   {acc:.2f}%")
    print(f"  Macro F1-Score:     {macro_f1:.3f}")
    print("  Confusion Matrix:")
    print("           " + "".join([f"{l:>10s}" for l in labels]))
    for idx, row in enumerate(conf_mat):
        print(f"  {labels[idx]:>8s}: " + "".join([f"{v:10d}" for v in row]))

    assert acc >= 90.0, f"Expected test accuracy >= 90%, got {acc}%"
    assert len(conf_mat) == 5 and len(conf_mat[0]) == 5, "Confusion matrix must be 5x5"
    print("✓ Model evaluation and confusion matrix fully verified.\n")

    # -------------------------------------------------------------------------
    # 5. Robustness & Error Handling
    # -------------------------------------------------------------------------
    print("--- 5. Robustness & Error Handling Verification ---")
    blank_img = np.zeros((128, 128), dtype=np.uint8)
    try:
        recognize_shape_fourier(blank_img)
        assert False, "Failed to raise error on blank image"
    except ValueError as val_err:
        print(f"  ✓ Blank image rejection: {val_err}")

    tiny_img = np.zeros((128, 128), dtype=np.uint8)
    tiny_img[60:63, 60:63] = 255  # 3x3 speckle
    try:
        recognize_shape_fourier(tiny_img)
        assert False, "Failed to raise error on tiny noise speckle"
    except ValueError as val_err:
        print(f"  ✓ Insufficient contour rejection: {val_err}")

    print("\n================ ALL FOURIER RECOGNITION TESTS PASSED! ================\n")


if __name__ == "__main__":
    run_fourier_verification()
