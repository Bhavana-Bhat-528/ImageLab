"""
Verification script for Module 4: Fourier Descriptor-Based Shape Recognition.
Tests:
1. End-to-end shape classification across all 7 categories (including Hexagon and Trapezium).
2. Invariance properties: Translation, Scale, and Rotation.
3. Harmonic truncation and contour reconstruction.
4. Independent test set evaluation and 7x7 Confusion Matrix integrity.
5. Open-set rejection of unfamiliar / irregular shapes as 'Unknown / Unrecognized Shape'.
6. Error handling on empty and corrupt inputs.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.fourier import (
    recognize_shape_fourier,
    evaluate_shape_model,
    SHAPE_CATEGORIES
)


def run_fourier_verification():
    print("================ RUNNING FOURIER RECOGNITION VALIDATION ================\n")

    samples_dir = backend_dir / "data" / "samples" / "recognition"
    assert samples_dir.exists(), f"Samples directory not found: {samples_dir}"

    # -------------------------------------------------------------------------
    # 1. Independent Test Set Classification (All 7 Categories)
    # -------------------------------------------------------------------------
    print("--- 1. Independent Test Set Classification (7 Categories, Zero Leakage) ---")
    test_files = [
        ("test_circle_01.png", "circle"),
        ("test_square_01.png", "square"),
        ("test_rectangle_01.png", "rectangle"),
        ("test_triangle_01.png", "triangle"),
        ("test_star_01.png", "star"),
        ("test_hexagon_01.png", "hexagon"),
        ("test_trapezium_01.png", "trapezium"),
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
        status = res["classification"]["recognition_status"]
        conf = res["classification"]["confidence_percentage"]
        dist = res["classification"]["nearest_distance"]
        thresh = res["classification"]["acceptance_threshold"]
        top_matches = res["classification"]["top_matches"]

        print(f"  [{fname:22s}] Expected: {expected_cat:10s} -> Predicted: {pred:10s} (Status: {status}, Conf: {conf:5.1f}%, Dist: {dist:.4f}, Thresh: {thresh:.4f})")
        assert status == "Recognized", f"Expected Recognized status for {fname}, got {status}"
        assert pred == expected_cat, f"Classification mismatch for {fname}: got {pred}, expected {expected_cat}"
        assert len(top_matches) > 0, "Top matches list is empty"
        assert len(res["descriptors"]) == 16, f"Expected 16 descriptors, got {len(res['descriptors'])}"
        assert res["images"]["spectrum_chart_base64"].startswith("data:image/png;base64,"), "Invalid spectrum chart"
        assert res["images"]["reconstructed_contour_base64"].startswith("data:image/png;base64,"), "Invalid reconstructed contour"

    print("✓ All 7 categories (including Hexagon and Trapezium) correctly classified with 100% precision.\n")

    # -------------------------------------------------------------------------
    # 2. Open-Set Rejection of Unfamiliar / Irregular Shapes
    # -------------------------------------------------------------------------
    print("--- 2. Open-Set Rejection of Unfamiliar Shapes ('Unknown / Unrecognized Shape') ---")
    unknown_files = [
        "test_unknown_01_amoeba.png",
        "test_unknown_02_crescent.png",
        "test_unknown_03_cross.png",
        "test_unknown_04_irregular.png"
    ]

    for u_fname in unknown_files:
        u_fpath = samples_dir / u_fname
        assert u_fpath.exists(), f"Unknown test sample missing: {u_fpath}"
        img = cv2.imread(str(u_fpath))

        res = recognize_shape_fourier(img, num_descriptors=16)
        status = res["classification"]["recognition_status"]
        pred = res["classification"]["predicted_category"]
        nearest_cat = res["classification"]["nearest_candidate_category"]
        dist = res["classification"]["nearest_distance"]
        thresh = res["classification"]["acceptance_threshold"]

        print(f"  [{u_fname:30s}] Status: {status:28s} (Nearest: {nearest_cat:10s}, Dist: {dist:.4f} > Thresh: {thresh:.4f})")
        assert status == "Unknown / Unrecognized Shape", f"Failed to reject unknown shape {u_fname}, got status {status}"
        assert pred == "Unknown / Unrecognized Shape", f"Failed to set predicted category to Unknown for {u_fname}"
        assert dist > thresh, f"Distance {dist} should exceed threshold {thresh}"

    print("✓ All unfamiliar / foreign shapes correctly rejected as 'Unknown / Unrecognized Shape'.\n")

    # -------------------------------------------------------------------------
    # 3. Invariance Verification (Rotation, Scale, Translation)
    # -------------------------------------------------------------------------
    print("--- 3. Mathematical Invariance Verification ---")
    base_sq = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(base_sq, (78, 78), (178, 178), 255, -1)

    rot_sq = np.zeros((256, 256), dtype=np.uint8)
    M_rot = cv2.getRotationMatrix2D((128, 128), 37, 1.0)
    rot_sq = cv2.warpAffine(base_sq, M_rot, (256, 256))

    scale_sq = np.zeros((256, 256), dtype=np.uint8)
    M_scale = cv2.getRotationMatrix2D((128, 128), 0, 0.7)
    scale_sq = cv2.warpAffine(base_sq, M_scale, (256, 256))

    trans_sq = np.zeros((256, 256), dtype=np.uint8)
    M_trans = np.float32([[1, 0, 25], [0, 1, -30]])
    trans_sq = cv2.warpAffine(base_sq, M_trans, (256, 256))

    d_base = np.array(recognize_shape_fourier(base_sq, num_descriptors=16)["descriptors"])
    d_rot = np.array(recognize_shape_fourier(rot_sq, num_descriptors=16)["descriptors"])
    d_scale = np.array(recognize_shape_fourier(scale_sq, num_descriptors=16)["descriptors"])
    d_trans = np.array(recognize_shape_fourier(trans_sq, num_descriptors=16)["descriptors"])

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
    # 4. Harmonic Truncation & Reconstruction Analysis
    # -------------------------------------------------------------------------
    print("--- 4. Harmonic Truncation & Reconstruction Analysis ---")
    hex_img = cv2.imread(str(samples_dir / "test_hexagon_01.png"))
    res_k4 = recognize_shape_fourier(hex_img, num_descriptors=4)
    res_k16 = recognize_shape_fourier(hex_img, num_descriptors=16)
    res_k32 = recognize_shape_fourier(hex_img, num_descriptors=32)

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
    # 5. Full Independent Test Set Evaluation & 7x7 Confusion Matrix
    # -------------------------------------------------------------------------
    print("--- 5. Full Independent Test Set Evaluation & 7x7 Confusion Matrix ---")
    eval_rep = evaluate_shape_model(num_descriptors=16)
    acc = eval_rep["overall_accuracy_pct"]
    macro_f1 = eval_rep["macro_f1"]
    conf_mat = eval_rep["confusion_matrix"]["matrix"]
    labels = eval_rep["confusion_matrix"]["labels"]

    print(f"  Training Library Size:    {eval_rep['train_samples_count']} prototypes")
    print(f"  Independent Test Set:     {eval_rep['test_samples_count']} specimens")
    print(f"  Unknown Rejection Rate:   {eval_rep['unknown_rejection_rate_pct']}% ({eval_rep['unknown_samples_tested']} tested)")
    print(f"  Overall Accuracy:         {acc:.2f}%")
    print(f"  Macro F1-Score:           {macro_f1:.3f}")
    print("  Confusion Matrix (7x7):")
    print("             " + "".join([f"{l:>11s}" for l in labels]))
    for idx, row in enumerate(conf_mat):
        print(f"  {labels[idx]:>10s}: " + "".join([f"{v:11d}" for v in row]))

    assert acc >= 95.0, f"Expected test accuracy >= 95%, got {acc}%"
    assert len(conf_mat) == 7 and len(conf_mat[0]) == 7, "Confusion matrix must be 7x7"
    assert eval_rep["unknown_rejection_rate_pct"] == 100.0, "All unknown specimens must be rejected"
    print("✓ 7-class model evaluation and confusion matrix fully verified.\n")

    # -------------------------------------------------------------------------
    # 6. Robustness & Error Handling
    # -------------------------------------------------------------------------
    print("--- 6. Robustness & Error Handling Verification ---")
    blank_img = np.zeros((128, 128), dtype=np.uint8)
    try:
        recognize_shape_fourier(blank_img)
        assert False, "Failed to raise error on blank image"
    except ValueError as val_err:
        print(f"  ✓ Blank image rejection: {val_err}")

    tiny_img = np.zeros((128, 128), dtype=np.uint8)
    tiny_img[60:63, 60:63] = 255
    try:
        recognize_shape_fourier(tiny_img)
        assert False, "Failed to raise error on tiny noise speckle"
    except ValueError as val_err:
        print(f"  ✓ Insufficient contour rejection: {val_err}")

    print("\n================ ALL FOURIER RECOGNITION TESTS PASSED! ================\n")


if __name__ == "__main__":
    run_fourier_verification()
