import sys
from pathlib import Path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import cv2
import numpy as np
from app.services.fourier import recognize_shape_fourier, evaluate_shape_model

def make_natural_starfish(size=256, arm_lengths=[85, 95, 82, 90, 87], r_inner=32, rotation_deg=18):
    canvas = np.zeros((size, size), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    theta = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    r = np.full_like(theta, float(r_inner))
    arm_angles = [0 + rotation_deg, 71 + rotation_deg, 146 + rotation_deg, 214 + rotation_deg, 289 + rotation_deg]
    for i, a0 in enumerate(arm_angles):
        ang_diff = np.abs((theta - np.deg2rad(a0) + np.pi) % (2 * np.pi) - np.pi)
        lobe_width = np.deg2rad(42)
        mask = ang_diff < lobe_width
        r[mask] += (arm_lengths[i] - r_inner) * 0.5 * (1 + np.cos(np.pi * ang_diff[mask] / lobe_width))
    pts = np.column_stack([cx + r * np.cos(theta), cy + r * np.sin(theta)]).astype(np.int32)
    cv2.fillPoly(canvas, [pts], 255)
    return canvas

def create_starfish_variants():
    return [
        ("starfish_standard", make_natural_starfish(rotation_deg=0)),
        ("starfish_rot45", make_natural_starfish(rotation_deg=45)),
        ("starfish_unequal", make_natural_starfish(arm_lengths=[105, 80, 95, 75, 90], rotation_deg=20)),
        ("starfish_thick", make_natural_starfish(r_inner=45, rotation_deg=10)),
        ("starfish_slender", make_natural_starfish(arm_lengths=[95, 95, 95, 95, 95], r_inner=24, rotation_deg=35))
    ]

def run_regression_suite():
    print("=" * 80)
    print("FULL REGRESSION & GENERALIZATION TEST SUITE")
    print("=" * 80)
    
    samples_dir = backend_dir / "data" / "samples" / "recognition"
    
    # -------------------------------------------------------------------------
    # 1. Natural Starfish Variants
    # -------------------------------------------------------------------------
    print("\n--- 1. Natural Starfish Variants (Rounded tips, curved edges, natural asymmetry) ---")
    sf_variants = create_starfish_variants()
    for name, img in sf_variants:
        res = recognize_shape_fourier(img, num_descriptors=16)
        cls = res["classification"]
        status = cls["recognition_status"]
        pred = cls["predicted_category"]
        dist = cls["nearest_distance"]
        tau = cls["acceptance_threshold"]
        sim = cls["confidence_percentage"]
        
        print(f"  [{name:20s}] -> Status: {status:10s} | Pred: {pred:10s} (d={dist:.4f}, tau={tau:.4f}, sim={sim:5.1f}%)")
        assert status == "Recognized", f"Expected Recognized for {name}, got {status}"
        assert pred == "star", f"Expected 'star' for {name}, got {pred}"
    print("✓ All natural starfish variants successfully recognized as 'star'.")
    
    # -------------------------------------------------------------------------
    # 2. Geometric Five-Pointed Stars
    # -------------------------------------------------------------------------
    print("\n--- 2. Geometric Five-Pointed Stars (Sharp CAD vertices) ---")
    geom_files = ["star_01.png", "star_02.png", "test_star_01.png"]
    for gname in geom_files:
        img = cv2.imread(str(samples_dir / gname))
        res = recognize_shape_fourier(img, num_descriptors=16)
        cls = res["classification"]
        print(f"  [{gname:20s}] -> Status: {cls['recognition_status']:10s} | Pred: {cls['predicted_category']:10s} (d={cls['nearest_distance']:.4f}, tau={cls['acceptance_threshold']:.4f})")
        assert cls["recognition_status"] == "Recognized"
        assert cls["predicted_category"] == "star"
    print("✓ All geometric stars successfully recognized as 'star'.")
    
    # -------------------------------------------------------------------------
    # 3. Hexagons (Verify Hexagon != Circle)
    # -------------------------------------------------------------------------
    print("\n--- 3. Hexagons (Verify Hexagon != Circle) ---")
    hex_files = ["test_hexagon_01.png", "test_hexagon_02.png", "test_hexagon_03.png"]
    for hname in hex_files:
        img = cv2.imread(str(samples_dir / hname))
        res = recognize_shape_fourier(img, num_descriptors=16)
        cls = res["classification"]
        print(f"  [{hname:20s}] -> Status: {cls['recognition_status']:10s} | Pred: {cls['predicted_category']:10s} (d={cls['nearest_distance']:.4f}, tau={cls['acceptance_threshold']:.4f})")
        assert cls["recognition_status"] == "Recognized"
        assert cls["predicted_category"] == "hexagon", f"Hexagon misclassified as {cls['predicted_category']}!"
    print("✓ Hexagons correctly classified with zero circle confusion.")
    
    # -------------------------------------------------------------------------
    # 4. Trapeziums (Verify Trapezium != Triangle)
    # -------------------------------------------------------------------------
    print("\n--- 4. Trapeziums (Verify Trapezium != Triangle) ---")
    trap_files = ["test_trapezium_01.png", "test_trapezium_02.png", "test_trapezium_03.png"]
    for tname in trap_files:
        img = cv2.imread(str(samples_dir / tname))
        res = recognize_shape_fourier(img, num_descriptors=16)
        cls = res["classification"]
        print(f"  [{tname:20s}] -> Status: {cls['recognition_status']:10s} | Pred: {cls['predicted_category']:10s} (d={cls['nearest_distance']:.4f}, tau={cls['acceptance_threshold']:.4f})")
        assert cls["recognition_status"] == "Recognized"
        assert cls["predicted_category"] == "trapezium", f"Trapezium misclassified as {cls['predicted_category']}!"
    print("✓ Trapeziums correctly classified with zero triangle confusion.")
    
    # -------------------------------------------------------------------------
    # 5. Circles, Triangles, Squares, Rectangles
    # -------------------------------------------------------------------------
    print("\n--- 5. Circles, Triangles, Squares, Rectangles ---")
    other_files = [
        ("test_circle_01.png", "circle"),
        ("test_triangle_01.png", "triangle"),
        ("test_square_01.png", "square"),
        ("test_rectangle_01.png", "rectangle"),
    ]
    for oname, expected_cat in other_files:
        img = cv2.imread(str(samples_dir / oname))
        res = recognize_shape_fourier(img, num_descriptors=16)
        cls = res["classification"]
        print(f"  [{oname:20s}] -> Expected: {expected_cat:10s} | Pred: {cls['predicted_category']:10s} (d={cls['nearest_distance']:.4f})")
        assert cls["recognition_status"] == "Recognized"
        assert cls["predicted_category"] == expected_cat
    print("✓ All standard geometric primitives accurately recognized.")
    
    # -------------------------------------------------------------------------
    # 6. Irregular Polygons & Foreign Outliers (Open-Set Rejection)
    # -------------------------------------------------------------------------
    print("\n--- 6. Irregular Polygons & Outliers (Must be Rejected as Unknown) ---")
    unknown_files = [
        "test_unknown_01_amoeba.png",
        "test_unknown_02_crescent.png",
        "test_unknown_03_cross.png",
        "test_unknown_04_irregular.png"
    ]
    for uname in unknown_files:
        img = cv2.imread(str(samples_dir / uname))
        res = recognize_shape_fourier(img, num_descriptors=16)
        cls = res["classification"]
        status = cls["recognition_status"]
        cand = cls["nearest_candidate_category"]
        dist = cls["nearest_distance"]
        tau = cls["acceptance_threshold"]
        print(f"  [{uname:28s}] -> Status: {status:28s} | Cand: {cand:10s} (d={dist:.4f} > tau={tau:.4f})")
        assert status == "Unknown / Unrecognized Shape", f"Expected Outlier rejection for {uname}, got {status}"
    print("✓ All 4 irregular foreign shapes correctly rejected as 'Unknown / Unrecognized Shape'.")

    # -------------------------------------------------------------------------
    # 7. Independent Test Set Evaluation Report
    # -------------------------------------------------------------------------
    print("\n--- 7. Independent Test Set Evaluation Report ---")
    eval_rep = evaluate_shape_model(num_descriptors=16)
    acc = eval_rep["overall_accuracy_pct"]
    rej_rate = eval_rep["unknown_rejection_rate_pct"]
    print(f"  Test Accuracy:         {acc:.2f}% across {eval_rep['test_samples_count']} specimens")
    print(f"  Unknown Rejection Rate: {rej_rate:.2f}% across {eval_rep['unknown_samples_tested']} outliers")
    print("  Confusion Matrix (7x7):")
    labels = eval_rep["confusion_matrix"]["labels"]
    mat = eval_rep["confusion_matrix"]["matrix"]
    print("             " + "".join([f"{l:>11s}" for l in labels]))
    for idx, row in enumerate(mat):
        print(f"  {labels[idx]:>10s}: " + "".join([f"{v:11d}" for v in row]))
        
    assert acc == 100.0, f"Expected 100% test accuracy, got {acc}%"
    assert rej_rate == 100.0, f"Expected 100% rejection rate, got {rej_rate}%"
    print("\n================ ALL REGRESSION TESTS PASSED (100%) ================")

if __name__ == "__main__":
    run_regression_suite()
