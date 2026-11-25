from typing import Optional
from pathlib import Path
import cv2
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from app.utils.image_io import validate_and_read_image, encode_image_to_base64
from app.services.enhancement import (
    process_image_enhancement,
    analyze_image_characteristics,
    compute_histogram_data
)
from app.config import settings

router = APIRouter(prefix="/enhancement", tags=["Module 1: Automatic Image Enhancement"])


@router.post("/analyze")
async def analyze_image_endpoint(
    file: UploadFile = File(...)
):
    """
    Analyzes input image characteristics (brightness, contrast, noise, sharpness)
    and returns metrics, identified defects, and 256-bin histogram data.
    """
    try:
        content = await file.read()
        filename = file.filename or "uploaded_image.png"
        image, meta = validate_and_read_image(
            file_bytes=content,
            filename=filename
        )

        characteristics = analyze_image_characteristics(image)
        hist_data = compute_histogram_data(image)
        orig_b64 = encode_image_to_base64(image)

        return {
            "status": "success",
            "module": "enhancement",
            "filename": filename,
            "file_metadata": meta,
            "characteristics": characteristics,
            "histogram": hist_data,
            "original_base64": orig_b64
        }
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image analysis error: {str(exc)}")


@router.post("/process")
async def process_enhancement(
    file: UploadFile = File(...),
    mode: str = Form("auto"),
    category: str = Form("auto"),
    operation: str = Form("auto"),
    # Common / Relevant parameters
    gamma: float = Form(1.0),
    equalization_mode: str = Form("clahe"),
    clahe_clip_limit: float = Form(2.0),
    clahe_tile_grid_size: int = Form(8),
    spatial_filter: str = Form("none"),
    kernel_size: int = Form(3),
    sharpen_strength: float = Form(0.0),
    # Optional parameters for backward compatibility
    log_c: Optional[float] = Form(None),
    contrast_r1: Optional[int] = Form(None),
    contrast_s1: int = Form(0),
    contrast_r2: Optional[int] = Form(None),
    contrast_s2: int = Form(255),
    slice_low: int = Form(100),
    slice_high: int = Form(180),
    slice_preserve_bg: bool = Form(True),
    bit_plane: int = Form(7),
    reference_file: Optional[UploadFile] = File(None),
    reference_sample: Optional[str] = Form(None),
    local_window_size: int = Form(8),
    smoothing_type: str = Form("box"),
    alpha_trim: float = Form(0.2),
    sharpening_type: str = Form("laplacian"),
    laplacian_mask: str = Form("4_neighbor"),
    unsharp_sigma: Optional[float] = Form(None),
    highboost_k: float = Form(1.0),
    gradient_operator: str = Form("sobel"),
    gradient_calc: str = Form("abs_sum"),
    gradient_mode: str = Form("sharpened")
):
    """
    Automatic Image Enhancement Execution Endpoint.
    Analyzes image characteristics, decides or accepts operations, executes the pipeline,
    and returns enhanced images (base64 data URL), histograms, diagnostics, and metrics.
    """
    try:
        content = await file.read()
        filename = file.filename or "uploaded_image.png"
        image, meta = validate_and_read_image(
            file_bytes=content,
            filename=filename
        )

        ref_image = None
        if reference_file is not None and reference_file.filename:
            ref_bytes = await reference_file.read()
            if len(ref_bytes) > 0:
                ref_image, _ = validate_and_read_image(
                    file_bytes=ref_bytes,
                    filename=reference_file.filename
                )
        elif reference_sample:
            sample_path = settings.SAMPLES_DIR / "enhancement" / reference_sample
            if not sample_path.exists():
                sample_path = settings.SAMPLES_DIR / "enhancement" / "ground_truth" / reference_sample
            if sample_path.exists():
                ref_image = cv2.imread(str(sample_path))

        result = process_image_enhancement(
            image=image,
            mode=mode.lower(),
            category=category.lower(),
            operation=operation.lower(),
            gamma=gamma,
            equalization_mode=equalization_mode.lower(),
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_grid_size=clahe_tile_grid_size,
            spatial_filter=spatial_filter.lower(),
            kernel_size=kernel_size,
            sharpen_strength=sharpen_strength,
            log_c=log_c,
            contrast_r1=contrast_r1,
            contrast_s1=contrast_s1,
            contrast_r2=contrast_r2,
            contrast_s2=contrast_s2,
            slice_low=slice_low,
            slice_high=slice_high,
            slice_preserve_bg=slice_preserve_bg,
            bit_plane=bit_plane,
            reference_image=ref_image,
            local_window_size=local_window_size,
            smoothing_type=smoothing_type.lower(),
            alpha_trim=alpha_trim,
            sharpening_type=sharpening_type.lower(),
            laplacian_mask=laplacian_mask.lower(),
            unsharp_sigma=unsharp_sigma,
            highboost_k=highboost_k,
            gradient_operator=gradient_operator.lower(),
            gradient_calc=gradient_calc.lower(),
            gradient_mode=gradient_mode.lower(),
            filename=filename
        )
        result["file_metadata"] = meta
        return result

    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Enhancement error: {str(exc)}")
