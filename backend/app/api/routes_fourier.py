from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query
from app.utils.image_io import validate_and_read_image
from app.services.fourier import recognize_shape_fourier, evaluate_shape_model

router = APIRouter(prefix="/fourier", tags=["Module 4: Fourier Descriptors Shape Recognition"])


@router.post("/process")
async def process_fourier(
    file: UploadFile = File(...),
    num_descriptors: int = Form(16),
    distance_metric: str = Form("euclidean"),
    normalize_scale: bool = Form(True),
    normalize_rotation: bool = Form(True)
):
    """
    Fourier Descriptor-Based Shape Recognition Endpoint:
    1. Preprocesses image: thresholding & noise removal.
    2. Extracts dominant contour and resamples uniformly to 128 points.
    3. Computes Discrete Fourier Transform z(n) -> Z(k).
    4. Normalizes for translation, scale, and rotation invariance.
    5. Reconstructs boundary from K truncated coefficients to demonstrate smoothing.
    6. Matches shape against training dataset library via Nearest-Neighbor classification.
    7. Evaluates performance against independent test split.
    """
    try:
        content = await file.read()
        filename = file.filename or "shape.png"
        image, meta = validate_and_read_image(
            file_bytes=content,
            filename=filename
        )
        result = recognize_shape_fourier(
            image=image,
            num_descriptors=num_descriptors,
            distance_metric=distance_metric,
            normalize_scale=normalize_scale,
            normalize_rotation=normalize_rotation,
            filename=filename
        )
        result["file_metadata"] = meta
        return result
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Fourier descriptor error: {str(exc)}")


@router.get("/evaluation")
async def get_fourier_evaluation(
    num_descriptors: int = Query(16, ge=4, le=64)
):
    """
    Returns the classification report and confusion matrix evaluated on the
    independent test split to assess model accuracy and boundary limitations.
    """
    try:
        return evaluate_shape_model(num_descriptors=num_descriptors)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model evaluation error: {str(exc)}")
