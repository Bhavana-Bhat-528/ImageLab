from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from app.utils.image_io import validate_and_read_image
from app.services.watershed import segment_cells_watershed

router = APIRouter(prefix="/watershed", tags=["Module 3: Morphological Watershed Segmentation"])


@router.post("/process")
async def process_watershed(
    file: UploadFile = File(...),
    threshold_method: str = Form("otsu"),
    threshold_val: int = Form(128),
    blur_ksize: int = Form(5),
    morph_ksize: int = Form(3),
    dist_ratio: float = Form(0.35),
    min_cell_area: int = Form(25),
    invert_polarity: str = Form("auto"),
    boundary_color: str = Form("red"),
    boundary_thickness: int = Form(2),
    manual_reference_count: Optional[int] = Form(None)
):
    """
    Morphological Watershed Segmentation Endpoint:
    1. Preprocesses microscopic specimen image (grayscale & Gaussian smoothing).
    2. Binarizes image via Otsu, manual threshold, or adaptive threshold.
    3. Cleans noise via morphological opening and expands background via dilation.
    4. Computes Euclidean distance transform and extracts sure foreground seeds.
    5. Generates connected-component markers and runs cv2.watershed.
    6. Identifies, counts, and measures individual cells.
    7. Compares against reference count and ground-truth masks.
    """
    try:
        content = await file.read()
        filename = file.filename or "microscopy.png"
        image, meta = validate_and_read_image(
            file_bytes=content,
            filename=filename
        )

        result = segment_cells_watershed(
            image=image,
            threshold_method=threshold_method,
            threshold_val=threshold_val,
            blur_ksize=blur_ksize,
            morph_ksize=morph_ksize,
            dist_ratio=dist_ratio,
            min_cell_area=min_cell_area,
            invert_polarity=invert_polarity,
            boundary_color=boundary_color,
            boundary_thickness=boundary_thickness,
            manual_reference_count=manual_reference_count,
            filename=filename
        )
        result["file_metadata"] = meta
        return result

    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Watershed segmentation error: {str(exc)}")
