from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from app.utils.image_io import validate_and_read_image
from app.services.boundary import extract_animal_boundary

router = APIRouter(prefix="/boundary", tags=["Module 2: Morphological Boundary Extraction"])


@router.post("/process")
async def process_boundary(
    file: UploadFile = File(...),
    se_shape: str = Form("rect"),
    se_size: int = Form(3),
    iterations: int = Form(1),
    segmentation_method: str = Form("grabcut"),
    box_x: Optional[int] = Form(None),
    box_y: Optional[int] = Form(None),
    box_w: Optional[int] = Form(None),
    box_h: Optional[int] = Form(None),
    margin_pct: float = Form(6.0),
    grabcut_iters: int = Form(5),
    overlay_color: str = Form("cyan"),
    display_thickness: int = Form(1)
):
    """
    Morphological Boundary Extraction Endpoint:
    1. Segments animal foreground (GrabCut with adjustable bounding box).
    2. Performs erosion and boundary set subtraction: beta(A) = A - (A (-) B).
    3. Generates multi-stage visualizations, overlays, metrics, and structuring element benchmarks.
    """
    try:
        content = await file.read()
        filename = file.filename or "animal.jpg"
        image, meta = validate_and_read_image(
            file_bytes=content,
            filename=filename
        )

        result = extract_animal_boundary(
            image=image,
            se_shape=se_shape,
            se_size=se_size,
            iterations=iterations,
            segmentation_method=segmentation_method,
            box_x=box_x,
            box_y=box_y,
            box_w=box_w,
            box_h=box_h,
            margin_pct=margin_pct,
            grabcut_iters=grabcut_iters,
            overlay_color=overlay_color,
            display_thickness=display_thickness,
            filename=filename
        )
        result["file_metadata"] = meta
        return result

    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Boundary extraction error: {str(exc)}")
