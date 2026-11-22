from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from app.utils.image_io import validate_and_read_image
from app.services.watershed import segment_cells_watershed

router = APIRouter(prefix="/watershed", tags=["Module 3: Morphological Watershed Segmentation"])


@router.post("/process")
async def process_watershed(
    file: UploadFile = File(...),
    threshold_val: int = Form(127),
    min_distance: int = Form(10)
):
    """
    Accepts an in-memory microscopic image upload, validates its integrity,
    and forwards to the watershed segmentation service placeholder.
    """
    try:
        content = await file.read()
        image, meta = validate_and_read_image(
            file_bytes=content,
            filename=file.filename or "upload.png"
        )
        result = segment_cells_watershed(
            image=image,
            threshold_val=threshold_val,
            min_distance=min_distance
        )
        result["input_metadata"] = meta
        return result
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
