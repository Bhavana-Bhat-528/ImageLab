from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from app.utils.image_io import validate_and_read_image
from app.services.boundary import extract_animal_boundary

router = APIRouter(prefix="/boundary", tags=["Module 2: Morphological Boundary Extraction"])


@router.post("/process")
async def process_boundary(
    file: UploadFile = File(...),
    se_shape: str = Form("rect"),
    se_size: int = Form(3)
):
    """
    Accepts an in-memory animal image upload, validates its integrity,
    and forwards to the boundary extraction service placeholder.
    """
    try:
        content = await file.read()
        image, meta = validate_and_read_image(
            file_bytes=content,
            filename=file.filename or "upload.png"
        )
        result = extract_animal_boundary(
            image=image,
            se_shape=se_shape,
            se_size=se_size
        )
        result["input_metadata"] = meta
        return result
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
