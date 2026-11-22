from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from app.utils.image_io import validate_and_read_image
from app.services.enhancement import analyze_and_enhance_image

router = APIRouter(prefix="/enhancement", tags=["Module 1: Automatic Image Enhancement"])


@router.post("/process")
async def process_enhancement(
    file: UploadFile = File(...),
    gamma: float = Form(1.0),
    kernel_size: int = Form(3),
    sharpen_strength: float = Form(1.0)
):
    """
    Accepts an in-memory image upload, validates its integrity and dimensions,
    and forwards to the enhancement service placeholder.
    """
    try:
        content = await file.read()
        image, meta = validate_and_read_image(
            file_bytes=content,
            filename=file.filename or "upload.png"
        )
        result = analyze_and_enhance_image(
            image=image,
            gamma=gamma,
            kernel_size=kernel_size,
            sharpen_strength=sharpen_strength
        )
        result["input_metadata"] = meta
        return result
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
