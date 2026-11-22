from fastapi import APIRouter, File, UploadFile, HTTPException
from app.utils.image_io import validate_and_read_image

router = APIRouter(prefix="/upload", tags=["In-Memory Upload & Validation"])


@router.post("/validate")
async def validate_uploaded_image(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file (PNG, JPG, JPEG, BMP), validates file type,
    size, dimensions, and image integrity strictly in-memory.
    Never permanently saves uploaded files to disk.
    """
    try:
        content = await file.read()
        _, metadata = validate_and_read_image(
            file_bytes=content,
            filename=file.filename or "uploaded_image.png"
        )
        return {
            "status": "success",
            "message": "Image validated and decoded into memory successfully.",
            "metadata": metadata
        }
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected validation error: {str(exc)}")
