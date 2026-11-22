from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from app.utils.image_io import validate_and_read_image
from app.services.fourier import compute_fourier_descriptors

router = APIRouter(prefix="/fourier", tags=["Module 4: Fourier Descriptors Shape Recognition"])


@router.post("/process")
async def process_fourier(
    file: UploadFile = File(...),
    num_descriptors: int = Form(16)
):
    """
    Accepts an in-memory object image upload, validates its integrity,
    and forwards to the Fourier descriptors service placeholder.
    """
    try:
        content = await file.read()
        image, meta = validate_and_read_image(
            file_bytes=content,
            filename=file.filename or "upload.png"
        )
        result = compute_fourier_descriptors(
            image=image,
            num_descriptors=num_descriptors
        )
        result["input_metadata"] = meta
        return result
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
