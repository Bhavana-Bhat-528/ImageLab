from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from app.utils.image_io import read_image_from_bytes
from app.services.enhancement import analyze_and_enhance_image

router = APIRouter(prefix="/enhancement", tags=["Module 1: Automatic Image Enhancement"])


@router.post("/process")
async def process_enhancement(
    file: UploadFile = File(...),
    gamma: float = Form(1.0),
    kernel_size: int = Form(3),
    sharpen_strength: float = Form(1.0)
):
    try:
        content = await file.read()
        image = read_image_from_bytes(content)
        result = analyze_and_enhance_image(
            image=image,
            gamma=gamma,
            kernel_size=kernel_size,
            sharpen_strength=sharpen_strength
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
