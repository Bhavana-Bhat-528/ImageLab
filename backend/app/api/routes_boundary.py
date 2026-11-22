from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from app.utils.image_io import read_image_from_bytes
from app.services.boundary import extract_animal_boundary

router = APIRouter(prefix="/boundary", tags=["Module 2: Morphological Boundary Extraction"])


@router.post("/process")
async def process_boundary(
    file: UploadFile = File(...),
    se_shape: str = Form("rect"),
    se_size: int = Form(3)
):
    try:
        content = await file.read()
        image = read_image_from_bytes(content)
        result = extract_animal_boundary(
            image=image,
            se_shape=se_shape,
            se_size=se_size
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
