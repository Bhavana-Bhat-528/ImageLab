from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from app.utils.image_io import read_image_from_bytes
from app.services.watershed import segment_cells_watershed

router = APIRouter(prefix="/watershed", tags=["Module 3: Morphological Watershed Segmentation"])


@router.post("/process")
async def process_watershed(
    file: UploadFile = File(...),
    threshold_val: int = Form(127),
    min_distance: int = Form(10)
):
    try:
        content = await file.read()
        image = read_image_from_bytes(content)
        result = segment_cells_watershed(
            image=image,
            threshold_val=threshold_val,
            min_distance=min_distance
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
