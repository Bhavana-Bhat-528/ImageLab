from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.config import settings

router = APIRouter(prefix="/samples", tags=["Sample Images"])


@router.get("/{module_name}")
async def list_sample_images(module_name: str):
    """
    Returns the list of local sample images available for a given module.
    """
    module_dir = settings.SAMPLES_DIR / module_name
    if not module_dir.exists() or not module_dir.is_dir():
        return {"module": module_name, "samples": []}
    
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [
        f.name for f in module_dir.iterdir()
        if f.is_file() and f.suffix.lower() in valid_exts
    ]
    return {"module": module_name, "samples": files}


@router.get("/{module_name}/{filename}")
async def get_sample_image(module_name: str, filename: str):
    """
    Serves a specific local sample image file.
    """
    file_path = settings.SAMPLES_DIR / module_name / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Sample image not found")
    return FileResponse(file_path)
