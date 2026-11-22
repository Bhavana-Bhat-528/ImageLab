from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.config import settings

router = APIRouter(prefix="/samples", tags=["Sample Datasets"])

CATEGORY_DIR_MAP = {
    "enhancement": "enhancement",
    "boundary": "boundary",
    "boundary-extraction": "boundary",
    "watershed": "watershed",
    "recognition": "recognition",
    "shape-recognition": "recognition",
    "fourier": "recognition",
}

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@router.get("")
async def list_all_sample_categories():
    """
    Lists all available sample dataset categories and their image counts.
    """
    categories = ["enhancement", "boundary", "watershed", "recognition"]
    result = {}
    for cat in categories:
        cat_dir = settings.SAMPLES_DIR / cat
        if cat_dir.exists() and cat_dir.is_dir():
            files = [
                f.name for f in cat_dir.iterdir()
                if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
            ]
            result[cat] = {
                "category": cat,
                "count": len(files),
                "samples": sorted(files)
            }
        else:
            result[cat] = {"category": cat, "count": 0, "samples": []}
    return {"status": "success", "categories": result}


@router.get("/{category}")
async def list_sample_images_by_category(category: str):
    """
    Lists all available benchmark sample images for a given category.
    """
    resolved_dir = CATEGORY_DIR_MAP.get(category.lower())
    if not resolved_dir:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown category '{category}'. Available categories: enhancement, boundary, watershed, recognition."
        )

    cat_dir = settings.SAMPLES_DIR / resolved_dir
    if not cat_dir.exists() or not cat_dir.is_dir():
        return {"category": category, "count": 0, "samples": []}

    files = [
        f.name for f in cat_dir.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
    ]
    return {
        "status": "success",
        "category": category,
        "resolved_category": resolved_dir,
        "count": len(files),
        "samples": sorted(files)
    }


@router.get("/{category}/{filename}")
async def get_sample_image(category: str, filename: str):
    """
    Serves a specific sample benchmark image file from the local repository.
    """
    resolved_dir = CATEGORY_DIR_MAP.get(category.lower())
    if not resolved_dir:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown category '{category}'."
        )

    file_path = settings.SAMPLES_DIR / resolved_dir / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Sample image '{filename}' not found in category '{category}'."
        )

    # Determine media type
    ext = file_path.suffix.lower()
    media_type = "image/png"
    if ext in (".jpg", ".jpeg"):
        media_type = "image/jpeg"
    elif ext == ".bmp":
        media_type = "image/bmp"
    elif ext in (".tif", ".tiff"):
        media_type = "image/tiff"

    return FileResponse(file_path, media_type=media_type)
