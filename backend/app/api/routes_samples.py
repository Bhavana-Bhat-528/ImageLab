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


def is_valid_input_sample(path: Path) -> bool:
    """
    Ensures the sample selector lists ONLY original benchmark input photographs/images.
    Excludes ground-truth masks, results, eroded masks, boundaries, and hidden files.
    """
    if not path.is_file():
        return False
    if path.suffix.lower() not in VALID_EXTENSIONS:
        return False
    name_lower = path.name.lower()
    if name_lower.startswith("."):
        return False
    # Filter out ground-truth masks, result outputs, eroded masks, boundary masks
    excluded_keywords = ["mask", "_gt", "ground_truth", "reference", "result", "boundary", "eroded", "processed"]
    if any(k in name_lower for k in excluded_keywords):
        return False
    return True


@router.get("")
async def list_all_sample_categories():
    """
    Lists all available sample dataset categories and their image counts.
    Strictly returns only original input sample images.
    """
    categories = ["enhancement", "boundary", "watershed", "recognition"]
    result = {}
    for cat in categories:
        cat_dir = settings.SAMPLES_DIR / cat
        if cat_dir.exists() and cat_dir.is_dir():
            files = [
                f.name for f in cat_dir.iterdir()
                if is_valid_input_sample(f)
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
    Strictly returns only original input images.
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
        if is_valid_input_sample(f)
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

    # First check category directory
    file_path = settings.SAMPLES_DIR / resolved_dir / filename
    # If not found, check ground_truth subfolder
    if not file_path.exists():
        file_path = settings.SAMPLES_DIR / resolved_dir / "ground_truth" / filename

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
