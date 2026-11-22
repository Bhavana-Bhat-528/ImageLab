from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from app.utils.image_io import validate_and_read_image
from app.services.enhancement import process_image_enhancement

router = APIRouter(prefix="/enhancement", tags=["Module 1: Automatic Image Enhancement"])


@router.post("/process")
async def process_enhancement(
    file: UploadFile = File(...),
    mode: str = Form("auto"),
    gamma: float = Form(1.0),
    equalization_mode: str = Form("clahe"),
    clahe_clip_limit: float = Form(2.0),
    clahe_tile_grid_size: int = Form(8),
    spatial_filter: str = Form("none"),
    kernel_size: int = Form(3),
    sharpen_strength: float = Form(0.0)
):
    """
    Automatic & Manual Image Enhancement Endpoint.
    Accepts an uploaded image file (PNG, JPG, BMP) purely in-memory, validates it,
    analyzes characteristics, decides or accepts operations, executes the pipeline,
    and returns enhanced images (base64 data URL), histograms, diagnostics, and metrics.
    """
    try:
        content = await file.read()
        filename = file.filename or "uploaded_image.png"
        image, meta = validate_and_read_image(
            file_bytes=content,
            filename=filename
        )

        result = process_image_enhancement(
            image=image,
            mode=mode.lower(),
            gamma=gamma,
            equalization_mode=equalization_mode.lower(),
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_grid_size=clahe_tile_grid_size,
            spatial_filter=spatial_filter.lower(),
            kernel_size=kernel_size,
            sharpen_strength=sharpen_strength,
            filename=filename
        )
        result["file_metadata"] = meta
        return result

    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Enhancement error: {str(exc)}")
