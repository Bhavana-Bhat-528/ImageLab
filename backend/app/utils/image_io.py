"""
In-memory Image I/O & Validation Utilities.
Uploaded images and processing outputs are handled strictly in-memory
without permanent disk storage, as required by ImageLab architecture.
"""

import base64
from pathlib import Path
from typing import Dict, Any, Tuple
import cv2
import numpy as np

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
MIN_DIMENSION = 16
MAX_DIMENSION = 8192


def validate_and_read_image(
    file_bytes: bytes,
    filename: str,
    as_grayscale: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Validates uploaded file type, size, dimensions, and image integrity.
    Decodes the image directly in memory into a NumPy ndarray without disk I/O.
    
    Returns:
        (image_array, metadata_dict)
    
    Raises:
        ValueError if validation or decoding fails.
    """
    if not file_bytes:
        raise ValueError("Uploaded file is empty (0 bytes).")
    
    # 1. Validate file size
    file_size = len(file_bytes)
    if file_size > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        curr_mb = file_size / (1024 * 1024)
        raise ValueError(
            f"File size ({curr_mb:.2f} MB) exceeds maximum allowed limit of {max_mb:.0f} MB."
        )

    # 2. Validate file extension
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        allowed_str = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValueError(
            f"Unsupported file format '{ext}'. Supported formats: {allowed_str}"
        )

    # 3. In-memory decode and integrity validation
    nparr = np.frombuffer(file_bytes, np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if as_grayscale else cv2.IMREAD_COLOR
    image = cv2.imdecode(nparr, flag)

    if image is None:
        raise ValueError(
            "Image integrity check failed: file bytes could not be decoded as a valid image."
        )

    # 4. Validate dimensions
    h, w = image.shape[:2]
    if h < MIN_DIMENSION or w < MIN_DIMENSION:
        raise ValueError(
            f"Image dimensions ({w}x{h}) are too small. Minimum required dimension is {MIN_DIMENSION}x{MIN_DIMENSION}."
        )
    if h > MAX_DIMENSION or w > MAX_DIMENSION:
        raise ValueError(
            f"Image dimensions ({w}x{h}) exceed maximum limit of {MAX_DIMENSION}x{MAX_DIMENSION}."
        )

    channels = 1 if len(image.shape) == 2 else image.shape[2]
    metadata = {
        "filename": filename,
        "format": ext.lstrip(".").upper(),
        "width": int(w),
        "height": int(h),
        "channels": int(channels),
        "dtype": str(image.dtype),
        "file_size_bytes": file_size,
        "memory_size_bytes": int(image.nbytes),
        "aspect_ratio": round(w / h, 3),
        "status": "valid_in_memory"
    }

    return image, metadata


def read_image_from_bytes(file_bytes: bytes, as_grayscale: bool = False) -> np.ndarray:
    """
    Decodes an uploaded image buffer from memory into a NumPy image array.
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if as_grayscale else cv2.IMREAD_COLOR
    image = cv2.imdecode(nparr, flag)
    if image is None:
        raise ValueError("Could not decode image from provided bytes.")
    return image


def encode_image_to_base64(image: np.ndarray, format_ext: str = ".png") -> str:
    """
    Encodes an in-memory NumPy image array to a Base64 data URL string.
    """
    success, buffer = cv2.imencode(format_ext, image)
    if not success:
        raise ValueError("Could not encode image to specified format.")
    b64_str = base64.b64encode(buffer).decode("utf-8")
    mime_type = "image/png" if format_ext.lower() == ".png" else "image/jpeg"
    return f"data:{mime_type};base64,{b64_str}"


def encode_image_to_bytes(image: np.ndarray, format_ext: str = ".png") -> bytes:
    """
    Encodes an in-memory NumPy image array into bytes for direct streaming.
    """
    success, buffer = cv2.imencode(format_ext, image)
    if not success:
        raise ValueError("Could not encode image.")
    return buffer.tobytes()
