"""
In-memory Image I/O utilities.
Uploaded images and processing outputs are converted strictly in-memory
as required: no permanent saving of uploaded files or processing results.
"""

import base64
import cv2
import numpy as np


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
