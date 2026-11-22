from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.api.routes_enhancement import router as enhancement_router
from app.api.routes_boundary import router as boundary_router
from app.api.routes_watershed import router as watershed_router
from app.api.routes_fourier import router as fourier_router
from app.api.routes_samples import router as samples_router
from app.api.routes_upload import router as upload_router

app = FastAPI(
    title="ImageLab API",
    description="Interactive Digital Image Processing Laboratory Backend",
    version="0.1.0"
)

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"}
    )

# Health Check Endpoint
@app.get("/api/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "app": "ImageLab",
        "version": "0.1.0",
        "modules": [
            "Automatic Image Enhancement",
            "Morphological Boundary Extraction",
            "Morphological Watershed Segmentation",
            "Object Recognition using Fourier Descriptors"
        ]
    }

# Register Routers
app.include_router(enhancement_router, prefix="/api")
app.include_router(boundary_router, prefix="/api")
app.include_router(watershed_router, prefix="/api")
app.include_router(fourier_router, prefix="/api")
app.include_router(samples_router, prefix="/api")
app.include_router(upload_router, prefix="/api")

# URL alias routes to match frontend route naming
app.include_router(boundary_router, prefix="/api/boundary-extraction")
app.include_router(fourier_router, prefix="/api/shape-recognition")

