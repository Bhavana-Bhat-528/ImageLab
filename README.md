# ImageLab – Interactive Digital Image Processing Laboratory

An interactive, local-first web application designed for learning, testing, and visualizing digital image processing techniques across four core modules:

1. **Automatic Image Enhancement** (Characteristic analysis, intensity transformation, spatial filtering, histogram comparison, MSE/PSNR/SSIM evaluation)
2. **Morphological Boundary Extraction** (Animal outer boundary extraction via binary segmentation, morphological erosion, set subtraction $\beta(A) = A - (A \ominus B)$, structuring element variations)
3. **Morphological Watershed Segmentation** (Separating touching cells in microscopic images, distance transform, marker generation, region labeling, count estimation)
4. **Object Recognition using Fourier Descriptors** (Boundary contour representation, invariant Fourier descriptors, shape similarity comparison, and classification across 5 categories)

---

## Architecture Overview

- **Frontend**: React 18, Vite, React Router 6, Vanilla CSS (Modern Slate Theme).
- **Backend**: Python 3.10+, FastAPI, OpenCV (`opencv-python-headless`), NumPy, SciPy, Matplotlib, scikit-image, scikit-learn.
- **Communication**: RESTful APIs (`multipart/form-data` upload and in-memory JSON / Base64 image streaming).
- **Persistence**: **Zero permanent image storage** on uploads — all processing is executed in-memory. Local sample images are retained in `backend/data/samples/`.

---

## Quickstart Guide (Windows)

### 1. Backend Setup

Open a terminal in the root directory:

```powershell
cd backend

# Create and activate Python virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
python run.py
```
Backend API will be live at: `http://127.0.0.1:8000`  
Interactive Swagger docs: `http://127.0.0.1:8000/docs`

---

### 2. Frontend Setup

Open a second terminal in the root directory:

```powershell
cd frontend

# Install npm packages
npm install

# Start Vite development server
npm run dev
```
Frontend UI will be live at: `http://localhost:5173`
