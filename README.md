# ImageLab – Interactive Digital Image Processing Laboratory

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI%20%7C%20OpenCV-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React%2018%20%7C%20Vite-61DAFB.svg)](https://reactjs.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)

**ImageLab** is a comprehensive, local-first web application designed for learning, experimenting with, and visualizing classic and modern Digital Image Processing (DIP) techniques. Built with an independent **Python FastAPI** backend and a responsive **React/Vite** frontend, ImageLab implements four core foundational laboratories:

1. **Automatic Image Enhancement:** Dynamic defect diagnostics, adaptive intensity mapping (power-law gamma correction, CLAHE, global histogram equalization), spatial noise filtering (bilateral, median, Gaussian), unsharp masking, and quantitative fidelity evaluation (MSE, PSNR, SSIM).
2. **Morphological Boundary Extraction:** Rigorous mathematical set subtraction $\beta(A) = A - (A \ominus B)$ with configurable structuring elements (rectangle, ellipse, cross) and sizes ($3\times 3$ to $15\times 15$), GrabCut animal foreground segmentation, and ground-truth IoU/Dice benchmarking.
3. **Morphological Watershed Segmentation:** Distance-transform marker-controlled watershed segmentation for separating touching and overlapping convex cells/particles, with an 8-stage visual pipeline, area statistics, and benchmark count accuracy verification.
4. **Object Recognition using Fourier Descriptors:** Invariant 1D Discrete Fourier Transform (DFT) contour analysis with uniform arc-length resampling, translation/scale/rotation normalization, harmonic frequency weighting ($\sqrt{k}$), contour reconstruction via Inverse DFT (IDFT), and open-set rejection thresholding ($\tau_c$) across a 7-class prototype library (Circle, Square, Rectangle, Triangle, Star/Starfish, Hexagon, Trapezium).

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Technology Stack & Dependencies](#technology-stack--dependencies)
- [Local Installation & Setup](#local-installation--setup)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [Running Integration & Quality Tests](#running-integration--quality-tests)
- [Sample Datasets & Attribution](#sample-datasets--attribution)
- [DIP Algorithm Formulations & Principles](#dip-algorithm-formulations--principles)
  - [Module 1: Automatic Image Enhancement](#module-1-automatic-image-enhancement)
  - [Module 2: Morphological Boundary Extraction](#module-2-morphological-boundary-extraction)
  - [Module 3: Morphological Watershed Segmentation](#module-3-morphological-watershed-segmentation)
  - [Module 4: Fourier Descriptor Shape Recognition](#module-4-fourier-descriptor-shape-recognition)
- [API Reference Documentation](#api-reference-documentation)
- [Data Handling & Zero Disk Persistence](#data-handling--zero-disk-persistence)
- [Educational Discussion & Algorithm Limitations](#educational-discussion--algorithm-limitations)

---

## System Architecture

ImageLab employs an asynchronous, client-server architecture with strict decoupling between the presentation layer and the image processing engine:

```
+-----------------------------------------------------------------------------------------+
|                                    React 18 Frontend                                    |
|                               (Vite + React Router + CSS)                               |
+--------------------+---------------------+--------------------+-------------------------+
                     |                     |                    |
        POST /process| (Multipart/FormData)|        GET /samples| (JSON / Images)
                     v                     v                    v
+-----------------------------------------------------------------------------------------+
|                                   FastAPI REST Backend                                  |
|                             (Uvicorn ASGI @ 127.0.0.1:8000)                             |
+--------------------+---------------------+--------------------+-------------------------+
| Enhancement Route  |   Boundary Route    |  Watershed Route   |  Fourier Descr. Route   |
| /api/enhancement/* |   /api/boundary/*   |   /api/watershed/* |     /api/fourier/*      |
+--------------------+---------------------+--------------------+-------------------------+
|                                  DIP Processing Layer                                   |
|   OpenCV 4 (cv2)   |   NumPy & SciPy     |    scikit-image    |      scikit-learn       |
+--------------------+---------------------+--------------------+-------------------------+
|                                  In-Memory Data Model                                   |
| Upload Bytes -> cv2.imdecode() -> DIP Transform -> cv2.imencode() -> Base64 Data URI   |
|                       (Zero Temporary Files Persisted to Disk)                          |
+-----------------------------------------------------------------------------------------+
```

### Key Architectural Tenets
1. **Local-First & Offline:** Runs entirely on Windows, macOS, or Linux without external cloud dependencies or third-party tracking.
2. **Zero Permanent Storage:** User-uploaded images and intermediate processing stages are held strictly in memory buffers (`io.BytesIO`, `numpy.ndarray`) and never written to disk.
3. **Stateless REST Communication:** Every analysis request provides its input and parameter overrides; the backend responds with visual artifacts encoded as standard `data:image/png;base64,...` data streams alongside rich structured numerical metadata.
4. **Academic Transparency:** Every module surfaces all intermediate computational representations (histograms, distance maps, markers, Fourier spectra, and eroded silhouettes) rather than presenting a black-box result.

---

## Technology Stack & Dependencies

### Backend Dependencies (`backend/requirements.txt`)
- **Python 3.10+**: Core programming language.
- **FastAPI (0.115+)**: High-performance, asynchronous REST API framework.
- **Uvicorn (0.34+)**: Production-grade ASGI server.
- **OpenCV (`opencv-python-headless` 4.10+)**: Computer vision and digital image processing engine.
- **NumPy (2.1+) & SciPy (1.14+)**: Vectorized array operations, distance transforms, and numerical linear algebra.
- **scikit-image (0.24+)**: Advanced image processing routines and SSIM evaluation.
- **scikit-learn (1.6+)**: Confusion matrices, classification reports, and machine learning metrics.
- **Matplotlib (3.9+)**: In-memory plotting for histogram distributions and harmonic power spectra.
- **Pydantic (2.10+)**: Request/response schema validation and settings management.
- **python-multipart**: Streaming multi-part form data processing for in-memory image uploads.

### Frontend Dependencies (`frontend/package.json`)
- **React 18.3+**: Declarative component-based UI library.
- **Vite 5.4+**: Next-generation frontend build tooling and HMR dev server.
- **React Router 6.28+**: Client-side application routing.
- **Lucide React**: Clean, modern UI iconography.
- **Vanilla CSS**: Custom design system adhering to a dark theme.

---

## Local Installation & Setup

### Prerequisites
- **Node.js**: v18.0.0 or higher ([Download Node.js](https://nodejs.org/))
- **Python**: v3.10 or higher ([Download Python](https://www.python.org/))
- **Git**: ([Download Git](https://git-scm.com/))

### Backend Setup

1. Open PowerShell or Terminal and navigate to the project directory:
   ```powershell
   cd backend
   ```

2. Create a Python virtual environment:
   ```powershell
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - **Windows PowerShell:**
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - **macOS / Linux:**
     ```bash
     source .venv/bin/activate
     ```

4. Install required Python packages:
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Start the FastAPI backend server:
   ```powershell
   python run.py
   ```
   *The backend will bind to `http://127.0.0.1:8000`. You can inspect the interactive OpenAPI Swagger docs at `http://127.0.0.1:8000/docs`.*

---

### Frontend Setup

1. Open a second terminal and navigate to the `frontend` folder:
   ```powershell
   cd frontend
   ```

2. Install dependencies:
   ```powershell
   npm install
   ```

3. Start the Vite development server:
   ```powershell
   npm run dev
   ```
   *The React UI will launch at `http://localhost:5173` and automatically proxy API calls to port `8000`.*

---

## Running Integration & Quality Tests

ImageLab includes a comprehensive test suite covering mathematical identities, benchmark ground-truth evaluations, invariance properties, and failure modes.

### 1. Master System Integration & Quality Test
To verify all 4 modules, in-memory zero-persistence guarantees, and edge-case error guards in a single run:
```powershell
cd backend
.venv\Scripts\python -u run_full_system_test.py
```

### 2. Individual Module Verification Suites
- **Fourier Recognition & Regression Suite (All 7 classes, natural starfish, and open-set rejection):**
  ```powershell
  .venv\Scripts\python test_starfish_regression.py
  .venv\Scripts\python verify_fourier.py
  ```
- **Watershed Segmentation Benchmark Suite (Coins count, microscopic cell IoU/Dice):**
  ```powershell
  .venv\Scripts\python verify_watershed.py
  ```
- **Boundary Extraction Suite (Erosion equation, structuring elements, animal masks):**
  ```powershell
  .venv\Scripts\python verify_boundary.py
  ```
- **Backend API Contract Suite (Uploads, sample retrieval, format rejection):**
  ```powershell
  .venv\Scripts\python test_api.py
  ```

### 3. Frontend Production Build Verification
To ensure all React pages, components, and CSS styles compile cleanly:
```powershell
cd frontend
npm run build
```

---

## Sample Datasets & Attribution

All sample datasets are stored locally in `backend/data/samples/` to enable full offline execution. Ground truth masks and reference annotations are segregated to avoid leaking evaluation targets into user-facing selectors.

| Module | Category / File | Description | Source / Origin | License |
| :--- | :--- | :--- | :--- | :--- |
| **Enhancement** | `01_low_contrast.png` | Compressed dynamic range ($[95, 145]$) | Derived from classic *Cameraman* | CC0 / Public Domain |
| **Enhancement** | `02_dark_underexposed.png` | Severe non-linear underexposure ($\gamma = 3.2$) | Derived from classic *Cameraman* | CC0 / Public Domain |
| **Enhancement** | `03_bright_overexposed.png` | Washed out highlight compression ($\gamma = 0.35$) | Derived from classic *Cameraman* | CC0 / Public Domain |
| **Enhancement** | `04_gaussian_noise.png` | Synthesized thermal sensor noise ($\sigma = 28$) | Derived from classic *Cameraman* | CC0 / Public Domain |
| **Enhancement** | `05_salt_and_pepper_noise.png` | Impulse dropouts ($p = 0.05$) | Derived from classic *Cameraman* | CC0 / Public Domain |
| **Enhancement** | `06_uneven_illumination.png` | Non-uniform spatial gradient on text document | scikit-image `data.page()` | BSD 3-Clause / CC0 |
| **Enhancement** | `07_blurred_out_of_focus.png` | Attenuated edge high frequencies ($\sigma = 1.6$) | Derived from classic *Cameraman* | CC0 / Public Domain |
| **Boundary** | `zebra.jpg` | High-contrast zebra coat against savannah | Muhammad Mahdi Karim (Wikimedia) | CC BY-SA 4.0 |
| **Boundary** | `elephant.jpg` | Curved trunk, tusks, and broad ear contours | Ikiwaner (Wikimedia Commons) | CC BY-SA 2.0 |
| **Boundary** | `dog.jpg` | Portrait canine foreground silhouette | PyTorch Torchvision Gallery | BSD 3-Clause |
| **Watershed** | `01_human_mitosis.png` | Clustered human cancer nuclei in mitosis | DKFZ / scikit-image `data.human_mitosis()` | BSD 3-Clause |
| **Watershed** | `02_cell_culture.png` | Confluent fluorescent cell membranes | scikit-image `data.cell()` | BSD 3-Clause |
| **Watershed** | `03_protein_transport_nuclei.png`| Nuclear translocation assay | Broad Bioimage Benchmark Collection | Open Science / BSD |
| **Watershed** | `04_overlapping_coins.png` | 24 partially overlapping circular boundaries | scikit-image `data.coins()` | CC0 / Public Domain |
| **Recognition** | 35 Training Prototypes | 5 prototypes across 7 classes (Circle, Square, Rect, Tri, Star, Hex, Trap) | Synthesized & Curated Vector Contours | CC0 / Public Domain |
| **Recognition** | 21 Independent Test Samples | 3 test specimens per class with unobserved rotations and scales | Synthesized Independent Set | CC0 / Public Domain |
| **Recognition** | 4 Outlier Unknowns | Amoeba, crescent, cross, and 7-vertex irregular polygon | Synthesized Outlier Benchmark | CC0 / Public Domain |

---

## DIP Algorithm Formulations & Principles

### Module 1: Automatic Image Enhancement

#### 1. Image Diagnostic Characterization
Before applying any enhancement operation, the input image $I(x, y)$ is evaluated across five objective statistical metrics:
- **Mean Luminance ($\mu$):** $\mu = \frac{1}{MN} \sum_{x,y} I(x, y)$
- **Standard Deviation ($\sigma$):** $\sigma = \sqrt{\frac{1}{MN} \sum_{x,y} (I(x, y) - \mu)^2}$
- **Noise Estimation ($\sigma_n$):** Computed via the median absolute deviation of the high-frequency Laplacian residuals:
  $$\sigma_n = \frac{\text{median}(|\nabla^2 I - \text{median}(\nabla^2 I)|)}{0.6745}$$
- **Sharpness Acuity ($S$):** Normalized variance of the discrete 2D Laplacian operator:
  $$S = \text{Var}\left(\frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}\right)$$
- **Histogram Uniformity / Entropy:** Cumulative dynamic range spread $[p_5, p_{95}]$.

#### 2. Adaptive Decision Matrix
An analytical decision mechanism selects and sequences operations based on the detected defects:
- **Dark Underexposure ($\mu < 95$):** Power-law gamma expansion with $\gamma = \max(0.35, \mu / 128.0) < 1.0$.
- **Bright Overexposure ($\mu > 165$):** Power-law gamma compression with $\gamma = \min(2.5, \mu / 110.0) > 1.0$.
- **Low Contrast ($\sigma < 42$ or dynamic range $< 135$):** Contrast-Limited Adaptive Histogram Equalization (CLAHE) with tile grid $8\times 8$ and clip limit $2.5$.
- **Impulse Noise:** $3\times 3$ or $5\times 5$ non-linear 2D Median filtering.
- **Sensor Noise:** Edge-preserving spatial Bilateral filtering ($d=7, \sigma_{\text{color}}=45, \sigma_{\text{space}}=45$).
- **Blur / Soft Edges:** Unsharp masking with Gaussian spatial subtraction:
  $$I_{\text{sharp}} = I + \alpha \cdot (I - G_\sigma * I), \quad \alpha \in [0.8, 1.5]$$

#### 3. Quantitative Evaluation Metrics
- **Mean Squared Error (MSE):** $\text{MSE} = \frac{1}{MN}\sum_{x=0}^{M-1}\sum_{y=0}^{N-1} [I_1(x,y) - I_2(x,y)]^2$
- **Peak Signal-to-Noise Ratio (PSNR):** $\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}_I^2}{\text{MSE}}\right)$
- **Structural Similarity Index (SSIM):**
  $$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$

---

### Module 2: Morphological Boundary Extraction

#### 1. Mathematical Morphology Identity
Morphological boundary extraction is implemented strictly in accordance with classical digital image processing principles:
$$\beta(A) = A - (A \ominus B) = A \cap (A \ominus B)^c$$
where:
- $A \subset \mathbb{Z}^2$ is the segmented binary foreground mask ($A(x,y) \in \{0, 255\}$).
- $B \subset \mathbb{Z}^2$ is the structuring element (SE) kernel of specified geometry and radius.
- $A \ominus B = \{z \in \mathbb{Z}^2 \mid (B)_z \subseteq A\}$ is the morphological erosion of $A$ by $B$.
- $\beta(A)$ is the extracted boundary perimeter.

#### 2. Structuring Element Geometries
Three discrete structuring element kernels are supported:
- **Rectangle (`cv2.MORPH_RECT`):** 8-connected symmetric square kernel, generating thicker orthogonal boundaries.
- **Cross (`cv2.MORPH_CROSS`):** 4-connected cross kernel, yielding thin minimal connectivity contours.
- **Ellipse (`cv2.MORPH_ELLIPSE`):** Discretized Euclidean disc, preserving isotropic, rotationally uniform boundary contours.

#### 3. Bounding-Box GrabCut Foreground Segmentation
For natural animal photographs, foreground masks are initialized using iterative energy-minimization graph cuts (`cv2.grabCut`):
1. User-specified or preset bounding box separates sure background from unknown candidate foreground.
2. 5-component Gaussian Mixture Models (GMM) model color distributions.
3. Graph cuts via max-flow/min-cut segment the foreground boundary without manual per-pixel rotoscoping.

---

### Module 3: Morphological Watershed Segmentation

#### 1. The 8-Stage Watershed Pipeline
Touching convex objects (such as clustered cell nuclei or overlapping coins) cannot be segmented by simple global thresholding. ImageLab implements marker-controlled morphological watershed segmentation:
1. **Grayscale Conversion & Gaussian Denoising:** $5\times 5$ Gaussian kernel attenuates sensor grain.
2. **Binary Thresholding:** Otsu adaptive binarization with automated image polarity detection.
3. **Morphological Opening ($A \circ B = (A \ominus B) \oplus B$):** $3\times 3$ elliptical kernel eliminates isolated debris.
4. **Morphological Dilation ($A \oplus B$):** Expands foreground to isolate sure background boundaries.
5. **Exact Euclidean Distance Transform:** Computes the Euclidean distance to the nearest background pixel:
   $$D(p) = \min_{q \in \text{Background}} \|p - q\|_2$$
6. **Sure Foreground Thresholding:** Identifies dominant regional peaks:
   $$\text{SureFG} = \{p \mid D(p) \ge \text{ratio} \cdot \max(D)\}$$
7. **Unknown Boundary Region Calculation:**
   $$\text{Unknown} = \text{SureBG} - \text{SureFG}$$
8. **Topological Marker Generation & Flooding:** Connected components label the foreground peaks as initial watershed seeds ($1, 2, \dots, K$); `cv2.watershed` floods gradient topographic basins until dam lines ($-1$) form between touching boundaries.

---

### Module 4: Fourier Descriptor Shape Recognition

#### 1. Uniform Arc-Length Resampling
To ensure discrete Fourier coefficients are invariant to contour point density and non-uniform boundary rasterization, boundary points are extracted via `cv2.findContours` and resampled to exactly $M = 128$ equidistant points along cumulative perimeter length $L$:
$$s(n) = \frac{n}{M} \cdot L, \quad n \in \{0, 1, \dots, M-1\}$$
Represented as a 1D complex coordinate sequence:
$$z(n) = x(n) + j \cdot y(n)$$

#### 2. Discrete Fourier Transform (DFT)
The discrete Fourier coefficients $Z(k)$ are computed as:
$$Z(k) = \sum_{n=0}^{M-1} z(n) e^{-j \frac{2\pi k n}{M}}, \quad k \in \{0, 1, \dots, M-1\}$$

#### 3. Invariance Normalization
- **Translation Invariance:** The DC coefficient $Z(0)$ corresponds to the contour center of mass. Setting $Z(0) = 0$ removes all dependency on spatial position.
- **Scale Invariance:** All harmonic magnitudes are divided by the magnitude of the dominant fundamental harmonic:
  $$s = \max(|Z(1)|, |Z(M-1)|), \quad Z_{\text{scaled}}(k) = \frac{Z(k)}{s}$$
- **Rotation and Starting-Point Invariance:** Phase information is discarded by taking the absolute magnitude spectrum $|Z_{\text{scaled}}(k)|$.
- **Harmonic Frequency Weighting ($\sqrt{k}$):** To prevent high-frequency corner signatures (such as the $k=5, 7$ harmonics for hexagons) from decaying below the noise floor due to $O(1/k^2)$ spectral decay, harmonics are weighted:
  $$f(k) = \sqrt{k} \cdot |Z_{\text{scaled}}(k)|$$

#### 4. Inverse DFT Contour Reconstruction
To demonstrate how low-frequency harmonics encode gross geometry while higher harmonics encode sharp corners, the contour is reconstructed from $K$ retained coefficients:
$$\hat{z}(n) = \frac{1}{M} \sum_{k \in \mathcal{K}} Z(k) e^{j \frac{2\pi k n}{M}}$$

#### 5. Open-Set Rejection Thresholding
To prevent foreign or irregular shapes from being falsely classified into known classes, an empirical acceptance threshold is calibrated per category $c$:
$$\tau_c = \max(0.065, \min(0.38, 1.3 \times R_c))$$
where $R_c$ is the maximum intra-class training distance. An unknown query $q$ is rejected as `'Unknown / Unrecognized Shape'` whenever:
$$\min_{i \in c^*} \|\mathbf{f}_q - \mathbf{f}_i\| > \tau_{c^*}$$

---

## API Reference Documentation

### General Endpoints

#### `GET /api/health`
System health check and module capability discovery.
- **Response:**
  ```json
  {
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
  ```

#### `GET /api/samples`
Lists all local benchmark sample files categorized by module.

#### `GET /api/samples/{category}/{filename}`
Streams the binary raw image bytes for a specified local sample image.

#### `POST /api/upload/validate`
Validates uploaded image file format, byte integrity, and dimensions strictly in memory without saving to disk.

---

### Module 1: Enhancement

#### `POST /api/enhancement/process`
Applies automated or manual image enhancement.
- **Form Parameters:**
  - `file`: Image file (`multipart/form-data`)
  - `mode`: `'auto'` or `'manual'`
  - `gamma`: Float (e.g., `1.0`)
  - `equalization_mode`: `'none'`, `'global'`, or `'clahe'`
  - `clahe_clip_limit`: Float (e.g., `2.0`)
  - `clahe_tile_grid_size`: Integer (e.g., `8`)
  - `spatial_filter`: `'none'`, `'gaussian'`, `'median'`, or `'bilateral'`
  - `kernel_size`: Odd integer (`3`, `5`, `7`)
  - `sharpen_strength`: Float (`0.0` to `2.5`)
- **Response:** JSON payload containing input/output statistical diagnostics, applied pipeline operations with justifications, quantitative metrics (MSE, PSNR, SSIM), 256-bin histogram arrays, and Base64 images.

---

### Module 2: Boundary Extraction

#### `POST /api/boundary/process`
Executes morphological foreground segmentation and boundary extraction.
- **Form Parameters:**
  - `file`: Image file (`multipart/form-data`)
  - `se_shape`: `'rect'`, `'ellipse'`, or `'cross'`
  - `se_size`: Odd integer (`3` to `15`)
  - `iterations`: Integer (`1` to `5`)
  - `segmentation_method`: `'grabcut'`, `'otsu'`, or `'adaptive'`
  - `margin_pct`: Float bounding box margin (`1.0` to `20.0`)
  - `overlay_color`: `'cyan'`, `'red'`, `'yellow'`, `'green'`, or `'magenta'`
  - `display_thickness`: Integer (`1` to `5`)
- **Response:** JSON payload containing foreground and boundary pixel counts, connected components, ground-truth IoU and Dice scores, cross-SE benchmark table, and Base64 mask viewports.

---

### Module 3: Watershed Segmentation

#### `POST /api/watershed/process`
Executes distance-transform marker-controlled watershed segmentation.
- **Form Parameters:**
  - `file`: Image file (`multipart/form-data`)
  - `threshold_method`: `'otsu'`, `'manual'`, or `'adaptive'`
  - `manual_threshold`: Integer (`0` to `255`)
  - `blur_ksize`: Odd integer (`3` to `11`)
  - `morph_ksize`: Odd integer (`3` to `9`)
  - `dist_ratio`: Float threshold ratio (`0.05` to `0.85`)
  - `min_cell_area`: Integer pixel threshold (`5` to `500`)
- **Response:** JSON payload containing estimated cell count, reference cell count comparison, per-region area and centroid table, ground-truth mask IoU/Dice, and Base64 images for all 8 visual pipeline stages.

---

### Module 4: Shape Recognition

#### `POST /api/fourier/process`
Computes invariant Fourier descriptors and classifies contour shapes with open-set rejection.
- **Form Parameters:**
  - `file`: Image file (`multipart/form-data`)
  - `num_descriptors`: Integer harmonics retained (`4` to `64`, default `16`)
  - `distance_metric`: `'euclidean'`, `'manhattan'`, or `'cosine'`
  - `normalize_scale`: Boolean (default `true`)
  - `normalize_rotation`: Boolean (default `true`)
- **Response:** JSON payload containing `recognition_status` (`'Recognized'` vs `'Unknown / Unrecognized Shape'`), `predicted_category`, nearest candidate match, Euclidean descriptor distance, acceptance threshold $\tau_c$, calibrated similarity percentage, top-5 matches table, spectrum bar chart Base64, and reconstructed contour Base64.

#### `GET /api/fourier/evaluation`
Retrieves pre-computed model performance on the independent test set, including overall accuracy, macro F1-score, $7\times 7$ confusion matrix, and out-of-distribution rejection rate.

---

## Data Handling & Zero Disk Persistence

ImageLab enforces a strict **zero permanent storage** policy for user uploads and intermediate results:

1. **In-Memory Decoding:** Uploaded `UploadFile` streams are buffered into `io.BytesIO` memory structures and decoded via `cv2.imdecode(np.frombuffer(...))`.
2. **In-Memory Encoding:** Visual outputs are serialized directly to memory via `cv2.imencode('.png', ...)` and encoded into standard base64 data URIs for client transmission.
3. **No File System Leakage:** Automated integration tests (`test_in_memory_and_zero_persistence()`) verify that the count of files on the server's local file system remains unchanged before, during, and after processing.
4. **Local Sample Access:** Only verified, openly licensed benchmark datasets are permanently stored in the read-only repository path `backend/data/samples/`.

---

## Educational Discussion & Algorithm Limitations

### 1. Internal Markings & Topological Holes
- **Observation:** 1D Fourier descriptors parameterize strictly the outermost external closed boundary curve $z(n)$.
- **Limitation:** A solid circular disc and a hollow circular ring (washer) of identical outer diameter yield identical normalized Fourier descriptors.
- **Mitigation:** In complex topological classification tasks, boundary Fourier descriptors should be supplemented with interior regional moments (Hu moments) or topological Euler characteristics.

### 2. Boundary Occlusion & Touching Boundaries
- **Observation:** Fourier descriptor extraction assumes a single unoccluded, continuous periodic boundary curve.
- **Limitation:** If two animals overlap or cell boundaries merge, contour tracing yields a fused envelope whose harmonics bear no resemblance to either individual object.
- **Mitigation:** Decompose overlapping silhouettes using Module 3 (Watershed Segmentation) prior to computing boundary descriptors.

### 3. Harmonic Truncation Trade-Off ($K$)
- **Observation:** Low harmonics ($K = 4 \text{ to } 8$) capture broad global aspect ratios, while higher harmonics capture localized vertices and corners.
- **Limitation:** Truncating $K$ too low causes squares and hexagons to smooth into circles; setting $K$ too high overfits to discrete single-pixel rasterization staircases.
- **Guideline:** $K = 16 \text{ to } 32$ provides an optimal balance between vertex preservation and noise immunity.

### 4. Over-Segmentation vs. Under-Segmentation in Watershed
- **Observation:** The Euclidean distance transform peak ratio controls marker density.
- **Limitation:** Setting $\text{dist\_ratio}$ too low ($\le 0.15$) causes local intensity ripples within a single cell nucleus to register as multiple markers, resulting in severe over-segmentation. Conversely, setting $\text{dist\_ratio}$ too high ($\ge 0.55$) merges closely touching cells into a single region.
- **Guideline:** Calibrate distance ratios between $0.25 \text{ and } 0.35$ for biological microscopy specimens.
