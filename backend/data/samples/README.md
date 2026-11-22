# ImageLab Standard Benchmark Datasets: Documentation, Attribution & Licensing

This directory contains verified, openly licensed, and benchmark datasets curated for the four modules of **ImageLab – Interactive Digital Image Processing Laboratory**.

All datasets are stored locally in the project repository to enable local-first offline experimentation. All user-uploaded images are processed strictly in-memory without persistent disk storage.

---

## 1. Automatic Image Enhancement (`/enhancement`)

These images evaluate automatic histogram analysis, dynamic intensity transformations (gamma correction, global/adaptive histogram equalization, contrast stretching), and spatial filtering (Laplacian sharpening, Gaussian smoothing, median filtering).

| File Name | Resolution | Format | Origin / Attribution | License | Benchmark Experimental Suitability |
|---|---|---|---|---|---|
| `01_low_contrast.png` | 512 × 512 | PNG Grayscale | Derived from classic standard test image *Cameraman* (scikit-image `data.camera()`) | Public Domain / CC0 | Compressed histogram restricted to dynamic range $[95, 145]$. Tests histogram stretching and CLAHE recovery of latent detail. |
| `02_dark_underexposed.png` | 512 × 512 | PNG Grayscale | Derived from *Cameraman* with severe power-law degradation ($\gamma = 3.2$) | Public Domain / CC0 | Heavy shadow clipping and underexposure. Evaluates power-law gamma expansion ($\gamma < 1.0$) and log transformation. |
| `03_bright_overexposed.png` | 512 × 512 | PNG Grayscale | Derived from *Cameraman* with high-key compressive gamma ($\gamma = 0.35$) | Public Domain / CC0 | Saturated highlights and washed-out dynamic range. Evaluates highlight compression and adaptive histogram equalization. |
| `04_gaussian_noise.png` | 512 × 512 | PNG Grayscale | Synthesized additive zero-mean Gaussian noise ($\sigma = 28$) on *Cameraman* | Public Domain / CC0 | Continuous sensor thermal/electronic noise. Tests spatial Gaussian vs. bilateral smoothing prior to sharpening. |
| `05_salt_and_pepper_noise.png` | 512 × 512 | PNG Grayscale | Synthesized impulse noise ($p = 0.05$) on *Cameraman* | Public Domain / CC0 | Saturated impulse dropouts. Demonstrates necessity of non-linear median filtering over linear box/Gaussian filters. |
| `06_uneven_illumination.png` | 191 × 384 | PNG Grayscale | Standard text document from scikit-image `data.page()` | BSD 3-Clause / CC0 | Strong non-uniform background gradient. Benchmarks local adaptive thresholding and homomorphic filtering. |
| `07_blurred_out_of_focus.png` | 512 × 512 | PNG Grayscale | Synthesized optical Gaussian blur ($5 \times 5, \sigma = 1.6$) on *Cameraman* | Public Domain / CC0 | Low spatial edge acuity and high-frequency attenuation. Evaluates unsharp masking and high-boost filtering. |

---

## 2. Morphological Boundary Extraction (`/boundary`)

These real-world animal photographs have clearly distinguishable foreground objects against natural backgrounds for morphological binarization, erosion ($A \ominus B$), and set subtraction $\beta(A) = A - (A \ominus B)$.

| File Name | Resolution | Format | Source & Author | License | Benchmark Experimental Suitability |
|---|---|---|---|---|---|
| `zebra.jpg` | 800 × 533 | JPEG RGB | Wikimedia Commons: *Plains Zebra (Equus quagga)* by Muhammad Mahdi Karim | Creative Commons Attribution-Share Alike 4.0 International (CC BY-SA 4.0) | High-contrast animal coat pattern with fine leg and tail contours. Demonstrates the impact of structuring element shapes (rect vs. cross vs. ellipse) on boundary connectivity. |
| `elephant.jpg` | 800 × 533 | JPEG RGB | Wikimedia Commons: *African Bush Elephant* by Ikiwaner | Creative Commons Attribution-Share Alike 2.0 Generic (CC BY-SA 2.0) | Massive silhouette featuring curved trunk, tusks, and wide ear geometries for topological boundary preservation. |
| `dog.jpg` | 500 × 500 | JPEG RGB | PyTorch Torchvision Gallery: *Canine portrait* (PyTorch Repository) | BSD 3-Clause License | Natural animal foreground with curved head, snout, and ears. Ideal for evaluating morphological boundary thickness as structuring element size varies from $3 \times 3$ to $15 \times 15$. |

---

## 3. Morphological Watershed Segmentation (`/watershed`)

Real microscopic imaging datasets featuring touching, clustered, or overlapping biological cells and convex particles to evaluate distance transforms, marker generation, and watershed region segmentation.

| File Name | Resolution | Format | Source & Author | License | Benchmark Experimental Suitability |
|---|---|---|---|---|---|
| `01_human_mitosis.png` | 512 × 512 | PNG Grayscale | *Human cells undergoing mitosis* from scikit-image `data.human_mitosis()`, originally produced in the laboratory of Christian Conrad, German Cancer Research Center (DKFZ) | BSD 3-Clause / Open Science | Standard peer-reviewed benchmark. Contains clustered and touching dividing cell nuclei requiring Euclidean distance transform peak markers to prevent over-segmentation. |
| `02_cell_culture.png` | 660 × 550 | PNG Grayscale | *Fluorescent cell culture* from scikit-image `data.cell()` | BSD 3-Clause / CC0 | Dense confluent cell culture with adjacent membranes. Tests Otsu thresholding and marker-controlled watershed segmentation. |
| `03_protein_transport_nuclei.png` | 360 × 366 | PNG Grayscale | *Nuclear translocation assay* from scikit-image `data.protein_transport()`, derived from Broad Bioimage Benchmark Collection | BSD 3-Clause / Open Science | Compact cluster of cell nuclei with varying intensities and close spatial proximities. |
| `04_overlapping_coins.png` | 303 × 384 | PNG Grayscale | Classic standard benchmark from scikit-image `data.coins()` | Public Domain / CC0 | Classic digital image processing benchmark of touching and partially occluded circular boundaries. |

---

## 4. Object Recognition using Fourier Descriptors (`/recognition`)

A standardized geometric shape dataset with 5 distinct planar shape categories and 4 variations per category (total 20 benchmark samples). Built with clean binary contours for computing invariant Fourier descriptors (invariant to translation, rotation, and scaling).

| Category | Samples | Variations & Rotations | License | Benchmark Experimental Suitability |
|---|---|---|---|---|
| **Circles** | `circle_01.png` - `circle_04.png` | Centered, translated, scaled, and eccentric ellipse variations | CC0 Public Domain | Circular harmonics: zeroth and fundamental frequency dominant; high harmonics near zero. Ideal ground truth for rotational invariance. |
| **Squares** | `square_01.png` - `square_04.png` | Upright, 30° rotated, 45° diamond, and scaled | CC0 Public Domain | 4-fold rotational symmetry with characteristic 4th-harmonic peaks in Fourier magnitude spectrum. |
| **Triangles** | `triangle_01.png` - `triangle_04.png` | Equilateral, right-angled, rotated (90°), and scalene | CC0 Public Domain | 3-fold vertex energy distribution; tests scale and orientation normalization across different aspect ratios. |
| **Rectangles** | `rectangle_01.png` - `rectangle_04.png` | Horizontal 2:1, vertical 1:2, rotated 25°, and wide slender | CC0 Public Domain | 2-fold symmetry; differentiates cleanly from square category by aspect ratio harmonic ratios. |
| **Stars** | `star_01.png` - `star_04.png` | Standard 5-point, rotated 5-point, 6-point hexagram, and slender 5-point | CC0 Public Domain | Complex multi-frequency boundary signatures with high frequency re-entrant corners. |
