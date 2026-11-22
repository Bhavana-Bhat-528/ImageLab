import React from 'react';
import { Link } from 'react-router-dom';

const modules = [
  {
    id: 1,
    title: 'Automatic Image Enhancement',
    route: '/enhancement',
    moduleTag: 'Module 1',
    category: 'Spatial & Intensity Transformations',
    description:
      'Analyzes input image characteristics (brightness, contrast, and histogram dispersion) and selects an appropriate intensity transformation (gamma correction, histogram equalization, contrast stretching) and spatial filter (unsharp masking, Gaussian smoothing, median filtering). Evaluates quantitative quality via MSE, PSNR, and SSIM metrics.',
    keyFeatures: [
      'Image characteristic analysis',
      'Dynamic intensity transformations',
      'Spatial filtering (Sharpening/Smoothing)',
      'Histogram comparison & MSE / PSNR / SSIM'
    ]
  },
  {
    id: 2,
    title: 'Morphological Boundary Extraction',
    route: '/boundary-extraction',
    moduleTag: 'Module 2',
    category: 'Mathematical Morphology',
    description:
      'Extracts and highlights the outer boundaries of animals and complex biological objects using binary segmentation, mathematical morphological erosion, set subtraction β(A) = A − (A ⊖ B), and color overlay visualization. Compares the geometric impact of varying structuring element sizes and shapes.',
    keyFeatures: [
      'Binary segmentation & thresholding',
      'Morphological erosion A ⊖ B',
      'Boundary set subtraction β(A)',
      'Rectangular, Cross, and Elliptical structuring elements'
    ]
  },
  {
    id: 3,
    title: 'Watershed Segmentation',
    route: '/watershed',
    moduleTag: 'Module 3',
    category: 'Region-Based Segmentation',
    description:
      'Separates clustered and touching cells in microscopic imaging through Otsu binarization, Euclidean distance transformation, local peak marker identification, and marker-controlled watershed segmentation. Accurately estimates individual cell populations and morphological region properties.',
    keyFeatures: [
      'Microscopic image preprocessing',
      'Euclidean distance transformation',
      'Local peak marker generation',
      'Touching boundary separation & cell counting'
    ]
  },
  {
    id: 4,
    title: 'Fourier Descriptor-Based Shape Recognition',
    route: '/shape-recognition',
    moduleTag: 'Module 4',
    category: 'Boundary Representation & Classification',
    description:
      'Extracts object contours and represents planar boundary curves using Discrete Fourier Transform descriptors invariant to translation, rotation, and scaling. Compares shape similarities across 5 distinct object categories and evaluates classification accuracy, precision, and recall.',
    keyFeatures: [
      'Planar boundary contour extraction',
      'Invariant Fourier descriptors computation',
      'Shape signature & frequency spectrum analysis',
      'Multi-class shape similarity matching'
    ]
  }
];

export default function Home() {
  return (
    <div className="home-container">
      {/* Hero Section */}
      <section className="hero-banner">
        <div className="hero-badge">Interactive Digital Image Processing Laboratory</div>
        <h1 className="hero-title">ImageLab</h1>
        <p className="hero-description">
          A local-first, interactive digital image processing laboratory combining classical computer vision,
          mathematical morphology, region segmentation, and spectral shape recognition. Process uploads purely in-memory
          with Python FastAPI and experiment with algorithms in real time.
        </p>

        <div className="hero-specs-row">
          <div className="spec-item">
            <span className="spec-dot"></span>
            <span>Zero-Persistence In-Memory Processing</span>
          </div>
          <div className="spec-item">
            <span className="spec-dot"></span>
            <span>Local Python FastAPI + OpenCV Engine</span>
          </div>
          <div className="spec-item">
            <span className="spec-dot"></span>
            <span>Quantitative Metrics & Visual Validation</span>
          </div>
        </div>
      </section>

      {/* Modules Grid */}
      <section className="modules-section">
        <div className="section-title-row">
          <div>
            <h2 className="section-heading">Laboratory Modules</h2>
            <p className="section-subheading">
              Select an experiment to open the interactive parameter control and analysis laboratory:
            </p>
          </div>
          <span className="badge-academic">4 Core Experiments</span>
        </div>

        <div className="modules-grid">
          {modules.map((mod) => (
            <article key={mod.id} className="module-card">
              <div className="card-top-bar">
                <span className="card-module-tag">{mod.moduleTag}</span>
                <span className="card-category-tag">{mod.category}</span>
              </div>

              <h3 className="card-title">{mod.title}</h3>
              <p className="card-description">{mod.description}</p>

              <div className="card-features-list">
                {mod.keyFeatures.map((feat, i) => (
                  <div key={i} className="card-feature-item">
                    <span className="feature-check">✓</span>
                    <span>{feat}</span>
                  </div>
                ))}
              </div>

              <div className="card-footer">
                <Link to={mod.route} className="btn btn-primary btn-explore">
                  <span>Explore</span>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <line x1="5" y1="12" x2="19" y2="12" />
                    <polyline points="12 5 19 12 12 19" />
                  </svg>
                </Link>
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}
