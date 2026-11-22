import React, { useState } from 'react';
import Breadcrumbs from '../components/Breadcrumbs';
import PageHeader from '../components/PageHeader';
import ImageUploadSection from '../components/ImageUploadSection';
import SampleSelector from '../components/SampleSelector';
import { processFourier } from '../services/api';

const SHAPE_PRESETS = [
  {
    filename: 'test_hexagon_01.png',
    label: 'Test Hexagon (Rotated 22°, Scale 0.88×)',
    category: 'Independent Test Specimen',
    description: '6-sided regular polygon with 22° rotation. Evaluates high-frequency corner harmonic resolution.'
  },
  {
    filename: 'test_trapezium_01.png',
    label: 'Test Trapezium (Rotated 18°, Scale 0.88×)',
    category: 'Independent Test Specimen',
    description: 'Asymmetric 4-sided trapezoid with non-parallel side slopes. Differentiated from triangle and rectangle.'
  },
  {
    filename: 'test_unknown_01_amoeba.png',
    label: 'Unfamiliar Amoeba Blob (Outlier)',
    category: 'Out-of-Distribution Outlier',
    description: 'Irregular wavy non-standard contour. Tests the open-set rejection threshold (must reject as Unknown).'
  },
  {
    filename: 'test_unknown_02_crescent.png',
    label: 'Unfamiliar Crescent Moon (Outlier)',
    category: 'Out-of-Distribution Outlier',
    description: 'Asymmetric crescent contour. Evaluates rejection mechanism against foreign geometric classes.'
  },
  {
    filename: 'test_star_01.png',
    label: 'Test Star (Rotated 18°, Scale 0.88×)',
    category: 'Independent Test Specimen',
    description: '5-point star with 18° rotation and scale variation. Evaluates rotational invariance of the Fourier magnitude spectrum.'
  },
  {
    filename: 'test_circle_01.png',
    label: 'Test Circle (Translated +14, -12)',
    category: 'Independent Test Specimen',
    description: 'Circular contour with spatial center offset. Tests translation invariance via DC component nullification Z(0) = 0.'
  },
  {
    filename: 'test_square_01.png',
    label: 'Test Square (Rotated 28°)',
    category: 'Independent Test Specimen',
    description: 'Regular 4-sided polygon with 28° arbitrary rotation. Tests harmonic corner preservation and rotational invariance.'
  },
  {
    filename: 'test_rectangle_01.png',
    label: 'Test Rectangle (Rotated 35°)',
    category: 'Independent Test Specimen',
    description: '1.7:1 aspect ratio rectangle rotated at 35°. Accurately differentiated from square via aspect ratio fundamental harmonic.'
  },
  {
    filename: 'test_triangle_01.png',
    label: 'Test Triangle (Rotated 42°)',
    category: 'Independent Test Specimen',
    description: 'Equilateral triangle oriented at 42°. Tests harmonic frequency distribution across 3 acute vertices.'
  },
  {
    filename: 'hexagon_01.png',
    label: 'Training Hexagon (Standard Prototype)',
    category: 'Reference Library Prototype',
    description: 'Standard regular 6-sided polygon prototype from the training library.'
  },
  {
    filename: 'trapezium_01.png',
    label: 'Training Trapezium (Standard Prototype)',
    category: 'Reference Library Prototype',
    description: 'Standard trapezoid prototype from the training library.'
  }
];

export default function ShapeRecognitionPage() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [activeSampleFilename, setActiveSampleFilename] = useState(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  // Parameter Controls State
  const [numDescriptors, setNumDescriptors] = useState(16);
  const [distanceMetric, setDistanceMetric] = useState('euclidean');
  const [normalizeScale, setNormalizeScale] = useState(true);
  const [normalizeRotation, setNormalizeRotation] = useState(true);

  // UI View Mode (grid vs split)
  const [viewMode, setViewMode] = useState('grid');
  const [activeStageTab, setActiveStageTab] = useState('reconstruction');

  // Active Results Data
  const [resultData, setResultData] = useState(null);

  const handleImageSelected = (imageData) => {
    setSelectedImage(imageData);
    setActiveSampleFilename(null);
    setErrorMessage(null);
    setResultData(null);
  };

  const handleSelectSample = (sampleData) => {
    setSelectedImage(sampleData);
    setActiveSampleFilename(sampleData.sampleFilename);
    setErrorMessage(null);
    setResultData(null);
  };

  const handleClearImage = () => {
    setSelectedImage(null);
    setActiveSampleFilename(null);
    setErrorMessage(null);
    setResultData(null);
  };

  const handleExecute = async () => {
    if (!selectedImage) return;
    setIsExecuting(true);
    setErrorMessage(null);

    try {
      let fileToUpload = selectedImage.file;

      if (!fileToUpload && selectedImage.previewUrl) {
        const resp = await fetch(selectedImage.previewUrl);
        const blob = await resp.blob();
        fileToUpload = new File([blob], selectedImage.meta?.name || 'shape.png', {
          type: blob.type || 'image/png'
        });
      }

      const formData = new FormData();
      formData.append('file', fileToUpload);
      formData.append('num_descriptors', numDescriptors.toString());
      formData.append('distance_metric', distanceMetric);
      formData.append('normalize_scale', normalizeScale ? 'true' : 'false');
      formData.append('normalize_rotation', normalizeRotation ? 'true' : 'false');

      const data = await processFourier(formData);
      setResultData(data);
    } catch (err) {
      setErrorMessage(err.message || 'Fourier descriptor shape recognition failed.');
    } finally {
      setIsExecuting(false);
    }
  };

  const handleReset = () => {
    setNumDescriptors(16);
    setDistanceMetric('euclidean');
    setNormalizeScale(true);
    setNormalizeRotation(true);
    setErrorMessage(null);
  };

  const STAGES_LIST = [
    { key: 'original_base64', id: 'original', num: 1, title: 'Original Specimen', desc: 'Raw shape raster input' },
    { key: 'binary_mask_base64', id: 'binary', num: 2, title: 'Preprocessed Binary Mask', desc: 'Otsu binarization with polarity check' },
    { key: 'contour_overlay_base64', id: 'contour', num: 3, title: 'Extracted Boundary & Centroid', desc: 'Uniformly resampled contour points (N=128)' },
    { key: 'reconstructed_contour_base64', id: 'reconstruction', num: 4, title: `Reconstructed Contour (K=${numDescriptors})`, desc: 'Inverse DFT from truncated low-frequency harmonics' },
    { key: 'spectrum_chart_base64', id: 'spectrum', num: 5, title: 'Fourier Magnitude Spectrum', desc: 'Normalized harmonic energy |Z(k)| · √k' }
  ];

  const isRecognized = resultData?.classification?.is_recognized ?? false;

  return (
    <div className="module-page-container">
      <Breadcrumbs items={[{ label: 'Fourier Descriptor-Based Shape Recognition' }]} />

      <PageHeader
        moduleNumber="4"
        title="Fourier Descriptor-Based Shape Recognition"
        description="Extracts planar object boundaries, computes invariant Fourier descriptors (invariant to translation, scale, rotation, and starting-point shift), reconstructs smoothed boundaries via inverse DFT, classifies shapes across 7 categories (Circle, Square, Rectangle, Triangle, Star, Hexagon, Trapezium), and rejects unfamiliar or foreign contours."
        tags={['Boundary Representation', 'Fourier Descriptors', 'Open-Set Recognition', 'Harmonic Reconstruction', 'k-NN Classification']}
        accentColor="amber"
      />

      <div className="module-layout-grid">
        {/* Left Column: Input and Parameters */}
        <div className="layout-sidebar">
          <ImageUploadSection
            selectedFile={selectedImage?.file}
            imagePreview={selectedImage?.previewUrl}
            imageMeta={selectedImage?.meta}
            onImageSelected={handleImageSelected}
            onClearImage={handleClearImage}
            label="Upload Shape Image"
            hint="Silhouetted object image (Circle, Square, Rectangle, Triangle, Star, Hexagon, Trapezium, or Unknown)"
            disabled={isExecuting}
          />

          <SampleSelector
            moduleName="shape-recognition"
            presets={SHAPE_PRESETS}
            onSelectSample={handleSelectSample}
            activeSampleFilename={activeSampleFilename}
          />

          {/* Fourier Parameters Panel */}
          <div className="section-card controls-section">
            <div className="section-card-header">
              <h3 className="section-card-title">3. Descriptor & Matching Controls</h3>
              <span className="badge-academic">DFT z(n) → Z(k)</span>
            </div>

            <div className="controls-form">
              {/* Retained Coefficients Slider */}
              <div className="control-group">
                <div className="control-label-row">
                  <label className="control-label">Retained Coefficients (K)</label>
                  <span className="control-value-badge" style={{ color: 'var(--accent-amber)' }}>
                    {numDescriptors} harmonics
                  </span>
                </div>
                <input
                  type="range"
                  className="control-range"
                  min="4"
                  max="64"
                  step="4"
                  value={numDescriptors}
                  onChange={(e) => setNumDescriptors(parseInt(e.target.value, 10))}
                  disabled={isExecuting}
                />
                <div className="range-bounds">
                  <span>4 (Coarse Ellipse)</span>
                  <span>16 (Balanced)</span>
                  <span>64 (Fine Detail)</span>
                </div>
                <span className="control-help">
                  Lower frequencies capture global silhouette; higher harmonics restore sharp corners and polygon vertices.
                </span>
              </div>

              {/* Similarity Metric */}
              <div className="control-group">
                <label className="control-label">Nearest-Neighbor Metric</label>
                <select
                  className="control-select"
                  value={distanceMetric}
                  onChange={(e) => setDistanceMetric(e.target.value)}
                  disabled={isExecuting}
                >
                  <option value="euclidean">Euclidean Distance (L2 Norm)</option>
                  <option value="manhattan">Manhattan Distance (L1 Norm)</option>
                  <option value="cosine">Cosine Spectral Distance</option>
                </select>
                <span className="control-help">
                  Metric used to query closest prototype in the 7-class training library.
                </span>
              </div>

              {/* Invariance Checkboxes */}
              <div className="control-group">
                <label className="control-label" style={{ marginBottom: '0.5rem' }}>
                  Invariance Normalizations
                </label>
                <div className="checkbox-row" style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                    <input
                      type="checkbox"
                      checked={normalizeScale}
                      onChange={(e) => setNormalizeScale(e.target.checked)}
                      disabled={isExecuting}
                    />
                    <span>Scale Invariance (Divide by |Z(1)|)</span>
                  </label>
                  <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                    <input
                      type="checkbox"
                      checked={normalizeRotation}
                      onChange={(e) => setNormalizeRotation(e.target.checked)}
                      disabled={isExecuting}
                    />
                    <span>Rotation & Phase Invariance (Magnitude |Z(k)|)</span>
                  </label>
                </div>
              </div>

              <div className="controls-actions" style={{ marginTop: '1rem' }}>
                <button
                  type="button"
                  className="btn btn-primary btn-execute"
                  disabled={!selectedImage || isExecuting}
                  onClick={handleExecute}
                >
                  {isExecuting ? (
                    <>
                      <span className="spinner-sm"></span>
                      Matching Descriptors...
                    </>
                  ) : (
                    <>
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <polygon points="5 3 19 12 5 21 5 3" />
                      </svg>
                      Extract Descriptors & Match
                    </>
                  )}
                </button>

                <button
                  type="button"
                  className="btn btn-outline"
                  onClick={handleReset}
                  disabled={isExecuting}
                >
                  Reset Defaults
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Visualizations, Classification & Evaluation */}
        <div className="layout-main">
          {errorMessage && (
            <div className="error-banner" role="alert">
              <div className="error-icon">!</div>
              <div className="error-content">
                <div className="error-title">Execution Error</div>
                <div className="error-detail">{errorMessage}</div>
              </div>
              <button type="button" className="error-close-btn" onClick={() => setErrorMessage(null)}>
                &times;
              </button>
            </div>
          )}

          {!selectedImage && (
            <div className="section-card empty-state-card">
              <h4 className="empty-state-title">No Shape Specimen Loaded</h4>
              <p className="empty-state-text">
                Select one of the shape presets (Hexagon, Trapezium, Circle, Square, Rectangle, Triangle, Star, or Unknown) or upload a custom shape silhouette to extract Fourier descriptors and execute open-set nearest-neighbor classification.
              </p>
            </div>
          )}

          {selectedImage && !resultData && !isExecuting && (
            <div className="section-card" style={{ textAlign: 'center', padding: '3.5rem 2rem' }}>
              <div style={{ maxWidth: '320px', margin: '0 auto 1.5rem auto' }}>
                <img
                  src={selectedImage.previewUrl}
                  alt="Awaiting Processing"
                  style={{ maxWidth: '100%', maxHeight: '220px', borderRadius: '6px', border: '1px solid var(--border-medium)' }}
                />
              </div>
              <h3 style={{ color: '#fff', fontSize: '1.1rem', marginBottom: '0.4rem' }}>
                Shape Ready for Fourier Recognition
              </h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', maxWidth: '480px', margin: '0 auto 1.5rem auto' }}>
                Click <strong>"Extract Descriptors & Match"</strong> to resample the outer contour, compute the complex Discrete Fourier Transform, normalize for scale and rotation invariance, and classify against the 7-class shape library with validation-based rejection.
              </p>
              <button
                type="button"
                className="btn btn-primary"
                onClick={handleExecute}
              >
                Extract Descriptors & Match
              </button>
            </div>
          )}

          {isExecuting && (
            <div className="section-card loading-state-card">
              <div className="academic-spinner"></div>
              <div className="loading-title">Computing Fourier Descriptors...</div>
              <div className="loading-sub">
                Interpolating uniform arc-length contour, transforming complex signal, and matching harmonic descriptors against training library with validation rejection.
              </div>
            </div>
          )}

          {resultData && (
            <div className="enhancement-results-container">
              {/* 1. Hero Classification Prediction Card */}
              <div
                className="shape-hero-card"
                style={{
                  borderColor: isRecognized ? 'rgba(52, 211, 153, 0.4)' : 'rgba(245, 158, 11, 0.5)',
                  background: isRecognized
                    ? 'linear-gradient(135deg, rgba(6, 78, 59, 0.35) 0%, rgba(15, 23, 42, 0.95) 100%)'
                    : 'linear-gradient(135deg, rgba(120, 53, 15, 0.35) 0%, rgba(15, 23, 42, 0.95) 100%)'
                }}
              >
                <div className="shape-hero-content">
                  <div className="shape-pred-block">
                    <span
                      className="shape-pred-text"
                      style={{
                        background: isRecognized
                          ? 'linear-gradient(135deg, #34d399 0%, #38bdf8 100%)'
                          : 'linear-gradient(135deg, #f87171 0%, #fbbf24 100%)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent'
                      }}
                    >
                      {resultData.classification?.predicted_category}
                    </span>
                    <div className="shape-pred-label">
                      <span className="shape-pred-title">
                        {isRecognized ? 'Shape Identified' : 'Out-of-Distribution Contour Rejected'}
                      </span>
                      <span className="shape-pred-subtitle">
                        Nearest candidate: <strong>{resultData.classification?.nearest_candidate_category}</strong> (Distance: {resultData.classification?.nearest_distance} vs Acceptance Threshold: {resultData.classification?.acceptance_threshold})
                      </span>
                    </div>
                  </div>

                  <div
                    className="shape-confidence-badge"
                    style={{
                      backgroundColor: isRecognized ? 'rgba(52, 211, 153, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                      borderColor: isRecognized ? 'rgba(52, 211, 153, 0.4)' : 'rgba(239, 68, 68, 0.4)',
                      color: isRecognized ? '#34d399' : '#f87171'
                    }}
                  >
                    <span>Status:</span>
                    <span style={{ fontSize: '0.95rem' }}>
                      {isRecognized ? `Recognized (${resultData.classification?.confidence_percentage}%)` : 'Unknown Shape'}
                    </span>
                  </div>
                </div>

                {resultData.classification?.status_note && (
                  <div style={{ marginTop: '0.85rem', fontSize: '0.8rem', color: isRecognized ? 'var(--text-secondary)' : '#fca5a5' }}>
                    {isRecognized ? '✓' : '⚠️'} <em>{resultData.classification.status_note}</em>
                  </div>
                )}
              </div>

              {/* 2. Visual Pipeline Viewports (5 Synchronized Stages) */}
              <div className="section-card">
                <div className="section-card-header">
                  <div>
                    <h3 className="section-card-title">4. Multi-Stage Fourier Pipeline (5 Stages)</h3>
                    <p className="section-card-sub" style={{ marginBottom: 0 }}>
                      From raw raster silhouette to boundary reconstruction and harmonic magnitude spectrum.
                    </p>
                  </div>

                  <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                    <button
                      type="button"
                      className={`btn btn-sm ${viewMode === 'grid' ? 'btn-primary' : 'btn-outline'}`}
                      onClick={() => setViewMode('grid')}
                    >
                      Grid View (All 5)
                    </button>
                    <button
                      type="button"
                      className={`btn btn-sm ${viewMode === 'split' ? 'btn-primary' : 'btn-outline'}`}
                      onClick={() => setViewMode('split')}
                    >
                      Reconstruction Split
                    </button>
                  </div>
                </div>

                {viewMode === 'grid' ? (
                  <div className="results-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', marginTop: '1rem' }}>
                    {STAGES_LIST.map((stage) => (
                      <div key={stage.id} className="result-viewport-card">
                        <div className="viewport-header">
                          <span className="viewport-stage-index">Stage {stage.num}</span>
                          <span className="viewport-label">{stage.title}</span>
                        </div>
                        <div className="viewport-canvas" style={{ height: '220px' }}>
                          <img
                            src={resultData.images[stage.key]}
                            alt={stage.title}
                            className="canvas-img"
                          />
                        </div>
                        <div style={{ padding: '0.45rem 0.75rem', fontSize: '0.725rem', color: 'var(--text-muted)', borderTop: '1px solid var(--border-subtle)', background: 'rgba(15, 23, 42, 0.4)' }}>
                          {stage.desc}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ marginTop: '1rem' }}>
                    <div className="results-grid results-grid-2">
                      <div className="result-viewport-card">
                        <div className="viewport-header">
                          <span className="viewport-stage-index">Extracted Boundary</span>
                          <span className="viewport-label">Original Contour & Centroid</span>
                        </div>
                        <div className="viewport-canvas" style={{ height: '340px' }}>
                          <img src={resultData.images.contour_overlay_base64} alt="Original Contour" className="canvas-img" />
                        </div>
                      </div>

                      <div className="result-viewport-card">
                        <div className="viewport-header">
                          <span className="viewport-stage-index">Truncated IDFT (K={resultData.parameters?.num_descriptors})</span>
                          <span className="viewport-label">Reconstructed Boundary Comparison</span>
                        </div>
                        <div className="viewport-canvas" style={{ height: '340px' }}>
                          <img src={resultData.images.reconstructed_contour_base64} alt="Reconstruction" className="canvas-img" />
                        </div>
                      </div>
                    </div>

                    <div style={{ marginTop: '1rem' }}>
                      <div className="result-viewport-card">
                        <div className="viewport-header">
                          <span className="viewport-stage-index">Harmonic Energy Spectrum</span>
                          <span className="viewport-label">Fourier Descriptors |Z(k)| · √k</span>
                        </div>
                        <div className="viewport-canvas" style={{ height: '260px' }}>
                          <img src={resultData.images.spectrum_chart_base64} alt="Spectrum Chart" className="canvas-img" />
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* 3. Shape Properties & Harmonic Reconstruction Fidelity */}
              <div className="section-card">
                <div className="section-card-header">
                  <h3 className="section-card-title">Shape Properties & Harmonic Reconstruction Fidelity</h3>
                  <span className="badge-academic">K = {resultData.parameters?.num_descriptors} Coefficients</span>
                </div>

                <div className="metrics-grid">
                  <div className="metric-box">
                    <span className="metric-label">Reconstruction MSE</span>
                    <span className="metric-value" style={{ color: 'var(--accent-cyan)' }}>
                      {resultData.shape_properties?.reconstruction_mse ?? 0}
                      <span className="metric-unit"> px²</span>
                    </span>
                    <span className="metric-description">Mean squared spatial approximation error</span>
                  </div>

                  <div className="metric-box">
                    <span className="metric-label">Spectral Energy Retained</span>
                    <span className="metric-value" style={{ color: 'var(--accent-emerald)' }}>
                      {resultData.shape_properties?.energy_retention_pct ?? 0}
                      <span className="metric-unit"> %</span>
                    </span>
                    <span className="metric-description">Low-frequency power percentage</span>
                  </div>

                  <div className="metric-box">
                    <span className="metric-label">Contour Perimeter</span>
                    <span className="metric-value">
                      {resultData.shape_properties?.contour_perimeter_px ?? 0}
                      <span className="metric-unit"> px</span>
                    </span>
                    <span className="metric-description">Boundary arc-length</span>
                  </div>

                  <div className="metric-box">
                    <span className="metric-label">Circularity Index</span>
                    <span className="metric-value">
                      {resultData.shape_properties?.circularity ?? 0}
                    </span>
                    <span className="metric-description">4πA / P² (1.0 = perfect circle)</span>
                  </div>
                </div>

                {/* Educational Note on Coefficient Truncation */}
                <div className="controls-placeholder-notice" style={{ marginTop: '1rem', borderColor: 'rgba(56, 189, 248, 0.3)', background: 'rgba(56, 189, 248, 0.05)' }}>
                  <span style={{ color: '#38bdf8', fontWeight: 600 }}>💡 Effect of Coefficient Count (K):</span>
                  <span style={{ color: 'var(--text-secondary)' }}>
                    At <strong>K = 4</strong>, the inverse DFT reconstructs only the best-fit fundamental ellipse. Increasing to <strong>K = 16</strong> restores polygonal edges and major corners (Hexagon, Trapezium, Square, Triangle). Higher harmonics (<strong>K = 32–64</strong>) accurately capture re-entrant star apexes and fine boundary lobes.
                  </span>
                </div>
              </div>

              {/* 4. Top-5 Nearest Neighbor Matches */}
              <div className="section-card">
                <div className="section-card-header">
                  <div>
                    <h3 className="section-card-title">Nearest-Neighbor Prototype Matches</h3>
                    <p className="section-card-sub" style={{ marginBottom: 0 }}>
                      Ranked Euclidean distances in normalized Fourier descriptor space against the 35-sample prototype library.
                    </p>
                  </div>
                  <span className="badge-academic">{resultData.parameters?.distance_metric?.toUpperCase()}</span>
                </div>

                <div className="cell-table-container" style={{ marginTop: '1rem' }}>
                  <table className="cell-table">
                    <thead>
                      <tr>
                        <th>Rank</th>
                        <th>Prototype Sample</th>
                        <th>Category</th>
                        <th>Descriptor Distance</th>
                        <th>Calibrated Match Ratio</th>
                      </tr>
                    </thead>
                    <tbody>
                      {resultData.classification?.top_matches?.map((m) => (
                        <tr key={m.rank}>
                          <td>
                            <span className={`rank-badge ${m.rank === 1 ? 'rank-1' : ''}`}>
                              #{m.rank}
                            </span>
                          </td>
                          <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{m.filename}</td>
                          <td style={{ textTransform: 'capitalize' }}>
                            <span style={{
                              color: m.category === resultData.classification.predicted_category
                                ? 'var(--accent-emerald)'
                                : 'var(--text-muted)'
                            }}>
                              {m.category}
                            </span>
                          </td>
                          <td style={{ fontFamily: 'var(--font-mono)' }}>{m.distance.toFixed(4)}</td>
                          <td>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
                              <span style={{ fontWeight: 600, color: m.similarity_pct > 70 ? 'var(--accent-emerald)' : m.similarity_pct > 0 ? 'var(--accent-amber)' : '#f87171' }}>
                                {m.similarity_pct}%
                              </span>
                              <div className="metric-progress-wrapper" style={{ width: '80px', margin: 0 }}>
                                <div
                                  className="metric-progress-fill"
                                  style={{
                                    width: `${m.similarity_pct}%`,
                                    backgroundColor: m.similarity_pct > 70 ? 'var(--accent-emerald)' : m.similarity_pct > 0 ? 'var(--accent-amber)' : '#f87171'
                                  }}
                                ></div>
                              </div>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* 5. Independent Test Split Evaluation & 7x7 Confusion Matrix */}
              {resultData.evaluation_dataset && (
                <div className="section-card">
                  <div className="section-card-header">
                    <div>
                      <h3 className="section-card-title">Independent Test Set Evaluation (7 Categories)</h3>
                      <p className="section-card-sub" style={{ marginBottom: 0 }}>
                        Evaluated across 21 independent test specimens generated with separate rotations, scales, and offsets, plus 4 out-of-distribution unknown shapes.
                      </p>
                    </div>
                    <span className="badge-academic" style={{ borderColor: 'var(--accent-emerald)', color: 'var(--accent-emerald)' }}>
                      Overall Accuracy: {resultData.evaluation_dataset.overall_accuracy_pct}%
                    </span>
                  </div>

                  <div className="metrics-grid" style={{ marginTop: '1rem' }}>
                    <div className="metric-box">
                      <span className="metric-label">Macro Precision</span>
                      <span className="metric-value" style={{ color: 'var(--accent-emerald)' }}>
                        {(resultData.evaluation_dataset.macro_precision * 100).toFixed(1)}
                        <span className="metric-unit"> %</span>
                      </span>
                      <span className="metric-description">Average across all 7 classes</span>
                    </div>

                    <div className="metric-box">
                      <span className="metric-label">Macro Recall</span>
                      <span className="metric-value" style={{ color: 'var(--accent-emerald)' }}>
                        {(resultData.evaluation_dataset.macro_recall * 100).toFixed(1)}
                        <span className="metric-unit"> %</span>
                      </span>
                      <span className="metric-description">True positive sensitivity</span>
                    </div>

                    <div className="metric-box">
                      <span className="metric-label">Macro F1-Score</span>
                      <span className="metric-value" style={{ color: 'var(--accent-cyan)' }}>
                        {resultData.evaluation_dataset.macro_f1}
                      </span>
                      <span className="metric-description">Harmonic mean of precision & recall</span>
                    </div>

                    <div className="metric-box">
                      <span className="metric-label">Outlier Rejection Rate</span>
                      <span className="metric-value" style={{ color: 'var(--accent-emerald)' }}>
                        {resultData.evaluation_dataset.unknown_rejection_rate_pct}
                        <span className="metric-unit"> %</span>
                      </span>
                      <span className="metric-description">
                        {resultData.evaluation_dataset.unknown_samples_tested} foreign shapes tested
                      </span>
                    </div>
                  </div>

                  {/* 7x7 Confusion Matrix Table */}
                  <div style={{ marginTop: '1.25rem' }}>
                    <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#fff', marginBottom: '0.5rem' }}>
                      Confusion Matrix (7x7: Rows = Ground Truth, Columns = Predicted)
                    </div>
                    <div style={{ overflowX: 'auto' }}>
                      <table className="confusion-table">
                        <thead>
                          <tr>
                            <th style={{ textAlign: 'left', background: 'rgba(15, 23, 42, 0.95)' }}>True \ Pred</th>
                            {resultData.evaluation_dataset.confusion_matrix?.labels?.map((lbl) => (
                              <th key={lbl}>{lbl}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {resultData.evaluation_dataset.confusion_matrix?.labels?.map((rowLabel, rowIdx) => (
                            <tr key={rowLabel}>
                              <td style={{ textAlign: 'left', fontWeight: 600, color: '#fff', background: 'rgba(15, 23, 42, 0.7)' }}>
                                {rowLabel}
                              </td>
                              {resultData.evaluation_dataset.confusion_matrix.matrix[rowIdx].map((val, colIdx) => {
                                const isHit = rowIdx === colIdx && val > 0;
                                const isMiss = rowIdx !== colIdx && val > 0;
                                return (
                                  <td
                                    key={colIdx}
                                    className={isHit ? 'cm-cell-hit' : isMiss ? 'cm-cell-miss' : 'cm-cell-zero'}
                                  >
                                    {val}
                                  </td>
                                );
                              })}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )}

              {/* 6. Scientific Limitations of Boundary-Based Recognition */}
              <div className="section-card">
                <div className="section-card-header">
                  <div>
                    <h3 className="section-card-title">Scientific Limitations of Boundary-Based Recognition</h3>
                    <p className="section-card-sub" style={{ marginBottom: 0 }}>
                      Fundamental theoretical and practical constraints of planar Fourier contour descriptors.
                    </p>
                  </div>
                  <span className="badge-academic" style={{ borderColor: 'var(--accent-amber)', color: 'var(--accent-amber)' }}>
                    Academic Analysis
                  </span>
                </div>

                <div className="limitations-grid">
                  {resultData.limitations?.map((item, idx) => (
                    <div key={idx} className="limitation-item-card">
                      <div className="limitation-item-title">
                        <span>⚠️ {item.topic}</span>
                      </div>
                      <div className="limitation-item-text">
                        <strong>Theoretical Phenomenon:</strong> {item.observation}
                      </div>
                      <div className="limitation-item-text">
                        <strong>DIP Impact:</strong> {item.impact}
                      </div>
                      <div className="limitation-item-mitigation">
                        <strong>Mitigation Strategy:</strong> {item.mitigation}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
