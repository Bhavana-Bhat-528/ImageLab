import React, { useState } from 'react';
import Breadcrumbs from '../components/Breadcrumbs';
import PageHeader from '../components/PageHeader';
import ImageUploadSection from '../components/ImageUploadSection';
import SampleSelector from '../components/SampleSelector';
import { processBoundary } from '../services/api';

const BOUNDARY_PRESETS = [
  {
    filename: 'zebra.jpg',
    label: 'Zebra (Equus quagga)',
    category: 'High-Contrast Animal Stripes',
    description: 'Distinctive striped coat pattern against grassland background. Tests morphological boundary connectivity.'
  },
  {
    filename: 'elephant.jpg',
    label: 'African Bush Elephant',
    category: 'Large Mammal Silhouette',
    description: 'Complex outer contour including curved trunk, tusks, and broad ear geometries.'
  },
  {
    filename: 'dog.jpg',
    label: 'Canine Portrait (Dog)',
    category: 'Domestic Animal Contour',
    description: 'Smooth animal foreground. Ideal for evaluating structuring element size impact (3x3 to 15x15).'
  }
];

export default function BoundaryExtractionPage() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [activeSampleFilename, setActiveSampleFilename] = useState(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  // Parameter Controls State
  const [seShape, setSeShape] = useState('ellipse');
  const [seSize, setSeSize] = useState(3);
  const [iterations, setIterations] = useState(1);
  const [segmentationMethod, setSegmentationMethod] = useState('grabcut');
  const [grabcutIters, setGrabcutIters] = useState(5);
  const [marginPct, setMarginPct] = useState(6.0);
  const [overlayColor, setOverlayColor] = useState('cyan');
  const [displayThickness, setDisplayThickness] = useState(1);

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
        fileToUpload = new File([blob], selectedImage.meta?.name || 'animal.jpg', {
          type: blob.type || 'image/jpeg'
        });
      }

      const formData = new FormData();
      formData.append('file', fileToUpload);
      formData.append('se_shape', seShape);
      formData.append('se_size', seSize.toString());
      formData.append('iterations', iterations.toString());
      formData.append('segmentation_method', segmentationMethod);
      formData.append('margin_pct', marginPct.toString());
      formData.append('grabcut_iters', grabcutIters.toString());
      formData.append('overlay_color', overlayColor);
      formData.append('display_thickness', displayThickness.toString());

      const data = await processBoundary(formData);
      setResultData(data);
    } catch (err) {
      setErrorMessage(err.message || 'Boundary extraction processing failed.');
    } finally {
      setIsExecuting(false);
    }
  };

  const handleReset = () => {
    setSeShape('ellipse');
    setSeSize(3);
    setIterations(1);
    setSegmentationMethod('grabcut');
    setGrabcutIters(5);
    setMarginPct(6.0);
    setOverlayColor('cyan');
    setDisplayThickness(1);
    setErrorMessage(null);
  };

  return (
    <div className="module-page-container">
      <Breadcrumbs items={[{ label: 'Morphological Boundary Extraction' }]} />

      <PageHeader
        moduleNumber="2"
        title="Morphological Boundary Extraction"
        description="Extracts and highlights outer animal boundaries using GrabCut segmentation, mathematical morphological erosion (A ⊖ B), set subtraction β(A) = A − (A ⊖ B), and color overlay onto original photographs. Evaluates the impact of structuring element shapes and sizes."
        tags={['GrabCut Segmentation', 'Mathematical Morphology', 'Binary Erosion', 'Boundary Set Subtraction', 'IoU / Dice']}
        accentColor="emerald"
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
            label="Upload Animal Image"
            hint="Zebra, elephant, dog or other animal photographs (min 3 real-world test sets)"
            disabled={isExecuting}
          />

          <SampleSelector
            moduleName="boundary"
            presets={BOUNDARY_PRESETS}
            onSelectSample={handleSelectSample}
            activeSampleFilename={activeSampleFilename}
          />

          {/* Morphological Parameters Panel */}
          <div className="section-card controls-section">
            <div className="section-card-header">
              <h3 className="section-card-title">3. Morphological Controls</h3>
              <span className="badge-academic">β(A) = A − (A ⊖ B)</span>
            </div>

            <div className="controls-form">
              {/* Segmentation Method */}
              <div className="control-group">
                <label className="control-label">Foreground Segmentation Method</label>
                <select
                  className="control-select"
                  value={segmentationMethod}
                  onChange={(e) => setSegmentationMethod(e.target.value)}
                  disabled={isExecuting}
                >
                  <option value="grabcut">OpenCV GrabCut (Adaptive Bounding Box)</option>
                  <option value="otsu">Otsu Global Binarization</option>
                  <option value="adaptive">Adaptive Gaussian Thresholding</option>
                </select>
                <span className="control-help">
                  GrabCut uses Gaussian Mixture Models to separate complex natural animal silhouettes.
                </span>
              </div>

              {segmentationMethod === 'grabcut' && (
                <>
                  <div className="control-group">
                    <div className="control-label-row">
                      <label className="control-label">GrabCut Bounding Box Margin</label>
                      <span className="control-value-badge">{marginPct}%</span>
                    </div>
                    <input
                      type="range"
                      className="control-range"
                      min="2"
                      max="18"
                      step="1"
                      value={marginPct}
                      onChange={(e) => setMarginPct(parseFloat(e.target.value))}
                      disabled={isExecuting}
                    />
                    <div className="range-bounds">
                      <span>2% (Tight)</span>
                      <span>6% (Standard)</span>
                      <span>18% (Wide)</span>
                    </div>
                  </div>

                  <div className="control-group">
                    <div className="control-label-row">
                      <label className="control-label">GrabCut Iterations</label>
                      <span className="control-value-badge">{grabcutIters} iters</span>
                    </div>
                    <input
                      type="range"
                      className="control-range"
                      min="1"
                      max="8"
                      step="1"
                      value={grabcutIters}
                      onChange={(e) => setGrabcutIters(parseInt(e.target.value, 10))}
                      disabled={isExecuting}
                    />
                  </div>
                </>
              )}

              {/* Structuring Element Shape */}
              <div className="control-group">
                <label className="control-label">Structuring Element Shape (B)</label>
                <select
                  className="control-select"
                  value={seShape}
                  onChange={(e) => setSeShape(e.target.value)}
                  disabled={isExecuting}
                >
                  <option value="ellipse">Ellipse / Disk (MORPH_ELLIPSE - Isotropic)</option>
                  <option value="cross">Cross (MORPH_CROSS - 4-connected)</option>
                  <option value="rect">Rectangle (MORPH_RECT - 8-connected)</option>
                </select>
                <span className="control-help">
                  Shape affects corner preservation and boundary thickness isotropy.
                </span>
              </div>

              {/* Structuring Element Size */}
              <div className="control-group">
                <div className="control-label-row">
                  <label className="control-label">Structuring Element Size</label>
                  <span className="control-value-badge">{seSize} × {seSize}</span>
                </div>
                <input
                  type="range"
                  className="control-range"
                  min="3"
                  max="15"
                  step="2"
                  value={seSize}
                  onChange={(e) => setSeSize(parseInt(e.target.value, 10))}
                  disabled={isExecuting}
                />
                <div className="range-bounds">
                  <span>3 × 3 (Fine)</span>
                  <span>7 × 7 (Medium)</span>
                  <span>15 × 15 (Thick)</span>
                </div>
              </div>

              {/* Erosion Iterations */}
              <div className="control-group">
                <div className="control-label-row">
                  <label className="control-label">Erosion Iterations</label>
                  <span className="control-value-badge">{iterations}×</span>
                </div>
                <input
                  type="range"
                  className="control-range"
                  min="1"
                  max="4"
                  step="1"
                  value={iterations}
                  onChange={(e) => setIterations(parseInt(e.target.value, 10))}
                  disabled={isExecuting}
                />
              </div>

              {/* Visual Styling */}
              <div className="control-group">
                <label className="control-label">Boundary Overlay Color</label>
                <select
                  className="control-select"
                  value={overlayColor}
                  onChange={(e) => setOverlayColor(e.target.value)}
                  disabled={isExecuting}
                >
                  <option value="cyan">Neon Cyan (High Contrast)</option>
                  <option value="yellow">High-Vis Yellow</option>
                  <option value="magenta">Vibrant Magenta</option>
                  <option value="green">Lime Green</option>
                  <option value="red">Pure Red</option>
                  <option value="white">Clean White</option>
                </select>
              </div>

              <div className="control-group">
                <div className="control-label-row">
                  <label className="control-label">Display Thickness</label>
                  <span className="control-value-badge">{displayThickness} px</span>
                </div>
                <input
                  type="range"
                  className="control-range"
                  min="1"
                  max="3"
                  step="1"
                  value={displayThickness}
                  onChange={(e) => setDisplayThickness(parseInt(e.target.value, 10))}
                  disabled={isExecuting}
                />
              </div>

              <div className="controls-actions" style={{ marginTop: '0.85rem' }}>
                <button
                  type="button"
                  className="btn btn-primary btn-execute"
                  disabled={!selectedImage || isExecuting}
                  onClick={handleExecute}
                >
                  {isExecuting ? (
                    <>
                      <span className="spinner-sm"></span>
                      Segmenting & Extracting...
                    </>
                  ) : (
                    <>
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <polygon points="5 3 19 12 5 21 5 3" />
                      </svg>
                      Extract Animal Boundary
                    </>
                  )}
                </button>

                <button
                  type="button"
                  className="btn btn-outline"
                  onClick={handleReset}
                  disabled={isExecuting}
                >
                  Reset
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Visualizations, Overlays & Quantitative Metrics */}
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
              <h4 className="empty-state-title">No Animal Photograph Loaded</h4>
              <p className="empty-state-text">
                Select one of the benchmark animal photographs (Zebra, Elephant, Dog) or upload a custom image to execute morphological boundary extraction.
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
                Photograph Ready for Boundary Extraction
              </h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', maxWidth: '480px', margin: '0 auto 1.5rem auto' }}>
                Click <strong>"Extract Animal Boundary"</strong> to execute GrabCut foreground segmentation, morphological erosion ($A \ominus B$), and set subtraction $\beta(A) = A - (A \ominus B)$.
              </p>
              <button
                type="button"
                className="btn btn-primary"
                onClick={handleExecute}
              >
                Start Boundary Extraction
              </button>
            </div>
          )}

          {isExecuting && (
            <div className="section-card loading-state-card">
              <div className="academic-spinner"></div>
              <div className="loading-title">Extracting Morphological Boundary...</div>
              <div className="loading-sub">
                Executing GrabCut Gaussian mixture models, structuring element structuring, and topological subtraction.
              </div>
            </div>
          )}

          {resultData && (
            <div className="enhancement-results-container">
              {/* 1. Complete 6-Stage Morphological Visual Pipeline */}
              <div className="section-card">
                <div className="section-card-header">
                  <h3 className="section-card-title">4. Multi-Stage Morphological Pipeline</h3>
                  <span className="badge-academic">
                    {resultData.metrics.image_dimensions} • {resultData.parameters.se_shape.toUpperCase()} ({resultData.parameters.se_size}x{resultData.parameters.se_size})
                  </span>
                </div>

                <div className="results-grid results-grid-3">
                  {/* Stage 1: Original */}
                  <div className="result-viewport-card">
                    <div className="viewport-header">
                      <span className="viewport-stage-index">Stage 1</span>
                      <span className="viewport-label">Original Photograph</span>
                    </div>
                    <div className="viewport-canvas">
                      <img src={resultData.images.original_base64} alt="Original" className="canvas-img" />
                    </div>
                  </div>

                  {/* Stage 2: Preprocessed / Bounding Box */}
                  <div className="result-viewport-card">
                    <div className="viewport-header">
                      <span className="viewport-stage-index">Stage 2</span>
                      <span className="viewport-label">GrabCut Box Initialization</span>
                    </div>
                    <div className="viewport-canvas">
                      <img src={resultData.images.preprocessed_base64} alt="Preprocessed" className="canvas-img" />
                    </div>
                  </div>

                  {/* Stage 3: Segmented Binary Mask */}
                  <div className="result-viewport-card">
                    <div className="viewport-header">
                      <span className="viewport-stage-index">Stage 3</span>
                      <span className="viewport-label">Foreground Binary Mask (A)</span>
                    </div>
                    <div className="viewport-canvas">
                      <img src={resultData.images.binary_mask_base64} alt="Binary Mask" className="canvas-img" />
                    </div>
                  </div>

                  {/* Stage 4: Eroded Binary Mask */}
                  <div className="result-viewport-card">
                    <div className="viewport-header">
                      <span className="viewport-stage-index">Stage 4</span>
                      <span className="viewport-label">Eroded Mask (A ⊖ B)</span>
                    </div>
                    <div className="viewport-canvas">
                      <img src={resultData.images.eroded_mask_base64} alt="Eroded Mask" className="canvas-img" />
                    </div>
                  </div>

                  {/* Stage 5: Extracted Boundary */}
                  <div className="result-viewport-card">
                    <div className="viewport-header">
                      <span className="viewport-stage-index">Stage 5</span>
                      <span className="viewport-label">Boundary: β(A) = A − (A ⊖ B)</span>
                    </div>
                    <div className="viewport-canvas">
                      <img src={resultData.images.boundary_mask_base64} alt="Extracted Boundary" className="canvas-img" />
                    </div>
                  </div>

                  {/* Stage 6: Final Overlay */}
                  <div className="result-viewport-card">
                    <div className="viewport-header">
                      <span className="viewport-stage-index">Stage 6</span>
                      <span className="viewport-label">Final Highlighted Overlay</span>
                    </div>
                    <div className="viewport-canvas">
                      <img src={resultData.images.overlay_base64} alt="Final Overlay" className="canvas-img" />
                    </div>
                  </div>
                </div>
              </div>

              {/* 2. Quantitative Topological Metrics */}
              <div className="section-card">
                <div className="section-card-header">
                  <h3 className="section-card-title">Quantitative Measurements & Connectivity</h3>
                  <span className="badge-academic">Topological Evaluation</span>
                </div>

                <div className="metrics-grid">
                  <div className="metric-box">
                    <span className="metric-label">Foreground Pixels |A|</span>
                    <span className="metric-value">{resultData.metrics.foreground_pixels.toLocaleString()}</span>
                    <span className="metric-description">
                      {resultData.metrics.foreground_percentage}% of total image
                    </span>
                  </div>

                  <div className="metric-box">
                    <span className="metric-label">Boundary Pixels |β(A)|</span>
                    <span className="metric-value">{resultData.metrics.boundary_pixels.toLocaleString()}</span>
                    <span className="metric-description">
                      Thickness Ratio: {resultData.metrics.boundary_to_foreground_ratio_pct}%
                    </span>
                  </div>

                  <div className="metric-box">
                    <span className="metric-label">Foreground Components</span>
                    <span className="metric-value">{resultData.metrics.foreground_connected_components}</span>
                    <span className="metric-description">Connected object islands</span>
                  </div>

                  <div className="metric-box">
                    <span className="metric-label">Boundary Components</span>
                    <span className="metric-value">{resultData.metrics.boundary_connected_components}</span>
                    <span className="metric-description">Closed perimeter loops</span>
                  </div>
                </div>

                {/* Ground Truth Accuracy Metrics (IoU & Dice) */}
                {resultData.metrics.ground_truth_evaluation.has_ground_truth ? (
                  <div className="metric-section-group" style={{ marginTop: '1.25rem', borderTop: '1px solid var(--border-medium)', paddingTop: '1rem' }}>
                    <div className="metric-group-title">
                      <span>Ground-Truth Mask Validation (IoU / Dice)</span>
                      <span className="badge-academic" style={{ borderColor: 'var(--accent-emerald)', color: 'var(--accent-emerald)' }}>
                        Validated vs {resultData.metrics.ground_truth_evaluation.reference_name}
                      </span>
                    </div>
                    <p className="metric-group-note" style={{ color: 'var(--accent-cyan)' }}>
                      {resultData.metrics.ground_truth_evaluation.interpretation}
                    </p>

                    <div className="metrics-grid">
                      <div className="metric-box" style={{ borderColor: 'rgba(52, 211, 153, 0.4)' }}>
                        <span className="metric-label">IoU (Jaccard Index)</span>
                        <span className="metric-value" style={{ color: 'var(--accent-emerald)' }}>
                          {(resultData.metrics.ground_truth_evaluation.iou * 100).toFixed(2)}
                          <span className="metric-unit"> %</span>
                        </span>
                        <span className="metric-description">Spatial intersection over union</span>
                      </div>

                      <div className="metric-box" style={{ borderColor: 'rgba(52, 211, 153, 0.4)' }}>
                        <span className="metric-label">Dice Coefficient (F1 Score)</span>
                        <span className="metric-value" style={{ color: 'var(--accent-emerald)' }}>
                          {(resultData.metrics.ground_truth_evaluation.dice * 100).toFixed(2)}
                          <span className="metric-unit"> %</span>
                        </span>
                        <span className="metric-description">Harmonic mean of precision & recall</span>
                      </div>

                      <div className="metric-box">
                        <span className="metric-label">True Overlap Pixels</span>
                        <span className="metric-value">
                          {resultData.metrics.ground_truth_evaluation.overlap_pixels.toLocaleString()}
                        </span>
                        <span className="metric-description">Target: {resultData.metrics.ground_truth_evaluation.ground_truth_pixels.toLocaleString()} px</span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="metric-section-group" style={{ marginTop: '1rem', borderTop: '1px solid var(--border-subtle)', paddingTop: '0.75rem' }}>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      ℹ️ Ground-truth reference mask active when analyzing standard animal benchmark photographs (`zebra.jpg`, `elephant.jpg`, `dog.jpg`).
                    </div>
                  </div>
                )}

                {/* Important Pedagogical Explanation */}
                <div className="controls-placeholder-notice" style={{ marginTop: '1rem', borderColor: 'var(--accent-amber)', background: 'rgba(245, 158, 11, 0.08)' }}>
                  <span style={{ color: 'var(--accent-amber)', fontWeight: 600 }}>⚠️ Critical Evaluation Note:</span>
                  <span style={{ color: 'var(--text-secondary)' }}>
                    {resultData.explanation.metric_accuracy_caveat}
                  </span>
                </div>
              </div>

              {/* 3. Structuring Element Cross-Configuration Comparison Table */}
              <div className="section-card">
                <div className="section-card-header">
                  <div>
                    <h3 className="section-card-title">Structuring Element Comparison Benchmark</h3>
                    <p className="section-card-sub" style={{ marginBottom: 0 }}>
                      Evaluates geometric effect of shapes and sizes on the exact same foreground mask.
                    </p>
                  </div>
                  <span className="badge-academic">9 Experimental Configurations</span>
                </div>

                <div style={{ overflowX: 'auto', marginTop: '1rem' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.825rem', textAlign: 'left' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--border-medium)', color: 'var(--text-muted)' }}>
                        <th style={{ padding: '0.65rem 0.85rem' }}>Shape</th>
                        <th style={{ padding: '0.65rem 0.85rem' }}>Kernel Size</th>
                        <th style={{ padding: '0.65rem 0.85rem' }}>Boundary Pixels</th>
                        <th style={{ padding: '0.65rem 0.85rem' }}>Thickness Ratio (%)</th>
                        <th style={{ padding: '0.65rem 0.85rem' }}>Connected Components</th>
                        <th style={{ padding: '0.65rem 0.85rem' }}>Geometric Character</th>
                      </tr>
                    </thead>
                    <tbody>
                      {resultData.se_benchmarks.map((row, idx) => {
                        const isCurrentActive = row.shape === resultData.parameters.se_shape && row.size === `${resultData.parameters.se_size}x${resultData.parameters.se_size}`;
                        return (
                          <tr
                            key={idx}
                            style={{
                              borderBottom: '1px solid var(--border-subtle)',
                              backgroundColor: isCurrentActive ? 'rgba(56, 189, 248, 0.12)' : 'transparent',
                              fontWeight: isCurrentActive ? 600 : 400
                            }}
                          >
                            <td style={{ padding: '0.65rem 0.85rem', color: isCurrentActive ? 'var(--accent-cyan)' : '#fff' }}>
                              {row.shape_label} {isCurrentActive && '★ (Active)'}
                            </td>
                            <td style={{ padding: '0.65rem 0.85rem', fontFamily: 'var(--font-mono)' }}>{row.size}</td>
                            <td style={{ padding: '0.65rem 0.85rem', fontFamily: 'var(--font-mono)' }}>{row.boundary_pixels.toLocaleString()}</td>
                            <td style={{ padding: '0.65rem 0.85rem', fontFamily: 'var(--font-mono)', color: 'var(--accent-emerald)' }}>{row.boundary_to_fg_ratio_pct}%</td>
                            <td style={{ padding: '0.65rem 0.85rem', fontFamily: 'var(--font-mono)' }}>{row.connected_components}</td>
                            <td style={{ padding: '0.65rem 0.85rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                              {row.shape === 'cross' && 'Preserves sharp acute corners and fine protruding details.'}
                              {row.shape === 'ellipse' && 'Isotropic, rotationally uniform boundary contours.'}
                              {row.shape === 'rect' && 'Thicker diagonal boundaries; square corner bias.'}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* 4. Pedagogical Explanation of Structuring Elements */}
              <div className="section-card">
                <div className="section-card-header">
                  <h3 className="section-card-title">Morphological Theory & Detail Preservation</h3>
                  <span className="badge-academic">Mathematical Morphology</span>
                </div>

                <div className="decision-block">
                  <div className="decision-subtitle">Mathematical Definition:</div>
                  <p className="decision-text" style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>
                    β(A) = A − (A ⊖ B) = {"{"} z | (B)_z ⊆ A and (B)_z ⊄ (A ∖ ∂A) {"}"}
                  </p>

                  <div className="decision-subtitle" style={{ marginTop: '0.85rem' }}>Active Structuring Element Impact:</div>
                  <p className="decision-text">
                    {resultData.explanation.structuring_element_effect}
                  </p>

                  <div className="decision-subtitle" style={{ marginTop: '0.85rem' }}>Kernel Size & Iteration Effects:</div>
                  <p className="decision-text">
                    {resultData.explanation.size_effect} Larger structuring elements or multiple erosion iterations broaden the outer boundary band.
                    However, oversized kernels erode thin animal appendages (such as elephant tusks, zebra legs, or dog whiskers) completely into background,
                    causing severed boundary topology and fragmentation.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
