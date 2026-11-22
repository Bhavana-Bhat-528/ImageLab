import React, { useState } from 'react';
import Breadcrumbs from '../components/Breadcrumbs';
import PageHeader from '../components/PageHeader';
import ImageUploadSection from '../components/ImageUploadSection';
import SampleSelector from '../components/SampleSelector';
import { processWatershed } from '../services/api';

const WATERSHED_PRESETS = [
  {
    filename: '01_human_mitosis.png',
    label: 'Human Mitosis Cells',
    category: 'Dividing Biological Cells',
    description: 'Microscopic human cancer cells undergoing mitosis with touching nuclear boundaries. Reference: 24 cells.',
    recommendedRatio: 0.35
  },
  {
    filename: '02_cell_culture.png',
    label: 'Fluorescent Cell Culture',
    category: 'Confluent Cells',
    description: 'Adjacent fluorescent cell membranes requiring distance transform marker separation. Reference: 26 cells.',
    recommendedRatio: 0.25
  },
  {
    filename: '03_protein_transport_nuclei.png',
    label: 'Nuclear Translocation Assay',
    category: 'Clustered Nuclei',
    description: 'Compact spatial aggregation of cell nuclei with varying intensities. Reference: 22 cells.',
    recommendedRatio: 0.35
  },
  {
    filename: '04_overlapping_coins.png',
    label: 'Overlapping Particles Benchmark',
    category: 'Touching Convex Objects',
    description: 'Classic DIP benchmark containing 24 touching/overlapping convex disks with ground-truth mask.',
    recommendedRatio: 0.28
  }
];

export default function WatershedPage() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [activeSampleFilename, setActiveSampleFilename] = useState(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  // Parameter Controls State
  const [thresholdMethod, setThresholdMethod] = useState('otsu');
  const [thresholdVal, setThresholdVal] = useState(128);
  const [blurKsize, setBlurKsize] = useState(5);
  const [morphKsize, setMorphKsize] = useState(3);
  const [distRatio, setDistRatio] = useState(0.35);
  const [minCellArea, setMinCellArea] = useState(25);
  const [invertPolarity, setInvertPolarity] = useState('auto');
  const [boundaryColor, setBoundaryColor] = useState('red');
  const [boundaryThickness, setBoundaryThickness] = useState(2);
  const [manualReferenceCount, setManualReferenceCount] = useState('');

  // UI View Mode (grid, split)
  const [viewMode, setViewMode] = useState('grid');
  const [activeStageTab, setActiveStageTab] = useState('overlay');

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

    // Apply preset recommended ratio if known
    const preset = WATERSHED_PRESETS.find((p) => p.filename === sampleData.sampleFilename);
    if (preset && preset.recommendedRatio !== undefined) {
      setDistRatio(preset.recommendedRatio);
    }
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
        fileToUpload = new File([blob], selectedImage.meta?.name || 'microscopy.png', {
          type: blob.type || 'image/png'
        });
      }

      const formData = new FormData();
      formData.append('file', fileToUpload);
      formData.append('threshold_method', thresholdMethod);
      formData.append('threshold_val', thresholdVal.toString());
      formData.append('blur_ksize', blurKsize.toString());
      formData.append('morph_ksize', morphKsize.toString());
      formData.append('dist_ratio', distRatio.toString());
      formData.append('min_cell_area', minCellArea.toString());
      formData.append('invert_polarity', invertPolarity);
      formData.append('boundary_color', boundaryColor);
      formData.append('boundary_thickness', boundaryThickness.toString());

      if (manualReferenceCount !== '' && !isNaN(parseInt(manualReferenceCount, 10))) {
        formData.append('manual_reference_count', parseInt(manualReferenceCount, 10).toString());
      }

      const data = await processWatershed(formData);
      setResultData(data);
    } catch (err) {
      setErrorMessage(err.message || 'Watershed segmentation processing failed.');
    } finally {
      setIsExecuting(false);
    }
  };

  const handleReset = () => {
    setThresholdMethod('otsu');
    setThresholdVal(128);
    setBlurKsize(5);
    setMorphKsize(3);
    setDistRatio(0.35);
    setMinCellArea(25);
    setInvertPolarity('auto');
    setBoundaryColor('red');
    setBoundaryThickness(2);
    setManualReferenceCount('');
    setErrorMessage(null);
  };

  const STAGES_LIST = [
    { key: 'original_base64', id: 'original', num: 1, title: 'Original Specimen', desc: 'Raw microscopic image' },
    { key: 'preprocessed_base64', id: 'prep', num: 2, title: 'Preprocessed Grayscale', desc: 'Gaussian noise reduction' },
    { key: 'binary_mask_base64', id: 'binary', num: 3, title: 'Clean Binary Mask', desc: 'Thresholded & opened (A ∘ B)' },
    { key: 'distance_transform_base64', id: 'dist', num: 4, title: 'Distance Transform', desc: 'Euclidean distance (cv2.DIST_L2)' },
    { key: 'sure_fg_base64', id: 'sure_fg', num: 5, title: 'Sure Foreground', desc: 'Distance peak marker seeds' },
    { key: 'sure_bg_base64', id: 'sure_bg', num: 6, title: 'Sure Background', desc: 'Morphological dilation (A ⊕ B)' },
    { key: 'markers_map_base64', id: 'markers', num: 7, title: 'Watershed Markers', desc: 'Labeled seeds & unknown zone' },
    { key: 'watershed_overlay_base64', id: 'overlay', num: 8, title: 'Watershed Segmented', desc: 'Partitioned cells & boundaries' }
  ];

  return (
    <div className="module-page-container">
      <Breadcrumbs items={[{ label: 'Watershed Segmentation' }]} />

      <PageHeader
        moduleNumber="3"
        title="Morphological Watershed Segmentation"
        description="Separates touching or overlapping cells in microscopic images and estimates individual cell count. Combines Gaussian smoothing, Otsu/adaptive binarization, morphological opening & dilation, Euclidean distance transform, connected-component marker seeding, and topological watershed flooding."
        tags={['Cell Counting', 'Distance Transform', 'Marker Generation', 'Watershed Transform', 'Microscopy']}
        accentColor="purple"
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
            label="Upload Microscopic Image"
            hint="Microscopic specimen images with touching cells (min 3 real-world test sets)"
            disabled={isExecuting}
          />

          <SampleSelector
            moduleName="watershed"
            presets={WATERSHED_PRESETS}
            onSelectSample={handleSelectSample}
            activeSampleFilename={activeSampleFilename}
          />

          {/* Watershed Parameter Controls Panel */}
          <div className="section-card controls-section">
            <div className="section-card-header">
              <h3 className="section-card-title">3. Watershed Parameters</h3>
              <span className="badge-academic">cv2.watershed</span>
            </div>

            <div className="controls-form">
              {/* Thresholding Method */}
              <div className="control-group">
                <label className="control-label">Binarization Method</label>
                <select
                  className="control-select"
                  value={thresholdMethod}
                  onChange={(e) => setThresholdMethod(e.target.value)}
                  disabled={isExecuting}
                >
                  <option value="otsu">Otsu Global Binarization (Automatic)</option>
                  <option value="manual">Manual Intensity Threshold</option>
                  <option value="adaptive">Adaptive Gaussian Thresholding</option>
                </select>
                <span className="control-help">
                  Otsu minimizes intra-class variance between cells and background.
                </span>
              </div>

              {/* Manual Threshold Value (if manual selected) */}
              {thresholdMethod === 'manual' && (
                <div className="control-group">
                  <div className="control-label-row">
                    <label className="control-label">Manual Threshold Value</label>
                    <span className="control-value-badge">{thresholdVal}</span>
                  </div>
                  <input
                    type="range"
                    className="control-range"
                    min="1"
                    max="254"
                    step="1"
                    value={thresholdVal}
                    onChange={(e) => setThresholdVal(parseInt(e.target.value, 10))}
                    disabled={isExecuting}
                  />
                  <div className="range-bounds">
                    <span>1 (Dark)</span>
                    <span>128</span>
                    <span>254 (Bright)</span>
                  </div>
                </div>
              )}

              {/* Distance Transform Threshold Ratio */}
              <div className="control-group">
                <div className="control-label-row">
                  <label className="control-label">Distance Transform Ratio (α)</label>
                  <span className="control-value-badge" style={{ color: 'var(--accent-cyan)' }}>
                    {(distRatio * 100).toFixed(0)}% of Max Dist
                  </span>
                </div>
                <input
                  type="range"
                  className="control-range"
                  min="0.10"
                  max="0.80"
                  step="0.01"
                  value={distRatio}
                  onChange={(e) => setDistRatio(parseFloat(e.target.value))}
                  disabled={isExecuting}
                />
                <div className="range-bounds">
                  <span>10% (Tolerant / Many Seeds)</span>
                  <span>35%</span>
                  <span>80% (Strict / Few Seeds)</span>
                </div>
                <span className="control-help">
                  Controls sure foreground peak isolation: Sure_FG = Dist &gt; α × max(Dist).
                </span>
              </div>

              {/* Minimum Cell Area Filter */}
              <div className="control-group">
                <div className="control-label-row">
                  <label className="control-label">Min Cell Area Filter</label>
                  <span className="control-value-badge">{minCellArea} px²</span>
                </div>
                <input
                  type="range"
                  className="control-range"
                  min="0"
                  max="200"
                  step="5"
                  value={minCellArea}
                  onChange={(e) => setMinCellArea(parseInt(e.target.value, 10))}
                  disabled={isExecuting}
                />
                <span className="control-help">
                  Suppresses subcellular fluorescent debris from forming spurious markers.
                </span>
              </div>

              {/* Gaussian Blur Kernel Size */}
              <div className="control-group">
                <div className="control-label-row">
                  <label className="control-label">Gaussian Blur Kernel</label>
                  <span className="control-value-badge">{blurKsize} × {blurKsize}</span>
                </div>
                <select
                  className="control-select"
                  value={blurKsize}
                  onChange={(e) => setBlurKsize(parseInt(e.target.value, 10))}
                  disabled={isExecuting}
                >
                  <option value={3}>3 × 3 (Sharp / Minimal Smoothing)</option>
                  <option value={5}>5 × 5 (Standard Smoothing)</option>
                  <option value={7}>7 × 7 (Heavy Noise Suppression)</option>
                  <option value={9}>9 × 9 (Extreme Blur)</option>
                </select>
              </div>

              {/* Morphological Kernel Size */}
              <div className="control-group">
                <div className="control-label-row">
                  <label className="control-label">Morphological Kernel (Opening & Dilation)</label>
                  <span className="control-value-badge">{morphKsize} × {morphKsize}</span>
                </div>
                <select
                  className="control-select"
                  value={morphKsize}
                  onChange={(e) => setMorphKsize(parseInt(e.target.value, 10))}
                  disabled={isExecuting}
                >
                  <option value={3}>3 × 3 (Fine Details Preserved)</option>
                  <option value={5}>5 × 5 (Moderate Boundary Smoothing)</option>
                  <option value={7}>7 × 7 (Large Boundary Separation)</option>
                </select>
              </div>

              {/* Polarity Check */}
              <div className="control-group">
                <label className="control-label">Specimen Background Polarity</label>
                <select
                  className="control-select"
                  value={invertPolarity}
                  onChange={(e) => setInvertPolarity(e.target.value)}
                  disabled={isExecuting}
                >
                  <option value="auto">Auto-Detect Background Polarity</option>
                  <option value="no">Normal (Dark Background, Bright Cells)</option>
                  <option value="yes">Invert (Bright Background, Dark Cells / Stains)</option>
                </select>
              </div>

              {/* Boundary Color and Thickness */}
              <div className="control-group">
                <label className="control-label">Watershed Boundary Color</label>
                <select
                  className="control-select"
                  value={boundaryColor}
                  onChange={(e) => setBoundaryColor(e.target.value)}
                  disabled={isExecuting}
                >
                  <option value="red">Vibrant Red</option>
                  <option value="yellow">High-Vis Yellow</option>
                  <option value="cyan">Neon Cyan</option>
                  <option value="green">Lime Green</option>
                  <option value="white">Clean White</option>
                  <option value="magenta">Vibrant Magenta</option>
                </select>
              </div>

              <div className="control-group">
                <div className="control-label-row">
                  <label className="control-label">Boundary Line Thickness</label>
                  <span className="control-value-badge">{boundaryThickness} px</span>
                </div>
                <input
                  type="range"
                  className="control-range"
                  min="1"
                  max="3"
                  step="1"
                  value={boundaryThickness}
                  onChange={(e) => setBoundaryThickness(parseInt(e.target.value, 10))}
                  disabled={isExecuting}
                />
              </div>

              {/* Manual Reference Count */}
              <div className="control-group">
                <label className="control-label">Manual Reference Cell Count (Optional)</label>
                <input
                  type="number"
                  className="control-input"
                  placeholder="e.g. 24 (leave blank for preset catalog)"
                  min="1"
                  max="1000"
                  value={manualReferenceCount}
                  onChange={(e) => setManualReferenceCount(e.target.value)}
                  disabled={isExecuting}
                />
                <span className="control-help">
                  Allows quantitative verification against manual microscope hemocytometer counts.
                </span>
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
                      Segmenting Cells...
                    </>
                  ) : (
                    <>
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <polygon points="5 3 19 12 5 21 5 3" />
                      </svg>
                      Run Watershed Segmentation
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

        {/* Right Column: Visualizations, Cell Counts, and Metrics */}
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
              <h4 className="empty-state-title">No Microscopic Specimen Loaded</h4>
              <p className="empty-state-text">
                Select one of the benchmark microscopy images (Human Mitosis, Cell Culture, Clustered Nuclei, or Overlapping Particles) or upload a custom image to execute distance-transform watershed segmentation.
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
                Specimen Ready for Watershed Segmentation
              </h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', maxWidth: '480px', margin: '0 auto 1.5rem auto' }}>
                Click <strong>"Run Watershed Segmentation"</strong> to execute grayscale binarization, morphological opening, Euclidean distance transform, sure marker labeling, and topological ridge extraction.
              </p>
              <button
                type="button"
                className="btn btn-primary"
                onClick={handleExecute}
              >
                Run Watershed Segmentation
              </button>
            </div>
          )}

          {isExecuting && (
            <div className="section-card loading-state-card">
              <div className="academic-spinner"></div>
              <div className="loading-title">Executing Morphological Watershed...</div>
              <div className="loading-sub">
                Computing Euclidean distance map, resolving touching cell markers, and establishing topological catchment basins.
              </div>
            </div>
          )}

          {resultData && (
            <div className="enhancement-results-container">
              {/* 1. Prominent Estimated Cell Count Hero Banner */}
              <div className="watershed-hero-card">
                <div className="watershed-hero-content">
                  <div className="watershed-count-block">
                    <span className="watershed-count-number">
                      {resultData.results.estimated_cell_count}
                    </span>
                    <div className="watershed-count-label">
                      <span className="watershed-count-title">Individual Cells Segmented</span>
                      <span className="watershed-count-subtitle">
                        {resultData.results.foreground_area_percentage}% specimen coverage • {resultData.results.total_foreground_pixels.toLocaleString()} fg pixels
                      </span>
                    </div>
                  </div>

                  <div className="watershed-comparison-badges">
                    {resultData.evaluation?.count_comparison?.reference_count !== undefined && (
                      <div
                        className={`watershed-pill ${
                          resultData.evaluation.count_comparison.status_color === 'green'
                            ? 'watershed-pill-success'
                            : resultData.evaluation.count_comparison.status_color === 'blue'
                            ? 'watershed-pill-info'
                            : 'watershed-pill-warning'
                        }`}
                      >
                        <span>
                          Ref: {resultData.evaluation.count_comparison.reference_count} cells
                        </span>
                        <span>•</span>
                        <span>
                          Δ = {resultData.evaluation.count_comparison.difference > 0 ? '+' : ''}
                          {resultData.evaluation.count_comparison.difference} (
                          {resultData.evaluation.count_comparison.percentage_error}% err)
                        </span>
                        <span>•</span>
                        <span>{resultData.evaluation.count_comparison.status}</span>
                      </div>
                    )}
                  </div>
                </div>

                {resultData.evaluation?.count_comparison?.notes && (
                  <div style={{ marginTop: '0.85rem', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                    💡 <em>{resultData.evaluation.count_comparison.notes}</em>
                  </div>
                )}
              </div>

              {/* 2. Ground Truth Spatial Overlap Metrics (IoU & Dice) */}
              {resultData.evaluation?.ground_truth_segmentation && (
                <div className="section-card">
                  <div className="section-card-header">
                    <div>
                      <h3 className="section-card-title">Ground-Truth Segmentation Accuracy</h3>
                      <p className="section-card-sub" style={{ marginBottom: 0 }}>
                        Spatial overlap against manually verified reference mask: {resultData.evaluation.ground_truth_segmentation.ground_truth_name}
                      </p>
                    </div>
                    <span className="badge-academic" style={{ borderColor: 'var(--accent-emerald)', color: 'var(--accent-emerald)' }}>
                      Mask Validated
                    </span>
                  </div>

                  <div className="metrics-grid" style={{ marginTop: '1rem' }}>
                    <div className="metric-box" style={{ borderColor: 'rgba(52, 211, 153, 0.4)' }}>
                      <span className="metric-label">IoU (Jaccard Index)</span>
                      <span className="metric-value" style={{ color: 'var(--accent-emerald)' }}>
                        {(resultData.evaluation.ground_truth_segmentation.iou * 100).toFixed(2)}
                        <span className="metric-unit"> %</span>
                      </span>
                      <div className="metric-progress-wrapper">
                        <div
                          className="metric-progress-fill"
                          style={{
                            width: `${Math.min(100, resultData.evaluation.ground_truth_segmentation.iou * 100)}%`,
                            backgroundColor: 'var(--accent-emerald)'
                          }}
                        ></div>
                      </div>
                      <span className="metric-description">Spatial intersection over union</span>
                    </div>

                    <div className="metric-box" style={{ borderColor: 'rgba(52, 211, 153, 0.4)' }}>
                      <span className="metric-label">Dice Similarity (F1 Score)</span>
                      <span className="metric-value" style={{ color: 'var(--accent-emerald)' }}>
                        {(resultData.evaluation.ground_truth_segmentation.dice * 100).toFixed(2)}
                        <span className="metric-unit"> %</span>
                      </span>
                      <div className="metric-progress-wrapper">
                        <div
                          className="metric-progress-fill"
                          style={{
                            width: `${Math.min(100, resultData.evaluation.ground_truth_segmentation.dice * 100)}%`,
                            backgroundColor: 'var(--accent-cyan)'
                          }}
                        ></div>
                      </div>
                      <span className="metric-description">Harmonic mean of precision & recall</span>
                    </div>

                    <div className="metric-box">
                      <span className="metric-label">True Overlap Pixels</span>
                      <span className="metric-value">
                        {resultData.evaluation.ground_truth_segmentation.overlap_pixels?.toLocaleString()}
                      </span>
                      <span className="metric-description">
                        Target GT: {resultData.evaluation.ground_truth_segmentation.ground_truth_pixels?.toLocaleString()} px
                      </span>
                    </div>

                    <div className="metric-box">
                      <span className="metric-label">Max Distance Transform</span>
                      <span className="metric-value" style={{ color: 'var(--accent-cyan)' }}>
                        {resultData.results.max_distance_transform_px}
                        <span className="metric-unit"> px</span>
                      </span>
                      <span className="metric-description">
                        Sure FG Seed Cutoff: {resultData.results.sure_fg_threshold_value} px
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* 3. Multi-Stage Visual Pipeline (8 Synchronized Stages) */}
              <div className="section-card">
                <div className="section-card-header">
                  <div>
                    <h3 className="section-card-title">4. Multi-Stage Watershed Pipeline (8 Stages)</h3>
                    <p className="section-card-sub" style={{ marginBottom: 0 }}>
                      Complete mathematical pipeline from raw specimen to distance-partitioned catchment basins.
                    </p>
                  </div>

                  <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                    <button
                      type="button"
                      className={`btn btn-sm ${viewMode === 'grid' ? 'btn-primary' : 'btn-outline'}`}
                      onClick={() => setViewMode('grid')}
                    >
                      Grid View (All 8)
                    </button>
                    <button
                      type="button"
                      className={`btn btn-sm ${viewMode === 'split' ? 'btn-primary' : 'btn-outline'}`}
                      onClick={() => setViewMode('split')}
                    >
                      Split View
                    </button>
                  </div>
                </div>

                {viewMode === 'grid' ? (
                  <div className="results-grid results-grid-4" style={{ marginTop: '1rem' }}>
                    {STAGES_LIST.map((stage) => (
                      <div key={stage.id} className="result-viewport-card">
                        <div className="viewport-header">
                          <span className="viewport-stage-index">Stage {stage.num}</span>
                          <span className="viewport-label">{stage.title}</span>
                        </div>
                        <div className="viewport-canvas">
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
                          <span className="viewport-stage-index">Input Specimen</span>
                          <span className="viewport-label">Original Image</span>
                        </div>
                        <div className="viewport-canvas" style={{ height: '360px' }}>
                          <img src={resultData.images.original_base64} alt="Original" className="canvas-img" />
                        </div>
                      </div>

                      <div className="result-viewport-card">
                        <div className="viewport-header">
                          <span className="viewport-stage-index">Final Output</span>
                          <span className="viewport-label">Segmented Cells & Watershed Ridges</span>
                        </div>
                        <div className="viewport-canvas" style={{ height: '360px' }}>
                          <img src={resultData.images.watershed_overlay_base64} alt="Segmented Overlay" className="canvas-img" />
                        </div>
                      </div>
                    </div>

                    {/* Stage selector tabs for inspecting intermediate representations */}
                    <div style={{ marginTop: '1rem' }}>
                      <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap', marginBottom: '0.75rem' }}>
                        {STAGES_LIST.map((st) => (
                          <button
                            key={st.id}
                            type="button"
                            className={`btn btn-sm ${activeStageTab === st.id ? 'btn-primary' : 'btn-outline'}`}
                            onClick={() => setActiveStageTab(st.id)}
                            style={{ fontSize: '0.75rem', padding: '0.25rem 0.6rem' }}
                          >
                            Stage {st.num}: {st.title}
                          </button>
                        ))}
                      </div>

                      {(() => {
                        const cur = STAGES_LIST.find((s) => s.id === activeStageTab) || STAGES_LIST[7];
                        return (
                          <div className="result-viewport-card">
                            <div className="viewport-header">
                              <span className="viewport-stage-index">Stage {cur.num} Detail</span>
                              <span className="viewport-label">{cur.title}</span>
                            </div>
                            <div className="viewport-canvas" style={{ height: '380px' }}>
                              <img src={resultData.images[cur.key]} alt={cur.title} className="canvas-img" />
                            </div>
                            <div style={{ padding: '0.65rem 0.85rem', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                              <strong>Mathematical Description:</strong> {cur.desc}
                            </div>
                          </div>
                        );
                      })()}
                    </div>
                  </div>
                )}
              </div>

              {/* 4. Cell Morphology Measurements Section & Table */}
              <div className="section-card">
                <div className="section-card-header">
                  <div>
                    <h3 className="section-card-title">Cell Morphology Statistics & Individual Region Table</h3>
                    <p className="section-card-sub" style={{ marginBottom: 0 }}>
                      Quantitative measurements: area, equivalent spherical diameter, centroid coordinates, and circularity index.
                    </p>
                  </div>
                  <span className="badge-academic">{resultData.results.cell_details?.length || 0} Regions Measured</span>
                </div>

                <div className="metrics-grid" style={{ marginTop: '1rem' }}>
                  <div className="metric-box">
                    <span className="metric-label">Mean Cell Area</span>
                    <span className="metric-value">
                      {resultData.results.statistics?.mean_cell_area_px}
                      <span className="metric-unit"> px²</span>
                    </span>
                    <span className="metric-description">Average footprint</span>
                  </div>

                  <div className="metric-box">
                    <span className="metric-label">Std Cell Area</span>
                    <span className="metric-value">
                      ±{resultData.results.statistics?.std_cell_area_px}
                      <span className="metric-unit"> px²</span>
                    </span>
                    <span className="metric-description">Area variation (σ)</span>
                  </div>

                  <div className="metric-box">
                    <span className="metric-label">Min Cell Area</span>
                    <span className="metric-value">
                      {resultData.results.statistics?.min_cell_area_px}
                      <span className="metric-unit"> px²</span>
                    </span>
                    <span className="metric-description">Smallest detected cell</span>
                  </div>

                  <div className="metric-box">
                    <span className="metric-label">Max Cell Area</span>
                    <span className="metric-value">
                      {resultData.results.statistics?.max_cell_area_px}
                      <span className="metric-unit"> px²</span>
                    </span>
                    <span className="metric-description">Largest cluster/cell</span>
                  </div>
                </div>

                {/* Individual Cell Measurements Table */}
                {resultData.results.cell_details && resultData.results.cell_details.length > 0 && (
                  <div style={{ marginTop: '1.25rem' }}>
                    <div className="cell-table-container">
                      <table className="cell-table">
                        <thead>
                          <tr>
                            <th>Cell ID</th>
                            <th>Area (px²)</th>
                            <th>Centroid (X, Y)</th>
                            <th>Eq. Diameter (px)</th>
                            <th>Circularity (4πA/P²)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {resultData.results.cell_details.map((cell) => (
                            <tr key={cell.id}>
                              <td style={{ fontWeight: 600, color: '#38bdf8' }}>#{cell.id}</td>
                              <td>{cell.area_px.toLocaleString()}</td>
                              <td>({cell.centroid_x}, {cell.centroid_y})</td>
                              <td>{cell.equivalent_diameter_px}</td>
                              <td>
                                <span style={{
                                  color: cell.circularity > 0.8 ? 'var(--accent-emerald)' : cell.circularity > 0.5 ? 'var(--accent-amber)' : 'var(--text-muted)'
                                }}>
                                  {cell.circularity}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <div style={{ fontSize: '0.725rem', color: 'var(--text-dim)', marginTop: '0.4rem', textAlign: 'right' }}>
                      Showing top {resultData.results.cell_details.length} segmented regions. Circularity: 1.0 = perfect circle.
                    </div>
                  </div>
                )}
              </div>

              {/* 5. Scientific Limitations of Microscopic Watershed Segmentation */}
              <div className="section-card">
                <div className="section-card-header">
                  <div>
                    <h3 className="section-card-title">Scientific Limitations of Watershed in Microscopy</h3>
                    <p className="section-card-sub" style={{ marginBottom: 0 }}>
                      Physical and biological phenomena impacting mathematical watershed accuracy and separation.
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
                        <span>⚠️ {item.category}</span>
                      </div>
                      <div className="limitation-item-text">
                        <strong>Physical Phenomenon:</strong> {item.observation}
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
