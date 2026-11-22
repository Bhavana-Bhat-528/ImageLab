import React, { useState } from 'react';
import Breadcrumbs from '../components/Breadcrumbs';
import PageHeader from '../components/PageHeader';
import ImageUploadSection from '../components/ImageUploadSection';
import SampleSelector from '../components/SampleSelector';
import HistogramComparison from '../components/HistogramComparison';
import { processEnhancement } from '../services/api';

const ENHANCEMENT_PRESETS = [
  {
    filename: '01_low_contrast.png',
    label: 'Low Contrast Scene',
    category: 'Contrast Deficient',
    description: 'Narrow dynamic range restricted to [95, 145]. Tests contrast stretching & CLAHE recovery.'
  },
  {
    filename: '02_dark_underexposed.png',
    label: 'Underexposed Image',
    category: 'Low Brightness',
    description: 'Low mean pixel intensity with severe shadow clipping. Tests power-law gamma expansion.'
  },
  {
    filename: '03_bright_overexposed.png',
    label: 'Overexposed Image',
    category: 'High Brightness',
    description: 'Saturated highlights. Tests compressive gamma correction and highlight restoration.'
  },
  {
    filename: '04_gaussian_noise.png',
    label: 'Gaussian Noise Image',
    category: 'Sensor Noise',
    description: 'Additive Gaussian noise (sigma=28). Tests spatial Gaussian and bilateral filtering.'
  },
  {
    filename: '05_salt_and_pepper_noise.png',
    label: 'Salt & Pepper Noise',
    category: 'Impulse Noise',
    description: 'High-frequency impulse noise (p=0.05). Tests non-linear median filtering.'
  },
  {
    filename: '06_uneven_illumination.png',
    label: 'Uneven Illumination',
    category: 'Non-Uniform Lighting',
    description: 'Non-uniform illumination gradient across document text.'
  }
];

export default function EnhancementPage() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [activeSampleFilename, setActiveSampleFilename] = useState(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  // Operational Mode: 'auto' | 'manual'
  const [mode, setMode] = useState('auto');

  // Manual Controls State
  const [gamma, setGamma] = useState(1.0);
  const [equalizationMode, setEqualizationMode] = useState('clahe');
  const [claheClipLimit, setClaheClipLimit] = useState(2.5);
  const [claheTileGridSize, setClaheTileGridSize] = useState(8);
  const [spatialFilter, setSpatialFilter] = useState('none');
  const [kernelSize, setKernelSize] = useState(3);
  const [sharpenStrength, setSharpenStrength] = useState(0.0);

  // Processing Results from API
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

      // If sample was selected without local file object, fetch its blob
      if (!fileToUpload && selectedImage.previewUrl) {
        const resp = await fetch(selectedImage.previewUrl);
        const blob = await resp.blob();
        fileToUpload = new File([blob], selectedImage.meta?.name || 'sample.png', {
          type: blob.type || 'image/png'
        });
      }

      const formData = new FormData();
      formData.append('file', fileToUpload);
      formData.append('mode', mode);
      formData.append('gamma', gamma.toString());
      formData.append('equalization_mode', equalizationMode);
      formData.append('clahe_clip_limit', claheClipLimit.toString());
      formData.append('clahe_tile_grid_size', claheTileGridSize.toString());
      formData.append('spatial_filter', spatialFilter);
      formData.append('kernel_size', kernelSize.toString());
      formData.append('sharpen_strength', sharpenStrength.toString());

      const data = await processEnhancement(formData);
      setResultData(data);
    } catch (err) {
      setErrorMessage(err.message || 'Image enhancement processing failed.');
    } finally {
      setIsExecuting(false);
    }
  };

  const handleResetManual = () => {
    setGamma(1.0);
    setEqualizationMode('clahe');
    setClaheClipLimit(2.5);
    setClaheTileGridSize(8);
    setSpatialFilter('none');
    setKernelSize(3);
    setSharpenStrength(0.0);
    setErrorMessage(null);
  };

  return (
    <div className="module-page-container">
      <Breadcrumbs items={[{ label: 'Automatic Image Enhancement' }]} />

      <PageHeader
        moduleNumber="1"
        title="Automatic Image Enhancement"
        description="Analyzes input image characteristics (brightness, contrast, noise, sharpness, and histogram dispersion) and selects an appropriate intensity transformation and spatial filter. Compares output quality using MSE, PSNR, and SSIM."
        tags={['Intensity Transformation', 'Spatial Filtering', 'Histogram Analysis', 'PSNR / SSIM', 'Fidelity Metrics']}
        accentColor="blue"
      />

      <div className="module-layout-grid">
        {/* Left Column: Input, Benchmark Samples, and Controls */}
        <div className="layout-sidebar">
          <ImageUploadSection
            selectedFile={selectedImage?.file}
            imagePreview={selectedImage?.previewUrl}
            imageMeta={selectedImage?.meta}
            onImageSelected={handleImageSelected}
            onClearImage={handleClearImage}
            label="Upload Test Image"
            hint="PNG, JPG, JPEG, or BMP (min 5 different test conditions)"
            disabled={isExecuting}
          />

          <SampleSelector
            moduleName="enhancement"
            presets={ENHANCEMENT_PRESETS}
            onSelectSample={handleSelectSample}
            activeSampleFilename={activeSampleFilename}
          />

          {/* Mode Switch & Controls */}
          <div className="section-card controls-section">
            <div className="section-card-header">
              <h3 className="section-card-title">3. Enhancement Controls</h3>
              <div className="mode-toggle-group">
                <button
                  type="button"
                  className={`btn-mode ${mode === 'auto' ? 'active' : ''}`}
                  onClick={() => setMode('auto')}
                  disabled={isExecuting}
                >
                  Adaptive Auto
                </button>
                <button
                  type="button"
                  className={`btn-mode ${mode === 'manual' ? 'active' : ''}`}
                  onClick={() => setMode('manual')}
                  disabled={isExecuting}
                >
                  Manual Controls
                </button>
              </div>
            </div>

            {mode === 'auto' ? (
              <div className="auto-mode-info-box">
                <div className="auto-info-title">
                  <span className="dot-auto"></span>
                  Adaptive Decision Engine
                </div>
                <p className="auto-info-desc">
                  The system analyzes statistical metrics (mean intensity, contrast spread, noise standard deviation,
                  sharpness, and clipping percentages) and selects only justified operations.
                </p>
                <div className="auto-rules-summary">
                  <div>• Dark images &rarr; Power-law gamma expansion (γ &lt; 1.0)</div>
                  <div>• Overexposed &rarr; Compressive gamma (γ &gt; 1.0)</div>
                  <div>• Low contrast &rarr; CLAHE adaptive equalization</div>
                  <div>• Impulse noise &rarr; Non-linear median filter (3x3)</div>
                  <div>• Gaussian noise &rarr; Bilateral / Gaussian smoothing</div>
                  <div>• Soft blur &rarr; High-boost unsharp masking</div>
                </div>
              </div>
            ) : (
              <div className="controls-form">
                {/* Gamma */}
                <div className="control-group">
                  <div className="control-label-row">
                    <label className="control-label">Gamma Correction (γ)</label>
                    <span className="control-value-badge">{gamma.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    className="control-range"
                    min="0.2"
                    max="3.0"
                    step="0.05"
                    value={gamma}
                    onChange={(e) => setGamma(parseFloat(e.target.value))}
                    disabled={isExecuting}
                  />
                  <div className="range-bounds">
                    <span>0.2 (Brighten)</span>
                    <span>1.0 (Identity)</span>
                    <span>3.0 (Darken)</span>
                  </div>
                </div>

                {/* Histogram Equalization / CLAHE */}
                <div className="control-group">
                  <label className="control-label">Contrast Equalization Mode</label>
                  <select
                    className="control-select"
                    value={equalizationMode}
                    onChange={(e) => setEqualizationMode(e.target.value)}
                    disabled={isExecuting}
                  >
                    <option value="none">None (Preserve Original Histogram)</option>
                    <option value="clahe">CLAHE (Contrast Limited Adaptive)</option>
                    <option value="global">Global Histogram Equalization</option>
                  </select>
                </div>

                {equalizationMode === 'clahe' && (
                  <>
                    <div className="control-group">
                      <div className="control-label-row">
                        <label className="control-label">CLAHE Clip Limit</label>
                        <span className="control-value-badge">{claheClipLimit.toFixed(1)}</span>
                      </div>
                      <input
                        type="range"
                        className="control-range"
                        min="1.0"
                        max="8.0"
                        step="0.5"
                        value={claheClipLimit}
                        onChange={(e) => setClaheClipLimit(parseFloat(e.target.value))}
                        disabled={isExecuting}
                      />
                    </div>

                    <div className="control-group">
                      <label className="control-label">CLAHE Tile Grid Size</label>
                      <select
                        className="control-select"
                        value={claheTileGridSize}
                        onChange={(e) => setClaheTileGridSize(parseInt(e.target.value, 10))}
                        disabled={isExecuting}
                      >
                        <option value={4}>4 × 4 Tiles</option>
                        <option value={8}>8 × 8 Tiles (Standard)</option>
                        <option value={16}>16 × 16 Tiles</option>
                      </select>
                    </div>
                  </>
                )}

                {/* Spatial Filter */}
                <div className="control-group">
                  <label className="control-label">Spatial Filter Selection</label>
                  <select
                    className="control-select"
                    value={spatialFilter}
                    onChange={(e) => setSpatialFilter(e.target.value)}
                    disabled={isExecuting}
                  >
                    <option value="none">None (No Spatial Filtering)</option>
                    <option value="gaussian">Gaussian Smoothing (Noise Reduction)</option>
                    <option value="median">Median Filter (Impulse Noise Removal)</option>
                    <option value="bilateral">Bilateral Filter (Edge-Preserving)</option>
                    <option value="unsharp">Unsharp Mask (Spatial Sharpening)</option>
                  </select>
                </div>

                {spatialFilter !== 'none' && (
                  <div className="control-group">
                    <label className="control-label">Filter Kernel Size</label>
                    <select
                      className="control-select"
                      value={kernelSize}
                      onChange={(e) => setKernelSize(parseInt(e.target.value, 10))}
                      disabled={isExecuting}
                    >
                      <option value={3}>3 × 3 Kernel (Fine)</option>
                      <option value={5}>5 × 5 Kernel</option>
                      <option value={7}>7 × 7 Kernel (Coarse)</option>
                    </select>
                  </div>
                )}

                {/* Sharpening Strength */}
                <div className="control-group">
                  <div className="control-label-row">
                    <label className="control-label">Sharpening Strength</label>
                    <span className="control-value-badge">{sharpenStrength.toFixed(1)}</span>
                  </div>
                  <input
                    type="range"
                    className="control-range"
                    min="0.0"
                    max="3.0"
                    step="0.2"
                    value={sharpenStrength}
                    onChange={(e) => setSharpenStrength(parseFloat(e.target.value))}
                    disabled={isExecuting}
                  />
                </div>
              </div>
            )}

            <div className="controls-actions" style={{ marginTop: '1.25rem' }}>
              <button
                type="button"
                className="btn btn-primary btn-execute"
                disabled={!selectedImage || isExecuting}
                onClick={handleExecute}
              >
                {isExecuting ? (
                  <>
                    <span className="spinner-sm"></span>
                    Executing in Python...
                  </>
                ) : (
                  <>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polygon points="5 3 19 12 5 21 5 3" />
                    </svg>
                    {mode === 'auto' ? 'Analyze & Enhance (Auto)' : 'Execute Custom Enhancement'}
                  </>
                )}
              </button>

              {mode === 'manual' && (
                <button
                  type="button"
                  className="btn btn-outline"
                  onClick={handleResetManual}
                  disabled={isExecuting}
                >
                  Reset
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Right Column: Visual Comparison, Diagnostics, Histograms & Metrics */}
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
              <h4 className="empty-state-title">No Image Loaded</h4>
              <p className="empty-state-text">
                Select one of the 6 benchmark sample images or upload a degraded photograph to start the enhancement workflow.
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
                Image Ready for Analysis
              </h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', maxWidth: '460px', margin: '0 auto 1.5rem auto' }}>
                Click <strong>"Analyze & Enhance"</strong> to evaluate mean luminance, contrast, noise floor, and edge sharpness in Python FastAPI.
              </p>
              <button
                type="button"
                className="btn btn-primary"
                onClick={handleExecute}
              >
                Start Automatic Enhancement
              </button>
            </div>
          )}

          {isExecuting && (
            <div className="section-card loading-state-card">
              <div className="academic-spinner"></div>
              <div className="loading-title">Processing In-Memory with OpenCV & NumPy...</div>
              <div className="loading-sub">
                Computing 256-bin histograms, estimating noise variance, and executing adaptive intensity transformations.
              </div>
            </div>
          )}

          {resultData && (
            <div className="enhancement-results-container">
              {/* 1. Visual Comparison Viewports */}
              <div className="section-card">
                <div className="section-card-header">
                  <h3 className="section-card-title">4. Visual Comparison Viewports</h3>
                  <span className="badge-academic">
                    Resolution: {resultData.input_characteristics.is_color ? 'Color BGR' : 'Grayscale'}
                  </span>
                </div>

                <div className="results-grid results-grid-2">
                  <div className="result-viewport-card">
                    <div className="viewport-header">
                      <span className="viewport-stage-index">Stage 1</span>
                      <span className="viewport-label">Original Input</span>
                    </div>
                    <div className="viewport-canvas">
                      <img
                        src={resultData.images.original_base64}
                        alt="Original"
                        className="canvas-img"
                      />
                    </div>
                  </div>

                  <div className="result-viewport-card">
                    <div className="viewport-header">
                      <span className="viewport-stage-index">Stage 2</span>
                      <span className="viewport-label">Enhanced Result</span>
                    </div>
                    <div className="viewport-canvas">
                      <img
                        src={resultData.images.enhanced_base64}
                        alt="Enhanced"
                        className="canvas-img"
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* 2. Detected Characteristics Diagnostics */}
              <div className="section-card">
                <div className="section-card-header">
                  <h3 className="section-card-title">Detected Image Characteristics</h3>
                  <span className="badge-academic">Statistical Diagnostics</span>
                </div>

                <div className="char-tags-row">
                  {resultData.input_characteristics.diagnostics.primary_defects.map((defect, i) => (
                    <span key={i} className="defect-badge">
                      {defect}
                    </span>
                  ))}
                </div>

                <div className="metrics-grid" style={{ marginTop: '1rem' }}>
                  <div className="metric-box">
                    <span className="metric-label">Mean Intensity</span>
                    <span className="metric-value">
                      {resultData.input_characteristics.mean_intensity}
                      <span className="metric-unit"> / 255</span>
                    </span>
                    <span className="metric-description">
                      Shadow Clip: {resultData.input_characteristics.shadow_clipping_pct}%
                    </span>
                  </div>

                  <div className="metric-box">
                    <span className="metric-label">Standard Deviation (Spread)</span>
                    <span className="metric-value">
                      {resultData.input_characteristics.std_intensity}
                    </span>
                    <span className="metric-description">
                      RMS Contrast: {resultData.input_characteristics.rms_contrast}
                    </span>
                  </div>

                  <div className="metric-box">
                    <span className="metric-label">Noise Floor (Sigma)</span>
                    <span className="metric-value">
                      {resultData.input_characteristics.noise_sigma}
                    </span>
                    <span className="metric-description">
                      {resultData.input_characteristics.has_salt_pepper_noise ? 'Impulse Noise Detected' : 'Continuous Noise'}
                    </span>
                  </div>

                  <div className="metric-box">
                    <span className="metric-label">Sharpness Score</span>
                    <span className="metric-value">
                      {resultData.input_characteristics.sharpness_score}
                    </span>
                    <span className="metric-description">
                      Variance of Laplacian
                    </span>
                  </div>
                </div>
              </div>

              {/* 3. Technique Selection & Explanations */}
              <div className="section-card">
                <div className="section-card-header">
                  <h3 className="section-card-title">Selected Techniques & Decision Rationale</h3>
                  <span className="badge-academic">
                    Mode: {resultData.mode.toUpperCase()}
                  </span>
                </div>

                <div className="decision-block">
                  <div className="decision-subtitle">Applied Operations:</div>
                  <ul className="decision-list">
                    {resultData.pipeline.selected_operations.map((op, i) => (
                      <li key={i} className="decision-item">
                        <strong>✓ {op}</strong>
                      </li>
                    ))}
                  </ul>

                  <div className="decision-subtitle" style={{ marginTop: '0.85rem' }}>Justification:</div>
                  <p className="decision-text">
                    {resultData.pipeline.justifications.join(' ')}
                  </p>

                  <div className="decision-subtitle" style={{ marginTop: '0.85rem' }}>Expected Improvement:</div>
                  <p className="decision-text" style={{ color: 'var(--accent-emerald)' }}>
                    {resultData.pipeline.expected_improvements.join(' ')}
                  </p>

                  {resultData.pipeline.potential_risks && resultData.pipeline.potential_risks.length > 0 && (
                    <>
                      <div className="decision-subtitle" style={{ marginTop: '0.85rem' }}>Potential Artifacts / Risks:</div>
                      <p className="decision-text" style={{ color: 'var(--accent-amber)' }}>
                        {resultData.pipeline.potential_risks.join(' ')}
                      </p>
                    </>
                  )}
                </div>
              </div>

              {/* 4. Histograms Comparison */}
              <HistogramComparison
                originalHist={resultData.histograms.original}
                enhancedHist={resultData.histograms.enhanced}
              />

              {/* 5. Quantitative Evaluation (MSE, PSNR, SSIM) */}
              <div className="section-card">
                <div className="section-card-header">
                  <div>
                    <h3 className="section-card-title">Quantitative Quality & Fidelity Metrics</h3>
                    <p className="section-card-sub" style={{ marginBottom: 0 }}>
                      Distinguishing input transformation distance from true ground-truth reconstruction.
                    </p>
                  </div>
                </div>

                {/* Relative to Input Metrics */}
                <div className="metric-section-group">
                  <div className="metric-group-title">
                    <span>A. Transformation Distance (Enhanced vs. Input)</span>
                    <span className="badge-academic">Relative Metric</span>
                  </div>
                  <p className="metric-group-note">
                    {resultData.metrics.relative_to_input.interpretation}
                  </p>

                  <div className="metrics-grid">
                    <div className="metric-box">
                      <span className="metric-label">Relative MSE</span>
                      <span className="metric-value">{resultData.metrics.relative_to_input.mse}</span>
                      <span className="metric-description">Squared pixel difference from raw input</span>
                    </div>
                    <div className="metric-box">
                      <span className="metric-label">Relative PSNR</span>
                      <span className="metric-value">
                        {resultData.metrics.relative_to_input.psnr}
                        <span className="metric-unit"> dB</span>
                      </span>
                      <span className="metric-description">Ratio relative to un-enhanced input</span>
                    </div>
                    <div className="metric-box">
                      <span className="metric-label">Relative SSIM</span>
                      <span className="metric-value">{resultData.metrics.relative_to_input.ssim}</span>
                      <span className="metric-description">Structural similarity to input (0 to 1)</span>
                    </div>
                  </div>
                </div>

                {/* Ground Truth Reference Metrics (When Available) */}
                {resultData.metrics.ground_truth_comparison.has_ground_truth ? (
                  <div className="metric-section-group" style={{ marginTop: '1.5rem', borderTop: '1px solid var(--border-medium)', paddingTop: '1.25rem' }}>
                    <div className="metric-group-title">
                      <span>B. Reconstruction Fidelity (vs. Clean Ground-Truth Reference)</span>
                      <span className="badge-academic" style={{ borderColor: 'var(--accent-emerald)', color: 'var(--accent-emerald)' }}>
                        Ground Truth Verified
                      </span>
                    </div>
                    <p className="metric-group-note" style={{ color: 'var(--accent-cyan)' }}>
                      Reference: <code>{resultData.metrics.ground_truth_comparison.reference_name}</code> — {resultData.metrics.ground_truth_comparison.interpretation}
                    </p>

                    <div className="metrics-grid">
                      <div className="metric-box">
                        <span className="metric-label">Input vs. Ground Truth PSNR</span>
                        <span className="metric-value">
                          {resultData.metrics.ground_truth_comparison.input_vs_ground_truth.psnr}
                          <span className="metric-unit"> dB</span>
                        </span>
                        <span className="metric-description">Degraded baseline PSNR</span>
                      </div>

                      <div className="metric-box">
                        <span className="metric-label">Enhanced vs. Ground Truth PSNR</span>
                        <span className="metric-value">
                          {resultData.metrics.ground_truth_comparison.enhanced_vs_ground_truth.psnr}
                          <span className="metric-unit"> dB</span>
                        </span>
                        <span className="metric-description">Restored signal PSNR</span>
                      </div>

                      <div className="metric-box" style={{ borderColor: resultData.metrics.ground_truth_comparison.psnr_gain_db >= 0 ? 'rgba(52, 211, 153, 0.4)' : 'rgba(244, 63, 94, 0.4)' }}>
                        <span className="metric-label">True PSNR Gain</span>
                        <span className="metric-value" style={{ color: resultData.metrics.ground_truth_comparison.psnr_gain_db >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}>
                          {resultData.metrics.ground_truth_comparison.psnr_gain_db > 0 ? '+' : ''}
                          {resultData.metrics.ground_truth_comparison.psnr_gain_db}
                          <span className="metric-unit"> dB</span>
                        </span>
                        <span className="metric-description">
                          {resultData.metrics.ground_truth_comparison.psnr_gain_db >= 0 ? 'Net Noise & Error Reduction' : 'Tone Remap Distance'}
                        </span>
                      </div>

                      <div className="metric-box" style={{ borderColor: resultData.metrics.ground_truth_comparison.ssim_gain >= 0 ? 'rgba(52, 211, 153, 0.4)' : 'rgba(244, 63, 94, 0.4)' }}>
                        <span className="metric-label">True SSIM Gain</span>
                        <span className="metric-value" style={{ color: resultData.metrics.ground_truth_comparison.ssim_gain >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}>
                          {resultData.metrics.ground_truth_comparison.ssim_gain > 0 ? '+' : ''}
                          {resultData.metrics.ground_truth_comparison.ssim_gain}
                        </span>
                        <span className="metric-description">
                          Baseline: {resultData.metrics.ground_truth_comparison.input_vs_ground_truth.ssim} &rarr; Enhanced: {resultData.metrics.ground_truth_comparison.enhanced_vs_ground_truth.ssim}
                        </span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="metric-section-group" style={{ marginTop: '1.25rem', borderTop: '1px solid var(--border-subtle)', paddingTop: '1rem' }}>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      ℹ️ Ground-truth reference metrics are active when benchmarking against controlled test samples (`01` through `05`), which have a known pristine reference (`clean_reference.png`).
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
