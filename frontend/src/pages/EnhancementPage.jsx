import React, { useState, useEffect } from 'react';
import Breadcrumbs from '../components/Breadcrumbs';
import PageHeader from '../components/PageHeader';
import ImageUploadSection from '../components/ImageUploadSection';
import SampleSelector from '../components/SampleSelector';
import HistogramComparison from '../components/HistogramComparison';
import { analyzeEnhancementImage, processEnhancement } from '../services/api';

const ENHANCEMENT_PRESETS = [
  {
    filename: '01_low_contrast.png',
    label: 'Low Contrast Scene',
    category: 'Contrast Deficient',
    description: 'Narrow dynamic range restricted to [95, 145]. Tests automatic contrast redistribution.'
  },
  {
    filename: '02_dark_underexposed.png',
    label: 'Underexposed Image',
    category: 'Low Brightness',
    description: 'Low mean pixel intensity with severe shadow clipping. Tests automatic gamma expansion.'
  },
  {
    filename: '03_bright_overexposed.png',
    label: 'Overexposed Image',
    category: 'High Brightness',
    description: 'Saturated highlights. Tests compressive gamma correction and highlight recovery.'
  },
  {
    filename: '04_gaussian_noise.png',
    label: 'Gaussian Noise Image',
    category: 'Sensor Noise',
    description: 'Additive sensor noise (sigma=28). Tests linear weighted smoothing.'
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
    description: 'Non-uniform illumination gradient across document text. Tests local contrast equalization.'
  },
  {
    filename: '07_blurred_out_of_focus.png',
    label: 'Blurred / Out-of-Focus',
    category: 'Spatial Blur',
    description: 'Low edge sharpness (Laplacian variance < 400). Tests unsharp masking & edge steepening.'
  }
];

export default function EnhancementPage() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [activeSampleFilename, setActiveSampleFilename] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isEnhancing, setIsEnhancing] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  // Operational Mode: 'auto' (Automatic Enhancement) | 'experiment' (Parameter Experimentation)
  const [mode, setMode] = useState('auto');

  // Image Analysis State (from /enhancement/analyze or process)
  const [analysisData, setAnalysisData] = useState(null);

  // Enhancement Results (from /enhancement/process)
  const [resultData, setResultData] = useState(null);

  // Relevant Parameter Experimentation Controls
  const [gamma, setGamma] = useState(1.0);
  const [equalizationMode, setEqualizationMode] = useState('clahe');
  const [claheClipLimit, setClaheClipLimit] = useState(2.5);
  const [spatialFilter, setSpatialFilter] = useState('none');
  const [kernelSize, setKernelSize] = useState(3);
  const [sharpenStrength, setSharpenStrength] = useState(0.0);

  // Helper to get image file object
  const getImageFile = async (imgData) => {
    if (!imgData) return null;
    if (imgData.file) return imgData.file;
    if (imgData.previewUrl) {
      const resp = await fetch(imgData.previewUrl);
      const blob = await resp.blob();
      return new File([blob], imgData.meta?.name || 'sample.png', {
        type: blob.type || 'image/png'
      });
    }
    return null;
  };

  // Run image analysis as soon as an image is selected/uploaded
  const triggerAnalysis = async (imgData) => {
    if (!imgData) return;
    setIsAnalyzing(true);
    setErrorMessage(null);
    try {
      const fileToUpload = await getImageFile(imgData);
      if (!fileToUpload) return;

      const formData = new FormData();
      formData.append('file', fileToUpload);

      const res = await analyzeEnhancementImage(formData);
      setAnalysisData(res.characteristics);
    } catch (err) {
      // Non-critical: if standalone analyze fails, it will still run on enhance
      console.warn('Initial analysis warning:', err.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleImageSelected = (imageData) => {
    setSelectedImage(imageData);
    setActiveSampleFilename(null);
    setErrorMessage(null);
    setResultData(null);
    setAnalysisData(null);
    triggerAnalysis(imageData);
  };

  const handleSelectSample = (sampleData) => {
    setSelectedImage(sampleData);
    setActiveSampleFilename(sampleData.sampleFilename);
    setErrorMessage(null);
    setResultData(null);
    setAnalysisData(null);
    triggerAnalysis(sampleData);
  };

  const handleClearImage = () => {
    setSelectedImage(null);
    setActiveSampleFilename(null);
    setErrorMessage(null);
    setResultData(null);
    setAnalysisData(null);
  };

  // Execute Automatic or Parameterized Enhancement
  const handleExecuteEnhancement = async (overrideMode = null) => {
    if (!selectedImage) return;
    const activeMode = overrideMode || mode;
    setIsEnhancing(true);
    setErrorMessage(null);

    try {
      const fileToUpload = await getImageFile(selectedImage);
      if (!fileToUpload) {
        throw new Error('Could not access image data for processing.');
      }

      const formData = new FormData();
      formData.append('file', fileToUpload);
      formData.append('mode', activeMode === 'auto' ? 'auto' : 'manual');

      if (activeMode !== 'auto') {
        formData.append('gamma', gamma.toString());
        formData.append('equalization_mode', equalizationMode);
        formData.append('clahe_clip_limit', claheClipLimit.toString());
        formData.append('spatial_filter', spatialFilter);
        formData.append('kernel_size', kernelSize.toString());
        formData.append('sharpen_strength', sharpenStrength.toString());
      }

      const data = await processEnhancement(formData);
      setResultData(data);
      setAnalysisData(data.input_characteristics);

      // If in auto mode, sync experiment sliders with auto-decided parameters
      if (activeMode === 'auto' && data.pipeline?.parameters) {
        const p = data.pipeline.parameters;
        if (p.gamma !== undefined) setGamma(p.gamma);
        if (p.equalization_mode !== undefined) setEqualizationMode(p.equalization_mode);
        if (p.clahe_clip_limit !== undefined) setClaheClipLimit(p.clahe_clip_limit);
        if (p.spatial_filter !== undefined) setSpatialFilter(p.spatial_filter);
        if (p.kernel_size !== undefined) setKernelSize(p.kernel_size);
        if (p.sharpen_strength !== undefined) setSharpenStrength(p.sharpen_strength);
      }
    } catch (err) {
      setErrorMessage(err.message || 'Image enhancement processing failed.');
    } finally {
      setIsEnhancing(false);
    }
  };

  const handleResetParameters = () => {
    if (resultData?.pipeline?.parameters) {
      const p = resultData.pipeline.parameters;
      setGamma(p.gamma ?? 1.0);
      setEqualizationMode(p.equalization_mode ?? 'clahe');
      setClaheClipLimit(p.clahe_clip_limit ?? 2.0);
      setSpatialFilter(p.spatial_filter ?? 'none');
      setKernelSize(p.kernel_size ?? 3);
      setSharpenStrength(p.sharpen_strength ?? 0.0);
    } else {
      setGamma(1.0);
      setEqualizationMode('clahe');
      setClaheClipLimit(2.0);
      setSpatialFilter('none');
      setKernelSize(3);
      setSharpenStrength(0.0);
    }
  };

  return (
    <div className="module-page-container">
      <Breadcrumbs items={[{ label: 'Automatic Image Enhancement' }]} />

      <PageHeader
        moduleNumber="1"
        title="Automatic Image Enhancement System"
        description="Analyzes input image characteristics (brightness, contrast, noise, and sharpness) and automatically selects and tunes an appropriate intensity transformation and spatial filter. Compare results with before/after viewports, conventional histograms, and quality metrics."
        tags={['Automatic Enhancement', 'Image Analysis', 'Tone Correction', 'Noise Reduction', 'Spatial Sharpening']}
        accentColor="blue"
      />

      <div className="module-layout-grid">
        {/* Left Column: Input Image, Enhancement Actions & Parameter Experimentation */}
        <div className="layout-sidebar">
          {/* 1. Input Image Section */}
          <ImageUploadSection
            selectedFile={selectedImage?.file}
            imagePreview={selectedImage?.previewUrl}
            imageMeta={selectedImage?.meta}
            onImageSelected={handleImageSelected}
            onClearImage={handleClearImage}
            label="1. Input Image"
            hint="PNG, JPG, JPEG, or BMP"
            disabled={isEnhancing || isAnalyzing}
          />

          <SampleSelector
            moduleName="enhancement"
            presets={ENHANCEMENT_PRESETS}
            onSelectSample={handleSelectSample}
            activeSampleFilename={activeSampleFilename}
          />

          {/* 2. Enhancement Control Card */}
          <div className="section-card controls-section">
            <div className="section-card-header">
              <h3 className="section-card-title">2. Enhancement Action</h3>
              <div className="mode-toggle-group">
                <button
                  type="button"
                  className={`btn-mode ${mode === 'auto' ? 'active' : ''}`}
                  onClick={() => setMode('auto')}
                  disabled={isEnhancing}
                >
                  Automatic Mode
                </button>
                <button
                  type="button"
                  className={`btn-mode ${mode === 'experiment' ? 'active' : ''}`}
                  onClick={() => setMode('experiment')}
                  disabled={isEnhancing}
                >
                  Experiment
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
                  The system evaluates image brightness, dynamic range, noise floor, and edge sharpness to select an optimal intensity transformation and spatial filter.
                </p>
                <div className="auto-rules-summary">
                  <div>• Dark images &rarr; Power-law gamma expansion (γ &lt; 1.0)</div>
                  <div>• Overexposed &rarr; Compressive gamma (γ &gt; 1.0)</div>
                  <div>• Low contrast &rarr; Local adaptive contrast enhancement</div>
                  <div>• Impulse noise &rarr; Order-statistic median filter</div>
                  <div>• Sensor noise &rarr; Linear weighted averaging smoothing</div>
                  <div>• Blurred edges &rarr; High-boost unsharp masking</div>
                  <div>• Well-balanced &rarr; Conservative pass without alteration</div>
                </div>

                <button
                  type="button"
                  className="btn btn-primary btn-execute"
                  style={{ marginTop: '0.85rem', width: '100%' }}
                  disabled={!selectedImage || isEnhancing}
                  onClick={() => handleExecuteEnhancement('auto')}
                >
                  {isEnhancing ? (
                    <>
                      <span className="spinner-sm"></span>
                      Enhancing Image...
                    </>
                  ) : (
                    <>
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <polygon points="5 3 19 12 5 21 5 3" />
                      </svg>
                      Enhance Image (Automatic)
                    </>
                  )}
                </button>
              </div>
            ) : (
              /* Compact Parameter Experimentation Area */
              <div className="controls-form">
                <div className="experiment-header-note">
                  Fine-tune relevant parameters to observe their direct impact on intensity distribution and visual fidelity.
                </div>

                {/* Gamma Value */}
                <div className="control-group">
                  <div className="control-label-row">
                    <label className="control-label">Intensity Gamma (γ)</label>
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
                    disabled={isEnhancing}
                  />
                  <div className="range-bounds">
                    <span>0.2 (Brighten)</span>
                    <span>1.0 (Identity)</span>
                    <span>3.0 (Darken)</span>
                  </div>
                </div>

                {/* Contrast Equalization */}
                <div className="control-group">
                  <label className="control-label">Contrast Equalization</label>
                  <select
                    className="control-select"
                    value={equalizationMode}
                    onChange={(e) => setEqualizationMode(e.target.value)}
                    disabled={isEnhancing}
                  >
                    <option value="none">None (Preserve Original Dynamic Range)</option>
                    <option value="clahe">Adaptive Local Equalization (CLAHE)</option>
                    <option value="global">Global Histogram Equalization</option>
                  </select>
                </div>

                {equalizationMode === 'clahe' && (
                  <div className="control-group">
                    <div className="control-label-row">
                      <label className="control-label">CLAHE Clip Limit</label>
                      <span className="control-value-badge">{claheClipLimit.toFixed(1)}</span>
                    </div>
                    <input
                      type="range"
                      className="control-range"
                      min="1.0"
                      max="6.0"
                      step="0.5"
                      value={claheClipLimit}
                      onChange={(e) => setClaheClipLimit(parseFloat(e.target.value))}
                      disabled={isEnhancing}
                    />
                  </div>
                )}

                {/* Spatial Smoothing Filter */}
                <div className="control-group">
                  <label className="control-label">Spatial Smoothing Filter</label>
                  <select
                    className="control-select"
                    value={spatialFilter}
                    onChange={(e) => setSpatialFilter(e.target.value)}
                    disabled={isEnhancing}
                  >
                    <option value="none">None (No Smoothing)</option>
                    <option value="median">Median Filter (Impulse Noise Removal)</option>
                    <option value="weighted_average">Weighted Averaging (Noise Smoothing)</option>
                  </select>
                </div>

                {spatialFilter !== 'none' && (
                  <div className="control-group">
                    <label className="control-label">Filter Kernel Size</label>
                    <select
                      className="control-select"
                      value={kernelSize}
                      onChange={(e) => setKernelSize(parseInt(e.target.value, 10))}
                      disabled={isEnhancing}
                    >
                      <option value={3}>3 × 3 Kernel (Fine)</option>
                      <option value={5}>5 × 5 Kernel (Moderate)</option>
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
                    disabled={isEnhancing}
                  />
                  <div className="range-bounds">
                    <span>0.0 (None)</span>
                    <span>1.4 (Standard)</span>
                    <span>3.0 (Aggressive)</span>
                  </div>
                </div>

                <div className="controls-actions" style={{ marginTop: '1.25rem' }}>
                  <button
                    type="button"
                    className="btn btn-primary btn-execute"
                    disabled={!selectedImage || isEnhancing}
                    onClick={() => handleExecuteEnhancement('experiment')}
                  >
                    {isEnhancing ? (
                      <>
                        <span className="spinner-sm"></span>
                        Applying...
                      </>
                    ) : (
                      'Apply Parameters'
                    )}
                  </button>

                  <button
                    type="button"
                    className="btn btn-outline"
                    onClick={handleResetParameters}
                    disabled={isEnhancing}
                  >
                    Reset
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Image Analysis, Visual Comparison, Decision Log, Histograms & Metrics */}
        <div className="layout-main">
          {errorMessage && (
            <div className="error-banner" role="alert">
              <div className="error-icon">!</div>
              <div className="error-content">
                <div className="error-title">Processing Error</div>
                <div className="error-detail">{errorMessage}</div>
              </div>
              <button type="button" className="error-close-btn" onClick={() => setErrorMessage(null)}>
                &times;
              </button>
            </div>
          )}

          {!selectedImage && (
            <div className="section-card empty-state-card">
              <h4 className="empty-state-title">No Image Selected</h4>
              <p className="empty-state-text">
                Select one of the 7 benchmark samples from the left panel or upload an image to analyze its characteristics and run automatic enhancement.
              </p>
            </div>
          )}

          {/* 1. Concise Image Characteristic Analysis Card */}
          {selectedImage && (
            <div className="section-card">
              <div className="section-card-header">
                <div>
                  <h3 className="section-card-title">Image Characteristic Analysis</h3>
                  <span className="section-card-sub" style={{ fontSize: '0.725rem' }}>
                    Quantitative diagnostics evaluating exposure, contrast, noise, and blur
                  </span>
                </div>
                {isAnalyzing && (
                  <span className="badge-academic">Analyzing...</span>
                )}
                {analysisData && !isAnalyzing && (
                  <span className="badge-academic">
                    {analysisData.is_color ? 'Color Image' : 'Grayscale'} ({analysisData.dimensions?.width} × {analysisData.dimensions?.height})
                  </span>
                )}
              </div>

              {analysisData ? (
                <div className="analysis-summary-grid">
                  {/* Brightness Diagnostic */}
                  <div className="analysis-metric-card">
                    <div className="analysis-metric-header">
                      <span className="analysis-metric-title">Brightness (Mean)</span>
                      <span className={`analysis-tag ${analysisData.diagnostics?.is_dark ? 'tag-warn' : analysisData.diagnostics?.is_bright ? 'tag-warn' : 'tag-good'}`}>
                        {analysisData.diagnostics?.is_dark ? 'Underexposed' : analysisData.diagnostics?.is_bright ? 'Overexposed' : 'Balanced'}
                      </span>
                    </div>
                    <div className="analysis-metric-num">{analysisData.mean_intensity} <small>/ 255</small></div>
                    <div className="analysis-metric-desc">
                      Shadow clipping: {analysisData.shadow_clipping_pct}% | Highlight clipping: {analysisData.highlight_clipping_pct}%
                    </div>
                  </div>

                  {/* Contrast Diagnostic */}
                  <div className="analysis-metric-card">
                    <div className="analysis-metric-header">
                      <span className="analysis-metric-title">Contrast (Std Dev)</span>
                      <span className={`analysis-tag ${analysisData.diagnostics?.is_low_contrast ? 'tag-warn' : 'tag-good'}`}>
                        {analysisData.diagnostics?.is_low_contrast ? 'Low Contrast' : 'Healthy Range'}
                      </span>
                    </div>
                    <div className="analysis-metric-num">σ = {analysisData.std_intensity}</div>
                    <div className="analysis-metric-desc">
                      Dynamic range: [{analysisData.min_intensity}, {analysisData.max_intensity}] (Span: {analysisData.dynamic_range})
                    </div>
                  </div>

                  {/* Noise Diagnostic */}
                  <div className="analysis-metric-card">
                    <div className="analysis-metric-header">
                      <span className="analysis-metric-title">Noise Level</span>
                      <span className={`analysis-tag ${analysisData.diagnostics?.is_noisy ? 'tag-warn' : 'tag-good'}`}>
                        {analysisData.has_salt_pepper_noise ? 'Impulse Noise' : analysisData.diagnostics?.is_noisy ? 'Sensor Noise' : 'Clean / Low Noise'}
                      </span>
                    </div>
                    <div className="analysis-metric-num">σ<sub>n</sub> = {analysisData.noise_sigma}</div>
                    <div className="analysis-metric-desc">
                      {analysisData.has_salt_pepper_noise ? 'Salt-and-pepper impulse spikes detected' : `Estimated additive Gaussian sigma: ${analysisData.noise_sigma}`}
                    </div>
                  </div>

                  {/* Sharpness Diagnostic */}
                  <div className="analysis-metric-card">
                    <div className="analysis-metric-header">
                      <span className="analysis-metric-title">Sharpness Acutance</span>
                      <span className={`analysis-tag ${analysisData.diagnostics?.is_severely_blurry ? 'tag-warn' : analysisData.diagnostics?.is_blurry ? 'tag-warn' : 'tag-good'}`}>
                        {analysisData.diagnostics?.is_severely_blurry ? 'Severe Blur' : analysisData.diagnostics?.is_blurry ? 'Mild Blur' : 'Sharp Edges'}
                      </span>
                    </div>
                    <div className="analysis-metric-num">{analysisData.sharpness_score}</div>
                    <div className="analysis-metric-desc">
                      Normalized sharpness index: {analysisData.norm_sharpness}
                    </div>
                  </div>
                </div>
              ) : (
                <div style={{ padding: '1rem', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                  Evaluating image characteristics...
                </div>
              )}

              {/* Detected Primary Defects */}
              {analysisData?.diagnostics?.primary_defects && (
                <div className="defects-banner">
                  <span className="defects-label">Identified Characteristics:</span>
                  <div className="defects-pills">
                    {analysisData.diagnostics.primary_defects.map((defect, i) => (
                      <span key={i} className="defect-pill">{defect}</span>
                    ))}
                  </div>
                </div>
              )}

              {/* Action button right beneath analysis if results not yet generated */}
              {!resultData && !isEnhancing && (
                <div style={{ marginTop: '1.25rem', display: 'flex', justifyContent: 'center' }}>
                  <button
                    type="button"
                    className="btn btn-primary"
                    style={{ padding: '0.65rem 2rem', fontSize: '0.9rem' }}
                    onClick={() => handleExecuteEnhancement('auto')}
                  >
                    Enhance Image Automatically
                  </button>
                </div>
              )}
            </div>
          )}

          {isEnhancing && (
            <div className="section-card loading-state-card">
              <div className="academic-spinner"></div>
              <div className="loading-title">Executing Enhancement Pipeline...</div>
              <div className="loading-sub">
                Applying selected transformations, redistributing contrast, and computing 256-bin histograms.
              </div>
            </div>
          )}

          {/* Results: Visual Comparison, Decision Log, Histograms & Metrics */}
          {resultData && (
            <div className="enhancement-results-container">
              {/* 2. Visual Comparison Viewports */}
              <div className="section-card">
                <div className="section-card-header">
                  <h3 className="section-card-title">Visual Comparison Viewports</h3>
                  <span className="badge-academic">
                    Mode: {resultData.mode.toUpperCase()}
                  </span>
                </div>

                <div className="comparison-viewports-grid">
                  <div className="viewport-panel">
                    <div className="viewport-label-row">
                      <span className="viewport-label">Original Photograph</span>
                      <span className="viewport-tag tag-input">Input</span>
                    </div>
                    <div className="viewport-image-box">
                      <img
                        src={resultData.images.original_base64}
                        alt="Original Input"
                        className="viewport-img"
                      />
                    </div>
                    <div className="viewport-stats">
                      <span>Mean: {resultData.input_characteristics.mean_intensity}</span>
                      <span>Std: {resultData.input_characteristics.std_intensity}</span>
                      <span>Range: {resultData.input_characteristics.dynamic_range}</span>
                    </div>
                  </div>

                  <div className="viewport-panel">
                    <div className="viewport-label-row">
                      <span className="viewport-label">Enhanced Result</span>
                      <span className="viewport-tag tag-enhanced">Output</span>
                    </div>
                    <div className="viewport-image-box">
                      <img
                        src={resultData.images.enhanced_base64}
                        alt="Enhanced Result"
                        className="viewport-img"
                      />
                    </div>
                    <div className="viewport-stats">
                      <span>Mean: {resultData.enhanced_characteristics.mean_intensity}</span>
                      <span>Std: {resultData.enhanced_characteristics.std_intensity}</span>
                      <span>Range: {resultData.enhanced_characteristics.dynamic_range}</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* 3. Selected Operations & Decision Rationale */}
              <div className="section-card">
                <div className="section-card-header">
                  <h3 className="section-card-title">Enhancement Pipeline & Decision Rationale</h3>
                  <span className="badge-academic" style={{ textTransform: 'uppercase' }}>
                    {resultData.pipeline.selected_operations?.length || 0} Operation(s)
                  </span>
                </div>

                <div className="decision-content-box">
                  <div className="decision-subtitle">Selected Operations:</div>
                  <ul className="decision-list">
                    {resultData.pipeline.selected_operations?.map((op, idx) => (
                      <li key={idx} className="decision-item">
                        <span className="decision-step-badge">{idx + 1}</span>
                        <span>{op}</span>
                      </li>
                    ))}
                  </ul>

                  {resultData.pipeline.justifications && resultData.pipeline.justifications.length > 0 && (
                    <>
                      <div className="decision-subtitle" style={{ marginTop: '0.85rem' }}>Rationale for Selection:</div>
                      <p className="decision-text">
                        {resultData.pipeline.justifications.join(' ')}
                      </p>
                    </>
                  )}

                  {resultData.pipeline.expected_improvements && resultData.pipeline.expected_improvements.length > 0 && (
                    <>
                      <div className="decision-subtitle" style={{ marginTop: '0.85rem' }}>Expected Improvements:</div>
                      <p className="decision-text" style={{ color: 'var(--accent-green)' }}>
                        {resultData.pipeline.expected_improvements.join(' ')}
                      </p>
                    </>
                  )}
                </div>
              </div>

              {/* 4. Conventional Histograms Comparison */}
              <HistogramComparison
                originalHist={resultData.histograms.original}
                enhancedHist={resultData.histograms.enhanced}
              />

              {/* 5. Quantitative Quality Metrics */}
              <div className="section-card">
                <div className="section-card-header">
                  <div>
                    <h3 className="section-card-title">Quantitative Quality & Fidelity Metrics</h3>
                    <p className="section-card-sub" style={{ marginBottom: 0 }}>
                      Evaluates transformation distance and ground-truth recovery metrics.
                    </p>
                  </div>
                </div>

                <div className="metric-section-group">
                  <div className="metric-group-title">
                    <span>Transformation Distance (Enhanced vs. Input)</span>
                    <span className="badge-academic">Relative Metric</span>
                  </div>
                  <div className="metrics-card-grid">
                    <div className="metric-box">
                      <div className="metric-label">Mean Squared Error (MSE)</div>
                      <div className="metric-value">{resultData.metrics.relative_to_input.mse}</div>
                    </div>
                    <div className="metric-box">
                      <div className="metric-label">Peak SNR (PSNR)</div>
                      <div className="metric-value">{resultData.metrics.relative_to_input.psnr} dB</div>
                    </div>
                    <div className="metric-box">
                      <div className="metric-label">Structural Similarity (SSIM)</div>
                      <div className="metric-value">{resultData.metrics.relative_to_input.ssim}</div>
                    </div>
                  </div>
                </div>

                {resultData.metrics.ground_truth_reference && (
                  <div className="metric-section-group" style={{ marginTop: '1.25rem' }}>
                    <div className="metric-group-title">
                      <span>Ground-Truth Reconstruction Fidelity ({resultData.metrics.ground_truth_reference.reference_image})</span>
                      <span className="badge-academic badge-gt">True Ground Truth</span>
                    </div>
                    <div className="metrics-card-grid">
                      <div className="metric-box">
                        <div className="metric-label">Ground-Truth MSE</div>
                        <div className="metric-value">{resultData.metrics.ground_truth_reference.mse}</div>
                      </div>
                      <div className="metric-box">
                        <div className="metric-label">Ground-Truth PSNR</div>
                        <div className="metric-value">{resultData.metrics.ground_truth_reference.psnr} dB</div>
                      </div>
                      <div className="metric-box">
                        <div className="metric-label">Ground-Truth SSIM</div>
                        <div className="metric-value">{resultData.metrics.ground_truth_reference.ssim}</div>
                      </div>
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
