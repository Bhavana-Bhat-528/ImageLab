import React, { useState } from 'react';
import Breadcrumbs from '../components/Breadcrumbs';
import PageHeader from '../components/PageHeader';
import ImageUploadSection from '../components/ImageUploadSection';
import SampleSelector from '../components/SampleSelector';
import ProcessingControlsPlaceholder from '../components/ProcessingControlsPlaceholder';
import ResultsAreaPlaceholder from '../components/ResultsAreaPlaceholder';

const ENHANCEMENT_PRESETS = [
  {
    filename: '01_low_contrast.png',
    label: 'Low Contrast Scene',
    category: 'Contrast Deficient',
    description: 'Under-utilized dynamic range restricted to [95, 145]. Tests contrast stretching & CLAHE.'
  },
  {
    filename: '02_dark_underexposed.png',
    label: 'Underexposed Image',
    category: 'Low Brightness',
    description: 'Low mean pixel intensity with clipped shadows. Tests gamma expansion (γ < 1.0).'
  },
  {
    filename: '03_bright_overexposed.png',
    label: 'Overexposed Image',
    category: 'High Brightness',
    description: 'Saturated highlights. Tests compressive gamma and histogram equalization.'
  },
  {
    filename: '04_gaussian_noise.png',
    label: 'Gaussian Noise Image',
    category: 'Sensor Noise',
    description: 'Zero-mean additive Gaussian noise (σ=28). Tests spatial Gaussian smoothing.'
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
    category: 'Non-Uniform Field',
    description: 'Non-uniform background lighting gradient across document.'
  }
];


export default function EnhancementPage() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [activeSampleFilename, setActiveSampleFilename] = useState(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  // Parameter controls state (placeholders for algorithm tuning)
  const [gamma, setGamma] = useState(1.2);
  const [kernelSize, setKernelSize] = useState(3);
  const [sharpenStrength, setSharpenStrength] = useState(1.0);
  const [transformationType, setTransformationType] = useState('auto');

  const handleImageSelected = (imageData) => {
    setSelectedImage(imageData);
    setActiveSampleFilename(null);
    setErrorMessage(null);
  };

  const handleSelectSample = (sampleData) => {
    setSelectedImage(sampleData);
    setActiveSampleFilename(sampleData.sampleFilename);
    setErrorMessage(null);
  };

  const handleClearImage = () => {
    setSelectedImage(null);
    setActiveSampleFilename(null);
    setErrorMessage(null);
  };

  const handleExecute = () => {
    if (!selectedImage) return;
    setIsExecuting(true);
    setErrorMessage(null);
    // Placeholder execution flow - does not simulate algorithms in JS
    setTimeout(() => {
      setIsExecuting(false);
    }, 1200);
  };

  const handleReset = () => {
    setGamma(1.2);
    setKernelSize(3);
    setSharpenStrength(1.0);
    setTransformationType('auto');
    setErrorMessage(null);
  };

  return (
    <div className="module-page-container">
      <Breadcrumbs items={[{ label: 'Automatic Image Enhancement' }]} />

      <PageHeader
        moduleNumber="1"
        title="Automatic Image Enhancement"
        description="Analyzes input image characteristics (brightness, contrast, and histogram dispersion) and selects an appropriate intensity transformation (gamma, equalization, contrast stretch) and spatial filter. Compares output quality using MSE, PSNR, and SSIM."
        tags={['Intensity Transformation', 'Spatial Filtering', 'Histogram Analysis', 'PSNR / SSIM']}
        accentColor="blue"
      />

      <div className="module-layout-grid">
        {/* Left Column: Input and Configuration */}
        <div className="layout-sidebar">
          <ImageUploadSection
            selectedFile={selectedImage?.file}
            imagePreview={selectedImage?.previewUrl}
            imageMeta={selectedImage?.meta}
            onImageSelected={handleImageSelected}
            onClearImage={handleClearImage}
            label="Upload Test Image"
            hint="PNG, JPG, TIFF, or BMP test images (min 5 different test sets)"
            disabled={isExecuting}
          />

          <SampleSelector
            moduleName="enhancement"
            presets={ENHANCEMENT_PRESETS}
            onSelectSample={handleSelectSample}
            activeSampleFilename={activeSampleFilename}
          />

          <ProcessingControlsPlaceholder
            title="3. Enhancement Controls"
            hasImage={Boolean(selectedImage)}
            isExecuting={isExecuting}
            onExecute={handleExecute}
            onReset={handleReset}
            executeLabel="Analyze & Enhance"
          >
            <div className="control-group">
              <label className="control-label">
                Transformation Selection Mode
              </label>
              <select
                className="control-select"
                value={transformationType}
                onChange={(e) => setTransformationType(e.target.value)}
                disabled={isExecuting}
              >
                <option value="auto">Auto-Select (Characteristic Analysis)</option>
                <option value="gamma">Power-Law (Gamma Correction)</option>
                <option value="equalization">Global Histogram Equalization</option>
                <option value="clahe">Contrast Limited Adaptive Equalization (CLAHE)</option>
                <option value="linear_stretch">Contrast Stretching (Min-Max Normalization)</option>
              </select>
              <span className="control-help">
                System evaluates histogram variance & skewness to pick the best transformation.
              </span>
            </div>

            <div className="control-group">
              <div className="control-label-row">
                <label className="control-label">Gamma Exponent (γ)</label>
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

            <div className="control-group">
              <label className="control-label">Spatial Filter Kernel</label>
              <select
                className="control-select"
                value={kernelSize}
                onChange={(e) => setKernelSize(parseInt(e.target.value, 10))}
                disabled={isExecuting}
              >
                <option value={3}>3 × 3 Kernel</option>
                <option value={5}>5 × 5 Kernel</option>
                <option value={7}>7 × 7 Kernel</option>
              </select>
            </div>

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
                step="0.1"
                value={sharpenStrength}
                onChange={(e) => setSharpenStrength(parseFloat(e.target.value))}
                disabled={isExecuting}
              />
            </div>
          </ProcessingControlsPlaceholder>
        </div>

        {/* Right Column: Visualizations & Quantitative Results */}
        <div className="layout-main">
          <ResultsAreaPlaceholder
            hasImage={Boolean(selectedImage)}
            imagePreview={selectedImage?.previewUrl}
            isExecuting={isExecuting}
            errorMessage={errorMessage}
            onDismissError={() => setErrorMessage(null)}
            moduleName="Automatic Enhancement"
            resultViews={[
              { label: "Original Input", key: "original" },
              { label: "Enhanced Output", key: "enhanced" },
              { label: "Histogram Comparison", key: "histograms" }
            ]}
            metrics={[
              { label: "MSE", value: null, description: "Mean Squared Error" },
              { label: "PSNR", value: null, unit: "dB", description: "Peak Signal-to-Noise Ratio" },
              { label: "SSIM", value: null, description: "Structural Similarity Index" },
              { label: "Mean Luminance", value: null, unit: "/ 255", description: "Input vs Enhanced Luminance" }
            ]}
          />
        </div>
      </div>
    </div>
  );
}
