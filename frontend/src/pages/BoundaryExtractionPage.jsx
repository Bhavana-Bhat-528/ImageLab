import React, { useState } from 'react';
import Breadcrumbs from '../components/Breadcrumbs';
import PageHeader from '../components/PageHeader';
import ImageUploadSection from '../components/ImageUploadSection';
import SampleSelector from '../components/SampleSelector';
import ProcessingControlsPlaceholder from '../components/ProcessingControlsPlaceholder';
import ResultsAreaPlaceholder from '../components/ResultsAreaPlaceholder';

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

  // Parameter controls state
  const [seShape, setSeShape] = useState('rect');
  const [seSize, setSeSize] = useState(3);
  const [overlayColor, setOverlayColor] = useState('cyan');
  const [thresholdMode, setThresholdMode] = useState('otsu');

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
    setTimeout(() => {
      setIsExecuting(false);
    }, 1200);
  };

  const handleReset = () => {
    setSeShape('rect');
    setSeSize(3);
    setOverlayColor('cyan');
    setThresholdMode('otsu');
    setErrorMessage(null);
  };

  return (
    <div className="module-page-container">
      <Breadcrumbs items={[{ label: 'Morphological Boundary Extraction' }]} />

      <PageHeader
        moduleNumber="2"
        title="Morphological Boundary Extraction"
        description="Extracts and highlights outer animal boundaries using binary segmentation, mathematical morphological erosion (A ⊖ B), set subtraction β(A) = A − (A ⊖ B), and color overlay onto original photographs. Evaluates the impact of structuring element shapes and sizes."
        tags={['Mathematical Morphology', 'Binary Erosion', 'Boundary Set Subtraction', 'Structuring Elements']}
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

          <ProcessingControlsPlaceholder
            title="3. Morphological Controls"
            hasImage={Boolean(selectedImage)}
            isExecuting={isExecuting}
            onExecute={handleExecute}
            onReset={handleReset}
            executeLabel="Extract Boundary"
          >
            <div className="control-group">
              <label className="control-label">
                Structuring Element Shape
              </label>
              <select
                className="control-select"
                value={seShape}
                onChange={(e) => setSeShape(e.target.value)}
                disabled={isExecuting}
              >
                <option value="rect">Rectangular (cv2.MORPH_RECT)</option>
                <option value="cross">Cross (cv2.MORPH_CROSS)</option>
                <option value="ellipse">Elliptical / Disk (cv2.MORPH_ELLIPSE)</option>
              </select>
              <span className="control-help">
                Shape determines connectivity and edge smoothing orientation.
              </span>
            </div>

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
                <span>9 × 9</span>
                <span>15 × 15 (Thick)</span>
              </div>
            </div>

            <div className="control-group">
              <label className="control-label">Segmentation Thresholding</label>
              <select
                className="control-select"
                value={thresholdMode}
                onChange={(e) => setThresholdMode(e.target.value)}
                disabled={isExecuting}
              >
                <option value="otsu">Otsu Automatic Binarization</option>
                <option value="adaptive">Adaptive Gaussian Thresholding</option>
                <option value="manual">Manual Fixed Threshold</option>
              </select>
            </div>

            <div className="control-group">
              <label className="control-label">Boundary Overlay Color</label>
              <select
                className="control-select"
                value={overlayColor}
                onChange={(e) => setOverlayColor(e.target.value)}
                disabled={isExecuting}
              >
                <option value="cyan">Neon Cyan (#00e5ff)</option>
                <option value="yellow">High-Vis Yellow (#ffea00)</option>
                <option value="magenta">Vibrant Magenta (#ff007f)</option>
                <option value="green">Lime Green (#00e676)</option>
              </select>
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
            moduleName="Boundary Extraction"
            resultViews={[
              { label: "Original Photograph", key: "original" },
              { label: "Preprocessed Binary Mask", key: "binary_mask" },
              { label: "Extracted Boundary Overlay", key: "boundary_overlay" }
            ]}
            metrics={[
              { label: "Foreground Pixel Count", value: null, description: "Total segmented foreground pixels" },
              { label: "Extracted Boundary Pixels", value: null, description: "Pixels in β(A) boundary contour" },
              { label: "Boundary Thickness Ratio", value: null, unit: "%", description: "Ratio of boundary to object area" },
              { label: "Connected Components", value: null, description: "Identified boundary topological count" }
            ]}
          />
        </div>
      </div>
    </div>
  );
}
