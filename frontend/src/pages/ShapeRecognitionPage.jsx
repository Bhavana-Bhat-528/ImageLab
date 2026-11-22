import React, { useState } from 'react';
import Breadcrumbs from '../components/Breadcrumbs';
import PageHeader from '../components/PageHeader';
import ImageUploadSection from '../components/ImageUploadSection';
import SampleSelector from '../components/SampleSelector';
import ProcessingControlsPlaceholder from '../components/ProcessingControlsPlaceholder';
import ResultsAreaPlaceholder from '../components/ResultsAreaPlaceholder';

const SHAPE_PRESETS = [
  {
    filename: 'circle_01.png',
    label: 'Category 1: Circle',
    category: 'Zero-Eccentricity Contour',
    description: 'Fundamental circular harmonic; tests rotational invariance of Fourier magnitude spectrum.'
  },
  {
    filename: 'square_01.png',
    label: 'Category 2: Square',
    category: '4-Fold Rotational Symmetry',
    description: 'Regular 4-sided polygon with prominent 4th harmonic frequency component.'
  },
  {
    filename: 'triangle_01.png',
    label: 'Category 3: Triangle',
    category: '3-Fold Vertex Distribution',
    description: 'Equilateral upright triangle; tests harmonic distribution across acute vertices.'
  },
  {
    filename: 'rectangle_01.png',
    label: 'Category 4: Rectangle',
    category: '2-Fold Aspect Ratio Asymmetry',
    description: 'Horizontal 2:1 rectangle; cleanly differentiated from square via aspect ratio harmonics.'
  },
  {
    filename: 'star_01.png',
    label: 'Category 5: Star',
    category: 'Re-Entrant Multi-Vertex Polygon',
    description: 'Classic 5-point star with alternating outer and inner vertex frequencies.'
  }
];


export default function ShapeRecognitionPage() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [activeSampleFilename, setActiveSampleFilename] = useState(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  // Parameter controls state
  const [numDescriptors, setNumDescriptors] = useState(16);
  const [distanceMetric, setDistanceMetric] = useState('euclidean');
  const [normalizeScale, setNormalizeScale] = useState(true);
  const [normalizeRotation, setNormalizeRotation] = useState(true);

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
    setNumDescriptors(16);
    setDistanceMetric('euclidean');
    setNormalizeScale(true);
    setNormalizeRotation(true);
    setErrorMessage(null);
  };

  return (
    <div className="module-page-container">
      <Breadcrumbs items={[{ label: 'Fourier Descriptor-Based Shape Recognition' }]} />

      <PageHeader
        moduleNumber="4"
        title="Fourier Descriptor-Based Shape Recognition"
        description="Extracts planar object boundaries, computes invariant Fourier descriptors (invariant to scale, rotation, and translation), and classifies unknown shapes against reference benchmarks across 5 distinct object categories."
        tags={['Boundary Representation', 'Fourier Descriptors', 'Invariant Shape Matching', 'k-NN Classification']}
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
            label="Upload Object Image"
            hint="Silhouetted object image across 5 categories (aeroplane, cup, tool, star, car)"
            disabled={isExecuting}
          />

          <SampleSelector
            moduleName="shape-recognition"
            presets={SHAPE_PRESETS}
            onSelectSample={handleSelectSample}
            activeSampleFilename={activeSampleFilename}
          />

          <ProcessingControlsPlaceholder
            title="3. Descriptor & Classifier Settings"
            hasImage={Boolean(selectedImage)}
            isExecuting={isExecuting}
            onExecute={handleExecute}
            onReset={handleReset}
            executeLabel="Extract Descriptors & Match"
          >
            <div className="control-group">
              <div className="control-label-row">
                <label className="control-label">Fourier Descriptors Count (K)</label>
                <span className="control-value-badge">{numDescriptors} harmonics</span>
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
                <span>4 (Coarse)</span>
                <span>16 (Balanced)</span>
                <span>64 (Fine detail)</span>
              </div>
              <span className="control-help">
                Lower frequencies capture global silhouette; higher harmonics capture localized indentations.
              </span>
            </div>

            <div className="control-group">
              <label className="control-label">Similarity Metric</label>
              <select
                className="control-select"
                value={distanceMetric}
                onChange={(e) => setDistanceMetric(e.target.value)}
                disabled={isExecuting}
              >
                <option value="euclidean">Euclidean Distance L2</option>
                <option value="manhattan">Manhattan Distance L1</option>
                <option value="cosine">Cosine Spectral Similarity</option>
              </select>
            </div>

            <div className="control-group">
              <label className="control-label" style={{ marginBottom: '0.5rem' }}>
                Invariance Normalizations
              </label>
              <div className="checkbox-row">
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={normalizeScale}
                    onChange={(e) => setNormalizeScale(e.target.checked)}
                    disabled={isExecuting}
                  />
                  <span>Scale Invariance (Divide by |F(1)|)</span>
                </label>
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={normalizeRotation}
                    onChange={(e) => setNormalizeRotation(e.target.checked)}
                    disabled={isExecuting}
                  />
                  <span>Rotation Invariance (Magnitude Spectrum |F(k)|)</span>
                </label>
              </div>
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
            moduleName="Shape Recognition"
            resultViews={[
              { label: "Original Input Image", key: "original" },
              { label: "Extracted Boundary Contour", key: "contour" },
              { label: "Fourier Magnitude Spectrum", key: "spectrum" }
            ]}
            metrics={[
              { label: "Predicted Category", value: null, description: "Nearest neighbor shape category" },
              { label: "Similarity Score", value: null, unit: "%", description: "Descriptor distance match ratio" },
              { label: "Boundary Perimeter", value: null, unit: "pts", description: "Sampled contour coordinate length" },
              { label: "Descriptor Energy", value: null, unit: "%", description: "Reconstruction energy retention" }
            ]}
          />
        </div>
      </div>
    </div>
  );
}
