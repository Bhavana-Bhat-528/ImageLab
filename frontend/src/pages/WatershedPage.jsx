import React, { useState } from 'react';
import Breadcrumbs from '../components/Breadcrumbs';
import PageHeader from '../components/PageHeader';
import ImageUploadSection from '../components/ImageUploadSection';
import SampleSelector from '../components/SampleSelector';
import ProcessingControlsPlaceholder from '../components/ProcessingControlsPlaceholder';
import ResultsAreaPlaceholder from '../components/ResultsAreaPlaceholder';

const WATERSHED_PRESETS = [
  {
    filename: '01_human_mitosis.png',
    label: 'Human Mitosis Cells',
    category: 'Dividing Biological Cells',
    description: 'Microscopic human cancer cells undergoing mitosis with touching nuclear boundaries.'
  },
  {
    filename: '02_cell_culture.png',
    label: 'Fluorescent Cell Culture',
    category: 'Confluent Cells',
    description: 'Adjacent fluorescent cell membranes requiring distance transform marker separation.'
  },
  {
    filename: '03_protein_transport_nuclei.png',
    label: 'Nuclear Translocation Assay',
    category: 'Clustered Nuclei',
    description: 'Compact spatial aggregation of cell nuclei with varying intensities.'
  },
  {
    filename: '04_overlapping_coins.png',
    label: 'Overlapping Particles Benchmark',
    category: 'Touching Convex Objects',
    description: 'Classic DIP benchmark for evaluating Euclidean distance transform watershed boundaries.'
  }
];


export default function WatershedPage() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [activeSampleFilename, setActiveSampleFilename] = useState(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  // Parameter controls state
  const [thresholdVal, setThresholdVal] = useState(128);
  const [minDistance, setMinDistance] = useState(15);
  const [distTransformMask, setDistTransformMask] = useState(5);
  const [compactness, setCompactness] = useState(0.0);

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
    setThresholdVal(128);
    setMinDistance(15);
    setDistTransformMask(5);
    setCompactness(0.0);
    setErrorMessage(null);
  };

  return (
    <div className="module-page-container">
      <Breadcrumbs items={[{ label: 'Watershed Segmentation' }]} />

      <PageHeader
        moduleNumber="3"
        title="Watershed Segmentation"
        description="Separates clustered and touching cells in microscopic imaging using Otsu binarization, Euclidean distance transformation, marker extraction, and marker-controlled watershed segmentation. Estimates individual cell counts and region geometry."
        tags={['Cell Segmentation', 'Distance Transform', 'Marker Generation', 'Watershed Transform']}
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

          <ProcessingControlsPlaceholder
            title="3. Watershed Parameters"
            hasImage={Boolean(selectedImage)}
            isExecuting={isExecuting}
            onExecute={handleExecute}
            onReset={handleReset}
            executeLabel="Run Watershed Segmentation"
          >
            <div className="control-group">
              <div className="control-label-row">
                <label className="control-label">Binarization Threshold</label>
                <span className="control-value-badge">{thresholdVal}</span>
              </div>
              <input
                type="range"
                className="control-range"
                min="0"
                max="255"
                value={thresholdVal}
                onChange={(e) => setThresholdVal(parseInt(e.target.value, 10))}
                disabled={isExecuting}
              />
              <div className="range-bounds">
                <span>0 (Dark Cells)</span>
                <span>128</span>
                <span>255 (Bright Cells)</span>
              </div>
            </div>

            <div className="control-group">
              <div className="control-label-row">
                <label className="control-label">Marker Peak Min Distance</label>
                <span className="control-value-badge">{minDistance} px</span>
              </div>
              <input
                type="range"
                className="control-range"
                min="5"
                max="50"
                value={minDistance}
                onChange={(e) => setMinDistance(parseInt(e.target.value, 10))}
                disabled={isExecuting}
              />
              <span className="control-help">
                Prevents over-segmentation by suppressing secondary local maxima.
              </span>
            </div>

            <div className="control-group">
              <label className="control-label">Distance Transform Mask</label>
              <select
                className="control-select"
                value={distTransformMask}
                onChange={(e) => setDistTransformMask(parseInt(e.target.value, 10))}
                disabled={isExecuting}
              >
                <option value={3}>3 × 3 Euclidean Metric (cv2.DIST_L2)</option>
                <option value={5}>5 × 5 Accurate Euclidean Metric</option>
              </select>
            </div>

            <div className="control-group">
              <div className="control-label-row">
                <label className="control-label">Watershed Compactness</label>
                <span className="control-value-badge">{compactness.toFixed(2)}</span>
              </div>
              <input
                type="range"
                className="control-range"
                min="0.0"
                max="1.0"
                step="0.05"
                value={compactness}
                onChange={(e) => setCompactness(parseFloat(e.target.value))}
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
            moduleName="Watershed Segmentation"
            resultViews={[
              { label: "Original Microscopic Image", key: "original" },
              { label: "Distance Transform & Markers", key: "distance_map" },
              { label: "Watershed Labeled Cells", key: "labeled_cells" }
            ]}
            metrics={[
              { label: "Estimated Cell Count", value: null, description: "Total segmented individual cell regions" },
              { label: "Mean Cell Area", value: null, unit: "px²", description: "Average segmented footprint" },
              { label: "Separated Touching Boundaries", value: null, description: "Resolved touching junction lines" },
              { label: "Segmentation Confidence", value: null, unit: "%", description: "Marker-to-basin consistency" }
            ]}
          />
        </div>
      </div>
    </div>
  );
}
