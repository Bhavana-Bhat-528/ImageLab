import React, { useState } from 'react';
import ImageDropzone from '../components/ImageDropzone';
import MetricCard from '../components/MetricCard';

export default function WatershedPage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [thresholdVal, setThresholdVal] = useState(128);
  const [minDistance, setMinDistance] = useState(15);

  const handleImageSelect = (file) => {
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Morphological Watershed Segmentation</h1>
        <p className="page-desc">
          Separation of touching microscopic cells via Otsu binarization, Euclidean distance transform, local peak marker extraction, and marker-controlled watershed segmentation.
        </p>
      </div>

      <div className="lab-layout">
        <div className="controls-panel">
          <h2 style={{ fontSize: '1.1rem', marginBottom: '1rem', color: '#fff' }}>Input & Parameters</h2>
          
          <ImageDropzone 
            onImageSelect={handleImageSelect}
            previewUrl={previewUrl}
            label="Upload Microscopic Image"
          />

          <div style={{ marginTop: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div>
              <label style={{ fontSize: '0.85rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>
                Binarization Threshold: {thresholdVal}
              </label>
              <input 
                type="range" 
                min="0" 
                max="255" 
                value={thresholdVal} 
                onChange={(e) => setThresholdVal(parseInt(e.target.value, 10))}
                style={{ width: '100%' }}
              />
            </div>

            <div>
              <label style={{ fontSize: '0.85rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>
                Marker Min Distance: {minDistance} px
              </label>
              <input 
                type="range" 
                min="5" 
                max="50" 
                value={minDistance} 
                onChange={(e) => setMinDistance(parseInt(e.target.value, 10))}
                style={{ width: '100%' }}
              />
            </div>

            <button className="btn btn-primary" style={{ marginTop: '0.5rem' }} disabled={!selectedFile}>
              Run Watershed Segmentation
            </button>
          </div>
        </div>

        <div className="preview-panel">
          <h2 style={{ fontSize: '1.1rem', marginBottom: '1rem', color: '#fff' }}>Segmentation Pipeline</h2>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', flex: 1, minHeight: '240px' }}>
            <div style={{ background: '#0f172a', border: '1px dashed var(--border-color)', borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', textAlign: 'center', padding: '0.5rem' }}>
              {previewUrl ? <img src={previewUrl} alt="Original" style={{ maxWidth: '100%', maxHeight: '200px', objectFit: 'contain' }} /> : 'Original Microscopic'}
            </div>
            <div style={{ background: '#0f172a', border: '1px dashed var(--border-color)', borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', textAlign: 'center', padding: '0.5rem' }}>
              Distance Map & Markers
            </div>
            <div style={{ background: '#0f172a', border: '1px dashed var(--border-color)', borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', textAlign: 'center', padding: '0.5rem' }}>
              Watershed Labeled Output
            </div>
          </div>

          <div style={{ marginTop: '1.5rem', display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
            <MetricCard label="Estimated Cell Count" value="--" description="Segmented individual cell regions" />
            <MetricCard label="Mean Cell Area" value="--" unit="px²" description="Average size per cell" />
            <MetricCard label="Overlap Ratio" value="--" description="Touching boundary ratio" />
          </div>
        </div>
      </div>
    </div>
  );
}
