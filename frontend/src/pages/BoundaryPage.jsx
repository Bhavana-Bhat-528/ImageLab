import React, { useState } from 'react';
import ImageDropzone from '../components/ImageDropzone';
import MetricCard from '../components/MetricCard';

export default function BoundaryPage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [seShape, setSeShape] = useState('rect');
  const [seSize, setSeSize] = useState(3);

  const handleImageSelect = (file) => {
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Morphological Boundary Extraction</h1>
        <p className="page-desc">
          Extract and highlight the outer boundaries of animals and objects using binary segmentation, mathematical morphological erosion, set subtraction: β(A) = A − (A ⊖ B), and overlay visualization.
        </p>
      </div>

      <div className="lab-layout">
        <div className="controls-panel">
          <h2 style={{ fontSize: '1.1rem', marginBottom: '1rem', color: '#fff' }}>Input & Parameters</h2>
          
          <ImageDropzone 
            onImageSelect={handleImageSelect}
            previewUrl={previewUrl}
            label="Upload Animal Image"
          />

          <div style={{ marginTop: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div>
              <label style={{ fontSize: '0.85rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>
                Structuring Element Shape
              </label>
              <select 
                value={seShape} 
                onChange={(e) => setSeShape(e.target.value)}
                style={{ width: '100%', padding: '0.5rem', borderRadius: '4px', background: '#0f172a', color: '#fff', border: '1px solid var(--border-color)' }}
              >
                <option value="rect">Rectangle / Box</option>
                <option value="cross">Cross</option>
                <option value="ellipse">Disk / Ellipse</option>
              </select>
            </div>

            <div>
              <label style={{ fontSize: '0.85rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>
                Structuring Element Size: {seSize}x{seSize}
              </label>
              <input 
                type="range" 
                min="3" 
                max="15" 
                step="2" 
                value={seSize} 
                onChange={(e) => setSeSize(parseInt(e.target.value, 10))}
                style={{ width: '100%' }}
              />
            </div>

            <button className="btn btn-primary" style={{ marginTop: '0.5rem' }} disabled={!selectedFile}>
              Extract Boundary
            </button>
          </div>
        </div>

        <div className="preview-panel">
          <h2 style={{ fontSize: '1.1rem', marginBottom: '1rem', color: '#fff' }}>Morphological Extraction Pipeline</h2>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', flex: 1, minHeight: '240px' }}>
            <div style={{ background: '#0f172a', border: '1px dashed var(--border-color)', borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', textAlign: 'center', padding: '0.5rem' }}>
              {previewUrl ? <img src={previewUrl} alt="Original" style={{ maxWidth: '100%', maxHeight: '200px', objectFit: 'contain' }} /> : 'Original Photo'}
            </div>
            <div style={{ background: '#0f172a', border: '1px dashed var(--border-color)', borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', textAlign: 'center', padding: '0.5rem' }}>
              Preprocessed Binary Mask
            </div>
            <div style={{ background: '#0f172a', border: '1px dashed var(--border-color)', borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', textAlign: 'center', padding: '0.5rem' }}>
              Final Boundary Overlay
            </div>
          </div>

          <div style={{ marginTop: '1.5rem', display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
            <MetricCard label="Foreground Pixels" value="--" description="Mask foreground pixel count" />
            <MetricCard label="Boundary Pixels" value="--" description="Extracted edge pixel count" />
            <MetricCard label="Connectivity" value="--" description="Connected components" />
          </div>
        </div>
      </div>
    </div>
  );
}
