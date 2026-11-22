import React, { useState } from 'react';
import ImageDropzone from '../components/ImageDropzone';
import MetricCard from '../components/MetricCard';

export default function FourierPage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [numDescriptors, setNumDescriptors] = useState(16);

  const handleImageSelect = (file) => {
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Shape Recognition via Fourier Descriptors</h1>
        <p className="page-desc">
          Object recognition using boundary representation, Discrete Fourier Transform descriptors (invariant to translation, rotation, and scale), and shape classification across 5 categories.
        </p>
      </div>

      <div className="lab-layout">
        <div className="controls-panel">
          <h2 style={{ fontSize: '1.1rem', marginBottom: '1rem', color: '#fff' }}>Input & Features</h2>
          
          <ImageDropzone 
            onImageSelect={handleImageSelect}
            previewUrl={previewUrl}
            label="Upload Object Image"
          />

          <div style={{ marginTop: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div>
              <label style={{ fontSize: '0.85rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>
                Number of Descriptors: {numDescriptors}
              </label>
              <input 
                type="range" 
                min="4" 
                max="64" 
                step="4" 
                value={numDescriptors} 
                onChange={(e) => setNumDescriptors(parseInt(e.target.value, 10))}
                style={{ width: '100%' }}
              />
            </div>

            <button className="btn btn-primary" style={{ marginTop: '0.5rem' }} disabled={!selectedFile}>
              Extract Descriptors & Classify
            </button>
          </div>
        </div>

        <div className="preview-panel">
          <h2 style={{ fontSize: '1.1rem', marginBottom: '1rem', color: '#fff' }}>Boundary & Spectrum Representation</h2>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', flex: 1, minHeight: '240px' }}>
            <div style={{ background: '#0f172a', border: '1px dashed var(--border-color)', borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', textAlign: 'center', padding: '0.5rem' }}>
              {previewUrl ? <img src={previewUrl} alt="Original" style={{ maxWidth: '100%', maxHeight: '200px', objectFit: 'contain' }} /> : 'Original Object'}
            </div>
            <div style={{ background: '#0f172a', border: '1px dashed var(--border-color)', borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', textAlign: 'center', padding: '0.5rem' }}>
              Segmented Contour
            </div>
            <div style={{ background: '#0f172a', border: '1px dashed var(--border-color)', borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', textAlign: 'center', padding: '0.5rem' }}>
              Fourier Descriptor Spectrum
            </div>
          </div>

          <div style={{ marginTop: '1.5rem', display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
            <MetricCard label="Predicted Class" value="--" description="Closest matching object category" />
            <MetricCard label="Similarity Score" value="--" description="Euclidean / Cosine descriptor distance" />
            <MetricCard label="Boundary Length" value="--" unit="pts" description="Number of contour coordinate points" />
          </div>
        </div>
      </div>
    </div>
  );
}
