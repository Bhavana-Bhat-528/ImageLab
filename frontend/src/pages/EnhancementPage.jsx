import React, { useState } from 'react';
import ImageDropzone from '../components/ImageDropzone';
import MetricCard from '../components/MetricCard';

export default function EnhancementPage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [gamma, setGamma] = useState(1.2);
  const [kernelSize, setKernelSize] = useState(3);
  const [sharpenStrength, setSharpenStrength] = useState(1.0);

  const handleImageSelect = (file) => {
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Automatic Image Enhancement</h1>
        <p className="page-desc">
          Automated characteristic analysis with intensity transformations (gamma correction, histogram equalization) and spatial filtering (Laplacian / unsharp masking).
        </p>
      </div>

      <div className="lab-layout">
        <div className="controls-panel">
          <h2 style={{ fontSize: '1.1rem', marginBottom: '1rem', color: '#fff' }}>Input & Controls</h2>
          
          <ImageDropzone 
            onImageSelect={handleImageSelect}
            previewUrl={previewUrl}
            label="Upload Test Image"
          />

          <div style={{ marginTop: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div>
              <label style={{ fontSize: '0.85rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>
                Gamma Value: {gamma}
              </label>
              <input 
                type="range" 
                min="0.2" 
                max="3.0" 
                step="0.1" 
                value={gamma} 
                onChange={(e) => setGamma(parseFloat(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>

            <div>
              <label style={{ fontSize: '0.85rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>
                Kernel Size: {kernelSize}x{kernelSize}
              </label>
              <select 
                value={kernelSize} 
                onChange={(e) => setKernelSize(parseInt(e.target.value, 10))}
                style={{ width: '100%', padding: '0.5rem', borderRadius: '4px', background: '#0f172a', color: '#fff', border: '1px solid var(--border-color)' }}
              >
                <option value={3}>3x3</option>
                <option value={5}>5x5</option>
                <option value={7}>7x7</option>
              </select>
            </div>

            <div>
              <label style={{ fontSize: '0.85rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.25rem' }}>
                Sharpening Strength: {sharpenStrength}
              </label>
              <input 
                type="range" 
                min="0.0" 
                max="3.0" 
                step="0.2" 
                value={sharpenStrength} 
                onChange={(e) => setSharpenStrength(parseFloat(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>

            <button className="btn btn-primary" style={{ marginTop: '0.5rem' }} disabled={!selectedFile}>
              Run Automatic Enhancement
            </button>
          </div>
        </div>

        <div className="preview-panel">
          <h2 style={{ fontSize: '1.1rem', marginBottom: '1rem', color: '#fff' }}>Comparison & Visualizations</h2>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', flex: 1, minHeight: '260px' }}>
            <div style={{ background: '#0f172a', border: '1px dashed var(--border-color)', borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
              {previewUrl ? <img src={previewUrl} alt="Original" style={{ maxWidth: '100%', maxHeight: '240px', objectFit: 'contain' }} /> : 'Original Image View'}
            </div>
            <div style={{ background: '#0f172a', border: '1px dashed var(--border-color)', borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
              Enhanced Output View
            </div>
          </div>

          <div style={{ marginTop: '1.5rem', display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
            <MetricCard label="MSE" value="--" description="Mean Squared Error" />
            <MetricCard label="PSNR" value="--" unit="dB" description="Peak Signal-to-Noise Ratio" />
            <MetricCard label="SSIM" value="--" description="Structural Similarity" />
          </div>
        </div>
      </div>
    </div>
  );
}
