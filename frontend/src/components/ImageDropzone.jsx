import React, { useRef } from 'react';

export default function ImageDropzone({ onImageSelect, previewUrl, label = "Upload Image" }) {
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      onImageSelect(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) {
      onImageSelect(file);
    }
  };

  return (
    <div 
      className="dropzone"
      onClick={() => fileInputRef.current?.click()}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      <input 
        ref={fileInputRef}
        type="file" 
        accept="image/*" 
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />
      <div className="dropzone-icon">📷</div>
      <div className="dropzone-text">{label}</div>
      <div className="dropzone-hint">Click or drag & drop (JPG, PNG, TIFF, BMP)</div>
      {previewUrl && (
        <div style={{ marginTop: '0.75rem' }}>
          <img 
            src={previewUrl} 
            alt="Upload Preview" 
            style={{ maxWidth: '100%', maxHeight: '180px', borderRadius: '4px', objectFit: 'contain' }} 
          />
        </div>
      )}
    </div>
  );
}
