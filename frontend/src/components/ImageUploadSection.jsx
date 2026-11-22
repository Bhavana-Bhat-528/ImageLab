import React, { useRef } from 'react';

export default function ImageUploadSection({
  selectedFile,
  imagePreview,
  imageMeta,
  onImageSelected,
  onClearImage,
  label = "Upload Image",
  hint = "Supports standard raster formats: PNG, JPG, JPEG, TIFF, BMP",
  disabled = false
}) {
  const fileInputRef = useRef(null);

  const handleFile = (file) => {
    if (!file || !file.type.startsWith('image/')) {
      alert('Please select a valid image file.');
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      onImageSelected({
        file,
        previewUrl: objectUrl,
        meta: {
          name: file.name,
          size: (file.size / 1024).toFixed(1) + ' KB',
          width: img.naturalWidth,
          height: img.naturalHeight,
          type: file.type || 'image/png'
        }
      });
    };
    img.onerror = () => {
      onImageSelected({
        file,
        previewUrl: objectUrl,
        meta: {
          name: file.name,
          size: (file.size / 1024).toFixed(1) + ' KB',
          width: 'Unknown',
          height: 'Unknown',
          type: file.type
        }
      });
    };
    img.src = objectUrl;
  };

  const handleInputChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFile(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (disabled) return;
    const file = e.dataTransfer.files?.[0];
    if (file) {
      handleFile(file);
    }
  };

  return (
    <div className="section-card upload-section">
      <div className="section-card-header">
        <h3 className="section-card-title">1. Input Image</h3>
        {selectedFile && (
          <button 
            type="button" 
            className="btn-text-danger" 
            onClick={onClearImage}
            title="Remove selected image"
          >
            Clear Image
          </button>
        )}
      </div>

      {!selectedFile ? (
        <div
          className="dropzone-area"
          onClick={() => fileInputRef.current?.click()}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          role="button"
          tabIndex={0}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/png,image/jpeg,image/tiff,image/bmp,image/webp"
            style={{ display: 'none' }}
            onChange={handleInputChange}
            disabled={disabled}
          />
          <div className="dropzone-icon">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
          </div>
          <p className="dropzone-label">{label}</p>
          <p className="dropzone-sub">{hint}</p>
          <button type="button" className="btn btn-secondary btn-sm" style={{ marginTop: '0.75rem' }}>
            Browse Local File
          </button>
        </div>
      ) : (
        <div className="uploaded-file-card">
          <div className="uploaded-thumbnail-wrapper">
            <img src={imagePreview} alt="Selected Input" className="uploaded-thumbnail" />
          </div>
          <div className="uploaded-details">
            <div className="uploaded-filename" title={imageMeta?.name}>{imageMeta?.name}</div>
            <div className="uploaded-metadata">
              <span>Resolution: {imageMeta?.width} × {imageMeta?.height} px</span>
              <span className="dot-sep">•</span>
              <span>Size: {imageMeta?.size}</span>
            </div>
            <div className="uploaded-status-badge">
              <span className="status-indicator-dot"></span>
              Memory Ready (In-Memory Buffer)
            </div>
          </div>
          <div className="uploaded-actions">
            <button
              type="button"
              className="btn btn-outline btn-xs"
              onClick={() => fileInputRef.current?.click()}
            >
              Replace
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/png,image/jpeg,image/tiff,image/bmp,image/webp"
              style={{ display: 'none' }}
              onChange={handleInputChange}
            />
          </div>
        </div>
      )}
    </div>
  );
}
