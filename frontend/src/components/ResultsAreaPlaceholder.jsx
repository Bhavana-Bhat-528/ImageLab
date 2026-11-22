import React from 'react';

export default function ResultsAreaPlaceholder({
  hasImage = false,
  imagePreview = null,
  isExecuting = false,
  errorMessage = null,
  onDismissError = null,
  resultViews = [
    { label: "Original Input", key: "original" },
    { label: "Preprocessed / Intermediate", key: "intermediate" },
    { label: "Final Result", key: "final" }
  ],
  metrics = [],
  moduleName = "Algorithm"
}) {
  return (
    <div className="section-card results-section">
      <div className="section-card-header">
        <div>
          <h3 className="section-card-title">4. Analytical Results & Visualizations</h3>
          <p className="section-card-sub">
            Scientific multi-stage visual comparison and quantitative metrics.
          </p>
        </div>
        <div className="results-header-badges">
          <span className="badge-academic">Zero-Persistence Memory Stream</span>
        </div>
      </div>

      {/* Error state */}
      {errorMessage && (
        <div className="error-banner" role="alert">
          <div className="error-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
          </div>
          <div className="error-content">
            <div className="error-title">Execution Error</div>
            <div className="error-detail">{errorMessage}</div>
          </div>
          {onDismissError && (
            <button type="button" className="error-close-btn" onClick={onDismissError}>
              &times;
            </button>
          )}
        </div>
      )}

      {/* Loading state */}
      {isExecuting && (
        <div className="loading-state-card">
          <div className="academic-spinner"></div>
          <div className="loading-title">Processing Image in Python FastAPI...</div>
          <div className="loading-sub">
            Streaming in-memory bytes to DIP module without persistent disk storage.
          </div>
        </div>
      )}

      {/* Empty state (no image uploaded/selected) */}
      {!hasImage && !isExecuting && (
        <div className="empty-state-card">
          <div className="empty-state-icon">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
              <circle cx="8.5" cy="8.5" r="1.5" />
              <polyline points="21 15 16 10 5 21" />
            </svg>
          </div>
          <h4 className="empty-state-title">No Image Loaded</h4>
          <p className="empty-state-text">
            Upload an image from your local system or select a benchmark sample to activate the analysis view.
          </p>
        </div>
      )}

      {/* Ready / Placeholder State (Image loaded, awaiting or showing execution slots) */}
      {hasImage && !isExecuting && (
        <div className="results-display-area">
          <div className={`results-grid results-grid-${resultViews.length}`}>
            {resultViews.map((view, idx) => (
              <div key={idx} className="result-viewport-card">
                <div className="viewport-header">
                  <span className="viewport-stage-index">Stage {idx + 1}</span>
                  <span className="viewport-label">{view.label}</span>
                </div>
                <div className="viewport-canvas">
                  {idx === 0 && imagePreview ? (
                    <img src={imagePreview} alt={view.label} className="canvas-img" />
                  ) : (
                    <div className="canvas-placeholder">
                      <div className="placeholder-icon">⚲</div>
                      <div className="placeholder-title">Awaiting {view.label}</div>
                      <div className="placeholder-sub">Output slot for {moduleName} processing</div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Quantitative Metrics Bar */}
          <div className="metrics-container">
            <div className="metrics-header">
              <h4 className="metrics-title">Quantitative Metrics & Evaluation</h4>
              <span className="badge-academic">Evaluation Metrics</span>
            </div>
            <div className="metrics-grid">
              {metrics.map((metric, i) => (
                <div key={i} className="metric-box">
                  <span className="metric-label">{metric.label}</span>
                  <span className="metric-value">
                    {metric.value ?? "--"}
                    {metric.unit && <span className="metric-unit"> {metric.unit}</span>}
                  </span>
                  {metric.description && (
                    <span className="metric-description">{metric.description}</span>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
