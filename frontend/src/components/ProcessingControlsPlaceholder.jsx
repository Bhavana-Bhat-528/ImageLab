import React from 'react';

export default function ProcessingControlsPlaceholder({
  title = "3. Processing & Algorithm Controls",
  children,
  onExecute,
  onReset,
  isExecuting = false,
  hasImage = false,
  executeLabel = "Execute Processing"
}) {
  return (
    <div className="section-card controls-section">
      <div className="section-card-header">
        <h3 className="section-card-title">{title}</h3>
        <span className="badge-academic">FastAPI Parameter Schema</span>
      </div>

      <div className="controls-form">
        {children}

        <div className="controls-actions">
          <button
            type="button"
            className="btn btn-primary btn-execute"
            disabled={!hasImage || isExecuting}
            onClick={onExecute}
          >
            {isExecuting ? (
              <>
                <span className="spinner-sm"></span>
                Processing In-Memory...
              </>
            ) : (
              <>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polygon points="5 3 19 12 5 21 5 3" />
                </svg>
                {executeLabel}
              </>
            )}
          </button>

          {onReset && (
            <button
              type="button"
              className="btn btn-outline"
              disabled={isExecuting}
              onClick={onReset}
            >
              Reset Parameters
            </button>
          )}
        </div>

        <div className="controls-placeholder-notice">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="16" x2="12" y2="12" />
            <line x1="12" y1="8" x2="12.01" y2="8" />
          </svg>
          <span>
            Parameters are wired to FastAPI REST endpoints. Algorithm execution will run via the Python backend in the next phase.
          </span>
        </div>
      </div>
    </div>
  );
}
