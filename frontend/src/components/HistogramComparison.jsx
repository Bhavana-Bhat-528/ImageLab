import React, { useState } from 'react';

export default function HistogramComparison({ originalHist, enhancedHist }) {
  const [activeChannel, setActiveChannel] = useState('luminance');

  if (!originalHist || !enhancedHist) {
    return (
      <div className="histogram-placeholder">
        <span>No histogram data available. Run enhancement to visualize intensity distribution.</span>
      </div>
    );
  }

  const isColor = Boolean(originalHist.isColor && enhancedHist.isColor);
  const origData = originalHist[activeChannel] || originalHist.luminance || [];
  const enhData = enhancedHist[activeChannel] || enhancedHist.luminance || [];

  // Normalize heights to 100 for SVG plotting
  const maxVal = Math.max(
    1,
    ...origData,
    ...enhData
  );

  const svgWidth = 480;
  const svgHeight = 160;
  const numBins = 256;
  const binWidth = svgWidth / numBins;

  // Build SVG path points
  const buildPath = (data) => {
    if (!data.length) return '';
    return data
      .map((val, i) => {
        const x = (i * binWidth).toFixed(1);
        const y = (svgHeight - (val / maxVal) * (svgHeight - 12)).toFixed(1);
        return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
      })
      .join(' ');
  };

  const origPath = buildPath(origData);
  const enhPath = buildPath(enhData);

  return (
    <div className="histogram-card">
      <div className="histogram-header">
        <div>
          <h4 className="histogram-title">Intensity Histograms (Before vs. After)</h4>
          <span className="histogram-sub">256-Bin Normalized Frequency Distribution</span>
        </div>
        {isColor && (
          <div className="histogram-channel-selector">
            {['luminance', 'red', 'green', 'blue'].map((ch) => (
              <button
                key={ch}
                type="button"
                className={`ch-btn ${activeChannel === ch ? 'active' : ''}`}
                onClick={() => setActiveChannel(ch)}
              >
                {ch.charAt(0).toUpperCase() + ch.slice(1)}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="histogram-legend">
        <div className="legend-item">
          <span className="legend-color-dot dot-original"></span>
          <span>Original Histogram</span>
        </div>
        <div className="legend-item">
          <span className="legend-color-dot dot-enhanced"></span>
          <span>Enhanced Histogram</span>
        </div>
      </div>

      <div className="histogram-svg-wrapper">
        <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} className="histogram-svg" preserveAspectRatio="none">
          {/* Subtle grid lines */}
          <line x1="0" y1={svgHeight / 2} x2={svgWidth} y2={svgHeight / 2} stroke="#1e293b" strokeDasharray="3,3" />
          <line x1={svgWidth / 4} y1="0" x2={svgWidth / 4} y2={svgHeight} stroke="#1e293b" strokeDasharray="3,3" />
          <line x1={svgWidth / 2} y1="0" x2={svgWidth / 2} y2={svgHeight} stroke="#1e293b" strokeDasharray="3,3" />
          <line x1={(3 * svgWidth) / 4} y1="0" x2={(3 * svgWidth) / 4} y2={svgHeight} stroke="#1e293b" strokeDasharray="3,3" />

          {/* Original Histogram Fill & Stroke (Translucent Cyan) */}
          {origPath && (
            <>
              <path
                d={`${origPath} L ${svgWidth} ${svgHeight} L 0 ${svgHeight} Z`}
                fill="rgba(56, 189, 248, 0.15)"
              />
              <path
                d={origPath}
                fill="none"
                stroke="#38bdf8"
                strokeWidth="1.5"
                strokeLinecap="round"
              />
            </>
          )}

          {/* Enhanced Histogram Fill & Stroke (Vibrant Emerald / Violet) */}
          {enhPath && (
            <>
              <path
                d={`${enhPath} L ${svgWidth} ${svgHeight} L 0 ${svgHeight} Z`}
                fill="rgba(52, 211, 153, 0.2)"
              />
              <path
                d={enhPath}
                fill="none"
                stroke="#34d399"
                strokeWidth="2"
                strokeLinecap="round"
              />
            </>
          )}
        </svg>

        <div className="histogram-x-axis">
          <span>0 (Shadows)</span>
          <span>64</span>
          <span>128 (Midtones)</span>
          <span>192</span>
          <span>255 (Highlights)</span>
        </div>
      </div>
    </div>
  );
}
