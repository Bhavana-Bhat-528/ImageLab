import React, { useState, useMemo } from 'react';

/**
 * Formats large pixel frequency numbers into clean readable strings (e.g. 1.2k, 45k, 1.2M).
 */
function formatFrequency(num) {
  if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
  if (num >= 10000) return (num / 1000).toFixed(0) + 'k';
  if (num >= 1000) return (num / 1000).toFixed(1) + 'k';
  return String(num);
}

/**
 * Calculates distribution statistics (mean, std dev, mode/peak, total pixels)
 * for a 256-bin histogram array.
 */
function calculateStats(bins) {
  if (!bins || !bins.length) return { mean: 0, std: 0, peakBin: 0, peakVal: 0, total: 0 };
  let total = 0;
  let weightedSum = 0;
  let peakVal = 0;
  let peakBin = 0;

  for (let i = 0; i < bins.length; i++) {
    const count = bins[i];
    total += count;
    weightedSum += i * count;
    if (count > peakVal) {
      peakVal = count;
      peakBin = i;
    }
  }

  const mean = total > 0 ? weightedSum / total : 0;
  let varianceSum = 0;
  for (let i = 0; i < bins.length; i++) {
    const diff = i - mean;
    varianceSum += diff * diff * bins[i];
  }
  const std = total > 0 ? Math.sqrt(varianceSum / total) : 0;

  return {
    mean: mean.toFixed(1),
    std: std.toFixed(1),
    peakBin,
    peakVal,
    total
  };
}

/**
 * Renders a standard, academic 256-bin rectangular bar histogram using SVG.
 * Includes X-axis (0-255), Y-axis (Pixel frequency), grid lines, rectangular bars,
 * and hover tooltip for inspecting exact bin values.
 */
function SingleHistogramChart({
  title,
  subtitle,
  badgeText,
  badgeColor,
  data,
  multiChannelData,
  activeChannel,
  isColor,
  sharedMaxY,
  barColor = '#38bdf8'
}) {
  const [hoveredBin, setHoveredBin] = useState(null);

  const stats = useMemo(() => calculateStats(data), [data]);

  // Dimensions for SVG canvas
  const svgWidth = 520;
  const svgHeight = 220;
  const margin = { top: 22, right: 16, bottom: 42, left: 56 };
  const plotWidth = svgWidth - margin.left - margin.right;
  const plotHeight = svgHeight - margin.top - margin.bottom;

  const numBins = 256;
  const binWidth = plotWidth / numBins;

  // Compute maximum Y for proper bar scaling (using either shared max or local max)
  const localMaxY = useMemo(() => {
    if (activeChannel === 'all' && multiChannelData) {
      const rMax = Math.max(...(multiChannelData.red || [1]));
      const gMax = Math.max(...(multiChannelData.green || [1]));
      const bMax = Math.max(...(multiChannelData.blue || [1]));
      return Math.max(1, rMax, gMax, bMax);
    }
    return Math.max(1, ...(data || [1]));
  }, [data, multiChannelData, activeChannel]);

  const maxY = sharedMaxY || localMaxY;

  // Y-axis tick intervals (0, 25%, 50%, 75%, 100%)
  const yTicks = [0, 0.25, 0.5, 0.75, 1.0].map((ratio) => {
    const val = Math.round(maxY * ratio);
    const yPos = margin.top + plotHeight - ratio * plotHeight;
    return { val, yPos };
  });

  // X-axis major ticks
  const xTicks = [
    { bin: 0, label: '0' },
    { bin: 64, label: '64' },
    { bin: 128, label: '128' },
    { bin: 192, label: '192' },
    { bin: 255, label: '255' }
  ];

  // Mouse interaction handler for interactive bin inspection
  const handleMouseMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clientX = e.clientX - rect.left;
    const svgX = (clientX / rect.width) * svgWidth;
    const plotX = svgX - margin.left;

    if (plotX >= 0 && plotX <= plotWidth) {
      const binIdx = Math.min(255, Math.max(0, Math.floor(plotX / binWidth)));
      setHoveredBin(binIdx);
    } else {
      setHoveredBin(null);
    }
  };

  const handleMouseLeave = () => {
    setHoveredBin(null);
  };

  return (
    <div className="histogram-panel">
      <div className="histogram-panel-header">
        <div>
          <div className="histogram-panel-title-row">
            <h5 className="histogram-panel-title">{title}</h5>
            {badgeText && (
              <span className="histogram-panel-badge" style={{ backgroundColor: badgeColor || 'var(--border-medium)' }}>
                {badgeText}
              </span>
            )}
          </div>
          {subtitle && <span className="histogram-panel-sub">{subtitle}</span>}
        </div>
        <div className="histogram-summary-pill">
          <span>μ = {stats.mean}</span>
          <span className="pill-divider">•</span>
          <span>σ = {stats.std}</span>
        </div>
      </div>

      <div
        className="histogram-svg-container"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      >
        <svg
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          className="histogram-standard-svg"
          preserveAspectRatio="xMidYMid meet"
        >
          {/* Background gridlines */}
          {yTicks.map((tick, idx) => (
            <g key={`ytick-${idx}`}>
              <line
                x1={margin.left}
                y1={tick.yPos}
                x2={margin.left + plotWidth}
                y2={tick.yPos}
                stroke="#1e293b"
                strokeDasharray={idx === 0 ? 'none' : '3,3'}
                strokeWidth={idx === 0 ? '1.5' : '1'}
              />
              <text
                x={margin.left - 8}
                y={tick.yPos + 3.5}
                textAnchor="end"
                className="axis-tick-text"
              >
                {formatFrequency(tick.val)}
              </text>
            </g>
          ))}

          {/* Y-axis label */}
          <text
            x={14}
            y={margin.top + plotHeight / 2}
            textAnchor="middle"
            transform={`rotate(-90 14 ${margin.top + plotHeight / 2})`}
            className="axis-title-text"
          >
            Frequency (Pixels)
          </text>

          {/* Histogram Bars */}
          {activeChannel === 'all' && multiChannelData ? (
            // Layered RGB Bars
            <>
              {/* Red Channel Bars */}
              {(multiChannelData.red || []).map((val, i) => {
                const barHeight = Math.min(plotHeight, (val / maxY) * plotHeight);
                const x = margin.left + i * binWidth;
                const y = margin.top + plotHeight - barHeight;
                return (
                  <rect
                    key={`r-${i}`}
                    x={x}
                    y={y}
                    width={Math.max(1, binWidth)}
                    height={barHeight}
                    fill="rgba(239, 68, 68, 0.45)"
                  />
                );
              })}
              {/* Green Channel Bars */}
              {(multiChannelData.green || []).map((val, i) => {
                const barHeight = Math.min(plotHeight, (val / maxY) * plotHeight);
                const x = margin.left + i * binWidth;
                const y = margin.top + plotHeight - barHeight;
                return (
                  <rect
                    key={`g-${i}`}
                    x={x}
                    y={y}
                    width={Math.max(1, binWidth)}
                    height={barHeight}
                    fill="rgba(34, 197, 94, 0.45)"
                  />
                );
              })}
              {/* Blue Channel Bars */}
              {(multiChannelData.blue || []).map((val, i) => {
                const barHeight = Math.min(plotHeight, (val / maxY) * plotHeight);
                const x = margin.left + i * binWidth;
                const y = margin.top + plotHeight - barHeight;
                return (
                  <rect
                    key={`b-${i}`}
                    x={x}
                    y={y}
                    width={Math.max(1, binWidth)}
                    height={barHeight}
                    fill="rgba(59, 130, 246, 0.55)"
                  />
                );
              })}
            </>
          ) : (
            // Single Channel Discrete Rectangular Bins
            (data || []).map((val, i) => {
              const barHeight = Math.min(plotHeight, (val / maxY) * plotHeight);
              const x = margin.left + i * binWidth;
              const y = margin.top + plotHeight - barHeight;
              const isHovered = hoveredBin === i;

              return (
                <rect
                  key={`bin-${i}`}
                  x={x}
                  y={y}
                  width={Math.max(1, binWidth)}
                  height={barHeight}
                  fill={isHovered ? '#f8fafc' : barColor}
                  className="histogram-bar-rect"
                />
              );
            })
          )}

          {/* X-axis baseline */}
          <line
            x1={margin.left}
            y1={margin.top + plotHeight}
            x2={margin.left + plotWidth}
            y2={margin.top + plotHeight}
            stroke="#475569"
            strokeWidth="1.5"
          />

          {/* X-axis ticks & labels */}
          {xTicks.map((tick) => {
            const xPos = margin.left + tick.bin * binWidth;
            return (
              <g key={`xtick-${tick.bin}`}>
                <line
                  x1={xPos}
                  y1={margin.top + plotHeight}
                  x2={xPos}
                  y2={margin.top + plotHeight + 5}
                  stroke="#475569"
                  strokeWidth="1.5"
                />
                <text
                  x={xPos}
                  y={margin.top + plotHeight + 16}
                  textAnchor="middle"
                  className="axis-tick-text"
                >
                  {tick.label}
                </text>
              </g>
            );
          })}

          {/* X-axis title */}
          <text
            x={margin.left + plotWidth / 2}
            y={svgHeight - 6}
            textAnchor="middle"
            className="axis-title-text"
          >
            Intensity Level r (0 – 255)
          </text>

          {/* Hover highlight cursor */}
          {hoveredBin !== null && (
            <g pointerEvents="none">
              <line
                x1={margin.left + hoveredBin * binWidth + binWidth / 2}
                y1={margin.top}
                x2={margin.left + hoveredBin * binWidth + binWidth / 2}
                y2={margin.top + plotHeight}
                stroke="#f1f5f9"
                strokeWidth="1"
                strokeDasharray="2,2"
              />
            </g>
          )}
        </svg>

        {/* Hover Tooltip display */}
        {hoveredBin !== null && data && (
          <div
            className="histogram-tooltip"
            style={{
              left: `${((margin.left + hoveredBin * binWidth) / svgWidth) * 100}%`
            }}
          >
            <div className="tooltip-bin">Intensity: <strong>{hoveredBin}</strong></div>
            <div className="tooltip-val">
              Count: <strong>{(data[hoveredBin] || 0).toLocaleString()}</strong> px
            </div>
            {multiChannelData && activeChannel === 'all' && (
              <div className="tooltip-sub">
                <span style={{ color: '#ef4444' }}>R: {multiChannelData.red?.[hoveredBin]}</span> •{' '}
                <span style={{ color: '#22c55e' }}>G: {multiChannelData.green?.[hoveredBin]}</span> •{' '}
                <span style={{ color: '#3b82f6' }}>B: {multiChannelData.blue?.[hoveredBin]}</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Distribution Metrics Footer */}
      <div className="histogram-metrics-footer">
        <div className="hist-metric-item">
          <span className="hist-metric-label">Peak Bin (Mode):</span>
          <span className="hist-metric-val">{stats.peakBin} ({formatFrequency(stats.peakVal)} px)</span>
        </div>
        <div className="hist-metric-item">
          <span className="hist-metric-label">Mean Intensity:</span>
          <span className="hist-metric-val">{stats.mean}</span>
        </div>
        <div className="hist-metric-item">
          <span className="hist-metric-label">Std Dev (Spread):</span>
          <span className="hist-metric-val">{stats.std}</span>
        </div>
        <div className="hist-metric-item">
          <span className="hist-metric-label">Total Pixels:</span>
          <span className="hist-metric-val">{stats.total.toLocaleString()}</span>
        </div>
      </div>
    </div>
  );
}

/**
 * Main HistogramComparison component.
 * Displays separate, side-by-side conventional bar histograms for Original and Enhanced images.
 * Provides RGB channel switching (Red, Green, Blue, Grayscale, Combined), and an optional
 * overlay comparison view.
 */
export default function HistogramComparison({ originalHist, enhancedHist }) {
  const [activeChannel, setActiveChannel] = useState('luminance');
  const [scaleMode, setScaleMode] = useState('shared'); // 'shared' or 'individual'
  const [viewMode, setViewMode] = useState('side-by-side'); // 'side-by-side' or 'overlay'

  if (!originalHist || !enhancedHist) {
    return (
      <div className="histogram-placeholder">
        <span>No histogram data available. Run enhancement to visualize intensity distribution.</span>
      </div>
    );
  }

  // Detect whether image is color (handling snake_case is_color or camelCase isColor)
  const isColor = Boolean(
    originalHist.is_color ??
    originalHist.isColor ??
    (originalHist.red && originalHist.red.length > 0)
  );

  // Extract channel data
  const getChannelData = (hist, ch) => {
    if (!hist) return [];
    if (ch === 'luminance') return hist.luminance || [];
    if (ch === 'red') return hist.red || hist.luminance || [];
    if (ch === 'green') return hist.green || hist.luminance || [];
    if (ch === 'blue') return hist.blue || hist.luminance || [];
    return hist.luminance || [];
  };

  const origData = getChannelData(originalHist, activeChannel);
  const enhData = getChannelData(enhancedHist, activeChannel);

  // Determine bar colors depending on the selected channel
  const getChannelColor = (ch, isEnhanced = false) => {
    switch (ch) {
      case 'red':
        return isEnhanced ? '#f87171' : '#dc2626';
      case 'green':
        return isEnhanced ? '#4ade80' : '#16a34a';
      case 'blue':
        return isEnhanced ? '#60a5fa' : '#2563eb';
      case 'luminance':
      default:
        return isEnhanced ? '#34d399' : '#38bdf8';
    }
  };

  // Shared max Y value for direct apples-to-apples visual comparison
  const sharedMaxY = useMemo(() => {
    if (scaleMode !== 'shared') return null;
    let max1 = 1;
    let max2 = 1;
    if (activeChannel === 'all' && isColor) {
      max1 = Math.max(
        1,
        ...(originalHist.red || []),
        ...(originalHist.green || []),
        ...(originalHist.blue || [])
      );
      max2 = Math.max(
        1,
        ...(enhancedHist.red || []),
        ...(enhancedHist.green || []),
        ...(enhancedHist.blue || [])
      );
    } else {
      max1 = Math.max(1, ...(origData || []));
      max2 = Math.max(1, ...(enhData || []));
    }
    return Math.max(max1, max2);
  }, [origData, enhData, originalHist, enhancedHist, activeChannel, isColor, scaleMode]);

  return (
    <div className="histogram-card">
      {/* Top Header & Controls */}
      <div className="histogram-header">
        <div>
          <h4 className="histogram-title">Intensity Histograms (Conventional Bar Distribution)</h4>
          <span className="histogram-sub">
            Standard 256-bin discrete rectangular frequency bars: Intensity [0–255] vs. Pixel Count
          </span>
        </div>

        <div className="histogram-controls-row">
          {/* Channel Selector for RGB images */}
          {isColor ? (
            <div className="histogram-channel-selector" title="Select color channel">
              <button
                type="button"
                className={`ch-btn ${activeChannel === 'luminance' ? 'active' : ''}`}
                onClick={() => setActiveChannel('luminance')}
              >
                Grayscale (Y)
              </button>
              <button
                type="button"
                className={`ch-btn ch-red ${activeChannel === 'red' ? 'active' : ''}`}
                onClick={() => setActiveChannel('red')}
              >
                Red (R)
              </button>
              <button
                type="button"
                className={`ch-btn ch-green ${activeChannel === 'green' ? 'active' : ''}`}
                onClick={() => setActiveChannel('green')}
              >
                Green (G)
              </button>
              <button
                type="button"
                className={`ch-btn ch-blue ${activeChannel === 'blue' ? 'active' : ''}`}
                onClick={() => setActiveChannel('blue')}
              >
                Blue (B)
              </button>
              <button
                type="button"
                className={`ch-btn ${activeChannel === 'all' ? 'active' : ''}`}
                onClick={() => setActiveChannel('all')}
              >
                RGB All
              </button>
            </div>
          ) : (
            <div className="histogram-mode-tag">
              <span className="tag-mono">Grayscale 8-bit Intensity</span>
            </div>
          )}

          {/* Scale synchronization toggle */}
          <div className="histogram-scale-selector" title="Y-axis scale mode">
            <button
              type="button"
              className={`scale-btn ${scaleMode === 'shared' ? 'active' : ''}`}
              onClick={() => setScaleMode('shared')}
            >
              Unified Y-Scale
            </button>
            <button
              type="button"
              className={`scale-btn ${scaleMode === 'individual' ? 'active' : ''}`}
              onClick={() => setScaleMode('individual')}
            >
              Auto-Scale
            </button>
          </div>
        </div>
      </div>

      {/* Legend & Channel Status */}
      <div className="histogram-legend">
        <div className="legend-item">
          <span className="legend-color-box" style={{ backgroundColor: getChannelColor(activeChannel, false) }}></span>
          <span>Original Histogram ({activeChannel === 'all' ? 'All Channels' : activeChannel.toUpperCase()})</span>
        </div>
        <div className="legend-item">
          <span className="legend-color-box" style={{ backgroundColor: getChannelColor(activeChannel, true) }}></span>
          <span>Enhanced Histogram ({activeChannel === 'all' ? 'All Channels' : activeChannel.toUpperCase()})</span>
        </div>
        <div className="legend-note">
          <span>Hover over any bin to inspect exact intensity value and pixel frequency.</span>
        </div>
      </div>

      {/* Side-by-Side Separate Histograms */}
      <div className="histogram-side-by-side-grid">
        {/* 1. Original Image Histogram */}
        <SingleHistogramChart
          title="Original Image Histogram"
          subtitle="Input intensity distribution prior to enhancement"
          badgeText="Before"
          badgeColor="#0284c7"
          data={origData}
          multiChannelData={originalHist}
          activeChannel={activeChannel}
          isColor={isColor}
          sharedMaxY={sharedMaxY}
          barColor={getChannelColor(activeChannel, false)}
        />

        {/* 2. Enhanced Image Histogram */}
        <SingleHistogramChart
          title="Enhanced Image Histogram"
          subtitle="Redistributed intensity distribution post-enhancement"
          badgeText="After"
          badgeColor="#059669"
          data={enhData}
          multiChannelData={enhancedHist}
          activeChannel={activeChannel}
          isColor={isColor}
          sharedMaxY={sharedMaxY}
          barColor={getChannelColor(activeChannel, true)}
        />
      </div>
    </div>
  );
}
