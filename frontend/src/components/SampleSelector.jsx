import React, { useState, useEffect } from 'react';
import { fetchSampleList, getSampleImageUrl } from '../services/api';

export default function SampleSelector({
  moduleName,
  presets = [],
  onSelectSample,
  activeSampleFilename = null
}) {
  const [remoteSamples, setRemoteSamples] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    fetchSampleList(moduleName)
      .then((data) => {
        if (isMounted && data.samples && data.samples.length > 0) {
          setRemoteSamples(data.samples);
        }
      })
      .catch(() => {
        // Fallback to presets if backend sample endpoint fails or is offline
      })
      .finally(() => {
        if (isMounted) setLoading(false);
      });

    return () => { isMounted = false; };
  }, [moduleName]);

  // Combine remote discovered samples with preset metadata
  const sampleList = remoteSamples.length > 0
    ? remoteSamples.map((filename) => {
        const matchedPreset = presets.find(p => p.filename === filename);
        return {
          filename,
          label: matchedPreset ? matchedPreset.label : filename.replace(/\.[^/.]+$/, ""),
          category: matchedPreset ? matchedPreset.category : "Sample Dataset",
          description: matchedPreset ? matchedPreset.description : "Pre-collected test benchmark image",
          url: getSampleImageUrl(moduleName, filename)
        };
      })
    : presets.map(p => ({
        ...p,
        url: p.url || getSampleImageUrl(moduleName, p.filename)
      }));

  const handleSampleClick = async (sample) => {
    try {
      // Fetch sample as in-memory blob to mimic standard file upload flow
      const response = await fetch(sample.url);
      if (response.ok) {
        const blob = await response.blob();
        const file = new File([blob], sample.filename, { type: blob.type || 'image/png' });
        const objectUrl = URL.createObjectURL(blob);
        const img = new Image();
        img.onload = () => {
          onSelectSample({
            file,
            previewUrl: objectUrl,
            sampleFilename: sample.filename,
            meta: {
              name: sample.filename,
              size: (blob.size / 1024).toFixed(1) + ' KB',
              width: img.naturalWidth,
              height: img.naturalHeight,
              type: blob.type || 'image/png',
              sampleInfo: sample.description
            }
          });
        };
        img.src = objectUrl;
      } else {
        // If image not physically on disk yet, provide metadata placeholder selection
        onSelectSample({
          file: null,
          previewUrl: sample.url,
          sampleFilename: sample.filename,
          meta: {
            name: sample.filename,
            size: 'Local sample benchmark',
            width: 'Standard',
            height: 'Standard',
            type: 'image/png',
            sampleInfo: sample.description
          }
        });
      }
    } catch {
      onSelectSample({
        file: null,
        previewUrl: sample.url,
        sampleFilename: sample.filename,
        meta: {
          name: sample.filename,
          size: 'Local sample benchmark',
          width: 'Standard',
          height: 'Standard',
          type: 'image/png',
          sampleInfo: sample.description
        }
      });
    }
  };

  return (
    <div className="section-card samples-section">
      <div className="section-card-header">
        <h3 className="section-card-title">2. Select Benchmark Sample</h3>
        <span className="badge-academic">{sampleList.length} benchmark sets</span>
      </div>
      <p className="section-card-sub">
        Choose from standard real-world laboratory test images configured for this experiment:
      </p>

      <div className="samples-grid">
        {sampleList.map((sample, idx) => {
          const isSelected = activeSampleFilename === sample.filename;
          return (
            <div
              key={idx}
              className={`sample-item-card ${isSelected ? 'active' : ''}`}
              onClick={() => handleSampleClick(sample)}
              role="button"
              tabIndex={0}
            >
              <div className="sample-badge">{sample.category || 'Benchmark'}</div>
              <div className="sample-title">{sample.label}</div>
              <div className="sample-desc">{sample.description}</div>
              <div className="sample-filename-code"><code>{sample.filename}</code></div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
