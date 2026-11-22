import React from 'react';
import { Link } from 'react-router-dom';

const modules = [
  {
    id: 1,
    title: 'Automatic Image Enhancement',
    path: '/enhancement',
    badge: 'Spatial & Intensity Processing',
    desc: 'Analyzes image characteristics, automatically chooses intensity transformations (gamma/contrast/equalization) and spatial filters (smoothing/sharpening), compares histograms, and evaluates MSE/PSNR/SSIM.',
    tag: 'Module 1'
  },
  {
    id: 2,
    title: 'Morphological Boundary Extraction',
    path: '/boundary',
    badge: 'Mathematical Morphology',
    desc: 'Extracts and highlights boundaries of objects and animals using binary segmentation, erosion, morphological set difference (A - (A ⊖ B)), and multi-size/shape structuring element evaluation.',
    tag: 'Module 2'
  },
  {
    id: 3,
    title: 'Morphological Watershed Segmentation',
    path: '/watershed',
    badge: 'Region Segmentation',
    desc: 'Separates touching cells in microscopic images using distance transform, marker-controlled watershed segmentation, and estimates individual cell counts.',
    tag: 'Module 3'
  },
  {
    id: 4,
    title: 'Shape Recognition with Fourier Descriptors',
    path: '/fourier',
    badge: 'Feature Extraction & Classification',
    desc: 'Extracts object boundaries, computes invariant Fourier descriptors, compares boundary representations, and matches/classifies shapes across 5 object categories.',
    tag: 'Module 4'
  }
];

export default function Home() {
  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Digital Image Processing Laboratory</h1>
        <p className="page-desc">
          An interactive, local-first platform designed to experiment, benchmark, and visualize classical and morphological computer vision algorithms in real time.
        </p>
      </div>

      <div className="grid-2">
        {modules.map((m) => (
          <div key={m.id} className="card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--accent-blue)', background: 'rgba(59, 130, 246, 0.1)', padding: '0.2rem 0.5rem', borderRadius: '4px' }}>
                {m.tag}
              </span>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                {m.badge}
              </span>
            </div>
            <h2 className="card-title">{m.title}</h2>
            <p className="card-desc">{m.desc}</p>
            <Link to={m.path} className="btn btn-outline" style={{ display: 'inline-flex' }}>
              Open Module &rarr;
            </Link>
          </div>
        ))}
      </div>
    </div>
  );
}
