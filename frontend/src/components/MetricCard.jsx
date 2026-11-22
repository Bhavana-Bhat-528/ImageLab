import React from 'react';

export default function MetricCard({ label, value, unit = '', description = '' }) {
  return (
    <div style={{
      background: 'rgba(15, 23, 42, 0.5)',
      border: '1px solid var(--border-color)',
      borderRadius: 'var(--radius-sm)',
      padding: '0.85rem 1rem',
      display: 'flex',
      flexDirection: 'column',
      gap: '0.25rem'
    }}>
      <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--text-muted)', letterSpacing: '0.05em' }}>
        {label}
      </div>
      <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#fff' }}>
        {value} <span style={{ fontSize: '0.85rem', fontWeight: 500, color: 'var(--accent-cyan)' }}>{unit}</span>
      </div>
      {description && (
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          {description}
        </div>
      )}
    </div>
  );
}
