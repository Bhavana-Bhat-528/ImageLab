import React from 'react';

export default function PageHeader({ moduleNumber, title, description, tags = [], accentColor = 'blue' }) {
  return (
    <div className={`page-header-banner page-header-${accentColor}`}>
      <div className="header-meta">
        {moduleNumber && (
          <span className="module-badge">
            Module {moduleNumber}
          </span>
        )}
        <div className="header-tags">
          {tags.map((tag, i) => (
            <span key={i} className="tech-tag">
              {tag}
            </span>
          ))}
        </div>
      </div>
      <h1 className="header-title">{title}</h1>
      <p className="header-description">{description}</p>
    </div>
  );
}
