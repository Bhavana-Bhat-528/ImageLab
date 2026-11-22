import React, { useState, useEffect } from 'react';
import { NavLink, Link } from 'react-router-dom';
import { checkBackendHealth } from '../services/api';

export default function Navbar() {
  const [backendStatus, setBackendStatus] = useState('checking');

  useEffect(() => {
    let isMounted = true;
    checkBackendHealth()
      .then(() => {
        if (isMounted) setBackendStatus('online');
      })
      .catch(() => {
        if (isMounted) setBackendStatus('offline');
      });
    return () => { isMounted = false; };
  }, []);

  return (
    <header className="navbar">
      <div className="navbar-left">
        <Link to="/" className="brand-link">
          <div className="brand-logo-badge">IL</div>
          <div className="brand-text">
            <span className="brand-title">ImageLab</span>
            <span className="brand-tagline">Interactive Digital Image Processing Laboratory</span>
          </div>
        </Link>
      </div>

      <nav className="navbar-nav" aria-label="Main Navigation">
        <NavLink to="/" end className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          Overview
        </NavLink>
        <NavLink to="/enhancement" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          Automatic Enhancement
        </NavLink>
        <NavLink to="/boundary-extraction" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          Boundary Extraction
        </NavLink>
        <NavLink to="/watershed" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          Watershed Segmentation
        </NavLink>
        <NavLink to="/shape-recognition" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          Shape Recognition
        </NavLink>
      </nav>

      <div className="navbar-right">
        <div 
          className={`server-status-pill ${backendStatus}`} 
          title={backendStatus === 'online' ? 'Python FastAPI REST API Online (Port 8000)' : 'Backend API Disconnected'}
        >
          <span className="status-ping-dot"></span>
          <span className="status-label">
            {backendStatus === 'online' ? 'FastAPI 8000' : 'Offline'}
          </span>
        </div>
      </div>
    </header>
  );
}
