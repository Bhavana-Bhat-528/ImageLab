import React, { useState, useEffect } from 'react';
import { NavLink } from 'react-router-dom';
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
      <div className="brand-section">
        <div className="brand-logo">IL</div>
        <div>
          <div className="brand-title">ImageLab</div>
          <div className="brand-subtitle">Digital Image Processing Lab</div>
        </div>
      </div>

      <nav className="nav-links">
        <NavLink to="/" end className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          Overview
        </NavLink>
        <NavLink to="/enhancement" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          1. Enhancement
        </NavLink>
        <NavLink to="/boundary" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          2. Boundary Extraction
        </NavLink>
        <NavLink to="/watershed" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          3. Watershed
        </NavLink>
        <NavLink to="/fourier" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          4. Fourier Descriptors
        </NavLink>
      </nav>

      <div className="server-status" title={backendStatus === 'online' ? 'FastAPI Backend Connected' : 'FastAPI Backend Disconnected'}>
        <span className={`status-dot ${backendStatus === 'online' ? 'online' : 'offline'}`}></span>
        <span>API: {backendStatus === 'online' ? 'Connected (8000)' : 'Offline'}</span>
      </div>
    </header>
  );
}
