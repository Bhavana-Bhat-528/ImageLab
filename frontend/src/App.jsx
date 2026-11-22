import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import EnhancementPage from './pages/EnhancementPage';
import BoundaryExtractionPage from './pages/BoundaryExtractionPage';
import WatershedPage from './pages/WatershedPage';
import ShapeRecognitionPage from './pages/ShapeRecognitionPage';
import './styles/App.css';

export default function App() {
  return (
    <Router>
      <div className="app-container">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/enhancement" element={<EnhancementPage />} />
            <Route path="/boundary-extraction" element={<BoundaryExtractionPage />} />
            <Route path="/watershed" element={<WatershedPage />} />
            <Route path="/shape-recognition" element={<ShapeRecognitionPage />} />
            
            {/* Backward compatibility redirects */}
            <Route path="/boundary" element={<Navigate to="/boundary-extraction" replace />} />
            <Route path="/fourier" element={<Navigate to="/shape-recognition" replace />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}
