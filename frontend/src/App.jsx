import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import EnhancementPage from './pages/EnhancementPage';
import BoundaryPage from './pages/BoundaryPage';
import WatershedPage from './pages/WatershedPage';
import FourierPage from './pages/FourierPage';
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
            <Route path="/boundary" element={<BoundaryPage />} />
            <Route path="/watershed" element={<WatershedPage />} />
            <Route path="/fourier" element={<FourierPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}
