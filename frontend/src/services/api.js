/**
 * Centralized API client for communicating with the FastAPI backend.
 * Provides helper methods for health check, module processing, and sample image fetching.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

export async function checkBackendHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error(`Health check failed with status: ${response.status}`);
  }
  return response.json();
}

export async function fetchSampleList(moduleName) {
  const response = await fetch(`${API_BASE_URL}/samples/${moduleName}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch samples for ${moduleName}`);
  }
  return response.json();
}

export function getSampleImageUrl(moduleName, filename) {
  return `${API_BASE_URL}/samples/${moduleName}/${filename}`;
}

export async function processEnhancement(formData) {
  const response = await fetch(`${API_BASE_URL}/enhancement/process`, {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Enhancement failed (${response.status})`);
  }
  return response.json();
}

export async function processBoundary(formData) {
  const response = await fetch(`${API_BASE_URL}/boundary/process`, {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Boundary extraction failed (${response.status})`);
  }
  return response.json();
}

export async function processWatershed(formData) {
  const response = await fetch(`${API_BASE_URL}/watershed/process`, {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Watershed segmentation failed (${response.status})`);
  }
  return response.json();
}

export async function validateUploadedImage(file) {
  const formData = new FormData();
  formData.append('file', file);
  const response = await fetch(`${API_BASE_URL}/upload/validate`, {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Upload validation failed (${response.status})`);
  }
  return response.json();
}

export async function processFourier(formData) {
  const response = await fetch(`${API_BASE_URL}/fourier/process`, {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Fourier processing failed (${response.status})`);
  }
  return response.json();
}
