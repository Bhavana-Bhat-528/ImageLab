/**
 * Centralized API client for communicating with the FastAPI backend.
 * Provides helper methods for health check, module processing, and sample image fetching.
 * Features built-in timeout handling to prevent indefinite loading states.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';
const DEFAULT_TIMEOUT_MS = 60000; // 60 seconds

async function fetchWithTimeout(url, options = {}, timeoutMs = DEFAULT_TIMEOUT_MS) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal
    });
    clearTimeout(timeoutId);
    return response;
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === 'AbortError') {
      throw new Error(`Request timed out after ${timeoutMs / 1000} seconds. Please verify the server is running and try again.`);
    }
    throw err;
  }
}

export async function checkBackendHealth() {
  const response = await fetchWithTimeout(`${API_BASE_URL}/health`, {}, 5000);
  if (!response.ok) {
    throw new Error(`Health check failed with status: ${response.status}`);
  }
  return response.json();
}

export async function fetchSampleList(moduleName) {
  const response = await fetchWithTimeout(`${API_BASE_URL}/samples/${moduleName}`, {}, 10000);
  if (!response.ok) {
    throw new Error(`Failed to fetch samples for ${moduleName}`);
  }
  return response.json();
}

export function getSampleImageUrl(moduleName, filename) {
  return `${API_BASE_URL}/samples/${moduleName}/${filename}`;
}

export async function processEnhancement(formData) {
  const response = await fetchWithTimeout(`${API_BASE_URL}/enhancement/process`, {
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
  const response = await fetchWithTimeout(`${API_BASE_URL}/boundary/process`, {
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
  const response = await fetchWithTimeout(`${API_BASE_URL}/watershed/process`, {
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
  const response = await fetchWithTimeout(`${API_BASE_URL}/upload/validate`, {
    method: 'POST',
    body: formData,
  }, 10000);
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Upload validation failed (${response.status})`);
  }
  return response.json();
}

export async function processFourier(formData) {
  const response = await fetchWithTimeout(`${API_BASE_URL}/fourier/process`, {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Fourier processing failed (${response.status})`);
  }
  return response.json();
}
