/**
 * API utilities for making HTTP requests
 */

export async function apiCall(url, options = {}) {
  const { body, headers = {}, ...restOptions } = options;
  const fetchOptions = {
    headers: { 'Content-Type': 'application/json', ...headers },
    ...restOptions,
  };
  
  if (body && typeof body === 'object') {
    fetchOptions.body = JSON.stringify(body);
  } else if (body) {
    fetchOptions.body = body;
  }
  
  const response = await fetch(url, fetchOptions);
  
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error || data.detail || `Failed: ${response.statusText}`);
  }
  
  return response.json();
}

