/**
 * Utility helper functions
 */

export function formatTimestamp(isoString) {
  if (!isoString) return '';
  const date = new Date(isoString);
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export function showStatusMessage(element, text, isError = false, duration = 3000) {
  element.textContent = text;
  element.className = `status-message ${isError ? 'error' : 'success'}`;
  setTimeout(() => {
    element.className = 'status-message';
  }, duration);
}

export function setStatus(text) {
  const statusEl = document.getElementById('status');
  if (statusEl) {
    statusEl.textContent = text;
  }
}

