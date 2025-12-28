/**
 * Hotkey Configuration Modal
 */

import { elements } from '../utils/dom.js';
import { state } from '../utils/state.js';
import { apiCall } from '../utils/api.js';
import { showStatusMessage } from '../utils/helpers.js';

const KEY_MAP = {
  ' ': 'space',
  'Enter': 'enter',
  'Tab': 'tab',
  'Escape': 'esc',
  'Backspace': 'backspace',
  'Delete': 'delete',
  'ArrowUp': 'up',
  'ArrowDown': 'down',
  'ArrowLeft': 'left',
  'ArrowRight': 'right',
};

const MODIFIERS = ['ctrl', 'control', 'shift', 'alt', 'meta', 'cmd'];

function formatHotkeyString(keys) {
  const parts = [];
  if (keys.includes('ctrl') || keys.includes('control')) parts.push('Ctrl');
  if (keys.includes('shift')) parts.push('Shift');
  if (keys.includes('alt')) parts.push('Alt');
  if (keys.includes('meta') || keys.includes('cmd')) parts.push('Cmd');
  
  const mainKey = keys.find(k => !MODIFIERS.includes(k));
  if (mainKey) {
    parts.push(mainKey.toUpperCase());
  }
  
  return parts.join('+');
}

function parseHotkeyString(hotkeyStr) {
  return hotkeyStr.toLowerCase().replace(/\s+/g, '').replace(/\+/g, '+');
}

function captureHotkeyHandler(e) {
  e.preventDefault();
  e.stopPropagation();
  
  const keys = [];
  if (e.ctrlKey) keys.push('ctrl');
  if (e.shiftKey) keys.push('shift');
  if (e.altKey) keys.push('alt');
  if (e.metaKey) keys.push('meta');
  
  let mainKey = '';
  if (e.key.length === 1) {
    mainKey = e.key.toLowerCase();
  } else if (e.key.startsWith('F') && e.key.length <= 3) {
    mainKey = e.key.toLowerCase();
  } else {
    mainKey = KEY_MAP[e.key] || e.key.toLowerCase();
  }
  
  if (mainKey && !keys.includes(mainKey)) {
    keys.push(mainKey);
  }
  
  if (keys.length > 0) {
    state.capturedKeys = keys;
    const displayStr = formatHotkeyString(keys);
    elements.hotkeyInput.value = displayStr;
    
    setTimeout(() => {
      state.isCapturingHotkey = false;
      elements.hotkeyCaptureBtn.textContent = 'Capture';
      document.removeEventListener('keydown', captureHotkeyHandler);
    }, 500);
  }
}

export async function loadHotkey() {
  try {
    const data = await apiCall('/ui/hotkeys/toggle_listening');
    state.currentHotkey = data.hotkey || '';
    updateHotkeyDisplay();
  } catch (e) {
    console.error('Error loading hotkey:', e);
    elements.currentHotkeyDisplay.textContent = 'Error loading';
  }
}

export function updateHotkeyDisplay() {
  elements.currentHotkeyDisplay.textContent = state.currentHotkey || 'Not set';
  elements.hotkeyInput.value = state.currentHotkey || '';
}

function handleHotkeyCapture() {
  if (state.isCapturingHotkey) {
    state.isCapturingHotkey = false;
    elements.hotkeyCaptureBtn.textContent = 'Capture';
    elements.hotkeyInput.placeholder = 'Press keys...';
    document.removeEventListener('keydown', captureHotkeyHandler);
  } else {
    state.isCapturingHotkey = true;
    elements.hotkeyCaptureBtn.textContent = 'Stop';
    elements.hotkeyInput.value = '';
    elements.hotkeyInput.placeholder = 'Press your hotkey combination...';
    state.capturedKeys = [];
    document.addEventListener('keydown', captureHotkeyHandler, true);
  }
}

async function saveHotkey() {
  const hotkeyValue = elements.hotkeyInput.value.trim();
  if (!hotkeyValue) {
    showStatusMessage(elements.hotkeyStatusMessage, 'Hotkey cannot be empty', true);
    return;
  }

  const apiHotkey = parseHotkeyString(hotkeyValue);
  
  elements.hotkeyModalSave.disabled = true;
  try {
    await apiCall('/ui/hotkeys/toggle_listening', {
      method: 'POST',
      body: { hotkey: apiHotkey },
    });
    showStatusMessage(elements.hotkeyStatusMessage, 'Hotkey saved successfully!');
    await loadHotkey();
    setTimeout(closeHotkeyModal, 1000);
  } catch (e) {
    console.error('Error saving hotkey:', e);
    showStatusMessage(elements.hotkeyStatusMessage, e.message || 'Failed to save hotkey', true);
  } finally {
    elements.hotkeyModalSave.disabled = false;
  }
}

export function closeHotkeyModal() {
  elements.hotkeyModal.classList.remove('active');
  state.isCapturingHotkey = false;
  elements.hotkeyCaptureBtn.textContent = 'Capture';
  document.removeEventListener('keydown', captureHotkeyHandler);
  elements.hotkeyStatusMessage.className = 'status-message';
}

export function setupHotkeyModal() {
  elements.toggleListeningBtn.oncontextmenu = async (e) => {
    e.preventDefault();
    elements.hotkeyModal.classList.add('active');
    await loadHotkey();
  };
  elements.hotkeyModalClose.onclick = closeHotkeyModal;
  elements.hotkeyModalCancel.onclick = closeHotkeyModal;
  elements.hotkeyModalSave.onclick = saveHotkey;
  elements.hotkeyCaptureBtn.onclick = handleHotkeyCapture;
  elements.hotkeyModal.onclick = (e) => {
    if (e.target === elements.hotkeyModal) {
      closeHotkeyModal();
    }
  };
}

