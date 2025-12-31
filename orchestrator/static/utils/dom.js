/**
 * DOM element references
 */

export const elements = {
  // Main UI elements
  historyPanel: document.getElementById('history-panel'),
  bilibiliPanel: document.getElementById('bilibili-panel'),
  bilibiliMessages: document.getElementById('bilibili-messages'),
  statusEl: document.getElementById('status'),

  // Buttons
  cancelBtn: document.getElementById('cancel-btn'),
  toggleListeningBtn: document.getElementById('toggle-listening-btn'),
  systemPromptBtn: document.getElementById('system-prompt-btn'),
  clearHistoryBtn: document.getElementById('clear-history-btn'),
  configBtn: document.getElementById('config-btn'),
  
  // Activity segments
  segments: {
    listening: document.getElementById('seg-listening'),
    transcribing: document.getElementById('seg-transcribing'),
    responding: document.getElementById('seg-responding'),
    executing_tools: document.getElementById('seg-executing_tools'),
    synthesizing: document.getElementById('seg-synthesizing'),
    playing: document.getElementById('seg-playing'),
  },
  
  // System Prompt Modal
  systemPromptModal: document.getElementById('system-prompt-modal'),
  modalClose: document.getElementById('modal-close'),
  modalCancel: document.getElementById('modal-cancel'),
  modalSave: document.getElementById('modal-save'),
  promptTextarea: document.getElementById('prompt-textarea'),
  promptFilePath: document.getElementById('prompt-file-path'),
  statusMessage: document.getElementById('status-message'),
  
  // Hotkey Modal
  hotkeyModal: document.getElementById('hotkey-modal'),
  hotkeyModalClose: document.getElementById('hotkey-modal-close'),
  hotkeyModalCancel: document.getElementById('hotkey-modal-cancel'),
  hotkeyModalSave: document.getElementById('hotkey-modal-save'),
  hotkeyInput: document.getElementById('hotkey-input'),
  hotkeyCaptureBtn: document.getElementById('hotkey-capture-btn'),
  currentHotkeyDisplay: document.getElementById('current-hotkey-display'),
  hotkeyStatusMessage: document.getElementById('hotkey-status-message'),
  
  // Config Modal
  configModal: document.getElementById('config-modal'),
  configModalClose: document.getElementById('config-modal-close'),
  configModalCancel: document.getElementById('config-modal-cancel'),
  configModalSave: document.getElementById('config-modal-save'),
  configAccordion: document.getElementById('config-accordion'),
  configStatusMessage: document.getElementById('config-status-message'),
};

