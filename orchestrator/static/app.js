/**
 * Main application entry point
 */

import { elements } from './utils/dom.js';
import { handleCancel, handleClearHistory, handleToggleListening } from './components/actions.js';
import { connect } from './components/websocket.js';
import { setupSystemPromptModal } from './modals/system-prompt.js';
import { setupHotkeyModal, loadHotkey } from './modals/hotkey.js';
import { setupConfigModal } from './modals/config.js';

function setupEventListeners() {
  // Action buttons
  elements.cancelBtn.onclick = handleCancel;
  elements.clearHistoryBtn.onclick = handleClearHistory;
  elements.toggleListeningBtn.onclick = handleToggleListening;
  
  // Modals
  setupSystemPromptModal();
  setupHotkeyModal();
  setupConfigModal();
}

async function init() {
  setupEventListeners();
  await loadHotkey();
  connect();
}

// Start the application when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
