/**
 * System Prompt Modal
 */

import { elements } from '../utils/dom.js';
import { apiCall } from '../utils/api.js';
import { showStatusMessage } from '../utils/helpers.js';

export async function loadSystemPrompt() {
  try {
    const data = await apiCall('/ui/system-prompt');
    elements.promptTextarea.value = data.prompt || '';
    elements.promptFilePath.textContent = data.file_path || 'Unknown';
  } catch (e) {
    console.error('Error loading system prompt:', e);
    showStatusMessage(elements.statusMessage, 'Failed to load system prompt', true);
  }
}

export async function saveSystemPrompt() {
  const prompt = elements.promptTextarea.value.trim();
  if (!prompt) {
    showStatusMessage(elements.statusMessage, 'Prompt cannot be empty', true);
    return;
  }

  elements.modalSave.disabled = true;
  try {
    await apiCall('/ui/system-prompt', {
      method: 'POST',
      body: { prompt },
    });
    showStatusMessage(elements.statusMessage, 'System prompt saved successfully!');
    await loadSystemPrompt();
  } catch (e) {
    console.error('Error saving system prompt:', e);
    showStatusMessage(elements.statusMessage, e.message || 'Failed to save system prompt', true);
  } finally {
    elements.modalSave.disabled = false;
  }
}

export function closeSystemPromptModal() {
  elements.systemPromptModal.classList.remove('active');
  elements.statusMessage.className = 'status-message';
}

export function setupSystemPromptModal() {
  elements.systemPromptBtn.onclick = async () => {
    elements.systemPromptModal.classList.add('active');
    await loadSystemPrompt();
  };
  elements.modalClose.onclick = closeSystemPromptModal;
  elements.modalCancel.onclick = closeSystemPromptModal;
  elements.modalSave.onclick = saveSystemPrompt;
  
  // Close on Escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && elements.systemPromptModal.classList.contains('active')) {
      closeSystemPromptModal();
    }
  });
  
  // Close on outside click
  elements.systemPromptModal.onclick = (e) => {
    if (e.target === elements.systemPromptModal) {
      closeSystemPromptModal();
    }
  };
}

