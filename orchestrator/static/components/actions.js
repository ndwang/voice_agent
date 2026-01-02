/**
 * Action handlers for UI buttons
 */

import { elements } from '../utils/dom.js';
import { setStatus } from '../utils/helpers.js';
import { apiCall } from '../utils/api.js';
import { fetchHistory } from './history.js';

export async function handleCancel() {
  elements.cancelBtn.disabled = true;
  setStatus('Cancelling…');
  try {
    await apiCall('/ui/cancel', { method: 'POST' });
  } finally {
    elements.cancelBtn.disabled = false;
  }
}

export async function handleClearHistory() {
  if (!confirm('Are you sure you want to clear the conversation history?')) {
    return;
  }
  
  elements.clearHistoryBtn.disabled = true;
  setStatus('Clearing history…');
  try {
    await apiCall('/ui/history/clear', { method: 'POST' });
    setStatus('History cleared');
    await fetchHistory();
  } catch (e) {
    console.error('Error clearing history:', e);
    setStatus('Error clearing history');
  } finally {
    elements.clearHistoryBtn.disabled = false;
  }
}

export async function handleToggleListening() {
  elements.toggleListeningBtn.disabled = true;
  try {
    await apiCall('/ui/listening/toggle', { method: 'POST' });
  } catch (e) {
    console.error('Error toggling listening:', e);
    setStatus('Error toggling listening');
  } finally {
    elements.toggleListeningBtn.disabled = false;
  }
}

export async function handleToggleBilibiliDanmaku() {
  elements.toggleBilibiliDanmakuBtn.disabled = true;
  try {
    await apiCall('/ui/bilibili/danmaku/toggle', { method: 'POST' });
  } catch (e) {
    console.error('Error toggling bilibili danmaku:', e);
    setStatus('Error toggling bilibili danmaku');
  } finally {
    elements.toggleBilibiliDanmakuBtn.disabled = false;
  }
}

export async function handleToggleBilibiliSuperChat() {
  elements.toggleBilibiliSuperChatBtn.disabled = true;
  try {
    await apiCall('/ui/bilibili/superchat/toggle', { method: 'POST' });
  } catch (e) {
    console.error('Error toggling bilibili superchat:', e);
    setStatus('Error toggling bilibili superchat');
  } finally {
    elements.toggleBilibiliSuperChatBtn.disabled = false;
  }
}

