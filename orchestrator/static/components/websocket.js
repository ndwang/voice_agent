/**
 * WebSocket connection and event handling
 */

import { state } from '../utils/state.js';
import { setStatus } from '../utils/helpers.js';
import {
  renderActivity,
  updateToggleListeningButton,
  updateToggleBilibiliDanmakuButton,
  updateToggleBilibiliSuperChatButton
} from './activity.js';
import { fetchHistory, updateStreamingMessage, clearStreamingMessages } from './history.js';
import { updateHotkeyDisplay } from '../modals/hotkey.js';
import { fetchBilibiliChat, addBilibiliMessage } from './bilibili.js';

// Helper to clear state after history fetch completes
async function fetchHistoryAndClearState() {
  await fetchHistory();
  // Clear state after transition completes
  state.currentTranscriptText = '';
  state.currentResponseText = '';
  state.sttInterim = '';
}

function handleWebSocketEvent(data) {
  switch (data.event) {
    case 'stt':
      if (data.stage === 'final') {
        state.currentTranscriptText = data.text || '';
        state.currentResponseText = '';
        state.sttInterim = '';
        updateStreamingMessage('user', state.currentTranscriptText);
      } else if (data.stage === 'interim') {
        state.sttInterim = data.text || '';
        const interimLine = state.currentTranscriptText 
          ? `${state.currentTranscriptText}\n[interim] ${state.sttInterim}`
          : `[interim] ${state.sttInterim}`;
        updateStreamingMessage('user', interimLine);
      }
      break;
      
    case 'llm_token':
      state.currentResponseText += data.token || '';
      updateStreamingMessage('assistant', state.currentResponseText);
      break;
      
    case 'llm_done':
      setStatus('LLM complete');
      // Don't clear state yet - renderHistory needs it for smooth transition
      // State will be cleared after transition completes
      // Small delay to ensure backend has saved to history
      setTimeout(() => {
        fetchHistoryAndClearState();
      }, 50);
      break;
      
    case 'llm_cancelled':
      setStatus('LLM cancelled');
      state.currentTranscriptText = '';
      state.currentResponseText = '';
      state.sttInterim = '';
      clearStreamingMessages();
      fetchHistory();
      break;
      
    case 'cancelled':
      setStatus('Speech cancelled');
      state.currentTranscriptText = '';
      state.currentResponseText = '';
      state.sttInterim = '';
      clearStreamingMessages();
      fetchHistory();
      break;
      
    case 'activity':
      renderActivity(data.state || {});
      updateToggleListeningButton();
      break;
      
    case 'listening_state_changed':
      if (data.enabled !== undefined) {
        renderActivity({ listening: data.enabled });
      }
      updateToggleListeningButton();
      break;
      
    case 'history_updated':
      // Only fetch if we're not already transitioning (llm_done handles its own fetch)
      // This prevents double-fetching when both llm_done and history_updated fire
      if (!state.currentResponseText && !state.currentTranscriptText) {
        fetchHistory();
      }
      break;
      
    case 'hotkey_registered':
      if (data.hotkey_id === 'toggle_listening') {
        state.currentHotkey = data.hotkey || '';
        updateHotkeyDisplay();
      }
      break;
      
    case 'hotkey_unregistered':
      if (data.hotkey_id === 'toggle_listening') {
        state.currentHotkey = '';
        updateHotkeyDisplay();
      }
      break;

    case 'bilibili_danmaku':
      addBilibiliMessage(data.message, false);
      break;

    case 'bilibili_superchat':
      addBilibiliMessage(data.message, true);
      break;

    case 'bilibili_danmaku_state_changed':
      if (data.enabled !== undefined) {
        state.bilibiliDanmakuEnabled = data.enabled;
      }
      updateToggleBilibiliDanmakuButton();
      break;

    case 'bilibili_superchat_state_changed':
      if (data.enabled !== undefined) {
        state.bilibiliSuperChatEnabled = data.enabled;
      }
      updateToggleBilibiliSuperChatButton();
      break;
  }
}

export function connect() {
  const wsProto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${wsProto}://${location.host}/ui/events`);

  ws.onopen = () => {
    setStatus('Connected');
    fetchHistory();
    fetchBilibiliChat();
  };

  ws.onclose = () => {
    setStatus('Disconnected – retrying…');
    renderActivity({
      transcribing: false,
      responding: false,
      synthesizing: false,
      playing: false,
    });
    setTimeout(connect, 1500);
  };

  ws.onerror = () => setStatus('Error – retrying…');

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleWebSocketEvent(data);
    } catch (e) {
      console.error('Bad message', e);
    }
  };
}

