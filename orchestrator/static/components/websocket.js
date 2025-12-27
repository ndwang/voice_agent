/**
 * WebSocket connection and event handling
 */

import { state } from '../utils/state.js';
import { setStatus } from '../utils/helpers.js';
import { renderActivity, updateToggleListeningButton } from './activity.js';
import { updateCurrentTranscript, updateCurrentResponse, clearLivePanel, commitCurrentTurn } from './live-panel.js';
import { fetchHistory } from './history.js';
import { updateHotkeyDisplay } from '../modals/hotkey.js';

function handleWebSocketEvent(data) {
  switch (data.event) {
    case 'stt':
      if (data.stage === 'final') {
        state.currentTranscriptText = data.text || '';
        state.currentResponseText = '';
        state.sttInterim = '';
        updateCurrentResponse();
      } else if (data.stage === 'interim') {
        state.sttInterim = data.text || '';
      }
      updateCurrentTranscript();
      break;
      
    case 'llm_token':
      state.currentResponseText += data.token || '';
      updateCurrentResponse();
      break;
      
    case 'llm_done':
      setStatus('LLM complete');
      commitCurrentTurn();
      break;
      
    case 'llm_cancelled':
      setStatus('LLM cancelled');
      clearLivePanel();
      break;
      
    case 'cancelled':
      setStatus('Speech cancelled');
      clearLivePanel();
      break;
      
    case 'activity':
      renderActivity(data.state || {});
      updateToggleListeningButton();
      break;
      
    case 'listening_state_changed':
      if (data.enabled !== undefined) {
        state.activityState.listening = data.enabled;
      }
      updateToggleListeningButton();
      break;
      
    case 'history_updated':
      fetchHistory();
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
  }
}

export function connect() {
  const wsProto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${wsProto}://${location.host}/ui/events`);

  ws.onopen = () => {
    setStatus('Connected');
    fetchHistory();
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

