/**
 * Live panel updates for current transcript and response
 */

import { elements } from '../utils/dom.js';
import { state } from '../utils/state.js';
import { fetchHistory } from './history.js';

export function updateCurrentTranscript() {
  const interimLine = state.sttInterim ? `\n[interim] ${state.sttInterim}` : '';
  elements.currentTranscript.textContent = `${state.currentTranscriptText}${interimLine}`.trim() || '';
}

export function updateCurrentResponse() {
  elements.currentResponse.textContent = state.currentResponseText || '';
}

export function clearLivePanel() {
  state.currentTranscriptText = '';
  state.sttInterim = '';
  state.currentResponseText = '';
  updateCurrentTranscript();
  updateCurrentResponse();
}

export function commitCurrentTurn() {
  if (state.currentTranscriptText.trim() && state.currentResponseText.trim()) {
    fetchHistory();
  }
  clearLivePanel();
}

