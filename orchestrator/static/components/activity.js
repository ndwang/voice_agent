/**
 * Activity rendering component
 */

import { elements } from '../utils/dom.js';
import { state } from '../utils/state.js';

export function renderActivity(newState) {
  state.activityState = { ...state.activityState, ...newState };
  Object.entries(state.activityState).forEach(([key, isOn]) => {
    if (elements.segments[key]) {
      elements.segments[key].classList.toggle('active', !!isOn);
    }
  });
}

export function updateToggleListeningButton() {
  const isListening = state.activityState.listening;
  elements.toggleListeningBtn.textContent = isListening ? 'Stop Listening' : 'Start Listening';
  elements.toggleListeningBtn.style.background = isListening ? '#1e40af' : '#991b1b';
  elements.toggleListeningBtn.style.borderColor = isListening ? '#3b82f6' : '#ef4444';
}

