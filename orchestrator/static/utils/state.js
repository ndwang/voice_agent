/**
 * Application state management
 */

export const state = {
  currentTranscriptText: '',
  currentResponseText: '',
  sttInterim: '',
  currentHotkey: '',
  isCapturingHotkey: false,
  capturedKeys: [],
  currentConfig: {},
  activityState: {
    listening: true,
    transcribing: false,
    responding: false,
    executing_tools: false,
    synthesizing: false,
    playing: false,
  },
};

