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
    synthesizing: false,
    playing: false,
  },
};

