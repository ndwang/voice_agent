"""
Constants used across orchestrator components.

UI event names and regex patterns for text processing.
"""
import re

# UI Event Names
# These are string constants for UI-related events (not in EventType enum)
UI_ACTIVITY = "ui.activity"
UI_HISTORY_UPDATED = "ui.history_updated"
UI_LISTENING_STATE_CHANGED = "ui.listening_state_changed"

# Text Processing Patterns
# Sentence detection pattern - matches punctuation followed by space/newline or end of string
SENTENCE_END_PATTERN = re.compile(r'[!?。！？](?=[\s\n]|$)')

# Thinking tag patterns for filtering reasoning/reasoning content
# Note: These match <think> tags used by some LLM models
THINKING_TAG_COMPLETE_PATTERN = re.compile(r'<think>.*?</think>', re.DOTALL)
THINKING_TAG_INCOMPLETE_PATTERN = re.compile(r'<think>.*$', re.DOTALL)

# Thinking tag strings
THINKING_TAG_OPEN = "<think>"
THINKING_TAG_CLOSE = "</think>"

