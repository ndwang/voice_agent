from enum import Enum, auto

class EventType(Enum):
    # System
    STARTUP = "system.startup"
    SHUTDOWN = "system.shutdown"
    STATE_CHANGED = "system.state_changed"
    HISTORY_UPDATED = "system.history_updated"
    LISTENING_STATE_CHANGED = "system.listening_state_changed"
    TURN_ENDED = "system.turn_ended"
    
    # Input
    SPEECH_START = "input.speech_start"
    TRANSCRIPT_INTERIM = "input.transcript.interim"
    TRANSCRIPT_FINAL = "input.transcript.final"
    INPUT_RECEIVED = "input.received"  # From queue consumer to interaction manager
    CRITICAL_INPUT = "input.critical"  # P0 message that should trigger interruption

    # Bilibili
    BILIBILI_DANMAKU = "bilibili.danmaku"
    BILIBILI_SUPERCHAT = "bilibili.superchat"

    # LLM
    LLM_REQUEST = "llm.request"
    LLM_TOKEN = "llm.token"
    LLM_RESPONSE_DONE = "llm.response_done"
    LLM_CANCELLED = "llm.cancelled"
    
    # TTS
    TTS_REQUEST = "tts.request"
    TTS_AUDIO_CHUNK = "tts.audio_chunk"
    
    # Subtitles
    SUBTITLE_REQUEST = "subtitle.request"

