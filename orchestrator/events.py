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
    BILIBILI_DANMAKU_STATE_CHANGED = "bilibili.danmaku_state_changed"
    BILIBILI_SUPERCHAT_STATE_CHANGED = "bilibili.superchat_state_changed"

    # LLM
    LLM_REQUEST = "llm.request"
    LLM_TOKEN = "llm.token"
    LLM_RESPONSE_DONE = "llm.response_done"
    LLM_CANCELLED = "llm.cancelled"  # DEPRECATED: Use VOICE_INTERRUPT or CRITICAL_INTERRUPT

    # Interrupts
    VOICE_INTERRUPT = "interrupt.voice"  # Voice detected during activity - cancel & expect voice input
    CRITICAL_INTERRUPT = "interrupt.critical"  # P0 item arrived - cancel & process critical item

    # Tool execution
    TOOL_CALL_REQUESTED = "tool.call_requested"
    TOOL_EXECUTING = "tool.executing"
    TOOL_RESULT = "tool.result"
    TOOL_ERROR = "tool.error"
    TOOL_INTERPRETATION_REQUEST = "tool.interpretation_request"

    # TTS
    TTS_REQUEST = "tts.request"
    TTS_AUDIO_CHUNK = "tts.audio_chunk"
    
    # Subtitles
    SUBTITLE_REQUEST = "subtitle.request"

