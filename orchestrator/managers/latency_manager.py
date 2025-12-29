import time
from typing import Optional
from datetime import datetime
from core.event_bus import EventBus, Event
from core.logging import get_logger
from core.settings import get_settings
from orchestrator.events import EventType
from orchestrator.managers.base import BaseManager

logger = get_logger(__name__)

class LatencyTracker(BaseManager):
    """
    Tracks and logs latency metrics across the pipeline.
    Subscribes to all relevant events to measure E2E latency.
    """
    
    def __init__(self, event_bus: EventBus):
        settings = get_settings()
        self.enabled = settings.orchestrator.enable_latency_tracking
        
        # State
        self.speech_end_time: Optional[float] = None
        self.llm_request_time: Optional[float] = None
        self.first_token_time: Optional[float] = None
        
        super().__init__(event_bus)

    def _register_handlers(self):
        if not self.enabled:
            return
        self.event_bus.subscribe(EventType.TRANSCRIPT_FINAL.value, self.on_transcript_final)
        self.event_bus.subscribe(EventType.LLM_REQUEST.value, self.on_llm_request)
        self.event_bus.subscribe(EventType.LLM_TOKEN.value, self.on_llm_token)
        self.event_bus.subscribe(EventType.TTS_AUDIO_CHUNK.value, self.on_tts_audio)

    async def on_transcript_final(self, event: Event):
        self.speech_end_time = time.time()
        self._log(f"SPEECH_END: {event.data.get('text')[:30]}...")

    async def on_llm_request(self, event: Event):
        self.llm_request_time = time.time()
        self._log("LLM_REQUEST_SENT")

    async def on_llm_token(self, event: Event):
        if self.first_token_time is None:
            self.first_token_time = time.time()
            if self.llm_request_time:
                latency = self.first_token_time - self.llm_request_time
                self._log(f"LLM_FIRST_TOKEN: {latency:.4f}s")

    async def on_tts_audio(self, event: Event):
        # Reset for next turn if we haven't already
        if self.speech_end_time:
            e2e = time.time() - self.speech_end_time
            self._log(f"E2E_FIRST_AUDIO: {e2e:.4f}s")
            self.speech_end_time = None  # Clear to avoid logging every chunk
            self.first_token_time = None

    def _log(self, msg: str):
        if self.enabled:
            logger.info(f"[LATENCY] {msg}")

