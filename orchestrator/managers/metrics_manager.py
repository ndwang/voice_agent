import uuid
import time
from collections import deque
from typing import Dict, Optional
from core.metrics.models import RequestTimeline
from core.logging import get_logger
from orchestrator.events import EventType
from core.event_bus import EventBus, Event
from core.metrics.history import MetricsHistory

logger = get_logger(__name__)

class MetricsManager:
    """
    Tracks real-time performance metrics for a single-user agent.
    Focuses on the hot path from speech-end to audio-start.
    """
    def __init__(self, event_bus: EventBus, max_history: int = 100):
        self.event_bus = event_bus
        self.active: Optional[RequestTimeline] = None
        self.history: MetricsHistory = MetricsHistory(max_history)
        self._last_transcript_final_time: Optional[float] = None

    def start(self):
        """Initialize and subscribe to events"""
        self._register_handlers()
        logger.info("Metrics Manager initialized")

    def _register_handlers(self):
        # Hot Path Events
        self.event_bus.subscribe(EventType.TRANSCRIPT_FINAL.value, self.on_transcript_final)
        self.event_bus.subscribe(EventType.INPUT_RECEIVED.value, self.on_input_received)
        self.event_bus.subscribe(EventType.LLM_REQUEST.value, self.on_llm_start)
        self.event_bus.subscribe(EventType.LLM_TOKEN.value, self.on_llm_token)
        self.event_bus.subscribe(EventType.TTS_REQUEST.value, self.on_tts_start)
        self.event_bus.subscribe(EventType.TTS_AUDIO_CHUNK.value, self.on_audio_start)

        # Cleanup/Completion Events
        self.event_bus.subscribe(EventType.LLM_RESPONSE_DONE.value, self.on_llm_end)
        self.event_bus.subscribe(EventType.TURN_ENDED.value, self.on_turn_end)

        # Interruption Events
        self.event_bus.subscribe(EventType.VOICE_INTERRUPT.value, self.on_cancel)
        self.event_bus.subscribe(EventType.CRITICAL_INTERRUPT.value, self.on_cancel)
        # Keep LLM_CANCELLED for backward compatibility during migration
        self.event_bus.subscribe(EventType.LLM_CANCELLED.value, self.on_cancel)

    async def on_transcript_final(self, event: Event):
        """Store timestamp when voice transcript is finalized (before queuing)"""
        self._last_transcript_final_time = event.timestamp

    async def on_input_received(self, event: Event):
        """Triggered when any input is received (voice, bilibili, etc.)"""
        source = event.data.get("source", "unknown")

        # For voice inputs, include transcript_final timestamp to measure queue wait time
        transcript_final = None
        if source == "voice" and self._last_transcript_final_time is not None:
            transcript_final = self._last_transcript_final_time
            self._last_transcript_final_time = None  # Clear after use

        self.active = RequestTimeline(
            request_id=str(uuid.uuid4())[:8],
            turn_start=event.timestamp,
            transcript=event.data.get("text", ""),
            transcript_final=transcript_final
        )

    async def on_llm_start(self, event: Event):
        if self.active:
            self.active.llm_start = event.timestamp

    async def on_llm_token(self, event: Event):
        """Capture only the first token"""
        if self.active and self.active.llm_first_token is None:
            self.active.llm_first_token = event.timestamp

    async def on_tts_start(self, event: Event):
        """Capture only the first sentence sent to synthesis"""
        if self.active and self.active.tts_start is None:
            self.active.tts_start = event.timestamp

    async def on_audio_start(self, event: Event):
        """Capture only the first audio chunk that hits the player"""
        if self.active and self.active.audio_start is None:
            self.active.audio_start = event.timestamp
            
            # Print immediate results to logs
            m = self.active.compute()
            queue_info = f"Queue: {m.get('queue_wait_ms', 0):.0f}ms | " if 'queue_wait_ms' in m else ""
            logger.info(
                f"[METRICS] ID:{m['request_id']} | "
                f"{queue_info}"
                f"E2E: {m.get('e2e_latency_ms', 0):.0f}ms | "
                f"TTFT: {m.get('llm_ttft_ms', 0):.0f}ms | "
                f"1st-Sentence: {m.get('first_sentence_ms', 0):.0f}ms | "
                f"TTS-TTFA: {m.get('tts_ttfa_ms', 0):.0f}ms"
            )

    async def on_llm_end(self, event: Event):
        """Capture generation stats and finalize turn"""
        if self.active:
            self.active.llm_end = event.timestamp
            self.active.completion_tokens = event.data.get("completion_tokens", 0)

    async def on_turn_end(self, event: Event):
        """Capture turn stats and add to history"""
        if self.active:
            metrics = self.active.compute()
            if "e2e_latency_ms" in metrics:
                self.history.add(metrics)
            self.active = None

    async def on_cancel(self, event: Event):
        """Wipe tracking on interruption - we don't care about cancelled stats"""
        if self.active:
            logger.debug(f"Metrics tracking cancelled for {self.active.request_id}")
            self.active = None
