from dataclasses import dataclass
from typing import Optional, Dict
import time

@dataclass
class RequestTimeline:
    request_id: str
    transcript: str

    # Timestamps (using time.perf_counter for precision)
    turn_start: float          # input.received (when input is dequeued and processed)
    transcript_final: Optional[float] = None  # transcript.final (voice only - when transcript ready)
    llm_start: Optional[float] = None     # llm.request
    llm_first_token: Optional[float] = None # llm.token (first)
    tts_start: Optional[float] = None     # tts.request (first sentence)
    audio_start: Optional[float] = None   # tts.audio_chunk (first sound)
    llm_end: Optional[float] = None       # llm.response_done

    completion_tokens: int = 0

    def compute(self) -> Dict:
        """Calculate durations in milliseconds and return as a dictionary"""
        m = {
            "request_id": self.request_id,
            "transcript": self.transcript[:100]  # Limit transcript length
        }

        # 0. Queue Wait Time (Voice only - TRANSCRIPT_FINAL to INPUT_RECEIVED)
        if self.transcript_final:
            m['queue_wait_ms'] = (self.turn_start - self.transcript_final) * 1000

        # 1. System Overhead (Internal logic before calling LLM)
        if self.llm_start:
            m['overhead_ms'] = (self.llm_start - self.turn_start) * 1000
            
        # 2. LLM Time to First Token (TTFT)
        if self.llm_start and self.llm_first_token:
            m['llm_ttft_ms'] = (self.llm_first_token - self.llm_start) * 1000
            
        # 3. First Sentence Latency (LLM Req -> First TTS Req)
        if self.llm_start and self.tts_start:
            m['first_sentence_ms'] = (self.tts_start - self.llm_start) * 1000
            
        # 4. TTS Synthesis Latency (First TTS Req -> First Audio)
        if self.tts_start and self.audio_start:
            m['tts_ttfa_ms'] = (self.audio_start - self.tts_start) * 1000
            
        # 5. Total End-to-End Latency (Silence period)
        if self.audio_start:
            m['e2e_latency_ms'] = (self.audio_start - self.turn_start) * 1000

        # 6. LLM Throughput
        if self.llm_first_token and self.llm_end and self.completion_tokens > 0:
            duration = self.llm_end - self.llm_first_token
            m['tokens_per_sec'] = self.completion_tokens / duration if duration > 0 else 0
            
        return m