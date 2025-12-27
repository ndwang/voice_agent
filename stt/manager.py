import asyncio
import json
import numpy as np
import time
from typing import Dict, Set, Optional, Any
from dataclasses import dataclass, field
from fastapi import WebSocket, WebSocketDisconnect
from core.logging import get_logger
from core.config import get_config
from stt.providers import FasterWhisperProvider, FunASRProvider

logger = get_logger(__name__)


@dataclass
class ClientStreamingState:
    """State for a single client's streaming session."""
    asr_cache: Dict[str, Any] = field(default_factory=dict)
    vad_cache: Dict[str, Any] = field(default_factory=dict)
    audio_buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    asr_chunk_buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))  # Buffer for ASR chunks
    vad_chunk_buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))  # Buffer for VAD chunks
    silence_start_time: Optional[float] = None  # Timestamp when silence started (ms)
    current_transcript: str = ""  # Accumulated transcript text
    is_speaking: bool = False  # Whether currently detecting speech
    last_interim_time: float = 0.0  # Last time interim transcript was sent (for batch mode)


class STTManager:
    """
    Manages STT transcription, client connections, and audio buffering.
    Decoupled from FastAPI WebSocket details.
    """

    def __init__(self):
        # Configuration
        self.language_code = get_config("stt", "language_code", default="zh")
        self.sample_rate = get_config("stt", "sample_rate", default=16000)
        self.interim_min_samples = get_config("stt", "interim_transcript_min_samples", default=int(0.3 * 16000))

        # Provider setup
        self.provider_name = get_config("stt", "provider", default="faster-whisper")
        self.provider = self._load_provider()
        
        # Check if streaming mode is enabled
        self.streaming_enabled = False
        if self.provider_name == "funasr" and isinstance(self.provider, FunASRProvider):
            self.streaming_enabled = self.provider.is_streaming

        # State
        self.connected_clients: Set[WebSocket] = set()
        self.clients_lock = asyncio.Lock()
        
        # Per-client streaming state
        self.client_states: Dict[WebSocket, ClientStreamingState] = {}
        self.client_states_lock = asyncio.Lock()
        
        # Silence threshold (use audio config or streaming config)
        if self.streaming_enabled:
            self.silence_threshold_ms = get_config(
                "stt", "providers", "funasr", "streaming", "silence_threshold_ms",
                default=get_config("audio", "silence_threshold_ms", default=500)
            )
            # Calculate chunk sizes for streaming
            streaming_config = get_config("stt", "providers", "funasr", "streaming", default={})
            chunk_size = streaming_config.get("chunk_size", [0, 10, 5])
            vad_chunk_size_ms = streaming_config.get("vad_chunk_size_ms", 200)
            # ASR chunk size: chunk_size[1] * 960 samples = 600ms at 16kHz
            self.asr_chunk_samples = int(chunk_size[1] * 960) if len(chunk_size) > 1 else 9600
            # VAD chunk size: 200ms = 3200 samples at 16kHz
            self.vad_chunk_samples = int(vad_chunk_size_ms * self.sample_rate / 1000)
            logger.info(f"Streaming chunk sizes: ASR={self.asr_chunk_samples} samples ({self.asr_chunk_samples/self.sample_rate*1000:.0f}ms), VAD={self.vad_chunk_samples} samples ({vad_chunk_size_ms}ms)")
        else:
            self.silence_threshold_ms = get_config("audio", "silence_threshold_ms", default=500)
            self.asr_chunk_samples = 0
            self.vad_chunk_samples = 0
    
    def _load_provider(self):
        logger.info(f"Initializing STT provider: {self.provider_name}...")
        try:
            if self.provider_name == "faster-whisper":
                model_path = get_config("stt", "providers", "faster-whisper", "model_path", default="faster-whisper-small")
                device = get_config("stt", "providers", "faster-whisper", "device", default=None)
                compute_type = get_config("stt", "providers", "faster-whisper", "compute_type", default=None)
                
                return FasterWhisperProvider(
                    model_path=model_path,
                    device=device,
                    compute_type=compute_type
                )
            elif self.provider_name == "funasr":
                model_name = get_config("stt", "providers", "funasr", "model_name", default="FunAudioLLM/Fun-ASR-Nano-2512")
                vad_model = get_config("stt", "providers", "funasr", "vad_model", default="fsmn-vad")
                vad_kwargs = get_config("stt", "providers", "funasr", "vad_kwargs", default={"max_single_segment_time": 30000})
                punc_model = get_config("stt", "providers", "funasr", "punc_model", default=None)
                device = get_config("stt", "providers", "funasr", "device", default=None)
                batch_size_s = get_config("stt", "providers", "funasr", "batch_size_s", default=0)
                streaming_config = get_config("stt", "providers", "funasr", "streaming", default=None)
                
                return FunASRProvider(
                    model_name=model_name,
                    vad_model=vad_model,
                    vad_kwargs=vad_kwargs,
                    punc_model=punc_model,
                    device=device,
                    batch_size_s=batch_size_s,
                    streaming_config=streaming_config
                )
            else:
                raise ValueError(f"Unknown STT provider: {self.provider_name}")
        except Exception as e:
            logger.error(f"Error loading STT provider: {e}", exc_info=True)
            raise

    async def add_client(self, websocket: WebSocket):
        """Register a new client."""
        async with self.clients_lock:
            self.connected_clients.add(websocket)
        
        # Initialize streaming state if streaming is enabled
        if self.streaming_enabled:
            await self._initialize_client_streaming(websocket)
        
        logger.info(f"Client connected. Total: {len(self.connected_clients)}")

    async def remove_client(self, websocket: WebSocket):
        """Unregister a client."""
        # Clean up streaming state
        if self.streaming_enabled:
            await self._cleanup_client_streaming(websocket)
        
        async with self.clients_lock:
            self.connected_clients.discard(websocket)
        logger.info(f"Client disconnected. Total: {len(self.connected_clients)}")
    
    async def _initialize_client_streaming(self, websocket: WebSocket):
        """Initialize streaming state for a client."""
        if not isinstance(self.provider, FunASRProvider):
            return
        
        try:
            cache_dict = self.provider.initialize_streaming()
            async with self.client_states_lock:
                self.client_states[websocket] = ClientStreamingState(
                    asr_cache=cache_dict["asr_cache"],
                    vad_cache=cache_dict["vad_cache"]
                )
        except Exception as e:
            logger.error(f"Error initializing streaming for client: {e}", exc_info=True)
    
    async def _cleanup_client_streaming(self, websocket: WebSocket):
        """Clean up streaming state for a client."""
        async with self.client_states_lock:
            self.client_states.pop(websocket, None)

    def _should_filter_transcript(self, text: str) -> bool:
        """
        Check if a transcript should be filtered out (just "嗯。").
        
        Args:
            text: The transcript text to check
            
        Returns:
            True if the transcript should be filtered out, False otherwise
        """
        if not text:
            return True
        
        # Just filter out "嗯。"
        if text.strip() == "嗯。":
            logger.debug(f"Filtering transcript: {repr(text)}")
            return True
        
        return False

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        message_str = json.dumps(message)
        disconnected = set()
        
        async with self.clients_lock:
            for client in self.connected_clients:
                try:
                    await client.send_text(message_str)
                except Exception as e:
                    logger.error(f"Error broadcasting to client: {e}")
                    disconnected.add(client)
            
            self.connected_clients.difference_update(disconnected)

    def transcribe_interim(self, audio_buffer: np.ndarray) -> Optional[str]:
        """Perform quick interim transcription without VAD."""
        # Skip interim for FunASR (too slow/non-streaming)
        if self.provider_name == "funasr":
            return None
            
        segments, _ = self.provider.transcribe(
            audio_buffer,
            language=self.language_code
        )
        return "".join([seg.text for seg in segments])

    async def process_audio_chunk(
        self,
        websocket: WebSocket,
        chunk: bytes
    ):
        """
        Process a chunk of audio for a specific client.
        Handles both streaming and non-streaming modes.
        
        Args:
            websocket: WebSocket connection for the client
            chunk: Audio chunk bytes
        """
        # Route to appropriate handler
        if self.streaming_enabled:
            await self._process_streaming_chunk(websocket, chunk)
        else:
            await self._process_batch_chunk(websocket, chunk)
    
    async def _process_streaming_chunk(self, websocket: WebSocket, chunk: bytes):
        """Process audio chunk in streaming mode."""
        if not isinstance(self.provider, FunASRProvider):
            return
        
        async with self.client_states_lock:
            state = self.client_states.get(websocket)
            if not state:
                # Initialize if missing
                await self._initialize_client_streaming(websocket)
                state = self.client_states.get(websocket)
                if not state:
                    logger.error("Failed to initialize streaming state")
                    return
        
        # Convert chunk to numpy array
        audio_chunk = np.frombuffer(chunk, dtype=np.float32)
        
        # Add to processing buffers
        state.asr_chunk_buffer = np.concatenate([state.asr_chunk_buffer, audio_chunk])
        state.vad_chunk_buffer = np.concatenate([state.vad_chunk_buffer, audio_chunk])
        
        current_time = time.time() * 1000
        
        # Process VAD when we have enough samples (200ms chunks)
        if state.vad_chunk_buffer.size >= self.vad_chunk_samples:
            vad_chunk = state.vad_chunk_buffer[:self.vad_chunk_samples].copy()
            state.vad_chunk_buffer = state.vad_chunk_buffer[self.vad_chunk_samples:]
            
            vad_result = self.provider.process_vad_chunk(
                vad_chunk,
                state.vad_cache,
                is_final=False
            )
            
            # Handle VAD results
            if vad_result:
                # Check for speech start: [[beg, -1]]
                if len(vad_result) > 0 and len(vad_result[0]) == 2:
                    beg, end = vad_result[0]
                    if beg != -1 and end == -1:
                        # Speech start detected
                        logger.info("Speech start detected")
                        # Only broadcast if we weren't already speaking (to avoid duplicate events)
                        if not state.is_speaking:
                            await self.broadcast({"type": "speech_start"})
                            # Reset caches when resuming from silence to avoid stale state
                            # This prevents weird characters/nonsense when speech resumes after long silence
                            logger.info("Resetting ASR and VAD caches after silence period")
                            cache_dict = self.provider.initialize_streaming()
                            state.asr_cache = cache_dict["asr_cache"]
                            state.vad_cache = cache_dict["vad_cache"]
                            
                            # Clear any buffered audio from before silence to discard old noise
                            state.asr_chunk_buffer = np.array([], dtype=np.float32)
                            state.vad_chunk_buffer = np.array([], dtype=np.float32)
                            state.current_transcript = ""
                            
                            # BUT preserve the VAD chunk that triggered detection (contains speech start)
                            # Add it back to asr_chunk_buffer as the starting point
                            state.asr_chunk_buffer = vad_chunk.copy()
                            
                        state.is_speaking = True
                        state.silence_start_time = None
                    elif beg == -1 and end != -1:
                        # Speech end detected
                        logger.info("Speech end detected")
                        state.is_speaking = False
                        state.silence_start_time = current_time
        
        # Process ASR when we have enough samples (600ms chunks)
        if state.asr_chunk_buffer.size >= self.asr_chunk_samples:
            asr_chunk = state.asr_chunk_buffer[:self.asr_chunk_samples].copy()
            state.asr_chunk_buffer = state.asr_chunk_buffer[self.asr_chunk_samples:]
            
            # Process with ASR
            asr_text = self.provider.process_streaming_chunk(
                asr_chunk,
                state.asr_cache,
                is_final=False
            )
            
            # Accumulate transcript if ASR produced text
            # ASR always returns text for this chunk only, so we always append it
            if asr_text:
                # Append new text to accumulated transcript (no space added - ASR handles spacing)
                if state.current_transcript:
                    state.current_transcript += asr_text
                else:
                    # First chunk - set it
                    state.current_transcript = asr_text
                
                # Send accumulated interim transcript
                await self.broadcast({
                    "type": "interim",
                    "text": state.current_transcript
                })
                logger.debug(f"Interim transcript (accumulated, {len(state.current_transcript)} chars): {state.current_transcript[:50]}...")
        
        # Check silence threshold
        if state.silence_start_time is not None:
            silence_duration = current_time - state.silence_start_time
            if silence_duration >= self.silence_threshold_ms:
                # Finalize transcript
                logger.info(f"Silence threshold exceeded ({silence_duration:.0f}ms), finalizing transcript")
                
                # Process any remaining buffered audio before finalizing
                # This ensures we capture all audio, not just complete chunks
                while state.asr_chunk_buffer.size >= self.asr_chunk_samples:
                    asr_chunk = state.asr_chunk_buffer[:self.asr_chunk_samples].copy()
                    state.asr_chunk_buffer = state.asr_chunk_buffer[self.asr_chunk_samples:]
                    
                    remaining_text = self.provider.process_streaming_chunk(
                        asr_chunk,
                        state.asr_cache,
                        is_final=False
                    )
                    if remaining_text:
                        # Accumulate remaining text (always append, no space)
                        if state.current_transcript:
                            state.current_transcript += remaining_text
                        else:
                            state.current_transcript = remaining_text
                        logger.debug(f"Processed remaining chunk, transcript: {state.current_transcript[:50]}...")
                
                # If there's still some audio left (less than a full chunk), process it
                if state.asr_chunk_buffer.size > 0:
                    remaining_text = self.provider.process_streaming_chunk(
                        state.asr_chunk_buffer,
                        state.asr_cache,
                        is_final=False
                    )
                    if remaining_text:
                        # Accumulate remaining text (always append, no space)
                        if state.current_transcript:
                            state.current_transcript += remaining_text
                        else:
                            state.current_transcript = remaining_text
                        logger.debug(f"Processed final partial chunk, transcript: {state.current_transcript[:50]}...")
                
                # Finalize ASR with is_final=True to get any final text
                # Use empty chunk since cache contains all state
                final_chunk = np.array([], dtype=np.float32)
                final_text = self.provider.process_streaming_chunk(
                    final_chunk,
                    state.asr_cache,
                    is_final=True
                )
                
                # Accumulate final text if present (always append, no space)
                if final_text:
                    if state.current_transcript:
                        state.current_transcript += final_text
                    else:
                        state.current_transcript = final_text
                
                # Apply punctuation to the complete transcript
                if isinstance(self.provider, FunASRProvider):
                    transcript_to_send = self.provider.apply_punctuation(state.current_transcript)
                else:
                    transcript_to_send = state.current_transcript
                
                logger.info(f"Finalizing transcript (accumulated, length: {len(transcript_to_send)} chars): {transcript_to_send[:100]}...")
                
                # Filter out short/noise transcripts
                if transcript_to_send and not self._should_filter_transcript(transcript_to_send):
                    await self.broadcast({
                        "type": "final",
                        "text": transcript_to_send
                    })
                    logger.info(f"Final transcript: {transcript_to_send[:50]}...")
                else:
                    # Filtered out or empty - don't send anything
                    if transcript_to_send:
                        logger.debug(f"Filtered out transcript: {transcript_to_send[:50]}...")
                
                # Reset state (clear current_transcript after sending final)
                cache_dict = self.provider.initialize_streaming()
                state.asr_cache = cache_dict["asr_cache"]
                state.vad_cache = cache_dict["vad_cache"]
                state.asr_chunk_buffer = np.array([], dtype=np.float32)
                state.vad_chunk_buffer = np.array([], dtype=np.float32)
                state.current_transcript = ""  # Clear after sending final
                state.silence_start_time = None
                state.is_speaking = False
    
    async def _process_batch_chunk(self, websocket: WebSocket, chunk: bytes):
        """Process audio chunk in batch mode (legacy behavior)."""
        # For batch mode, maintain per-client buffer
        # This is a simplified version - in practice, batch mode might need its own state management
        # For now, we'll use a simple approach with a temporary buffer per client
        async with self.client_states_lock:
            if websocket not in self.client_states:
                self.client_states[websocket] = ClientStreamingState()
            state = self.client_states[websocket]
        
        # Add to buffer
        new_data = np.frombuffer(chunk, dtype=np.float32)
        state.audio_buffer = np.concatenate([state.audio_buffer, new_data])
        
        # Run interim transcription periodically (throttled)
        current_time = time.time() * 1000
        INTERIM_THROTTLE_MS = 500
        
        if (state.audio_buffer.size > self.interim_min_samples and 
            current_time - state.last_interim_time > INTERIM_THROTTLE_MS):
            
            interim_text = self.transcribe_interim(state.audio_buffer)
            if interim_text and interim_text.strip():
                await self.broadcast({
                    "type": "interim",
                    "text": interim_text
                })
                state.last_interim_time = current_time



