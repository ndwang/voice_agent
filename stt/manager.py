import asyncio
import json
import numpy as np
import time
from typing import Dict, Set, Optional, Tuple, Any
from fastapi import WebSocket, WebSocketDisconnect
from core.logging import get_logger
from core.config import get_config
from stt.providers import FasterWhisperProvider, FunASRProvider

logger = get_logger(__name__)


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
        
        flush_cmd = get_config("stt", "flush_command", default="\x00")
        if isinstance(flush_cmd, str):
            self.flush_command = flush_cmd.encode('latin-1')
        else:
            self.flush_command = bytes([flush_cmd]) if isinstance(flush_cmd, int) else flush_cmd

        # Provider setup
        self.provider_name = get_config("stt", "provider", default="faster-whisper")
        self.provider = self._load_provider()

        # State
        self.connected_clients: Set[WebSocket] = set()
        self.clients_lock = asyncio.Lock()
        
        # We need to track buffer per-client or global? 
        # Original code used per-request scope variables in the endpoint.
        # Since we might have multiple audio sources, we'll keep state in the client handler, 
        # but the Manager can handle the heavy lifting of transcription.
    
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
                device = get_config("stt", "providers", "funasr", "device", default=None)
                batch_size_s = get_config("stt", "providers", "funasr", "batch_size_s", default=0)
                
                return FunASRProvider(
                    model_name=model_name,
                    vad_model=vad_model,
                    vad_kwargs=vad_kwargs,
                    device=device,
                    batch_size_s=batch_size_s
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
        logger.info(f"Client connected. Total: {len(self.connected_clients)}")

    async def remove_client(self, websocket: WebSocket):
        """Unregister a client."""
        async with self.clients_lock:
            self.connected_clients.discard(websocket)
        logger.info(f"Client disconnected. Total: {len(self.connected_clients)}")

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

    def transcribe_final(self, audio_buffer: np.ndarray) -> str:
        """Perform full transcription with VAD."""
        if audio_buffer.size == 0:
            return ""
            
        segments, _ = self.provider.transcribe(
            audio_buffer,
            language=self.language_code,
            vad_filter=True
        )
        return "".join([seg.text for seg in segments])

    async def process_audio_chunk(self, 
                                  chunk: bytes, 
                                  audio_buffer: np.ndarray, 
                                  last_interim_time: float) -> Tuple[np.ndarray, float]:
        """
        Process a chunk of audio:
        1. Add to buffer
        2. Check for flush command
        3. Run interim transcription if needed
        4. Broadcast results
        
        Returns:
            (new_buffer, new_last_interim_time)
        """
        INTERIM_THROTTLE_MS = 500
        
        # Check for flush command first
        if chunk == self.flush_command:
            logger.info("Flush command received.")
            final_text = self.transcribe_final(audio_buffer)
            
            if final_text.strip():
                await self.broadcast({
                    "type": "final",
                    "text": final_text
                })
                logger.info(f"Final transcript: {final_text[:50]}...")
            else:
                 await self.broadcast({
                    "type": "final",
                    "text": ""
                })
            
            # Reset buffer
            return np.array([], dtype=np.float32), last_interim_time
        
        # Standard audio processing
        new_data = np.frombuffer(chunk, dtype=np.float32)
        audio_buffer = np.concatenate([audio_buffer, new_data])
        
        current_time = time.time() * 1000
        if (audio_buffer.size > self.interim_min_samples and 
            current_time - last_interim_time > INTERIM_THROTTLE_MS):
            
            interim_text = self.transcribe_interim(audio_buffer)
            if interim_text and interim_text.strip():
                await self.broadcast({
                    "type": "interim",
                    "text": interim_text
                })
                last_interim_time = current_time
                
        return audio_buffer, last_interim_time



