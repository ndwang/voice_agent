import asyncio
import json
import time
from typing import Dict, Optional, AsyncIterator, Any
from fastapi import WebSocket, WebSocketDisconnect
from core.logging import get_logger
from core.config import get_config
from tts.providers import EdgeTTSProvider, ChatTTSProvider
from tts.base import TTSProvider

logger = get_logger(__name__)


class TTSManager:
    """
    Manages TTS sessions, provider interactions, and streaming state.
    """

    def __init__(self):
        # Configuration
        self.output_sample_rate = get_config("tts", "output_sample_rate")
        if self.output_sample_rate is None:
            self.output_sample_rate = get_config("audio", "output", "sample_rate", default=16000)

        # Provider setup
        self.provider_name = get_config("tts", "provider", default="edge-tts")
        self.provider: TTSProvider = self._load_provider()
        
        # Default settings
        self.default_voice = get_config("tts", "providers", "edge-tts", "voice", default="zh-CN-XiaoxiaoNeural")
        self.default_rate = get_config("tts", "providers", "edge-tts", "rate", default="+0%")
        self.default_pitch = get_config("tts", "providers", "edge-tts", "pitch", default="+0Hz")

    def _load_provider(self) -> TTSProvider:
        logger.info(f"Initializing TTS provider: {self.provider_name}...")
        try:
            if self.provider_name == "edge-tts":
                return EdgeTTSProvider(
                    default_voice=get_config("tts", "providers", "edge-tts", "voice", default="zh-CN-XiaoxiaoNeural"),
                    default_rate=get_config("tts", "providers", "edge-tts", "rate", default="+0%"),
                    default_pitch=get_config("tts", "providers", "edge-tts", "pitch", default="+0Hz"),
                    output_sample_rate=self.output_sample_rate
                )
            elif self.provider_name == "chattts":
                return ChatTTSProvider(
                    output_sample_rate=self.output_sample_rate,
                    model_source=get_config("tts", "providers", "chattts", "model_source", default="local"),
                    device=get_config("tts", "providers", "chattts", "device", default=None)
                )
            else:
                raise ValueError(f"Unknown TTS provider: {self.provider_name}")
        except Exception as e:
            logger.error(f"Error loading TTS provider: {e}", exc_info=True)
            raise

    async def list_voices(self) -> list:
        """List available voices."""
        return await self.provider.list_voices()

    async def find_voices(self, **filters) -> list:
        """Find voices by attributes (Edge TTS only)."""
        if hasattr(self.provider, "find_voices"):
            return await self.provider.find_voices(**filters)
        return []

    async def synthesize(self, text: str, **kwargs) -> bytes:
        """Synthesize complete audio (non-streaming)."""
        return await self.provider.synthesize(text, **kwargs)

    async def handle_websocket_session(self, websocket: WebSocket):
        """
        Handle a WebSocket session for streaming synthesis.
        Manages the protocol loop: receive text -> synthesize -> stream audio.
        """
        await websocket.accept()
        
        text_buffer = []
        provider_params = {}
        
        # Keepalive state
        KEEPALIVE_INTERVAL = 2.0
        last_ping_time = time.time()
        
        try:
            while True:
                # 1. Manage Keepalive Pings
                current_time = time.time()
                if current_time - last_ping_time >= KEEPALIVE_INTERVAL:
                    try:
                        await websocket.send_text(json.dumps({"type": "ping"}))
                        last_ping_time = current_time
                    except Exception:
                        break  # Connection closed
                
                # 2. Wait for message with timeout
                try:
                    # Short timeout to allow ping loop to run
                    message = await asyncio.wait_for(websocket.receive_text(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    break
                
                # 3. Process Message
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    await websocket.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
                    continue
                
                msg_type = data.get("type")
                
                if msg_type == "text":
                    await self._handle_text_message(websocket, data, text_buffer, provider_params)
                
                elif msg_type == "config":
                    self._update_params(data, provider_params)
                    await websocket.send_text(json.dumps({"type": "config_updated"}))
                    
                elif msg_type == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                    last_ping_time = time.time()
                    
                elif msg_type == "pong":
                    last_ping_time = time.time()
        
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"WebSocket session error: {e}", exc_info=True)
            try:
                await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
            except:
                pass

    def _update_params(self, data: dict, params: dict):
        """Update provider parameters from message data."""
        keys = ["voice", "rate", "pitch", "speaker", "temperature", "top_p", "top_k", "stream_speed"]
        for k in keys:
            if k in data:
                params[k] = data[k]

    async def _handle_text_message(self, websocket: WebSocket, data: dict, text_buffer: list, params: dict):
        """Handle incoming text for synthesis."""
        text = data.get("text", "")
        finalize = data.get("finalize", False)
        
        # Update one-off params
        self._update_params(data, params)
        
        if text:
            text_buffer.append(text)
            
        # Immediate synthesis if we have buffer (stream as we go)
        if text_buffer:
            full_text = "".join(text_buffer)
            text_buffer.clear()
            
            # Ack receipt
            await websocket.send_text(json.dumps({"type": "status", "message": "received"}))
            
            await self._synthesize_and_stream(websocket, full_text, params)
        elif finalize:
            # Empty text but finalize requested -> just say done
            await websocket.send_text(json.dumps({"type": "done"}))

    async def _synthesize_and_stream(self, websocket: WebSocket, text: str, params: dict):
        """Synthesize text and stream audio chunks to client."""
        try:
            # Heartbeat task to keep connection alive while waiting for TTS engine
            heartbeat_task = asyncio.create_task(self._send_processing_heartbeats(websocket))
            
            try:
                chunk_count = 0
                async for audio_chunk in self.provider.synthesize_stream(text, **params):
                    # Cancel heartbeat once we start getting audio
                    if not heartbeat_task.cancelled():
                        heartbeat_task.cancel()
                    
                    await websocket.send_bytes(audio_chunk)
                    chunk_count += 1
                
                await websocket.send_text(json.dumps({"type": "done"}))
                
            finally:
                if not heartbeat_task.cancelled():
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass
                        
        except Exception as e:
            logger.error(f"Synthesis error: {e}", exc_info=True)
            _, error_msg = self.provider.parse_error(e)
            await websocket.send_text(json.dumps({"type": "error", "message": error_msg}))

    async def _send_processing_heartbeats(self, websocket: WebSocket):
        """Send ping_processing messages while waiting for TTS."""
        try:
            while True:
                await asyncio.sleep(0.5)
                await websocket.send_text(json.dumps({"type": "ping_processing"}))
        except asyncio.CancelledError:
            pass
        except Exception:
            pass


