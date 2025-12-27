import asyncio
import json
import websockets
from core.event_bus import EventBus, Event
from core.logging import get_logger
from core.config import get_config
from orchestrator.events import EventType
from orchestrator.managers.base import BaseManager
from orchestrator.utils.event_helpers import publish_activity
from audio.audio_player import AudioPlayer

logger = get_logger(__name__)

class TTSManager(BaseManager):
    """
    Manages TTS synthesis and audio playback.
    
    Listens to TTS_REQUEST events, sends text to TTS Service via WebSocket,
    receives audio, and plays it.
    """
    
    def __init__(self, event_bus: EventBus):
        self.url = get_config("services", "tts_websocket_url", default="ws://localhost:8003/synthesize/stream")
        self.audio_player = AudioPlayer(on_play_state=self._on_play_state_changed)
        self.websocket = None
        self._receiver_task = None
        self._synthesizing = False
        self._connecting = False  # Track connection state to avoid duplicate connections
        self._current_sample_rate = 16000  # Default, will be updated by audio_config
        super().__init__(event_bus)
        
    def _register_handlers(self):
        self.event_bus.subscribe(EventType.TTS_REQUEST.value, self.on_tts_request)
        self.event_bus.subscribe(EventType.LLM_REQUEST.value, self.on_llm_request)  # Pre-connect on LLM request
        self.event_bus.subscribe(EventType.LLM_CANCELLED.value, self.on_cancel)
        self.event_bus.subscribe(EventType.LLM_RESPONSE_DONE.value, self.on_llm_done)
        self.event_bus.subscribe(EventType.SPEECH_START.value, self.on_speech_start)
        
    async def _on_play_state_changed(self, is_playing: bool):
        """Callback when audio playback state changes."""
        await publish_activity(self.event_bus, {"playing": is_playing})
    
    async def on_llm_request(self, event: Event):
        """Pre-connect WebSocket when LLM request is published to reduce latency."""
        # Pre-connect WebSocket in background to avoid blocking
        if not self.websocket and not self._connecting:
            # Use create_task to connect asynchronously without blocking
            asyncio.create_task(self._ensure_connected())
    
    async def on_llm_done(self, event: Event):
        """Finalize TTS stream when LLM response is complete."""
        if self.websocket and self._synthesizing:
            try:
                # Send finalize message to TTS service
                await self.websocket.send(json.dumps({
                    "type": "text",
                    "text": "",
                    "finalize": True
                }))
            except Exception as e:
                self.logger.error(f"Failed to finalize TTS stream: {e}")
    
    async def on_cancel(self, event: Event):
        """Cancel TTS synthesis and stop all audio playback."""
        self.logger.info("Cancelling TTS synthesis and playback")
        
        # Stop audio playback and clear queue
        await self.audio_player.stop()
        
        # Cancel receiver loop to stop processing incoming audio chunks
        if self._receiver_task and not self._receiver_task.done():
            self._receiver_task.cancel()
            try:
                await self._receiver_task
            except asyncio.CancelledError:
                pass
            self._receiver_task = None
        
        # Close WebSocket connection to stop TTS service from sending more chunks
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                self.logger.debug(f"Error closing TTS WebSocket: {e}")
            self.websocket = None
        self._connecting = False
        
        # Reset activity states
        self._synthesizing = False
        await publish_activity(self.event_bus, {"synthesizing": False, "playing": False})
    
    async def on_speech_start(self, event: Event):
        """Interrupt TTS playback and synthesis when user starts speaking."""
        self.logger.info("User speech detected - interrupting TTS playback")
        
        # Stop audio playback and clear queue
        await self.audio_player.stop()
        
        # Cancel receiver loop to stop processing incoming audio chunks
        if self._receiver_task and not self._receiver_task.done():
            self._receiver_task.cancel()
            try:
                await self._receiver_task
            except asyncio.CancelledError:
                pass
            self._receiver_task = None
        
        # Close WebSocket connection to stop TTS service from sending more chunks
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                self.logger.debug(f"Error closing TTS WebSocket: {e}")
            self.websocket = None
        self._connecting = False
        
        # Reset activity states
        self._synthesizing = False
        await publish_activity(self.event_bus, {"synthesizing": False, "playing": False})
            
    async def on_tts_request(self, event: Event):
        text = event.data.get("text")
        if not text:
            return
        
        # Set synthesizing state when TTS request is received
        if not self._synthesizing:
            self._synthesizing = True
            await publish_activity(self.event_bus, {"synthesizing": True})
            
        # Ensure connection (will be fast if already pre-connected)
        await self._ensure_connected()
            
        if self.websocket:
            try:
                # Send text to TTS Service
                await self.websocket.send(json.dumps({
                    "type": "text", 
                    "text": text,
                    "finalize": False
                }))
            except Exception as e:
                self.logger.error(f"Failed to send to TTS service: {e}")
                # Connection may have failed, reset and try to reconnect
                self.websocket = None
                self._synthesizing = False
                await publish_activity(self.event_bus, {"synthesizing": False})
                
    async def _ensure_connected(self):
        """Ensure WebSocket is connected. Returns immediately if already connected."""
        if self.websocket:
            return  # Already connected
        
        if self._connecting:
            # Connection in progress, wait for it
            while self._connecting and not self.websocket:
                await asyncio.sleep(0.01)  # Small delay to avoid busy waiting
            return
        
        # Start connection
        await self._connect()
    
    async def _connect(self):
        """Establish WebSocket connection to TTS service."""
        if self._connecting:
            return  # Already connecting
        
        self._connecting = True
        try:
            self.logger.debug("Connecting to TTS service...")
            self.websocket = await websockets.connect(self.url)
            # Start receiver task and track it
            self._receiver_task = asyncio.create_task(self._receiver_loop())
            self.logger.debug("Connected to TTS service")
        except Exception as e:
            self.logger.error(f"Could not connect to TTS service: {e}")
            self.websocket = None
        finally:
            self._connecting = False

    async def _receiver_loop(self):
        try:
            async for message in self.websocket:
                if isinstance(message, bytes):
                    # Play audio
                    # When first chunk arrives, synthesizing is done
                    if self._synthesizing:
                        self._synthesizing = False
                        await publish_activity(self.event_bus, {"synthesizing": False})
                    
                    await self.audio_player.play_audio_chunk(message, source_sample_rate=self._current_sample_rate)
                    await self.event_bus.publish(Event(EventType.TTS_AUDIO_CHUNK.value, {"size": len(message)}))
                elif isinstance(message, str):
                    # Handle text messages (e.g., "done", "error", "audio_config")
                    try:
                        data = json.loads(message) if message else {}
                        msg_type = data.get("type")
                        
                        if msg_type == "audio_config":
                            self._current_sample_rate = data.get("sample_rate", 16000)
                            self.logger.info(f"Received audio config: sample_rate={self._current_sample_rate}")
                        elif msg_type == "done":
                            # TTS synthesis complete
                            if self._synthesizing:
                                self._synthesizing = False
                                await publish_activity(self.event_bus, {"synthesizing": False})
                    except json.JSONDecodeError:
                        pass
        except asyncio.CancelledError:
            self.logger.info("TTS receiver loop cancelled")
            raise
        except websockets.exceptions.ConnectionClosed:
            self.logger.info("TTS WebSocket connection closed")
            self.websocket = None
            self._receiver_task = None
            self._connecting = False
            # Reset synthesizing state when connection closes
            if self._synthesizing:
                self._synthesizing = False
                await publish_activity(self.event_bus, {"synthesizing": False})
        except Exception as e:
            self.logger.error(f"TTS Receiver error: {e}")
            self.websocket = None
            self._receiver_task = None
            self._connecting = False
            # Reset activity states on error
            self._synthesizing = False
            await publish_activity(self.event_bus, {"synthesizing": False, "playing": False})

