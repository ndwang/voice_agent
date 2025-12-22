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
        self._synthesizing = False
        super().__init__(event_bus)
        
    def _register_handlers(self):
        self.event_bus.subscribe(EventType.TTS_REQUEST.value, self.on_tts_request)
        self.event_bus.subscribe(EventType.LLM_CANCELLED.value, self.on_cancel)
        self.event_bus.subscribe(EventType.LLM_RESPONSE_DONE.value, self.on_llm_done)
        
    async def _on_play_state_changed(self, is_playing: bool):
        """Callback when audio playback state changes."""
        await publish_activity(self.event_bus, {"playing": is_playing})
    
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
        await self.audio_player.stop()
        # Reset activity states
        self._synthesizing = False
        await publish_activity(self.event_bus, {"synthesizing": False, "playing": False})
        if self.websocket:
            # We might want to send a clear message to TTS server if supported
            pass
            
    async def on_tts_request(self, event: Event):
        text = event.data.get("text")
        if not text:
            return
        
        # Set synthesizing state when TTS request is received
        if not self._synthesizing:
            self._synthesizing = True
            await publish_activity(self.event_bus, {"synthesizing": True})
            
        # Ensure connection
        if not self.websocket:
            await self._connect()
            
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
                self._synthesizing = False
                await publish_activity(self.event_bus, {"synthesizing": False})
                
    async def _connect(self):
        try:
            self.websocket = await websockets.connect(self.url)
            # Start receiver task
            asyncio.create_task(self._receiver_loop())
        except Exception as e:
            self.logger.error(f"Could not connect to TTS service: {e}")

    async def _receiver_loop(self):
        try:
            async for message in self.websocket:
                if isinstance(message, bytes):
                    # Play audio
                    # When first chunk arrives, synthesizing is done
                    if self._synthesizing:
                        self._synthesizing = False
                        await publish_activity(self.event_bus, {"synthesizing": False})
                    
                    await self.audio_player.play_audio_chunk(message)
                    await self.event_bus.publish(Event(EventType.TTS_AUDIO_CHUNK.value, {"size": len(message)}))
                elif isinstance(message, str):
                    # Handle text messages (e.g., "done", "error")
                    try:
                        data = json.loads(message) if message else {}
                        if data.get("type") == "done":
                            # TTS synthesis complete
                            if self._synthesizing:
                                self._synthesizing = False
                                await publish_activity(self.event_bus, {"synthesizing": False})
                    except json.JSONDecodeError:
                        pass
        except websockets.exceptions.ConnectionClosed:
            self.logger.info("TTS WebSocket connection closed")
            self.websocket = None
            # Reset synthesizing state when connection closes
            if self._synthesizing:
                self._synthesizing = False
                await publish_activity(self.event_bus, {"synthesizing": False})
        except Exception as e:
            self.logger.error(f"TTS Receiver error: {e}")
            self.websocket = None
            # Reset activity states on error
            self._synthesizing = False
            await publish_activity(self.event_bus, {"synthesizing": False, "playing": False})

