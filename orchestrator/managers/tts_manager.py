import asyncio
import json
import websockets
from core.event_bus import EventBus, Event
from core.logging import get_logger
from core.settings import get_settings
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
        settings = get_settings()
        self.url = settings.services.tts_websocket_url
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
        if self.websocket:
            try:
                # Send finalize message to TTS service
                # This tells the TTS service that all sentences have been sent
                self.logger.debug("Sending finalize message to TTS service")
                await self.websocket.send(json.dumps({
                    "type": "text",
                    "text": "",
                    "finalize": True
                }))
            except websockets.exceptions.ConnectionClosed as e:
                self.logger.warning(f"Connection closed while finalizing TTS stream: {e}")
                self.websocket = None
            except Exception as e:
                self.logger.error(f"Failed to finalize TTS stream: {e}", exc_info=True)
                # Don't reset websocket here - might still be usable for future requests
    
    async def on_cancel(self, event: Event):
        """Cancel TTS synthesis and stop all audio playback."""
        if not self._synthesizing and not self.audio_player.playing:
            return

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
                try:
                    if isinstance(message, bytes):
                        # Play audio                        
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
                                # TTS synthesis complete for one sentence
                                self.logger.debug("Received 'done' message from TTS service")
                                if self._synthesizing:
                                    self._synthesizing = False
                                    await publish_activity(self.event_bus, {"synthesizing": False})
                            elif msg_type == "error":
                                # TTS service reported an error
                                error_msg = data.get("message", "Unknown error")
                                self.logger.warning(f"TTS service error: {error_msg}")
                                # Don't close connection, allow other sentences to be processed
                        except json.JSONDecodeError:
                            self.logger.warning(f"Failed to parse message from TTS service: {message[:100]}")
                except Exception as e:
                    # Log error but continue processing - don't let one bad message close the connection
                    self.logger.error(f"Error processing message from TTS service: {e}", exc_info=True)
                    continue
        except asyncio.CancelledError:
            self.logger.info("TTS receiver loop cancelled")
            raise
        except websockets.exceptions.ConnectionClosed as e:
            # Log connection close with details
            was_synthesizing = self._synthesizing
            self.logger.warning(
                f"TTS WebSocket connection closed unexpectedly "
                f"(code: {e.code}, reason: {e.reason}, was_synthesizing: {was_synthesizing})"
            )
            self.websocket = None
            self._receiver_task = None
            self._connecting = False
            # Reset synthesizing state when connection closes
            if self._synthesizing:
                self._synthesizing = False
                await publish_activity(self.event_bus, {"synthesizing": False})
        except Exception as e:
            self.logger.error(f"TTS Receiver error: {e}", exc_info=True)
            self.websocket = None
            self._receiver_task = None
            self._connecting = False
            # Reset activity states on error
            if self._synthesizing:
                self._synthesizing = False
                await publish_activity(self.event_bus, {"synthesizing": False, "playing": False})

