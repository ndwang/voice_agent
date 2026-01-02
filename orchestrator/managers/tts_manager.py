import asyncio
import json
import websockets
from core.event_bus import EventBus, Event
from core.logging import get_logger
from core.config import get_config
from core.settings import get_settings
from core.settings.reload_result import ReloadResult
from orchestrator.events import EventType
from orchestrator.managers.base import BaseManager
from orchestrator.core.activity_state import get_activity_state
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
        self.activity_state = get_activity_state()  # Access centralized activity state
        super().__init__(event_bus)
        
    def on_config_changed(self, changes: dict) -> ReloadResult:
        """
        Handle configuration changes.

        Args:
            changes: Dict with changed config sections

        Returns:
            ReloadResult with status and details
        """
        result = ReloadResult(handler_name="TTSManager", success=True)

        try:
            if "audio" in changes:
                audio_changes = changes.get("audio", {})

                # Hot-reloadable: output device
                if "output" in audio_changes:
                    settings = get_settings()
                    new_device = settings.audio.output.device
                    old_device = self.audio_player.output_device
                    self.audio_player.output_device = new_device
                    result.changes_applied.append(f"audio.output.device: {old_device} -> {new_device}")
                    logger.info(f"Output device updated: {new_device}")

                # Restart-required: input device
                if "input" in audio_changes:
                    result.restart_required.append("audio.input requires AudioDriver restart")

            if "services" in changes:
                # Update WebSocket URL
                new_url = get_config("services", "tts_websocket_url", default="ws://localhost:8003/synthesize/stream")
                if new_url != self.url:
                    self.url = new_url
                    # Close existing connection, will reconnect on next request
                    if self.websocket:
                        logger.info("Closing TTS WebSocket connection for URL update")
                        # Note: We can't await here since this is sync, but the connection will close naturally
                        # and reconnect on next TTS request
                    result.changes_applied.append(f"services.tts_websocket_url: {self.url}")

            if "tts" in changes:
                tts_changes = changes.get("tts", {})

                # Restart-required: provider change
                if "provider" in tts_changes:
                    result.restart_required.append("tts.provider requires TTS service restart")

                # Restart-required: server URL (for GPT-SoVITS backend)
                if "providers" in tts_changes and "gpt-sovits" in tts_changes.get("providers", {}):
                    if "server_url" in tts_changes["providers"]["gpt-sovits"]:
                        result.restart_required.append("tts.providers.gpt-sovits.server_url requires TTS service restart")

        except Exception as e:
            result.success = False
            result.errors.append(f"Failed to reload TTS config: {str(e)}")
            logger.error(f"TTS config reload error: {e}", exc_info=True)

        return result

    def _register_handlers(self):
        self.event_bus.subscribe(EventType.TTS_REQUEST.value, self.on_tts_request)
        self.event_bus.subscribe(EventType.LLM_REQUEST.value, self.on_llm_request)  # Pre-connect on LLM request
        self.event_bus.subscribe(EventType.VOICE_INTERRUPT.value, self.on_cancel)
        self.event_bus.subscribe(EventType.CRITICAL_INTERRUPT.value, self.on_cancel)
        # Keep LLM_CANCELLED for backward compatibility during migration
        self.event_bus.subscribe(EventType.LLM_CANCELLED.value, self.on_cancel)
        self.event_bus.subscribe(EventType.LLM_RESPONSE_DONE.value, self.on_llm_done)
        
    async def _on_play_state_changed(self, is_playing: bool):
        """Callback when audio playback state changes."""
        await self.activity_state.update({"playing": is_playing})
        if not is_playing:
            await self.event_bus.publish(Event(EventType.TURN_ENDED.value))
    
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
        await self.activity_state.update({"synthesizing": False, "playing": False})
    
    async def on_tts_request(self, event: Event):
        text = event.data.get("text")
        if not text:
            return
        
        # Set synthesizing state when TTS request is received
        if not self._synthesizing:
            self._synthesizing = True
            await self.activity_state.update({"synthesizing": True})
            
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
                await self.activity_state.update({"synthesizing": False})
                
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
                                    await self.activity_state.update({"synthesizing": False})
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
                await self.activity_state.update({"synthesizing": False})
        except Exception as e:
            self.logger.error(f"TTS Receiver error: {e}", exc_info=True)
            self.websocket = None
            self._receiver_task = None
            self._connecting = False
            # Reset activity states on error
            if self._synthesizing:
                self._synthesizing = False
                await self.activity_state.update({"synthesizing": False, "playing": False})

