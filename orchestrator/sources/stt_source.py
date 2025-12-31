import json
import asyncio
import websockets
from typing import Optional
from core.settings import get_settings
from core.logging import get_logger
from core.event_bus import Event
from orchestrator.events import EventType
from orchestrator.sources.base import BaseSource
from orchestrator.core.activity_state import get_activity_state

logger = get_logger(__name__)

class STTSource(BaseSource):
    """
    WebSocket client that receives transcripts from STT service
    and publishes them to the event bus.
    """

    def __init__(self, event_bus):
        super().__init__(event_bus)
        settings = get_settings()
        self.url = settings.services.stt_transcript_url
        self.websocket = None
        self._reconnect_delay = 1.0
        self._connect_task = None
        self.accumulated_transcript = ""  # Accumulate interim transcripts
        self.activity_state = get_activity_state()  # Access centralized activity state

    async def start(self):
        """Start the STT source."""
        if self.running:
            return
        self.running = True
        self._connect_task = asyncio.create_task(self._connect_loop())
        logger.info(f"STT Source started, connecting to {self.url}")

    async def stop(self):
        """Stop the STT source."""
        self.running = False
        if self._connect_task:
            self._connect_task.cancel()
            try:
                await self._connect_task
            except asyncio.CancelledError:
                pass
        
        if self.websocket:
            try:
                await self.websocket.close()
            except:
                pass
        logger.info("STT Source stopped")

    async def _connect_loop(self):
        """Main connection loop with auto-reconnect."""
        while self.running:
            try:
                async with websockets.connect(self.url) as ws:
                    self.websocket = ws
                    self._reconnect_delay = 1.0
                    logger.info("Connected to STT service")
                    
                    async for message in ws:
                        await self._handle_message(message)
                        
            except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError):
                logger.warning(f"STT connection lost. Retrying in {self._reconnect_delay}s...")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 10.0)
            except Exception as e:
                logger.error(f"STT Source error: {e}", exc_info=True)
                await asyncio.sleep(5.0)

    async def _handle_message(self, message):
        """Process incoming WebSocket message."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "final":
                text = data.get("text", "")
                # Use provided final text, or fall back to accumulated transcript
                # STT manager should send the full accumulated transcript, but we keep accumulated as backup
                final_text = text if text.strip() else self.accumulated_transcript
                
                if final_text.strip():
                    # Update activity: transcribing done
                    await self.activity_state.update({"transcribing": False})
                    await self.publish(EventType.TRANSCRIPT_FINAL, {"text": final_text})
                    # Reset accumulated transcript after final
                    self.accumulated_transcript = ""
            
            elif msg_type == "interim":
                text = data.get("text", "")
                if text.strip():
                    # STT manager should send accumulated transcripts, but we also accumulate here as backup
                    # Update accumulated transcript (use the longer one, as STT manager sends accumulated)
                    if len(text) >= len(self.accumulated_transcript):
                        self.accumulated_transcript = text
                    else:
                        # If new text is shorter, it might be a partial update - append if not already included
                        if text not in self.accumulated_transcript:
                            self.accumulated_transcript += text

                    # Update activity: transcribing active
                    await self.activity_state.update({"transcribing": True})
                    # Send the accumulated transcript (STT manager should already send accumulated, but use ours as fallback)
                    await self.publish(EventType.TRANSCRIPT_INTERIM, {"text": self.accumulated_transcript})
            
            elif msg_type == "speech_start":
                # Reset accumulated transcript on speech start
                self.accumulated_transcript = ""
                await self.publish(EventType.SPEECH_START, {})
                
        except json.JSONDecodeError:
            pass
        except Exception as e:
            logger.error(f"Error handling STT message: {e}")

