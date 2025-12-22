import json
import asyncio
import websockets
from typing import Optional
from core.config import get_config
from core.logging import get_logger
from core.event_bus import Event
from orchestrator.events import EventType
from orchestrator.sources.base import BaseSource
from orchestrator.utils.event_helpers import publish_activity

logger = get_logger(__name__)

class STTSource(BaseSource):
    """
    WebSocket client that receives transcripts from STT service
    and publishes them to the event bus.
    """
    
    def __init__(self, event_bus):
        super().__init__(event_bus)
        self.url = get_config("services", "stt_websocket_url", default="ws://localhost:8001/ws/transcribe")
        self.websocket = None
        self._reconnect_delay = 1.0
        self._connect_task = None

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
                self._reconnect_delay = min(self._reconnect_delay * 2, 60.0)
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
                if text.strip():
                    # Publish activity: transcribing done
                    await publish_activity(self.event_bus, {"transcribing": False})
                    await self.publish(EventType.TRANSCRIPT_FINAL, {"text": text})
            
            elif msg_type == "interim":
                text = data.get("text", "")
                if text.strip():
                    # Publish activity: transcribing active
                    await publish_activity(self.event_bus, {"transcribing": True})
                    await self.publish(EventType.TRANSCRIPT_INTERIM, {"text": text})
            
            elif msg_type == "speech_start":
                await self.publish(EventType.SPEECH_START, {})
                
        except json.JSONDecodeError:
            pass
        except Exception as e:
            logger.error(f"Error handling STT message: {e}")

