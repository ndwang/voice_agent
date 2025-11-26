"""
STT Client

WebSocket client for receiving transcripts from STT server.
The orchestrator connects to STT server to receive transcripts broadcast by the server.
"""
import json
import asyncio
import websockets
from typing import Callable, Optional
from pathlib import Path
import logging
import sys

# Add project root to path to import config_loader
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config_loader import get_config
from orchestrator.logging_config import setup_logging, get_logger

# Set up logging
setup_logging()
logger = get_logger(__name__)


class STTClient:
    """WebSocket client for STT server."""
    
    def __init__(self, on_transcript: Callable[[str], None]):
        """
        Initialize STT client.
        
        Args:
            on_transcript: Async callback function called when transcript is received
        """
        self.on_transcript = on_transcript
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.running = False
        self._connect_task: Optional[asyncio.Task] = None
    
    async def connect(self, url: Optional[str] = None):
        """Connect to STT server WebSocket and listen for transcripts."""
        url = url or get_config("services", "stt_websocket_url", default="ws://localhost:8001/ws/transcribe")
        logger.info(f"STT Client: Connecting to {url}...")
        
        while True:
            try:
                async with websockets.connect(url) as websocket:
                    self.websocket = websocket
                    self.running = True
                    logger.info("STT Client: Connected to STT server")
                    
                    # Listen for transcript messages
                    async for message in websocket:
                        try:
                            # STT server sends text messages (JSON)
                            if isinstance(message, str):
                                data = json.loads(message)
                            else:
                                data = json.loads(message.decode('utf-8'))
                            
                            # Only process final transcripts (ignore interim)
                            if data.get("type") == "final":
                                text = data.get("text", "")
                                speech_end_time = data.get("speech_end_time")  # Optional timestamp from STT server
                                stt_latency = data.get("stt_latency")  # Optional STT processing latency
                                if text.strip():
                                    # Call the callback (which should be async)
                                    if asyncio.iscoroutinefunction(self.on_transcript):
                                        # Check if callback accepts additional parameters
                                        import inspect
                                        sig = inspect.signature(self.on_transcript)
                                        param_count = len(sig.parameters)
                                        if param_count > 2:
                                            await self.on_transcript(text, speech_end_time, stt_latency)
                                        elif param_count > 1:
                                            await self.on_transcript(text, speech_end_time)
                                        else:
                                            await self.on_transcript(text)
                                    else:
                                        self.on_transcript(text)
                        except json.JSONDecodeError:
                            logger.warning(f"STT Client: Invalid JSON: {message}")
                        except Exception as e:
                            logger.error(f"STT Client: Error processing message: {e}", exc_info=True)
            
            except websockets.exceptions.ConnectionClosed:
                logger.warning("STT Client: Connection closed")
                self.running = False
                await asyncio.sleep(5)  # Wait before reconnecting
            except Exception as e:
                logger.error(f"STT Client: Connection error: {e}", exc_info=True)
                self.running = False
                await asyncio.sleep(5)  # Wait before reconnecting
    
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self.running and self.websocket is not None
    
    async def close(self):
        """Close connection."""
        self.running = False
        if self.websocket:
            try:
                await self.websocket.close()
            except:
                pass

