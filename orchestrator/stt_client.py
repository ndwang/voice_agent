"""
STT Client

WebSocket client for receiving transcripts from STT server.
The orchestrator connects to STT server to receive transcripts broadcast by the server.
"""
import json
import asyncio
import websockets
from typing import Callable, Optional, Dict, Any
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
    
    def __init__(self, on_transcript: Callable[[str], None], on_event: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Initialize STT client.
        
        Args:
            on_transcript: Async callback function called when a final transcript is received.
            on_event: Optional callback invoked for any STT event (e.g., interim/final updates).
        """
        self.on_transcript = on_transcript
        self.on_event = on_event
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.running = False
        self._connect_task: Optional[asyncio.Task] = None
        self._reconnect_delay = 1.0  # Start with 1 second delay
        self._max_reconnect_delay = 60.0  # Max 60 seconds between retries
        self._consecutive_failures = 0
        self._last_error_logged = None
    
    async def connect(self, url: Optional[str] = None):
        """Connect to STT server WebSocket and listen for transcripts."""
        url = url or get_config("services", "stt_websocket_url", default="ws://localhost:8001/ws/transcribe")
        logger.info(f"STT Client: Connecting to {url}...")
        
        while True:
            try:
                async with websockets.connect(url) as websocket:
                    self.websocket = websocket
                    self.running = True
                    # Reset reconnect delay and failure count on successful connection
                    self._reconnect_delay = 1.0
                    self._consecutive_failures = 0
                    self._last_error_logged = None
                    logger.info("STT Client: Connected to STT server")
                    
                    # Listen for transcript messages
                    async for message in websocket:
                        try:
                            # STT server sends text messages (JSON)
                            if isinstance(message, str):
                                data = json.loads(message)
                            else:
                                data = json.loads(message.decode('utf-8'))
                            
                            # Fire generic event callback if provided (interim/final)
                            if self.on_event:
                                if asyncio.iscoroutinefunction(self.on_event):
                                    await self.on_event(data)
                                else:
                                    self.on_event(data)
                            
                            # Only process final transcripts (ignore interim for transcript callback)
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
                self.running = False
                self._consecutive_failures += 1
                logger.warning(f"STT Client: Connection closed. Reconnecting in {self._reconnect_delay:.1f}s...")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)
            
            except (ConnectionRefusedError, OSError, websockets.exceptions.InvalidStatusCode) as e:
                # Handle connection refused and other OS-level connection errors
                # Suppress full traceback for connection errors
                self.running = False
                self._consecutive_failures += 1
                error_msg = str(e)
                error_type = type(e).__name__
                
                # Only log detailed message on first failure or if error message changed
                if self._last_error_logged != error_msg or self._consecutive_failures == 1:
                    if isinstance(e, ConnectionRefusedError):
                        logger.error(
                            f"STT Client: Cannot connect to STT server at {url}. "
                            f"The server may not be running or the address is incorrect."
                        )
                    else:
                        logger.error(
                            f"STT Client: Connection error ({error_type}): {error_msg}"
                        )
                    self._last_error_logged = error_msg
                else:
                    # For repeated same errors, log less verbosely (no traceback)
                    if self._consecutive_failures % 10 == 0:
                        # Log every 10th attempt to avoid spam
                        logger.warning(
                            f"STT Client: Still cannot connect after {self._consecutive_failures} attempts. "
                            f"Retrying in {self._reconnect_delay:.1f}s..."
                        )
                
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)
            
            except asyncio.TimeoutError:
                self.running = False
                self._consecutive_failures += 1
                if self._consecutive_failures == 1 or self._consecutive_failures % 10 == 0:
                    logger.warning(
                        f"STT Client: Connection timeout to {url}. "
                        f"Retrying in {self._reconnect_delay:.1f}s... (attempt {self._consecutive_failures})"
                    )
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)
            
            except websockets.exceptions.InvalidURI as e:
                # Invalid URI is a configuration error, don't retry
                logger.error(f"STT Client: Invalid WebSocket URL '{url}': {e}")
                self.running = False
                break  # Don't retry on invalid URI
            
            except Exception as e:
                # Catch-all for other unexpected errors
                # Only show full traceback on first occurrence or if it's a different error
                self.running = False
                self._consecutive_failures += 1
                error_msg = f"{type(e).__name__}: {str(e)}"
                
                if self._last_error_logged != error_msg or self._consecutive_failures == 1:
                    logger.error(
                        f"STT Client: Unexpected connection error: {error_msg}",
                        exc_info=True  # Full traceback only for first occurrence
                    )
                    self._last_error_logged = error_msg
                else:
                    # For repeated same errors, just log the message
                    logger.warning(
                        f"STT Client: Still encountering error (attempt {self._consecutive_failures}): {error_msg}"
                    )
                
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)
    
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

