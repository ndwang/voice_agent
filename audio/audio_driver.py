"""
Audio Driver Service

Captures microphone input and streams audio chunks continuously to the STT server.
"""
import argparse
import asyncio
import numpy as np
import sounddevice as sd
import websockets
import httpx
from typing import Optional

from core.config import get_config
from core.logging import setup_logging, get_logger

# Set up logging
setup_logging()
logger = get_logger(__name__)


class AudioDriver:
    """
    Audio driver that captures microphone input and streams to STT server.
    Can be started/stopped programmatically or run standalone.
    """
    
    def __init__(self, stt_url: Optional[str] = None, device: Optional[int] = None, event_bus=None):
        """
        Initialize audio driver.
        
        Args:
            stt_url: WebSocket URL for STT server (default: from config)
            device: Audio input device index (default: from config)
            event_bus: Optional EventBus for in-process communication
        """
        self.stt_url = stt_url or get_config("services", "stt_websocket_url", default="ws://localhost:8001/ws/transcribe")
        self.orchestrator_base_url = get_config("services", "orchestrator_base_url", default="http://localhost:8000")
        self.listening_status_poll_interval = get_config("audio", "listening_status_poll_interval", default=1.0)
        self.sample_rate = get_config("audio", "input", "sample_rate", default=16000)
        self.channels = get_config("audio", "input", "channels", default=1)
        self.dtype = get_config("audio", "dtype", default="float32")
        block_size_ms = get_config("audio", "block_size_ms", default=100)
        self.block_size = int(self.sample_rate * (block_size_ms / 1000))
        self.input_device = device if device is not None else get_config("audio", "input", "device", default=None)
        
        self.audio_queue = asyncio.Queue()
        self.running = False
        self._connect_task = None
        self._stream = None
        self._device_index = None
        self._device_name = None
        self._reconnect_delay = 1.0
        self.event_bus = event_bus
    
    def _audio_callback(self, indata, frames, time, status):
        """This is called from a separate thread by sounddevice."""
        if status:
            logger.warning(f"Audio callback status: {status}")
        chunk = indata.copy()
        self.audio_queue.put_nowait(chunk)
    
    async def _poll_listening_status(self, listening_enabled_flag):
        """
        Periodically poll orchestrator for listening status.
        
        Args:
            listening_enabled_flag: Dict with 'enabled' key to update
        """
        async with httpx.AsyncClient(timeout=5.0) as client:
            while self.running:
                try:
                    response = await client.get(f"{self.orchestrator_base_url}/ui/listening/status")
                    if response.status_code == 200:
                        data = response.json()
                        listening_enabled_flag['enabled'] = data.get('enabled', True)
                    else:
                        logger.warning(f"Failed to get listening status: {response.status_code}")
                except Exception as e:
                    logger.debug(f"Error polling listening status: {e}")
                    # On error, assume listening is enabled to avoid blocking
                    listening_enabled_flag['enabled'] = True
                
                await asyncio.sleep(self.listening_status_poll_interval)
    
    def _initialize_audio(self):
        """
        Initialize audio hardware.
        This is called before connecting to STT server.
            
        Returns:
            tuple: (device_index, device_name)
        """
        # List available input devices
        logger.info("Available audio input devices:")
        logger.info("-" * 80)
        devices = sd.query_devices()
        default_input = sd.default.device[0]  # Get default input device index
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                marker = " (DEFAULT)" if i == default_input else ""
                logger.info(f"  [{i}] {device['name']}{marker}")
                logger.info(f"      Channels: {device['max_input_channels']}, "
                      f"Sample Rate: {device['default_samplerate']:.0f} Hz")
        
        logger.info("-" * 80)
        
        # Determine which device to use
        if self.input_device is not None:
            device_index = self.input_device
            device_name = devices[device_index]['name']
            logger.info(f"Using specified device [{device_index}]: {device_name}")
        else:
            device_index = default_input
            device_name = devices[device_index]['name']
            logger.info(f"Using default input device [{device_index}]: {device_name}")
        
        logger.info("Audio initialization complete.")
        return device_index, device_name
    
    async def _stream_mic_to_server(self, websocket, device_index):
        """
        Continuously forwards audio chunks from the microphone to the STT server.
        
        Args:
            websocket: WebSocket connection to STT server
            device_index: Audio input device index
        """
        # Listening status flag (shared with polling/event task)
        listening_enabled_flag = {'enabled': True}
        
        # Start status tracking task
        if self.event_bus:
            # Use event bus if available (in-process)
            logger.info("Using event bus for listening status updates")
            status_task = asyncio.create_task(self._listen_for_status_changes(listening_enabled_flag))
        else:
            # Fall back to polling (standalone mode)
            logger.info(f"Using HTTP polling for listening status updates ({self.listening_status_poll_interval}s)")
            status_task = asyncio.create_task(self._poll_listening_status(listening_enabled_flag))
        
        logger.info("Opening microphone stream...")

        try:
            self._stream = sd.InputStream(
                device=device_index,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=self.block_size,
                callback=self._audio_callback
            )
            self._stream.start()
            logger.info("Microphone is live. Streaming audio continuously.")
            
            try:
                while self.running:
                    # Get audio chunk from the queue
                    chunk = await self.audio_queue.get()
                    chunk_bytes = chunk.tobytes()
                    
                    # Check listening status
                    listening_enabled = listening_enabled_flag['enabled']
                    
                    # Send audio chunk to server only if listening is enabled
                    if listening_enabled:
                        try:
                            await websocket.send(chunk_bytes)
                        except (websockets.exceptions.ConnectionClosed, ConnectionError):
                            # Connection lost - propagate to trigger reconnection
                            raise
                    
                    # Periodically yield to event loop
                    await asyncio.sleep(0)
            finally:
                # Clean up status tracking task
                status_task.cancel()
                try:
                    await status_task
                except asyncio.CancelledError:
                    pass
                
        except (websockets.exceptions.ConnectionClosed, ConnectionError):
            # Connection errors should propagate to trigger reconnection
            raise
        except Exception as e:
            logger.error(f"Error in microphone stream: {e}", exc_info=True)
            # Re-raise to trigger reconnection
            raise
    
    async def _listen_for_status_changes(self, listening_enabled_flag):
        """Subscribe to event bus for listening status changes."""
        if not self.event_bus:
            return

        async def handle_change(event):
            listening_enabled_flag['enabled'] = event.data.get('enabled', True)

        # Import constant here to avoid top-level orchestrator dependency in standalone mode
        try:
            from orchestrator.core.constants import UI_LISTENING_STATE_CHANGED
            topic = UI_LISTENING_STATE_CHANGED
        except ImportError:
            topic = "ui.listening_state_changed"

        self.event_bus.subscribe(topic, handle_change)
        try:
            # Keep the task alive until cancelled
            while self.running:
                await asyncio.sleep(1)
        finally:
            self.event_bus.unsubscribe(topic, handle_change)
    
    async def start(self):
        """Start the audio driver."""
        if self.running:
            return
        
        self.running = True
        
        # Initialize audio hardware
        logger.info("Initializing audio hardware...")
        try:
            self._device_index, self._device_name = self._initialize_audio()
        except Exception as e:
            logger.error(f"Failed to initialize audio: {e}", exc_info=True)
            logger.error("Cannot proceed without audio initialization.")
            self.running = False
            return
        
        logger.info(f"Audio driver started, connecting to {self.stt_url}")
        self._connect_task = asyncio.create_task(self._connect_loop())
    
    async def stop(self):
        """Stop the audio driver."""
        self.running = False
        
        if self._connect_task:
            self._connect_task.cancel()
            try:
                await self._connect_task
            except asyncio.CancelledError:
                pass
        
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except:
                pass
            self._stream = None
        
        logger.info("Audio driver stopped")
    
    async def _connect_loop(self):
        """Main connection loop with auto-reconnect."""
        while self.running:
            try:
                async with websockets.connect(self.stt_url) as websocket:
                    self._reconnect_delay = 1.0
                    logger.info("Connected to STT service")
                    
                    # Stream audio to server
                    await self._stream_mic_to_server(websocket, self._device_index)
                        
            except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError):
                logger.warning(f"STT connection lost. Retrying in {self._reconnect_delay}s...")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 10.0)
            except websockets.exceptions.InvalidURI:
                logger.error(f"Invalid WebSocket URI: {self.stt_url}")
                logger.error("Cannot retry with invalid URI. Exiting.")
                break
            except Exception as e:
                logger.error(f"Audio driver error: {e}", exc_info=True)
                await asyncio.sleep(5.0)


# Standalone main function for backward compatibility
async def main(stt_url=None, device=None):
    """
    Main function to initialize audio, then connect to WebSocket and run concurrent tasks.
    Automatically reconnects if connection is lost.
    
    Args:
        stt_url: WebSocket URL for STT server (default: uses STT_WEBSOCKET_URL)
        device: Audio input device index (default: uses INPUT_DEVICE)
    """
    driver = AudioDriver(stt_url=stt_url, device=device)
    await driver.start()
    
    try:
        # Keep running until interrupted
        while driver.running:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nStopping audio driver...")
    finally:
        await driver.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio driver service - captures microphone and streams to STT")
    parser.add_argument(
        "--stt-url",
        type=str,
        default=None,
        help="STT server WebSocket URL (default: from config)"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio input device index (default: system default)"
    )
    args = parser.parse_args()
    
    try:
        asyncio.run(main(stt_url=args.stt_url, device=args.device))
    except KeyboardInterrupt:
        logger.info("\nStopping audio driver...")
