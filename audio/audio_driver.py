"""
Audio Driver Service

Captures microphone input and streams audio to the STT server.
Performs Voice Activity Detection (VAD) to only send audio when speech is detected.
Receives transcriptions from the STT server and displays them in real-time.
"""
import argparse
import asyncio
import ctypes
import json
import numpy as np
import os
import sounddevice as sd
import sys
import time
import torch
import websockets
import matplotlib.pyplot as plt
import logging
import httpx
from collections import deque
from pathlib import Path

# Windows-specific: ctypes for console API

# Configure logging with time info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

# Add project root to path to import config_loader
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config_loader import get_config

# --- Configuration ---
STT_WEBSOCKET_URL = get_config("services", "stt_websocket_url", default="ws://localhost:8001/ws/transcribe")
ORCHESTRATOR_BASE_URL = get_config("services", "orchestrator_base_url", default="http://localhost:8000")
LISTENING_STATUS_POLL_INTERVAL = get_config("audio", "listening_status_poll_interval", default=1.0)
SAMPLE_RATE = get_config("audio", "sample_rate", default=16000)
CHANNELS = get_config("audio", "channels", default=1)
DTYPE = get_config("audio", "dtype", default="float32")
# BLOCK_SIZE_MS determines VAD sensitivity. 50-100ms is good.
BLOCK_SIZE_MS = get_config("audio", "block_size_ms", default=100)
BLOCK_SIZE = int(SAMPLE_RATE * (BLOCK_SIZE_MS / 1000))
FLUSH_COMMAND_STR = get_config("audio", "flush_command", default="\x00")
# Convert string to bytes (YAML "\x00" is already a null byte character)
if isinstance(FLUSH_COMMAND_STR, str):
    FLUSH_COMMAND = FLUSH_COMMAND_STR.encode('latin-1')
else:
    FLUSH_COMMAND = bytes([FLUSH_COMMAND_STR]) if isinstance(FLUSH_COMMAND_STR, int) else FLUSH_COMMAND_STR
# Set to None to use default device, or specify device index/name
INPUT_DEVICE = get_config("audio", "input_device", default=None)

# VAD Configuration
SILENCE_THRESHOLD_MS = get_config("audio", "silence_threshold_ms", default=500)  # 500ms of silence triggers a "flush"
VAD_MIN_SPEECH_PROB = get_config("audio", "vad_min_speech_prob", default=0.5)
silence_blocks = int(SILENCE_THRESHOLD_MS / BLOCK_SIZE_MS)

# Global audio queue
audio_queue = asyncio.Queue()

# Global waveform buffer for plotting
waveform_buffer = deque(maxlen=int(SAMPLE_RATE * 2))  # Store 2 seconds of audio

# Plot configuration
PLOT_WINDOW_SECONDS = get_config("audio", "plot_window_seconds", default=2)  # Show last 2 seconds of audio
PLOT_UPDATE_INTERVAL_MS = get_config("audio", "plot_update_interval_ms", default=50)  # Update plot every 50ms

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

class TerminalDisplay:
    """
    Centralized terminal display manager that handles all output updates atomically.
    Prevents cursor position conflicts by using a lock and managing cursor position internally.
    """
    def __init__(self):
        self._lock = asyncio.Lock()
        self._initialized = False
        self._anchor_saved = False
        self._last_update_time = 0.0
        self._update_interval = 0.05  # Update display at most every 50ms (20 FPS)
        # Ensure unbuffered output on Windows
        if sys.platform == 'win32':
            # Set stdout to unbuffered mode
            try:
                if hasattr(sys.stdout, 'reconfigure'):
                    sys.stdout.reconfigure(line_buffering=True)
            except (ValueError, OSError):
                pass
    
    def _write(self, text: str):
        """Write text directly to stdout with explicit flushing."""
        sys.stdout.write(text)
        sys.stdout.flush()
        
        # On Windows, use console API to force refresh if available
        if sys.platform == 'win32':
            try:
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
                if handle:
                    kernel32.FlushConsoleInputBuffer(handle)
            except (AttributeError, OSError):
                pass
    
    async def initialize(self):
        """Initialize the display with 3 fixed lines: status, probability, transcript."""
        async with self._lock:
            self._write(f"{RED}Speaking not detected{RESET}\n")
            self._write(f"Speech probability: 0.000\n")
            self._write("Transcript: ")  # Transcript line - cursor stays on line 3
            self._write("\033[s")  # Save cursor position (on transcript line)
            self._anchor_saved = True
            self._initialized = True
    
    async def update_status_and_probability(self, status_text: str, probability: float):
        """Update both status and probability lines atomically in a single operation."""
        current_time = time.time()
        if current_time - self._last_update_time < self._update_interval:
            return
        
        async with self._lock:
            if not self._anchor_saved:
                return
            output = (
                "\033[u"  # Restore to anchor (transcript line)
                "\033[2A\r" + status_text + "\033[K"  # Move up 2, update status (line 1)
                "\033[1B\rSpeech probability: " + f"{probability:.3f}" + "\033[K"  # Move down 1, update probability (line 2)
                "\033[1B"  # Move back down to anchor
                "\033[s"  # Re-save anchor
            )
            self._write(output)
            self._last_update_time = current_time
            await asyncio.sleep(0)  # Yield to event loop
    
    async def update_transcript(self, text: str, is_final: bool = False):
        """Update the transcript line (line 3)."""
        async with self._lock:
            if not self._anchor_saved:
                return
            prefix = "FINAL:   " if is_final else "INTERIM: "
            output = (
                "\033[u"  # Restore to anchor
                "\r" + prefix + text + "\033[K"
                "\033[s"  # Re-save anchor
            )
            self._write(output)

# Global display manager instance
display = TerminalDisplay()

def audio_callback(indata, frames, time, status):
    """This is called from a separate thread by sounddevice."""
    if status:
        # Note: logging from callback thread is safe, but we use print here
        # because this is a real-time callback and we want immediate output
        print(f"Audio callback status: {status}", flush=True)
    chunk = indata.copy()
    audio_queue.put_nowait(chunk)
    # Add to waveform buffer for plotting
    waveform_buffer.extend(chunk.flatten())

async def listen_to_server(websocket):
    """
    Receives JSON messages from the STT server and updates the transcript via display manager.
    """
    logger.info("Listening to STT server...")
    try:
        async for message in websocket:
            data = json.loads(message)
            if data['type'] == 'interim':
                await display.update_transcript(data['text'], is_final=False)
            elif data['type'] == 'final':
                await display.update_transcript(data['text'], is_final=True)
            
    except websockets.exceptions.ConnectionClosed:
        # Connection closed - will be handled by main retry loop
        raise
    except Exception as e:
        logger.error(f"Error listening to server: {e}", exc_info=True)
        raise

def setup_plot():
    """Set up the matplotlib plot for real-time waveform display."""
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Real-time Audio Waveform')
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3)
    
    # Initialize empty line
    line, = ax.plot([], [], 'b-', linewidth=0.5)
    
    return fig, ax, line

def update_plot(fig, ax, line):
    """Update the plot with current waveform data."""
    if len(waveform_buffer) == 0:
        return
    
    # Get current waveform data
    waveform_data = np.array(waveform_buffer)
    
    # Create time axis
    time_axis = np.linspace(0, len(waveform_data) / SAMPLE_RATE, len(waveform_data))
    
    # Update the plot
    line.set_data(time_axis, waveform_data)
    
    # Update x-axis limits to show last PLOT_WINDOW_SECONDS
    if len(waveform_data) > 0:
        max_time = len(waveform_data) / SAMPLE_RATE
        ax.set_xlim(max(0, max_time - PLOT_WINDOW_SECONDS), max_time)
    
    # Redraw
    fig.canvas.draw()
    fig.canvas.flush_events()

async def plot_updater(fig, ax, line):
    """Async task to periodically update the plot."""
    try:
        while True:
            update_plot(fig, ax, line)
            await asyncio.sleep(PLOT_UPDATE_INTERVAL_MS / 1000.0)
    except asyncio.CancelledError:
        pass

async def poll_listening_status(listening_enabled_flag):
    """
    Periodically poll orchestrator for listening status.
    
    Args:
        listening_enabled_flag: Dict with 'enabled' key to update
    """
    async with httpx.AsyncClient(timeout=5.0) as client:
        while True:
            try:
                response = await client.get(f"{ORCHESTRATOR_BASE_URL}/ui/listening/status")
                if response.status_code == 200:
                    data = response.json()
                    listening_enabled_flag['enabled'] = data.get('enabled', True)
                else:
                    logger.warning(f"Failed to get listening status: {response.status_code}")
            except Exception as e:
                logger.debug(f"Error polling listening status: {e}")
                # On error, assume listening is enabled to avoid blocking
                listening_enabled_flag['enabled'] = True
            
            await asyncio.sleep(LISTENING_STATUS_POLL_INTERVAL)


async def stream_mic_to_server(websocket, enable_plot=False):
    """
    Manages VAD and sends audio from the mic to the server.
    
    Args:
        websocket: WebSocket connection to STT server
        enable_plot: If True, show real-time waveform plot
    """
    # Listening status flag (shared with polling task)
    listening_enabled_flag = {'enabled': True}
    
    # Start polling task
    polling_task = asyncio.create_task(poll_listening_status(listening_enabled_flag))
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
    if INPUT_DEVICE is not None:
        device_index = INPUT_DEVICE
        device_name = devices[device_index]['name']
        logger.info(f"Using specified device [{device_index}]: {device_name}")
    else:
        device_index = default_input
        device_name = devices[device_index]['name']
        logger.info(f"Using default input device [{device_index}]: {device_name}")
    
    logger.info("Loading VAD model...")
    try:
        vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            skip_validation=True
        )
    except Exception as e:
        logger.error(f"Error loading VAD model: {e}", exc_info=True)
        return

    logger.info("Opening microphone stream...")
    
    # Set up the plot if enabled
    fig = None
    ax = None
    line = None
    if enable_plot:
        logger.info("Initializing waveform plot...")
        fig, ax, line = setup_plot()
        plt.show(block=False)
    
    is_speaking = False
    silence_counter = 0

    try:
        with sd.InputStream(
            device=device_index,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=BLOCK_SIZE,
            callback=audio_callback
        ):
            logger.info("\n--- Microphone is live. Start speaking. ---\n")
            if enable_plot:
                logger.info("Waveform plot window opened. Press Ctrl+C to stop.\n")
            else:
                logger.info("Press Ctrl+C to stop.\n")
            
            # Initialize display with 3 fixed lines: status, probability, transcript
            await display.initialize()
            
            # VAD requires exactly 512 samples for 16kHz
            VAD_CHUNK_SIZE = 512
            vad_buffer = np.array([], dtype=np.float32)
            last_speech_prob = 0.0  # Track last known speech probability
            
            # Start plot updater task if plotting is enabled
            plot_task = None
            if enable_plot:
                plot_task = asyncio.create_task(plot_updater(fig, ax, line))
            
            # Start terminal flush task to ensure output is visible even without plotting
            async def terminal_flusher():
                """Periodically flush terminal output to ensure visibility."""
                try:
                    while True:
                        sys.stdout.flush()
                        await asyncio.sleep(PLOT_UPDATE_INTERVAL_MS / 1000.0)
                except asyncio.CancelledError:
                    pass
            
            flush_task = asyncio.create_task(terminal_flusher())
            
            loop_counter = 0
            YIELD_INTERVAL = 10
            
            try:
                while True:
                    # 1. Get audio chunk from the queue
                    chunk = await audio_queue.get()
                    chunk_1d = chunk.flatten()
                    
                    # 2. Add chunk to VAD buffer
                    vad_buffer = np.concatenate([vad_buffer, chunk_1d])
                    
                    # 3. Run VAD on 512-sample chunks
                    # Process all complete 512-sample chunks in the buffer
                    while len(vad_buffer) >= VAD_CHUNK_SIZE:
                        vad_chunk = vad_buffer[:VAD_CHUNK_SIZE]
                        last_speech_prob = vad_model(torch.from_numpy(vad_chunk), SAMPLE_RATE).item()
                        # Remove processed chunk from buffer
                        vad_buffer = vad_buffer[VAD_CHUNK_SIZE:]
                    
                    # Use the last known speech probability
                    speech_prob = last_speech_prob

                    # Check listening status
                    listening_enabled = listening_enabled_flag['enabled']
                    
                    # VAD Logic - Determine status text
                    status_text = ""
                    if not listening_enabled:
                        # Listening disabled - don't send audio
                        status_text = f"{RED}Listening disabled{RESET}"
                        # Reset speaking state if we were speaking
                        if is_speaking:
                            is_speaking = False
                            silence_counter = 0
                    elif speech_prob > VAD_MIN_SPEECH_PROB:
                        # Speaking
                        if not is_speaking:
                            # Silence→Speech transition: Send speech_start message for interruption
                            is_speaking = True
                            if listening_enabled:
                                try:
                                    await websocket.send(json.dumps({"type": "speech_start"}))
                                except (websockets.exceptions.ConnectionClosed, ConnectionError):
                                    # Connection lost - propagate to trigger reconnection
                                    raise
                        status_text = f"{GREEN}Speaking detected{RESET}"
                        
                        # Send audio to server only if listening is enabled
                        if listening_enabled:
                            try:
                                await websocket.send(chunk.tobytes())
                            except (websockets.exceptions.ConnectionClosed, ConnectionError):
                                # Connection lost - propagate to trigger reconnection
                                raise
                        silence_counter = 0
                    else:
                        # Not speaking
                        if is_speaking:
                            # We were speaking, but now we're not
                            silence_counter += 1
                            if silence_counter > silence_blocks:
                                # Silence threshold reached - change to red status
                                if listening_enabled:
                                    try:
                                        await websocket.send(FLUSH_COMMAND)
                                    except (websockets.exceptions.ConnectionClosed, ConnectionError):
                                        # Connection lost - propagate to trigger reconnection
                                        raise
                                is_speaking = False
                                silence_counter = 0
                        status_text = f"{RED}Speaking not detected{RESET}"
                    
                    await display.update_status_and_probability(status_text, speech_prob)
                    
                    # Periodically yield to event loop
                    loop_counter += 1
                    if loop_counter >= YIELD_INTERVAL:
                        loop_counter = 0
                        await asyncio.sleep(0)
            except KeyboardInterrupt:
                raise
            finally:
                # Clean up tasks
                polling_task.cancel()
                try:
                    await polling_task
                except asyncio.CancelledError:
                    pass
                
                flush_task.cancel()
                try:
                    await flush_task
                except asyncio.CancelledError:
                    pass
                
                if plot_task is not None:
                    plot_task.cancel()
                    try:
                        await plot_task
                    except asyncio.CancelledError:
                        pass
                if fig is not None:
                    plt.close(fig)
                
    except (websockets.exceptions.ConnectionClosed, ConnectionError):
        # Connection errors should propagate to trigger reconnection
        raise
    except Exception as e:
        logger.error(f"Error in microphone stream: {e}", exc_info=True)
        if enable_plot:
            plt.close('all')  # Close all plots on error
        # Re-raise to trigger reconnection
        raise


async def main(stt_url=None, device=None, enable_plot=False):
    """
    Main function to connect to WebSocket and run concurrent tasks.
    Automatically reconnects if connection is lost.
    
    Args:
        stt_url: WebSocket URL for STT server (default: uses STT_WEBSOCKET_URL)
        device: Audio input device index (default: uses INPUT_DEVICE)
        enable_plot: If True, show real-time waveform plot
    """
    # Update global configuration if provided
    global STT_WEBSOCKET_URL, INPUT_DEVICE
    if stt_url:
        STT_WEBSOCKET_URL = stt_url
    if device is not None:
        INPUT_DEVICE = device
    
    logger.info(f"Connecting to STT server at {STT_WEBSOCKET_URL}...")
    
    while True:
        try:
            async with websockets.connect(STT_WEBSOCKET_URL) as websocket:
                logger.info("Successfully connected to STT server.")
                
                # Run the two tasks concurrently
                listen_task = asyncio.create_task(listen_to_server(websocket))
                stream_task = asyncio.create_task(stream_mic_to_server(websocket, enable_plot=enable_plot))
                
                try:
                    await asyncio.gather(listen_task, stream_task)
                except (websockets.exceptions.ConnectionClosed, ConnectionError) as e:
                    logger.warning(f"Connection lost: {e}")
                    logger.info("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)
                    continue
                except KeyboardInterrupt:
                    # User interrupted - exit gracefully
                    raise
                except Exception as e:
                    logger.error(f"Error during operation: {e}", exc_info=True)
                    logger.info("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)
                    continue
            
        except websockets.exceptions.InvalidURI:
            logger.error(f"Invalid WebSocket URI: {STT_WEBSOCKET_URL}")
            logger.error("Cannot retry with invalid URI. Exiting.")
            break
        except ConnectionRefusedError:
            logger.warning(f"Connection refused. Is the STT server running at {STT_WEBSOCKET_URL}?")
            logger.info("Retrying in 5 seconds...")
            await asyncio.sleep(5)
        except KeyboardInterrupt:
            logger.info("\nStopping audio driver...")
            break
        except Exception as e:
            logger.error(f"Failed to connect: {e}", exc_info=True)
            logger.info("Retrying in 5 seconds...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio driver service - captures microphone and streams to STT")
    parser.add_argument(
        "--stt-url",
        type=str,
        default=None,
        help=f"STT server WebSocket URL (default: {STT_WEBSOCKET_URL})"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio input device index (default: system default)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Enable real-time waveform plot (default: disabled)"
    )
    args = parser.parse_args()
    
    try:
        asyncio.run(main(stt_url=args.stt_url, device=args.device, enable_plot=args.plot))
    except KeyboardInterrupt:
        logger.info("\nStopping audio driver...")
