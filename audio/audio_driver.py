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
from collections import deque

# Windows-specific: ctypes for console API

# --- Configuration ---
STT_WEBSOCKET_URL = "ws://localhost:8001/ws/transcribe"
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = 'float32'
# BLOCK_SIZE_MS determines VAD sensitivity. 50-100ms is good.
BLOCK_SIZE_MS = 100
BLOCK_SIZE = int(SAMPLE_RATE * (BLOCK_SIZE_MS / 1000))
FLUSH_COMMAND = b'\x00' # Must match the server's command
# Set to None to use default device, or specify device index/name
INPUT_DEVICE = None

# VAD Configuration
SILENCE_THRESHOLD_MS = 500 # 500ms of silence triggers a "flush"
VAD_MIN_SPEECH_PROB = 0.5
silence_blocks = int(SILENCE_THRESHOLD_MS / BLOCK_SIZE_MS)

# Global audio queue
audio_queue = asyncio.Queue()

# Global waveform buffer for plotting
waveform_buffer = deque(maxlen=int(SAMPLE_RATE * 2))  # Store 2 seconds of audio

# Plot configuration
PLOT_WINDOW_SECONDS = 2  # Show last 2 seconds of audio
PLOT_UPDATE_INTERVAL_MS = 50  # Update plot every 50ms

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
        print(f"Audio callback status: {status}", flush=True)
    chunk = indata.copy()
    audio_queue.put_nowait(chunk)
    # Add to waveform buffer for plotting
    waveform_buffer.extend(chunk.flatten())

async def listen_to_server(websocket):
    """
    Receives JSON messages from the STT server and updates the transcript via display manager.
    """
    print("Listening to STT server...")
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
        print(f"\nError listening to server: {e}")
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

async def stream_mic_to_server(websocket, enable_plot=False):
    """
    Manages VAD and sends audio from the mic to the server.
    
    Args:
        websocket: WebSocket connection to STT server
        enable_plot: If True, show real-time waveform plot
    """
    # List available input devices
    print("Available audio input devices:")
    print("-" * 80)
    devices = sd.query_devices()
    default_input = sd.default.device[0]  # Get default input device index
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            marker = " (DEFAULT)" if i == default_input else ""
            print(f"  [{i}] {device['name']}{marker}")
            print(f"      Channels: {device['max_input_channels']}, "
                  f"Sample Rate: {device['default_samplerate']:.0f} Hz")
    
    print("-" * 80)
    
    # Determine which device to use
    if INPUT_DEVICE is not None:
        device_index = INPUT_DEVICE
        device_name = devices[device_index]['name']
        print(f"Using specified device [{device_index}]: {device_name}")
    else:
        device_index = default_input
        device_name = devices[device_index]['name']
        print(f"Using default input device [{device_index}]: {device_name}")
    
    print("Loading VAD model...")
    try:
        vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
    except Exception as e:
        print(f"Error loading VAD model: {e}")
        print("Please ensure you have an internet connection for the first run,")
        print("and that PyTorch is installed (pip install torch).")
        return

    print("Opening microphone stream...")
    
    # Set up the plot if enabled
    fig = None
    ax = None
    line = None
    if enable_plot:
        print("Initializing waveform plot...")
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
            print("\n--- Microphone is live. Start speaking. ---\n")
            if enable_plot:
                print("Waveform plot window opened. Press Ctrl+C to stop.\n")
            else:
                print("Press Ctrl+C to stop.\n")
            
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

                    # VAD Logic - Determine status text
                    status_text = ""
                    if speech_prob > VAD_MIN_SPEECH_PROB:
                        # Speaking
                        if not is_speaking:
                            is_speaking = True
                        status_text = f"{GREEN}Speaking detected{RESET}"
                        
                        # Send audio to server
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
        print(f"\nError in microphone stream: {e}")
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
    
    print(f"Connecting to STT server at {STT_WEBSOCKET_URL}...")
    
    while True:
        try:
            async with websockets.connect(STT_WEBSOCKET_URL) as websocket:
                print("Successfully connected to STT server.")
                
                # Run the two tasks concurrently
                listen_task = asyncio.create_task(listen_to_server(websocket))
                stream_task = asyncio.create_task(stream_mic_to_server(websocket, enable_plot=enable_plot))
                
                try:
                    await asyncio.gather(listen_task, stream_task)
                except (websockets.exceptions.ConnectionClosed, ConnectionError) as e:
                    print(f"\nConnection lost: {e}")
                    print("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)
                    continue
                except KeyboardInterrupt:
                    # User interrupted - exit gracefully
                    raise
                except Exception as e:
                    print(f"\nError during operation: {e}")
                    print("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)
                    continue
            
        except websockets.exceptions.InvalidURI:
            print(f"Error: Invalid WebSocket URI: {STT_WEBSOCKET_URL}")
            print("Cannot retry with invalid URI. Exiting.")
            break
        except ConnectionRefusedError:
            print(f"Connection refused. Is the STT server running at {STT_WEBSOCKET_URL}?")
            print("Retrying in 5 seconds...")
            await asyncio.sleep(5)
        except KeyboardInterrupt:
            print("\nStopping audio driver...")
            break
        except Exception as e:
            print(f"Failed to connect: {e}")
            print("Retrying in 5 seconds...")
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
        print("\nStopping audio driver...")
