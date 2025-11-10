import argparse
import asyncio
import json
import numpy as np
import sounddevice as sd
import torch
import websockets
import matplotlib.pyplot as plt
from collections import deque

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
    Receives JSON messages from the STT server and prints them.
    Updates the transcript line (3rd line) in place.
    Transcript persists on screen and updates in place for each new sentence.
    """
    print("Listening to STT server...")
    try:
        async for message in websocket:
            data = json.loads(message)
            # Update transcript line (3rd line) in place
            # Restore to saved position (which is on transcript line), then update
            print("\033[u", end="", flush=True)  # Restore to saved position (transcript line)
            if data['type'] == 'interim':
                # Update interim transcript on line 3, stays in place
                print("\rINTERIM: " + data['text'] + "\033[K", end="", flush=True)
            elif data['type'] == 'final':
                # Update final transcript on line 3, stays in place (no newline)
                # Next sentence will update this same line
                print("\rFINAL:   " + data['text'] + "\033[K", end="", flush=True)
            
    except websockets.exceptions.ConnectionClosed as e:
        print(f"\nConnection to server closed: {e}")
    except Exception as e:
        print(f"\nError listening to server: {e}")

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
            print("\n--- Microphone is live. Start speaking (in Chinese). ---\n")
            if enable_plot:
                print("Waveform plot window opened. Press Ctrl+C to stop.\n")
            else:
                print("Press Ctrl+C to stop.\n")
            
            # Initialize 3 fixed lines: status, probability, transcript
            # All three lines will update in place
            print(f"{RED}Speaking not detected{RESET}")
            print(f"Speech probability: 0.000")
            print("Transcript: ", end="", flush=True)  # Transcript line - print without newline so cursor stays on line 3
            # Save cursor position on transcript line (line 3) for transcript updates
            print("\033[s", end="", flush=True)  # Save cursor position (on transcript line)
            
            # VAD requires exactly 512 samples for 16kHz
            VAD_CHUNK_SIZE = 512
            vad_buffer = np.array([], dtype=np.float32)
            last_speech_prob = 0.0  # Track last known speech probability
            
            # Start plot updater task if plotting is enabled
            plot_task = None
            if enable_plot:
                plot_task = asyncio.create_task(plot_updater(fig, ax, line))
            
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

                    # 4. Update status lines - use carriage return and line feed to update in place
                    # We need to move up to the status line (2 lines up from blank line)
                    # Then update status, move down 1, update probability, stay there
                    
                    # 5. VAD Logic - Update status line
                    status_text = ""
                    if speech_prob > VAD_MIN_SPEECH_PROB:
                        # Speaking
                        if not is_speaking:
                            is_speaking = True
                        status_text = f"{GREEN}Speaking detected{RESET}"
                        
                        # Send audio to server
                        await websocket.send(chunk.tobytes())
                        silence_counter = 0
                    else:
                        # Not speaking
                        if is_speaking:
                            # We were speaking, but now we're not
                            silence_counter += 1
                            if silence_counter > silence_blocks:
                                # Silence threshold reached - change to red status
                                await websocket.send(FLUSH_COMMAND)
                                is_speaking = False
                                silence_counter = 0
                        status_text = f"{RED}Speaking not detected{RESET}"
                    
                    # Update status and probability lines (lines 1 and 2)
                    # Restore to saved position (transcript line), then update status lines
                    print("\033[u", end="", flush=True)  # Restore to saved position (transcript line)
                    print("\033[2A\r" + status_text + "\033[K", end="", flush=True)  # Move up 2, update status line (line 1)
                    print("\033[1B\rSpeech probability: " + f"{speech_prob:.3f}" + "\033[K", end="", flush=True)  # Move down 1, update probability line (line 2)
                    print("\033[1B", end="", flush=True)  # Move down 1 back to transcript line (line 3)
                    # Cursor is now back on transcript line, ready for next update
            except KeyboardInterrupt:
                raise
            finally:
                # Clean up plot task if it was started
                if plot_task is not None:
                    plot_task.cancel()
                    try:
                        await plot_task
                    except asyncio.CancelledError:
                        pass
                if fig is not None:
                    plt.close(fig)
                
    except Exception as e:
        print(f"\nError in microphone stream: {e}")
        if enable_plot:
            plt.close('all')  # Close all plots on error


async def main(enable_plot=False):
    """
    Main function to connect to WebSocket and run concurrent tasks.
    
    Args:
        enable_plot: If True, show real-time waveform plot
    """
    print(f"Connecting to STT server at {STT_WEBSOCKET_URL}...")
    try:
        async with websockets.connect(STT_WEBSOCKET_URL) as websocket:
            print("Successfully connected to STT server.")
            
            # Run the two tasks concurrently
            listen_task = asyncio.create_task(listen_to_server(websocket))
            stream_task = asyncio.create_task(stream_mic_to_server(websocket, enable_plot=enable_plot))
            
            await asyncio.gather(listen_task, stream_task)
            
    except websockets.exceptions.InvalidURI:
        print(f"Error: Invalid WebSocket URI: {STT_WEBSOCKET_URL}")
    except ConnectionRefusedError:
        print(f"Error: Connection refused. Is the STT server running at {STT_WEBSOCKET_URL}?")
    except Exception as e:
        print(f"Failed to connect: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice agent driver with real-time STT")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Enable real-time waveform plot (default: disabled)"
    )
    args = parser.parse_args()
    
    try:
        asyncio.run(main(enable_plot=args.plot))
    except KeyboardInterrupt:
        print("\nStopping test driver...")