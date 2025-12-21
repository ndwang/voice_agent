import os
import json
import time
import numpy as np
import uvicorn
import asyncio
import logging
import sys
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Add project root to path to import config_loader
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config_loader import get_config
from stt.providers import FasterWhisperProvider, FunASRProvider

# Configure logging with time info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

# --- Configuration ---
# 1. Server Configuration
HOST = get_config("stt", "host", default="0.0.0.0")
PORT = get_config("stt", "port", default=8001)

# 2. Transcription Configuration
LANGUAGE_CODE = get_config("stt", "language_code", default="zh")  # Chinese
SAMPLE_RATE = get_config("stt", "sample_rate", default=16000)  # 16kHz
INTERIM_TRANSCRIPT_MIN_SAMPLES = get_config("stt", "interim_transcript_min_samples", default=int(0.3 * 16000))  # Min audio (0.3s) to run an interim transcription
FLUSH_COMMAND_STR = get_config("stt", "flush_command", default="\x00")
# Convert string to bytes (YAML "\x00" is already a null byte character)
if isinstance(FLUSH_COMMAND_STR, str):
    FLUSH_COMMAND = FLUSH_COMMAND_STR.encode('latin-1')
else:
    FLUSH_COMMAND = bytes([FLUSH_COMMAND_STR]) if isinstance(FLUSH_COMMAND_STR, int) else FLUSH_COMMAND_STR

# 3. Provider Configuration
STT_PROVIDER = get_config("stt", "provider", default="faster-whisper")

# --- Global STT Provider ---
# Load the provider once at startup
logger.info(f"Initializing STT provider: {STT_PROVIDER}...")
try:
    if STT_PROVIDER == "faster-whisper":
        # Faster-Whisper configuration
        model_path = get_config("stt", "providers", "faster-whisper", "model_path", default="faster-whisper-small")
        device = get_config("stt", "providers", "faster-whisper", "device", default=None)
        compute_type = get_config("stt", "providers", "faster-whisper", "compute_type", default=None)
        
        stt_provider = FasterWhisperProvider(
            model_path=model_path,
            device=device,
            compute_type=compute_type
        )
    elif STT_PROVIDER == "funasr":
        # FunASR configuration
        model_name = get_config("stt", "providers", "funasr", "model_name", default="FunAudioLLM/Fun-ASR-Nano-2512")
        vad_model = get_config("stt", "providers", "funasr", "vad_model", default="fsmn-vad")
        vad_kwargs = get_config("stt", "providers", "funasr", "vad_kwargs", default={"max_single_segment_time": 30000})
        device = get_config("stt", "providers", "funasr", "device", default=None)
        batch_size_s = get_config("stt", "providers", "funasr", "batch_size_s", default=0)
        
        stt_provider = FunASRProvider(
            model_name=model_name,
            vad_model=vad_model,
            vad_kwargs=vad_kwargs,
            device=device,
            batch_size_s=batch_size_s
        )
    else:
        raise ValueError(f"Unknown STT provider: {STT_PROVIDER}. Supported: faster-whisper, funasr")
    
    logger.info(f"STT provider '{STT_PROVIDER}' loaded successfully.")
except Exception as e:
    logger.error(f"Error loading STT provider: {e}", exc_info=True)
    # Exit if provider fails to load
    exit(1)

# --- FastAPI Server ---
app = FastAPI()

# Track all connected clients for broadcasting transcripts
connected_clients = set()
clients_lock = asyncio.Lock()


async def broadcast_to_clients(message: dict):
    """Broadcast transcript message to all connected clients."""
    message_str = json.dumps(message)
    async with clients_lock:
        disconnected = set()
        for client in connected_clients:
            try:
                await client.send_text(message_str)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}", exc_info=True)
                disconnected.add(client)
    # Remove disconnected clients
    connected_clients.difference_update(disconnected)


@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time STT.
    Receives raw audio bytes (16kHz, float32) and broadcasts interim and final transcripts to all connected clients.
    Supports both clients that send audio and clients that only listen (like orchestrator).
    """
    # Accept the WebSocket connection first
    await websocket.accept()
    logger.info("Client connected to STT server.")
    
    # Add client to connected clients set
    async with clients_lock:
        connected_clients.add(websocket)
    
    # Each client that sends audio gets its own audio buffer
    audio_buffer = np.array([], dtype=np.float32)
    last_interim_time = 0
    INTERIM_THROTTLE_MS = 500  # Minimum time between interim transcripts (ms)
    is_audio_client = False  # Track if this client sends audio

    try:
        while True:
            try:
                # Use receive() to handle both bytes and text messages
                # Set a timeout to allow checking for disconnections
                message = await asyncio.wait_for(websocket.receive(), timeout=1.0)
            except asyncio.TimeoutError:
                # Timeout is normal for listener-only clients - just continue
                continue
            
            # Check message type
            if "bytes" in message:
                # Client is sending audio data
                is_audio_client = True
                data = message["bytes"]
                
                # Check for the "flush" command
                if data == FLUSH_COMMAND:
                    logger.info("Flush command received. Finalizing transcription.")
                    
                    if audio_buffer.size > 0:
                        # Run final transcription with VAD filter
                        segments, info = stt_provider.transcribe(
                            audio_buffer,
                            language=LANGUAGE_CODE,
                            vad_filter=True
                        )
                        final_text = "".join([seg.text for seg in segments])
                        
                        # Broadcast final transcript to all clients
                        if final_text.strip():
                            await broadcast_to_clients({
                                "type": "final", 
                                "text": final_text
                            })
                            logger.info(f"Broadcast final transcript: {final_text[:50]}...")
                    else:
                        # Broadcast empty final if no audio was received
                        await broadcast_to_clients({
                            "type": "final", 
                            "text": ""
                        })
                    
                    # Clear the buffer for the next turn
                    audio_buffer = np.array([], dtype=np.float32)
                
                # Process incoming audio data
                else:
                    # Convert raw bytes to float32 numpy array
                    new_chunk = np.frombuffer(data, dtype=np.float32)
                    # Append to the client's buffer
                    audio_buffer = np.concatenate([audio_buffer, new_chunk])

                    # Send interim transcription if buffer has enough audio
                    # Skip interim transcripts for FunASR (non-streaming model, slower)
                    # Throttle to avoid sending too frequently
                    current_time = time.time() * 1000  # Convert to milliseconds
                    if (STT_PROVIDER != "funasr" and  # Skip interim for FunASR
                        audio_buffer.size > INTERIM_TRANSCRIPT_MIN_SAMPLES and 
                        current_time - last_interim_time > INTERIM_THROTTLE_MS):
                        # Run non-VAD transcription for speed
                        segments, info = stt_provider.transcribe(
                            audio_buffer,
                            language=LANGUAGE_CODE
                        )
                        interim_text = "".join([seg.text for seg in segments])
                        
                        if interim_text.strip():
                            # Broadcast interim transcript to all clients
                            await broadcast_to_clients({"type": "interim", "text": interim_text})
                            last_interim_time = current_time
            
            elif "text" in message:
                # Client sent a text message (e.g., keepalive or control message)
                try:
                    data = json.loads(message["text"])
                    msg_type = data.get("type")
                    
                    if msg_type == "speech_start":
                        # Broadcast speech_start event to all clients for interruption
                        await broadcast_to_clients({"type": "speech_start"})
                        logger.debug("Broadcast speech_start event")
                except json.JSONDecodeError:
                    # Not a JSON message, ignore
                    pass
                except Exception as e:
                    logger.warning(f"Error processing text message: {e}")

    except WebSocketDisconnect:
        logger.info("Client disconnected from STT server.")
    except Exception as e:
        logger.error(f"An error occurred in STT WebSocket: {e}", exc_info=True)
    finally:
        # Remove client from connected clients set
        async with clients_lock:
            connected_clients.discard(websocket)
        # Clean up client-specific resources
        if is_audio_client:
            del audio_buffer
        logger.info("STT client connection closed.")

@app.get("/")
async def root():
    return {"message": "STT Server is running. Connect to /ws/transcribe for real-time transcription."}

if __name__ == "__main__":
    logger.info(f"Starting STT server on {HOST}:{PORT}...")
    uvicorn.run(app, host=HOST, port=PORT)