import os
import json
import time
import numpy as np
import torch
import uvicorn
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel

# --- Configuration ---
# 1. Model Configuration
MODEL_SIZE = "small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "int8" if DEVICE == "cuda" else "default"

# 2. Server Configuration
HOST = "0.0.0.0"
PORT = 8001

# 3. Transcription Configuration
LANGUAGE_CODE = "zh"  # Chinese
SAMPLE_RATE = 16000  # 16kHz
INTERIM_TRANSCRIPT_MIN_SAMPLES = int(0.3 * SAMPLE_RATE) # Min audio (0.3s) to run an interim transcription
FLUSH_COMMAND = b'\x00' # Special byte command to finalize transcription

# --- Global Model ---
# Load the model once at startup
print(f"Loading STT model '{MODEL_SIZE}' on {DEVICE} ({COMPUTE_TYPE})...")
try:
    stt_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("STT model loaded successfully.")
except Exception as e:
    print(f"Error loading STT model: {e}")
    # Exit if model fails to load
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
                print(f"Error broadcasting to client: {e}")
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
    print("Client connected to STT server.")
    
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
                    print("Flush command received. Finalizing transcription.")
                    if audio_buffer.size > 0:
                        # Run final transcription with VAD filter
                        segments, info = stt_model.transcribe(
                            audio_buffer,
                            language=LANGUAGE_CODE,
                            vad_filter=True
                        )
                        final_text = "".join([seg.text for seg in segments])
                        
                        # Broadcast final transcript to all clients
                        if final_text.strip():
                            await broadcast_to_clients({"type": "final", "text": final_text})
                            print(f"Broadcast final transcript: {final_text[:50]}...")
                    else:
                        # Broadcast empty final if no audio was received
                        await broadcast_to_clients({"type": "final", "text": ""})
                    
                    # Clear the buffer for the next turn
                    audio_buffer = np.array([], dtype=np.float32)
                
                # Process incoming audio data
                else:
                    # Convert raw bytes to float32 numpy array
                    new_chunk = np.frombuffer(data, dtype=np.float32)
                    # Append to the client's buffer
                    audio_buffer = np.concatenate([audio_buffer, new_chunk])

                    # Send interim transcription if buffer has enough audio
                    # Throttle to avoid sending too frequently
                    current_time = time.time() * 1000  # Convert to milliseconds
                    if (audio_buffer.size > INTERIM_TRANSCRIPT_MIN_SAMPLES and 
                        current_time - last_interim_time > INTERIM_THROTTLE_MS):
                        # Run non-VAD transcription for speed
                        segments, info = stt_model.transcribe(
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
                # For now, we ignore text messages from clients
                pass

    except WebSocketDisconnect:
        print("Client disconnected from STT server.")
    except Exception as e:
        print(f"An error occurred in STT WebSocket: {e}")
    finally:
        # Remove client from connected clients set
        async with clients_lock:
            connected_clients.discard(websocket)
        # Clean up client-specific resources
        if is_audio_client:
            del audio_buffer
        print("STT client connection closed.")

@app.get("/")
async def root():
    return {"message": "STT Server is running. Connect to /ws/transcribe for real-time transcription."}

if __name__ == "__main__":
    print(f"Starting STT server on {HOST}:{PORT}...")
    uvicorn.run(app, host=HOST, port=PORT)