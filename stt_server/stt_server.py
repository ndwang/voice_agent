import os
import numpy as np
import torch
import uvicorn
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

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time STT.
    Receives raw audio bytes (16kHz, float32) and sends JSON transcripts.
    """
    # Accept the WebSocket connection first
    await websocket.accept()
    print("Client connected to STT server.")
    # Each client gets its own audio buffer
    audio_buffer = np.array([], dtype=np.float32)

    try:
        while True:
            # 1. Receive data from the client
            data = await websocket.receive_bytes()

            # 2. Check for the "flush" command
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
                    await websocket.send_json({"type": "final", "text": final_text})
                else:
                    # Send empty final if no audio was received
                    await websocket.send_json({"type": "final", "text": ""})
                
                # Clear the buffer for the next turn
                audio_buffer = np.array([], dtype=np.float32)
            
            # 3. Process incoming audio data
            else:
                # Convert raw bytes to float32 numpy array
                new_chunk = np.frombuffer(data, dtype=np.float32)
                # Append to the client's buffer
                audio_buffer = np.concatenate([audio_buffer, new_chunk])

                # Optimization: Only run interim transcription if:
                # 1. The buffer has enough audio
                # 2. The client is still listening (e.g., in a "LISTENING" state)
                # (For now, we just check buffer size)
                if audio_buffer.size > INTERIM_TRANSCRIPT_MIN_SAMPLES:
                    # Run non-VAD transcription for speed
                    segments, info = stt_model.transcribe(
                        audio_buffer,
                        language=LANGUAGE_CODE
                    )
                    interim_text = "".join([seg.text for seg in segments])
                    
                    if interim_text.strip():
                        await websocket.send_json({"type": "interim", "text": interim_text})

    except WebSocketDisconnect:
        print("Client disconnected from STT server.")
    except Exception as e:
        print(f"An error occurred in STT WebSocket: {e}")
    finally:
        # Clean up client-specific resources
        del audio_buffer
        print("STT client connection closed.")

@app.get("/")
async def root():
    return {"message": "STT Server is running. Connect to /ws/transcribe for real-time transcription."}

if __name__ == "__main__":
    print(f"Starting STT server on {HOST}:{PORT}...")
    uvicorn.run(app, host=HOST, port=PORT)