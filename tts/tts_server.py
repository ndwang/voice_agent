"""
TTS Service

FastAPI server for Text-to-Speech with multiple backend support.
Supports streaming audio synthesis via WebSocket.
"""
import os
import json
import asyncio
import logging
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from typing import Optional
import uvicorn

from tts.base import TTSProvider
from tts.providers import EdgeTTSProvider, ChatTTSProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
HOST = "0.0.0.0"
PORT = 8003

# TTS Provider Configuration
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "edge-tts")  # "edge-tts" or "chattts"
OUTPUT_SAMPLE_RATE = 16000  # Output sample rate (matches audio player)

# Edge TTS Configuration
EDGE_TTS_VOICE = os.getenv("TTS_VOICE", "zh-CN-XiaoxiaoNeural")
EDGE_TTS_RATE = os.getenv("TTS_RATE", "+0%")
EDGE_TTS_PITCH = os.getenv("TTS_PITCH", "+0Hz")

# ChatTTS Configuration
CHATTTS_MODEL_SOURCE = os.getenv("CHATTTS_MODEL_SOURCE", "local")  # "local", "huggingface", "custom"
CHATTTS_DEVICE = os.getenv("CHATTTS_DEVICE", None)  # "cuda", "cpu", or None for auto

# --- Initialize TTS Provider ---
tts_provider: Optional[TTSProvider] = None

if TTS_PROVIDER == "edge-tts":
    tts_provider = EdgeTTSProvider(
        default_voice=EDGE_TTS_VOICE,
        default_rate=EDGE_TTS_RATE,
        default_pitch=EDGE_TTS_PITCH,
        output_sample_rate=OUTPUT_SAMPLE_RATE
    )
elif TTS_PROVIDER == "chattts":
    tts_provider = ChatTTSProvider(
        output_sample_rate=OUTPUT_SAMPLE_RATE,
        model_source=CHATTTS_MODEL_SOURCE,
        device=CHATTTS_DEVICE
    )
else:
    raise ValueError(f"Unknown TTS provider: {TTS_PROVIDER}. Supported: edge-tts, chattts")

if tts_provider is None:
    raise ValueError("TTS provider not initialized. Please check configuration.")

# --- FastAPI Server ---
app = FastAPI()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "TTS Service is running",
        "provider": TTS_PROVIDER,
        "output_sample_rate": OUTPUT_SAMPLE_RATE
    }


@app.get("/voices")
async def list_voices():
    """List all available voices for the current provider."""
    try:
        voices = await tts_provider.list_voices()
        return {"voices": voices}
    except Exception as e:
        status_code, error_message = tts_provider.parse_error(e)
        return JSONResponse(
            status_code=status_code,
            content={"error": f"Failed to list voices: {error_message}"}
        )


@app.get("/voices/find")
async def find_voices(
    gender: Optional[str] = None,
    language: Optional[str] = None,
    locale: Optional[str] = None
):
    """
    Find voices by attributes (Edge TTS only).
    
    Query parameters:
    - gender: Filter by gender (e.g., "Female", "Male")
    - language: Filter by language code (e.g., "zh", "en")
    - locale: Filter by locale (e.g., "zh-CN", "en-US")
    """
    if TTS_PROVIDER != "edge-tts":
        return JSONResponse(
            status_code=400,
            content={"error": "Voice filtering is only available for Edge TTS provider"}
        )
    
    try:
        filters = {}
        if gender:
            filters["Gender"] = gender
        if language:
            filters["Language"] = language
        if locale:
            filters["Locale"] = locale
        
        if not filters:
            return JSONResponse(
                status_code=400,
                content={"error": "At least one filter parameter (gender, language, locale) is required"}
            )
        
        # EdgeTTSProvider has find_voices method
        if hasattr(tts_provider, "find_voices"):
            voices = await tts_provider.find_voices(**filters)
            return {"voices": voices}
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "find_voices method not available"}
            )
    except Exception as e:
        status_code, error_message = tts_provider.parse_error(e)
        return JSONResponse(
            status_code=status_code,
            content={"error": f"Failed to find voices: {error_message}"}
        )


class SynthesizeRequest(BaseModel):
    """Request model for TTS synthesis."""
    text: str
    voice: Optional[str] = None
    rate: Optional[str] = None
    pitch: Optional[str] = None
    speaker: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None


@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    """
    Synthesize text to speech (non-streaming).
    
    Returns complete audio as binary data (int16 PCM format, 16kHz, mono).
    """
    # Log request info
    logger.info(f"Synthesis request: {json.dumps(request.model_dump(), ensure_ascii=False)}")
    
    try:
        # Build provider parameters
        provider_params = {}
        if request.voice is not None:
            provider_params["voice"] = request.voice
        if request.rate is not None:
            provider_params["rate"] = request.rate
        if request.pitch is not None:
            provider_params["pitch"] = request.pitch
        if request.speaker is not None:
            provider_params["speaker"] = request.speaker
        if request.temperature is not None:
            provider_params["temperature"] = request.temperature
        if request.top_p is not None:
            provider_params["top_p"] = request.top_p
        if request.top_k is not None:
            provider_params["top_k"] = request.top_k
        
        # Validate text
        if not request.text or not isinstance(request.text, str):
            raise HTTPException(
                status_code=400,
                detail="Text must be a non-empty string"
            )
        
        # Synthesize audio
        audio_bytes = await tts_provider.synthesize(request.text, **provider_params)
        
        # Return audio as binary response
        return Response(
            content=audio_bytes,
            media_type="audio/pcm",
            headers={
                "Content-Type": "audio/pcm",
                "Sample-Rate": str(OUTPUT_SAMPLE_RATE),
                "Channels": "1",
                "Format": "int16"
            }
        )
    except Exception as e:
        status_code, error_message = tts_provider.parse_error(e)
        raise HTTPException(
            status_code=status_code,
            detail=f"Synthesis failed: {error_message}"
        )


@app.websocket("/synthesize/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for streaming TTS synthesis.
    
    Receives JSON messages:
    - {"type": "text", "text": "...", "finalize": False} - Text to synthesize
    - {"type": "text", "text": "", "finalize": True} - Finalize synthesis
    - {"type": "ping"} - Keepalive ping (responds with pong)
    
    Sends:
    - Audio chunks as bytes (int16 PCM format, 16kHz, mono)
    - JSON messages: {"type": "done"}, {"type": "pong"}, or {"type": "error", "message": "..."}
    """
    await websocket.accept()
    
    # Per-connection state
    text_buffer = []
    # Provider-specific parameters (will be passed to synthesize_stream)
    provider_params = {}
    
    # Keepalive ping interval (seconds)
    KEEPALIVE_INTERVAL = 2.0  # Send ping every 20 seconds
    last_ping_time = asyncio.get_event_loop().time()
    
    try:
        while True:
            # Check if we need to send a keepalive ping before waiting for messages
            current_time = asyncio.get_event_loop().time()
            time_since_last_ping = current_time - last_ping_time
            
            if time_since_last_ping >= KEEPALIVE_INTERVAL:
                # Send keepalive ping proactively
                try:
                    await websocket.send_text(json.dumps({"type": "ping"}))
                    last_ping_time = current_time
                    logger.debug("Sent keepalive ping")
                except Exception:
                    # Connection likely closed
                    break
            
            # Calculate timeout: use shorter of remaining time until next ping or max receive timeout
            # Use very long timeout to prevent connection timeouts (1 hour)
            time_until_next_ping = max(0, KEEPALIVE_INTERVAL - time_since_last_ping)
            receive_timeout = min(time_until_next_ping if time_until_next_ping > 0 else KEEPALIVE_INTERVAL, 3600.0)
            
            # Receive message with timeout
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=receive_timeout)
            except asyncio.TimeoutError:
                # Timeout is normal - continue loop to check for ping or receive again
                continue
            except Exception:
                break
            
            # Parse JSON message
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
                continue
            
            msg_type = data.get("type")
            
            if msg_type == "text":
                text = data.get("text", "")
                finalize = data.get("finalize", False)
                
                # Update provider-specific parameters if provided
                # These will be passed to synthesize_stream
                for param in ["voice", "rate", "pitch", "speaker", "temperature", "top_p", "top_k", "stream_speed"]:
                    if param in data:
                        provider_params[param] = data[param]
                
                # Add text to buffer
                if text:
                    text_buffer.append(text)
                
                # Synthesize immediately when we have text (don't wait for finalize)
                # This enables true streaming: TTS starts as soon as first token arrives
                if text_buffer:
                    full_text = "".join(text_buffer)
                    text_buffer.clear()
                    
                    # Track synthesis start time (only on first synthesis)
                    import time
                    if not hasattr(websocket, '_synthesis_started'):
                        synthesis_start_time = time.perf_counter()
                        websocket._synthesis_started = True
                        websocket._synthesis_start_time = synthesis_start_time
                    else:
                        synthesis_start_time = websocket._synthesis_start_time
                    
                    first_audio_sent = getattr(websocket, '_first_audio_sent', False)
                    
                    # Log request info
                    request_info = {
                        "text": full_text,
                        "finalize": finalize,
                        **provider_params
                    }
                    logger.info(f"WebSocket synthesis request: {json.dumps(request_info, ensure_ascii=False)}")
                    
                    # --- IMMEDIATE ACK: Send status message instantly to reset client timer ---
                    await websocket.send_text(json.dumps({"type": "status", "message": "received"}))
                    
                    # Synthesize audio using the configured provider
                    try:
                        # Track last keepalive time for long synthesis
                        last_keepalive_time = synthesis_start_time
                        KEEPALIVE_DURING_SYNTHESIS = 10.0  # Send keepalive every 10 seconds during synthesis
                        chunk_count = 0
                        
                        # --- PARALLEL HEARTBEAT: Keep socket alive while waiting for Edge-TTS connection ---
                        async def send_heartbeats():
                            """Background task to send heartbeats while waiting for TTS provider to connect."""
                            try:
                                while True:
                                    await asyncio.sleep(0.5)  # Send every 0.5 seconds (beating the 1s timeout)
                                    await websocket.send_text(json.dumps({"type": "ping_processing"}))
                            except asyncio.CancelledError:
                                pass
                            except Exception as e:
                                # Connection likely closed
                                logger.debug(f"Heartbeat task stopped: {e}")
                        
                        # Start the heartbeat task BEFORE calling TTS provider
                        heartbeat_task = asyncio.create_task(send_heartbeats())
                        
                        try:
                            # Stream audio chunks from provider
                            # The heartbeat_task runs in the background while this line waits for Edge-TTS connection
                            async for audio_chunk in tts_provider.synthesize_stream(full_text, **provider_params):
                                # Once we have audio, we can cancel the heartbeat
                                if not heartbeat_task.cancelled():
                                    heartbeat_task.cancel()
                                    try:
                                        await heartbeat_task
                                    except asyncio.CancelledError:
                                        pass
                                
                                # Track first audio chunk latency
                                if not first_audio_sent:
                                    first_audio_time = time.perf_counter()
                                    time_to_first_audio = first_audio_time - synthesis_start_time
                                    logger.info(f"TTS time-to-first-audio: {time_to_first_audio*1000:.0f}ms")
                                    first_audio_sent = True
                                    websocket._first_audio_sent = True
                                
                                # Send audio chunk
                                try:
                                    await websocket.send_bytes(audio_chunk)
                                    chunk_count += 1
                                except Exception as send_error:
                                    logger.error(f"Error sending audio chunk: {send_error}")
                                    raise
                                
                                # Send periodic keepalive during long synthesis
                                current_time = time.perf_counter()
                                if current_time - last_keepalive_time >= KEEPALIVE_DURING_SYNTHESIS:
                                    try:
                                        await websocket.send_text(json.dumps({
                                            "type": "progress",
                                            "message": f"Synthesizing... ({chunk_count} chunks sent)"
                                        }))
                                        last_keepalive_time = current_time
                                        logger.debug(f"Sent synthesis progress: {chunk_count} chunks")
                                    except Exception:
                                        # If we can't send keepalive, connection might be closed
                                        logger.warning("Failed to send synthesis progress keepalive")
                            
                            # Calculate total synthesis time
                            total_synthesis_time = time.perf_counter() - synthesis_start_time
                            logger.info(f"TTS total synthesis time: {total_synthesis_time*1000:.0f}ms, {chunk_count} chunks")
                            
                            # Send completion message
                            await websocket.send_text(json.dumps({
                                "type": "done"
                            }))
                            
                        except Exception as e:
                            logger.error(f"Error during TTS synthesis: {e}", exc_info=True)
                            status_code, error_message = tts_provider.parse_error(e)
                            try:
                                await websocket.send_text(json.dumps({
                                    "type": "error",
                                    "message": error_message
                                }))
                            except Exception:
                                # Connection might already be closed
                                logger.warning("Could not send error message, connection may be closed")
                        finally:
                            # Ensure heartbeat is killed
                            if not heartbeat_task.cancelled():
                                heartbeat_task.cancel()
                                try:
                                    await heartbeat_task
                                except asyncio.CancelledError:
                                    pass
                    except Exception as outer_e:
                        # Handle any errors that occur before or during synthesis setup
                        logger.error(f"Error in synthesis setup: {outer_e}", exc_info=True)
                        # If heartbeat_task was created, clean it up
                        if 'heartbeat_task' in locals() and not heartbeat_task.cancelled():
                            heartbeat_task.cancel()
                            try:
                                await heartbeat_task
                            except asyncio.CancelledError:
                                pass
                        # Try to send error message
                        try:
                            status_code, error_message = tts_provider.parse_error(outer_e)
                            await websocket.send_text(json.dumps({
                                "type": "error",
                                "message": error_message
                            }))
                        except Exception:
                            logger.warning("Could not send error message, connection may be closed")
                
                elif finalize:
                        # Finalize requested - synthesize any remaining buffered text
                        if text_buffer:
                            full_text = "".join(text_buffer)
                            text_buffer.clear()
                            
                            # Use existing synthesis start time if available
                            if hasattr(websocket, '_synthesis_start_time'):
                                synthesis_start_time = websocket._synthesis_start_time
                            else:
                                import time
                                synthesis_start_time = time.perf_counter()
                                websocket._synthesis_start_time = synthesis_start_time
                            
                            first_audio_sent = getattr(websocket, '_first_audio_sent', False)
                            
                            # Synthesize remaining text
                            async for audio_chunk in tts_provider.synthesize_stream(full_text, **provider_params):
                                if not first_audio_sent:
                                    import time
                                    first_audio_time = time.perf_counter()
                                    time_to_first_audio = first_audio_time - synthesis_start_time
                                    logger.info(f"TTS time-to-first-audio: {time_to_first_audio*1000:.0f}ms")
                                    first_audio_sent = True
                                    websocket._first_audio_sent = True
                                
                                await websocket.send_bytes(audio_chunk)
                            
                            await websocket.send_text(json.dumps({"type": "done"}))
                        else:
                            # Empty text but finalize requested
                            await websocket.send_text(json.dumps({
                                "type": "done"
                            }))
            
            elif msg_type == "config":
                # Update provider configuration
                for param in ["voice", "rate", "pitch", "speaker", "temperature", "top_p", "top_k", "stream_speed"]:
                    if param in data:
                        provider_params[param] = data[param]
                
                await websocket.send_text(json.dumps({
                    "type": "config_updated"
                }))
            
            elif msg_type == "ping":
                # Respond to client ping with pong
                await websocket.send_text(json.dumps({
                    "type": "pong"
                }))
                last_ping_time = asyncio.get_event_loop().time()
            
            elif msg_type == "pong":
                # Client responded to our ping - update last ping time
                last_ping_time = asyncio.get_event_loop().time()
            
            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}"
                }))
    
    except Exception as e:
        # Try to send error message, but ignore if websocket is already closed
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Server error: {str(e)}"
            }))
        except Exception:
            pass


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "provider": TTS_PROVIDER}


if __name__ == "__main__":
    logger.info(f"Starting TTS server on {HOST}:{PORT}...")
    logger.info(f"Provider: {TTS_PROVIDER}")
    logger.info(f"Output sample rate: {OUTPUT_SAMPLE_RATE} Hz")
    # Configure uvicorn to send WebSocket pings for keepalive
    # Fast ping intervals to prevent client timeout during Edge-TTS connection establishment
    # ws_ping_timeout=None disables timeout enforcement to prevent connection drops
    uvicorn.run(
        app, 
        host=HOST, 
        port=PORT,
        ws_ping_interval=5.0,  # Ping every 5 seconds
        ws_ping_timeout=None   # Disable ping timeout to prevent connection drops
    )

