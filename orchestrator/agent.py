"""
Agent

Main orchestration logic for the voice agent.
Coordinates STT, LLM, TTS, and OCR services.
"""
import asyncio
import json
import httpx
import websockets
from typing import Optional, Dict
from fastapi import FastAPI
import uvicorn
import logging

from orchestrator.config import Config
from orchestrator.logging_config import setup_logging, get_logger
from orchestrator.context_manager import ContextManager
from orchestrator.ocr_client import OCRClient
from orchestrator.latency_tracker import LatencyTracker
from audio.audio_player import AudioPlayer
from orchestrator.stt_client import STTClient

# Set up logging
log_level = getattr(logging, Config.LOG_LEVEL, logging.INFO)
setup_logging(level=log_level)
logger = get_logger(__name__)


class Agent:
    """Main voice agent orchestrator."""
    
    def __init__(self):
        """Initialize voice agent."""
        self.context_manager = ContextManager()
        self.audio_player = AudioPlayer()
        self.ocr_client = OCRClient()
        self.stt_client = STTClient(self.on_transcript)
        self.running = False
        self.tts_websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.tts_receiver_task: Optional[asyncio.Task] = None
        self.tts_receiver_running = False
        
        # Latency tracking
        self.latency_tracker = LatencyTracker() if Config.ENABLE_LATENCY_TRACKING else None
        self._current_speech_end_time: Optional[float] = None
        self._llm_first_token_received = False
        self._tts_first_audio_received = False
        self._latest_latency_results: Optional[Dict] = None
    
    async def on_transcript(self, text: str, speech_end_time: Optional[float] = None, stt_latency: Optional[float] = None):
        """
        Handle transcript from STT.
        
        Args:
            text: Transcript text
            speech_end_time: Optional timestamp when speech ended (from STT server)
            stt_latency: Optional STT processing latency in seconds
        """
        logger.info(f"Transcript received: {text}")
        
        # Track speech end time for latency measurement
        if self.latency_tracker:
            self.latency_tracker.start_round()
            transcript_received_time = self.latency_tracker.mark("transcript_received")
            
            # Convert speech_end_time to perf_counter time base
            # If we have speech_end_time from STT (time.time()) and stt_latency,
            # we can calculate when speech ended relative to transcript_received
            if speech_end_time and stt_latency is not None:
                # speech_end_time is in time.time() (wall clock)
                # transcript_received_time is in time.perf_counter() (monotonic)
                # We need to convert: speech_end = transcript_received - stt_latency
                # This gives us speech_end in perf_counter time base
                import time
                self._current_speech_end_time = transcript_received_time - stt_latency
                self.latency_tracker.marks["speech_end"] = self._current_speech_end_time
            elif speech_end_time:
                # If we have speech_end_time but no stt_latency, estimate
                # Assume minimal network delay and use transcript_received - small offset
                import time
                # Estimate: speech ended slightly before transcript received
                # Use a small default STT latency estimate (e.g., 0.3s)
                estimated_stt_latency = 0.3
                self._current_speech_end_time = transcript_received_time - estimated_stt_latency
                self.latency_tracker.marks["speech_end"] = self._current_speech_end_time
            else:
                # Use current time as fallback (slightly less accurate)
                self._current_speech_end_time = self.latency_tracker.mark("speech_end")
            
            # Record STT latency if provided
            if stt_latency is not None:
                self.latency_tracker.measurements["stt_latency"].append(stt_latency)
                if self.latency_tracker.current_round is not None:
                    self.latency_tracker.current_round["stt_latency"] = stt_latency
            
            self._llm_first_token_received = False
            self._tts_first_audio_received = False
            
        # Add to conversation history
        self.context_manager.add_user_message(text)
        
        # Process with LLM and TTS
        await self.process_user_input(text)
    
    async def fetch_ocr_texts(self) -> str:
        """
        Fetch all OCR texts from the OCR service.
        
        Returns:
            All OCR texts as a single string
        """
        return await self.ocr_client.get_all_texts()
    
    async def process_user_input(self, user_text: str):
        """Process user input through LLM and TTS pipeline."""
        try:
            # Format context for LLM
            context_data = self.context_manager.format_context_for_llm(user_text)
            
            # Track LLM start
            if self.latency_tracker:
                self.latency_tracker.start("llm_total")
            
            # Stream LLM response
            async for token in self.stream_llm_response(context_data["prompt"]):
                # Stream token to TTS
                await self.stream_tts_text(token)
            
            # Track LLM completion
            if self.latency_tracker:
                self.latency_tracker.end("llm_total")
            
            # Finalize TTS
            await self.finalize_tts()
            
            # Calculate and log latencies if tracking is enabled
            if self.latency_tracker:
                round_data = self.latency_tracker.end_round()
                if round_data:
                    # Debug: Log what measurements we have
                    logger.debug(f"Latency measurements captured: {list(round_data.keys())}")
                    
                    # Store latest results for API access
                    self._latest_latency_results = round_data.copy()
                    
                    # Log formatted results
                    logger.info("=" * 60)
                    logger.info("Latency Measurement Results")
                    logger.info("=" * 60)
                    logger.info(self.latency_tracker.format_round(round_data))
                    logger.info("=" * 60)
        
        except Exception as e:
            logger.error(f"Error processing user input: {e}", exc_info=True)
    
    async def stream_llm_response(self, prompt: str) -> str:
        """Stream LLM response tokens."""
        full_response = ""
        try:
            logger.info("LLM response starting...")
            
            # Track LLM request start (for time-to-first-token)
            if self.latency_tracker:
                self.latency_tracker.start("llm_time_to_first_token")
            
            async with httpx.AsyncClient() as client:
                # Use SSE streaming endpoint
                async with client.stream(
                    "POST",
                    Config.get_llm_stream_url(),
                    json={
                        "prompt": prompt
                    },
                    timeout=60.0
                ) as response:
                    response.raise_for_status()
                    
                    current_event = None
                    async for line in response.aiter_lines():
                        # Skip empty lines
                        if not line.strip():
                            continue
                        
                        # Parse SSE format: event: <type> and data: <json>
                        if line.startswith("event: "):
                            current_event = line[7:].strip()
                            continue
                            
                        if line.startswith("data: "):
                            data_str = line[6:].strip()  # Remove "data: " prefix
                            try:
                                data = json.loads(data_str)
                                
                                if current_event == "token":
                                    # Data format: {"token": "..."}
                                    token = data.get("token", "")
                                    if token:
                                        # Track first token latency
                                        if self.latency_tracker and not self._llm_first_token_received:
                                            self.latency_tracker.end("llm_time_to_first_token")
                                            self.latency_tracker.mark("llm_first_token")
                                            self._llm_first_token_received = True
                                        
                                        full_response += token
                                        yield token
                                elif current_event == "done":
                                    # Data format: {"status": "complete"}
                                    break
                                elif current_event == "error":
                                    # Data format: {"error": "...", "status_code": ...}
                                    error = data.get("error", "Unknown error")
                                    logger.error(f"LLM error: {error}")
                                    break
                                
                                # Reset event after processing
                                current_event = None
                            except json.JSONDecodeError:
                                current_event = None
                                continue
        
        except Exception as e:
            logger.error(f"Error streaming LLM response: {e}", exc_info=True)
        finally:
            # Always log and add to context, even if there was an error
            if full_response:
                logger.info(f"LLM response: {full_response}")
                self.context_manager.add_assistant_message(full_response)
    
    async def _tts_receiver_loop(self):
        """Background task to receive audio chunks from TTS WebSocket and queue them for playback."""
        logger.info("TTS receiver loop started")
        consecutive_timeouts = 0
        MAX_CONSECUTIVE_TIMEOUTS = 300  # Allow up to 5 minutes of no messages (for long synthesis)
        
        try:
            while self.tts_receiver_running:
                if self.tts_websocket is None:
                    await asyncio.sleep(0.1)
                    continue
                
                try:
                    # Wait for message with timeout to allow checking if we should continue
                    # Use very long timeout to prevent connection timeouts (1 hour)
                    message = await asyncio.wait_for(self.tts_websocket.recv(), timeout=3600.0)
                    
                    # Reset timeout counter on successful receive
                    consecutive_timeouts = 0
                    
                    if isinstance(message, bytes):
                        # Audio chunk - queue it for playback (non-blocking)
                        # Track first audio chunk received
                        if self.latency_tracker and not self._tts_first_audio_received:
                            self.latency_tracker.end("tts_time_to_first_audio")
                            self.latency_tracker.mark("tts_first_audio")
                            self._tts_first_audio_received = True
                        
                        await self.audio_player.play_audio_chunk(message, latency_tracker=self.latency_tracker)
                    else:
                        # JSON message
                        try:
                            data = json.loads(message)
                            msg_type = data.get("type")
                            
                            if msg_type == "done":
                                logger.debug("TTS synthesis chunk complete")
                            elif msg_type == "error":
                                error_msg = data.get("message", "Unknown error")
                                logger.error(f"TTS error: {error_msg}")
                            elif msg_type == "progress":
                                # Progress update during long synthesis
                                progress_msg = data.get("message", "")
                                logger.debug(f"TTS synthesis progress: {progress_msg}")
                            elif msg_type == "ping":
                                # Respond to server ping with pong
                                try:
                                    await self.tts_websocket.send(json.dumps({"type": "pong"}))
                                except Exception:
                                    pass
                            elif msg_type == "pong":
                                # Server responded to our ping
                                logger.debug("Received pong from TTS server")
                        except json.JSONDecodeError:
                            logger.warning(f"Received non-JSON text message: {message}")
                
                except asyncio.TimeoutError:
                    # No message received - this is normal during long synthesis
                    consecutive_timeouts += 1
                    if consecutive_timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
                        logger.warning(f"TTS WebSocket: No messages received for {MAX_CONSECUTIVE_TIMEOUTS * 5} seconds, connection may be dead")
                        # Don't break - continue waiting in case synthesis is still ongoing
                        # The connection will be detected as closed on the next recv attempt
                    continue
                except (websockets.exceptions.ConnectionClosed, ConnectionError, OSError) as e:
                    # Connection closed - log and reset connection state
                    logger.warning(f"TTS WebSocket connection closed: {e}")
                    # Reset websocket reference so it can be reconnected
                    self.tts_websocket = None
                    # Continue loop to wait for reconnection (don't break)
                    # The loop will check if tts_websocket is None and wait
                    continue
                except Exception as e:
                    logger.error(f"Error in TTS receiver loop: {e}", exc_info=True)
                    await asyncio.sleep(0.1)
        
        except Exception as e:
            logger.error(f"TTS receiver loop error: {e}", exc_info=True)
        finally:
            logger.info("TTS receiver loop stopped")
    
    async def stream_tts_text(self, text: str):
        """Stream text chunk to TTS service (non-blocking)."""
        max_retries = 2
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                # Connect to TTS WebSocket if not already connected
                if not hasattr(self, 'tts_websocket') or self.tts_websocket is None:
                    logger.info("TTS streaming starting...")
                    
                    # Configure websocket with ping_interval and ping_timeout for keepalive
                    self.tts_websocket = await websockets.connect(
                        Config.get_tts_websocket_url(),
                        # Increase these to ensure the library doesn't close the connection 
                        # if the server is busy generating for a few seconds.
                        ping_interval=None,  # Let the Server drive the Pings (Server-side pings)
                        ping_timeout=None,   # Disable client-side strict enforcement
                        close_timeout=10     # Give 10s for a graceful close
                    )
                    logger.info("TTS WebSocket connected")
                    
                    # Start receiver task if not already running or if it has stopped
                    if not self.tts_receiver_running:
                        # Receiver not running - start it
                        self.tts_receiver_running = True
                        self.tts_receiver_task = asyncio.create_task(self._tts_receiver_loop())
                        logger.debug("TTS receiver task started")
                    elif self.tts_receiver_task and self.tts_receiver_task.done():
                        # Receiver task exists but has stopped - restart it
                        logger.info("TTS receiver task stopped, restarting...")
                        # Clean up the old task
                        try:
                            if not self.tts_receiver_task.cancelled():
                                self.tts_receiver_task.cancel()
                                try:
                                    await self.tts_receiver_task
                                except asyncio.CancelledError:
                                    pass
                        except Exception as e:
                            logger.warning(f"Error cleaning up old receiver task: {e}")
                        
                        # Start new receiver task
                        self.tts_receiver_running = True
                        self.tts_receiver_task = asyncio.create_task(self._tts_receiver_loop())
                        logger.debug("TTS receiver task restarted")
                
                # Track TTS start when sending first text chunk (not when connecting)
                # This ensures we track from when we actually send text, not when connection is established
                if self.latency_tracker and not self._tts_first_audio_received:
                    self.latency_tracker.start("tts_time_to_first_audio")
                
                # Send text chunk (TTS server will synthesize immediately)
                # Audio chunks will be received by the background receiver task
                await self.tts_websocket.send(json.dumps({
                    "type": "text",
                    "text": text,
                    "finalize": False
                }))
                logger.debug(f"TTS text chunk sent: {text[:50]}...")
                # Success - break out of retry loop
                break
            
            except (websockets.exceptions.ConnectionClosed, ConnectionError, OSError) as e:
                logger.warning(f"TTS WebSocket connection error: {e}")
                # Reset connection and retry
                if hasattr(self, 'tts_websocket') and self.tts_websocket is not None:
                    try:
                        await self.tts_websocket.close()
                    except:
                        pass
                    self.tts_websocket = None
                
                retry_count += 1
                if retry_count <= max_retries:
                    logger.info(f"Retrying TTS connection (attempt {retry_count}/{max_retries})...")
                    await asyncio.sleep(0.5)  # Brief delay before retry
                else:
                    logger.error(f"Failed to send TTS text after {max_retries} retries")
            
            except Exception as e:
                logger.error(f"Error streaming TTS: {e}", exc_info=True)
                # Reset TTS connection
                if hasattr(self, 'tts_websocket') and self.tts_websocket is not None:
                    try:
                        await self.tts_websocket.close()
                    except:
                        pass
                    self.tts_websocket = None
                # Don't retry on unexpected errors
                break
    
    async def finalize_tts(self):
        """Finalize TTS synthesis. Audio chunks will be received by the background receiver task."""
        try:
            if not hasattr(self, 'tts_websocket') or self.tts_websocket is None:
                logger.warning("TTS WebSocket not connected, cannot finalize")
                return
            
            logger.info("Finalizing TTS...")
            # Send finalize message (may trigger synthesis if there's buffered text)
            # The background receiver task will handle receiving any remaining audio chunks
            try:
                await self.tts_websocket.send(json.dumps({
                    "type": "text",
                    "text": "",
                    "finalize": True
                }))
            except (websockets.exceptions.ConnectionClosed, ConnectionError, OSError) as e:
                logger.warning(f"TTS WebSocket connection closed during finalize: {e}")
                # Connection may have closed, but synthesis might still complete
                # The receiver loop will handle any remaining audio if connection is re-established
                return
            
            # Give the receiver task a moment to process the finalize message
            # and any remaining audio chunks (longer wait for long synthesis)
            await asyncio.sleep(1.0)
        
        except Exception as e:
            logger.error(f"Error finalizing TTS: {e}", exc_info=True)
    
    async def start(self):
        """Start the agent."""
        self.running = True
        # Connect to STT server
        asyncio.create_task(self.stt_client.connect())
        logger.info("Voice Agent started")
    
    async def stop(self):
        """Stop the agent."""
        self.running = False
        
        # Stop TTS receiver task
        self.tts_receiver_running = False
        if self.tts_receiver_task:
            self.tts_receiver_task.cancel()
            try:
                await self.tts_receiver_task
            except asyncio.CancelledError:
                pass
        
        # Close TTS WebSocket
        if self.tts_websocket:
            try:
                await self.tts_websocket.close()
            except:
                pass
            self.tts_websocket = None
        
        # Stop audio playback
        await self.audio_player.stop()
        
        # Close STT client connection
        await self.stt_client.close()


# --- FastAPI Server ---
app = FastAPI()
agent: Optional[Agent] = None


@app.on_event("startup")
async def startup_event():
    """Start agent on server startup."""
    global agent
    try:
        agent = Agent()
        asyncio.create_task(agent.start())
    except Exception as e:
        logger.error(f"Error starting agent: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Stop agent on server shutdown."""
    global agent
    if agent:
        await agent.stop()


@app.get("/")
async def root():
    return {"message": "Orchestrator Service is running"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy"
    }


@app.get("/ocr/texts")
async def get_ocr_texts():
    """Fetch all OCR texts from the OCR service."""
    if not agent:
        return {"error": "Agent not initialized"}
    
    texts = await agent.fetch_ocr_texts()
    return {
        "texts": texts,
        "count": len(texts.split("\n")) if texts else 0
    }


@app.get("/latency/latest")
async def get_latest_latency():
    """Get latest latency measurement results."""
    if not agent:
        return {"error": "Agent not initialized"}
    
    if not Config.ENABLE_LATENCY_TRACKING:
        return {"error": "Latency tracking is not enabled"}
    
    if agent._latest_latency_results is None:
        return {"message": "No latency measurements yet"}
    
    # Format results with readable names
    formatted_results = {}
    for key, value in agent._latest_latency_results.items():
        if isinstance(value, float):
            formatted_results[key] = {
                "seconds": value,
                "milliseconds": value * 1000,
                "formatted": agent.latency_tracker.format_latency(value) if agent.latency_tracker else f"{value*1000:.0f}ms"
            }
        else:
            formatted_results[key] = value
    
    return {
        "results": formatted_results,
        "raw": agent._latest_latency_results
    }


if __name__ == "__main__":
    logger.info(f"Starting Orchestrator server on {Config.ORCHESTRATOR_HOST}:{Config.ORCHESTRATOR_PORT}...")
    uvicorn.run(app, host=Config.ORCHESTRATOR_HOST, port=Config.ORCHESTRATOR_PORT)

