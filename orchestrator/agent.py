"""
Agent

Main orchestration logic for the voice agent.
Coordinates STT, LLM, TTS, and OCR services.
"""
import asyncio
import json
import httpx
import websockets
from typing import Optional
from fastapi import FastAPI
import uvicorn
import logging

from orchestrator.config import Config
from orchestrator.logging_config import setup_logging, get_logger
from orchestrator.context_manager import ContextManager
from orchestrator.ocr_client import OCRClient
from audio.audio_player import AudioPlayer
from orchestrator.stt_client import STTClient

# Set up logging
setup_logging()
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
    
    async def on_transcript(self, text: str):
        """Handle transcript from STT."""
        logger.info(f"Transcript received: {text}")
        
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
            
            # Stream LLM response
            async for token in self.stream_llm_response(context_data["prompt"]):
                # Stream token to TTS
                await self.stream_tts_text(token)
            
            # Finalize TTS
            await self.finalize_tts()
        
        except Exception as e:
            logger.error(f"Error processing user input: {e}", exc_info=True)
    
    async def stream_llm_response(self, prompt: str) -> str:
        """Stream LLM response tokens."""
        full_response = ""
        try:
            logger.info("LLM response starting...")
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
        try:
            while self.tts_receiver_running:
                if self.tts_websocket is None:
                    await asyncio.sleep(0.1)
                    continue
                
                try:
                    # Wait for message with timeout to allow checking if we should continue
                    message = await asyncio.wait_for(self.tts_websocket.recv(), timeout=1.0)
                    
                    if isinstance(message, bytes):
                        # Audio chunk - queue it for playback (non-blocking)
                        await self.audio_player.play_audio_chunk(message)
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
                    # No message received, continue loop
                    continue
                except (websockets.exceptions.ConnectionClosed, ConnectionError, OSError) as e:
                    logger.warning(f"TTS WebSocket connection closed: {e}")
                    break
                except Exception as e:
                    logger.error(f"Error in TTS receiver loop: {e}", exc_info=True)
                    await asyncio.sleep(0.1)
        
        except Exception as e:
            logger.error(f"TTS receiver loop error: {e}", exc_info=True)
        finally:
            logger.info("TTS receiver loop stopped")
    
    async def stream_tts_text(self, text: str):
        """Stream text chunk to TTS service (non-blocking)."""
        try:
            # Connect to TTS WebSocket if not already connected
            if not hasattr(self, 'tts_websocket') or self.tts_websocket is None:
                logger.info("TTS streaming starting...")
                # Configure websocket with ping_interval and ping_timeout for keepalive
                self.tts_websocket = await websockets.connect(
                    Config.get_tts_websocket_url(),
                    ping_interval=20,  # Send ping every 20 seconds
                    ping_timeout=10    # Wait 10 seconds for pong response
                )
                
                # Start receiver task if not already running
                if not self.tts_receiver_running:
                    self.tts_receiver_running = True
                    self.tts_receiver_task = asyncio.create_task(self._tts_receiver_loop())
            
            # Send text chunk (TTS server will synthesize immediately)
            # Audio chunks will be received by the background receiver task
            await self.tts_websocket.send(json.dumps({
                "type": "text",
                "text": text,
                "finalize": False
            }))
            logger.debug(f"TTS text chunk sent: {text[:50]}...")
        
        except Exception as e:
            logger.error(f"Error streaming TTS: {e}", exc_info=True)
            # Reset TTS connection
            if hasattr(self, 'tts_websocket'):
                try:
                    await self.tts_websocket.close()
                except:
                    pass
                self.tts_websocket = None
    
    async def finalize_tts(self):
        """Finalize TTS synthesis. Audio chunks will be received by the background receiver task."""
        try:
            if not hasattr(self, 'tts_websocket') or self.tts_websocket is None:
                logger.warning("TTS WebSocket not connected, cannot finalize")
                return
            
            logger.info("Finalizing TTS...")
            # Send finalize message (may trigger synthesis if there's buffered text)
            # The background receiver task will handle receiving any remaining audio chunks
            await self.tts_websocket.send(json.dumps({
                "type": "text",
                "text": "",
                "finalize": True
            }))
            
            # Give the receiver task a moment to process the finalize message
            # and any remaining audio chunks
            await asyncio.sleep(0.5)
        
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


if __name__ == "__main__":
    logger.info(f"Starting Orchestrator server on {Config.ORCHESTRATOR_HOST}:{Config.ORCHESTRATOR_PORT}...")
    uvicorn.run(app, host=Config.ORCHESTRATOR_HOST, port=Config.ORCHESTRATOR_PORT)

