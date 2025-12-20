"""
Agent

Main orchestration logic for the voice agent.
Coordinates STT, LLM, TTS, and OCR services.
"""
import asyncio
import json
import re
import websockets
from typing import Optional, Dict, Set, List
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import logging
import sys
import time
from datetime import datetime

# Add project root to path to import config_loader
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config_loader import get_config
from orchestrator.logging_config import setup_logging, get_logger
from orchestrator.context_manager import ContextManager
from orchestrator.ocr_client import OCRClient
from audio.audio_player import AudioPlayer
from orchestrator.stt_client import STTClient
from llm.providers import GeminiProvider, OllamaProvider
from llm.base import LLMProvider

# Set up logging
log_level = getattr(logging, get_config("orchestrator", "log_level", default="INFO").upper(), logging.INFO)
setup_logging(level=log_level)
logger = get_logger(__name__)


class Agent:
    """Main voice agent orchestrator."""
    
    def __init__(self):
        """Initialize voice agent."""
        # Load system prompt file path from config if available
        system_prompt_file = get_config("orchestrator", "system_prompt_file", default=None)
        self.context_manager = ContextManager(system_prompt_file=system_prompt_file)
        self.audio_player = AudioPlayer(on_play_state=self.on_play_state_changed)
        self.ocr_client = OCRClient()
        self.stt_client = STTClient(self.on_transcript, on_event=self.on_stt_event)
        self.running = False
        self.tts_websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.tts_receiver_task: Optional[asyncio.Task] = None
        self.tts_receiver_running = False
        self.cancel_event = asyncio.Event()
        self.event_subscribers: Set[asyncio.Queue] = set()
        self.event_lock = asyncio.Lock()
        self.activity_state: Dict[str, bool] = {
            "listening": True,
            "transcribing": False,
            "responding": False,
            "synthesizing": False,
            "playing": False,
        }
        
        # Initialize LLM provider directly
        self.llm_provider = self._initialize_llm_provider()
        
        # Check if thinking should be disabled
        provider_name = get_config("llm", "provider", default="ollama")
        self.disable_thinking = get_config("llm", "providers", provider_name, "disable_thinking", default=False) if provider_name == "ollama" else False
        
        # Latency tracking state
        self.speech_end_time: Optional[float] = None
        self.llm_request_sent_time: Optional[float] = None
        self.first_token_time: Optional[float] = None
        self.llm_full_response_time: Optional[float] = None
        self.first_tts_request_time: Optional[float] = None
        self.first_audio_received_time: Optional[float] = None
        self.first_audio_queued_time: Optional[float] = None

        # Track how many TTS synthesis requests are in flight
        self.pending_tts_requests: int = 0

        # Enable/disable latency tracking logs
        self.enable_latency_tracking = get_config("orchestrator", "enable_latency_tracking", default=False)

    def _latency_log(self, message: str):
        """Helper to conditionally emit latency logs."""
        if self.enable_latency_tracking:
            logger.info(message)
    
    async def subscribe_events(self) -> asyncio.Queue:
        """Create a subscriber queue for UI event streaming."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        async with self.event_lock:
            self.event_subscribers.add(queue)
        return queue
    
    async def unsubscribe_events(self, queue: asyncio.Queue):
        """Remove a subscriber queue."""
        async with self.event_lock:
            self.event_subscribers.discard(queue)
    
    async def publish_event(self, event: Dict):
        """Publish an event to all subscribers without blocking."""
        async with self.event_lock:
            subscribers = list(self.event_subscribers)
        
        for queue in subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Drop oldest item to make room, then retry once
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    logger.warning("UI event queue full; dropping event")
    
    async def set_activity(self, **kwargs):
        """Update activity flags and broadcast to UI."""
        self.activity_state.update(kwargs)
        await self.publish_event({
            "event": "activity",
            "state": self.activity_state.copy()
        })
    
    async def on_play_state_changed(self, is_playing: bool):
        """Callback from AudioPlayer to indicate playback activity."""
        await self.set_activity(playing=is_playing)
    
    def _initialize_llm_provider(self) -> LLMProvider:
        """Initialize LLM provider based on configuration."""
        provider_name = get_config("llm", "provider", default="ollama")
        
        if provider_name == "gemini":
            model = get_config("llm", "providers", "gemini", "model", default="gemini-2.5-flash")
            api_key = get_config("llm", "providers", "gemini", "api_key", default="")
            api_key = api_key if api_key else None  # Empty string becomes None
            logger.info(f"Initializing Gemini provider with model: {model}")
            return GeminiProvider(model=model, api_key=api_key)
            
        elif provider_name == "ollama":
            model = get_config("llm", "providers", "ollama", "model", default="llama3")
            base_url = get_config("llm", "providers", "ollama", "base_url", default="http://localhost:11434")
            timeout = float(get_config("llm", "providers", "ollama", "timeout", default=300.0))
            disable_thinking = get_config("llm", "providers", "ollama", "disable_thinking", default=False)
            logger.info(f"Initializing Ollama provider with model: {model}, base_url: {base_url}, disable_thinking: {disable_thinking}")
            return OllamaProvider(model=model, base_url=base_url, timeout=timeout, disable_thinking=disable_thinking)
            
        else:
            raise ValueError(f"Unknown LLM provider: {provider_name}. Supported: gemini, ollama")
    
    async def on_stt_event(self, event: Dict):
        """Forward STT interim/final events to UI subscribers."""
        try:
            await self.publish_event({
                "event": "stt",
                "stage": event.get("type"),
                "text": event.get("text", "")
            })
            
            stage = event.get("type")
            if stage == "interim":
                await self.set_activity(transcribing=True)
            elif stage == "final":
                await self.set_activity(transcribing=False)
        except Exception as e:
            logger.warning(f"Failed to publish STT event: {e}", exc_info=True)
    
    async def on_transcript(self, text: str):
        """
        Handle transcript from STT.
        
        Args:
            text: Transcript text
        """
        # Clear any previous cancel signals for the new interaction
        self.cancel_event.clear()
        
        # Reset latency tracking for new interaction
        self.speech_end_time = time.time()
        self.llm_request_sent_time = None
        self.first_token_time = None
        self.llm_full_response_time = None
        self.first_tts_request_time = None
        self.first_audio_received_time = None
        self.first_audio_queued_time = None
        
        readable_time = datetime.fromtimestamp(self.speech_end_time).isoformat()
        self._latency_log(f"[LATENCY] SPEECH_END: timestamp={self.speech_end_time:.6f}, readable_time={readable_time}, text={text[:50]}")
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
            process_start_time = time.time()
            readable_time = datetime.fromtimestamp(process_start_time).isoformat()
            self._latency_log(f"[LATENCY] PROCESS_START: timestamp={process_start_time:.6f}, readable_time={readable_time}")
            
            # Format context for LLM
            context_data = self.context_manager.format_context_for_llm(user_text)
            
            # Buffer to accumulate tokens into sentences
            sentence_buffer = ""
            
            # Sentence-ending punctuation (English and Chinese)
            # Pattern matches punctuation followed by whitespace or end of string
            # This ensures we only match complete sentences
            sentence_end_pattern = re.compile(r'[!?。！？](?=[\s\n]|$)')
            
            # Stream LLM response - pass messages and system_prompt for stateless providers
            async for token in self.stream_llm_response(
                messages=context_data["messages"],
                system_prompt=context_data.get("system_prompt")
            ):
                if self.cancel_event.is_set():
                    break
                
                # Accumulate tokens in buffer
                sentence_buffer += token
                
                # Check if we have a complete sentence
                # Find all sentence endings in the buffer
                match = sentence_end_pattern.search(sentence_buffer)
                
                while match:
                    # Found a complete sentence ending
                    # Position after the punctuation mark
                    punct_pos = match.end()
                    
                    # Extract the complete sentence (including the punctuation)
                    # Skip trailing whitespace after punctuation for sentence boundary
                    sentence_end_pos = punct_pos
                    while sentence_end_pos < len(sentence_buffer) and sentence_buffer[sentence_end_pos].isspace():
                        sentence_end_pos += 1
                    
                    complete_sentence = sentence_buffer[:sentence_end_pos].strip()
                    if complete_sentence:
                        # Send complete sentence to TTS
                        await self.stream_tts_text(complete_sentence)
                        logger.debug(f"Sent complete sentence to TTS: {complete_sentence[:100]}...")
                    
                    # Remove sent sentence from buffer
                    sentence_buffer = sentence_buffer[sentence_end_pos:]
                    
                    # Check for more complete sentences in remaining buffer
                    match = sentence_end_pattern.search(sentence_buffer)
            
            # After streaming completes, send any remaining text in buffer
            if self.cancel_event.is_set():
                await self.stop_tts_stream()
                return
            
            if sentence_buffer.strip():
                await self.stream_tts_text(sentence_buffer.strip())
                logger.debug(f"Sent final sentence to TTS: {sentence_buffer.strip()[:100]}...")
            
            # Finalize TTS
            await self.finalize_tts()
            
            # Log latency summary
            process_end_time = time.time()
            readable_time = datetime.fromtimestamp(process_end_time).isoformat()
            self._latency_log(f"[LATENCY] PROCESS_END: timestamp={process_end_time:.6f}, readable_time={readable_time}")
            
            # Summary of key latency metrics
            if self.llm_request_sent_time and self.first_token_time:
                llm_to_first_token = self.first_token_time - self.llm_request_sent_time
                llm_request_readable = datetime.fromtimestamp(self.llm_request_sent_time).isoformat()
                first_token_readable = datetime.fromtimestamp(self.first_token_time).isoformat()
                self._latency_log(f"[LATENCY] SUMMARY: llm_request_to_first_token={llm_to_first_token:.6f}s (request={llm_request_readable}, first_token={first_token_readable})")
            
            if self.llm_request_sent_time and self.llm_full_response_time:
                llm_to_full_response = self.llm_full_response_time - self.llm_request_sent_time
                llm_request_readable = datetime.fromtimestamp(self.llm_request_sent_time).isoformat()
                full_response_readable = datetime.fromtimestamp(self.llm_full_response_time).isoformat()
                self._latency_log(f"[LATENCY] SUMMARY: llm_request_to_full_response={llm_to_full_response:.6f}s (request={llm_request_readable}, full_response={full_response_readable})")
            
            if self.first_tts_request_time and self.first_audio_received_time:
                tts_to_first_audio = self.first_audio_received_time - self.first_tts_request_time
                tts_request_readable = datetime.fromtimestamp(self.first_tts_request_time).isoformat()
                first_audio_readable = datetime.fromtimestamp(self.first_audio_received_time).isoformat()
                self._latency_log(f"[LATENCY] SUMMARY: tts_request_to_first_audio={tts_to_first_audio:.6f}s (request={tts_request_readable}, first_audio={first_audio_readable})")
            
            if self.speech_end_time and self.first_audio_queued_time:
                e2e_latency = self.first_audio_queued_time - self.speech_end_time
                speech_end_readable = datetime.fromtimestamp(self.speech_end_time).isoformat()
                audio_queued_readable = datetime.fromtimestamp(self.first_audio_queued_time).isoformat()
                self._latency_log(f"[LATENCY] SUMMARY: e2e_transcript_to_audio_queued={e2e_latency:.6f}s (transcript={speech_end_readable}, audio_queued={audio_queued_readable})")
        
        except Exception as e:
            logger.error(f"Error processing user input: {e}", exc_info=True)
    
    async def stream_llm_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Stream LLM response tokens directly from provider.
        
        Args:
            messages: List of message dicts with "role" and "content" keys.
                     Must include system message (if any) and conversation history.
                     The last message should be the current user message.
            system_prompt: Optional system prompt.
        """
        full_response = ""
        thinking_buffer = ""  # Buffer to handle thinking tags that span chunks
        
        try:
            # Log when LLM request is sent (right before starting the async iteration)
            self.llm_request_sent_time = time.time()
            readable_time = datetime.fromtimestamp(self.llm_request_sent_time).isoformat()
            self._latency_log(f"[LATENCY] LLM_REQUEST_SENT: timestamp={self.llm_request_sent_time:.6f}, readable_time={readable_time}")
            logger.info("LLM response starting...")
            await self.set_activity(responding=True)

            # Stream directly from provider - pass messages and system_prompt for stateless operation
            async for token in self.llm_provider.generate_stream(
                messages=messages,
                system_prompt=system_prompt
            ):
                if self.cancel_event.is_set():
                    logger.info("LLM stream cancelled; stopping token processing")
                    await self.set_activity(responding=False)
                    break
                
                if token:
                    full_response += token
                    
                    # Log first token
                    if self.first_token_time is None:
                        self.first_token_time = time.time()
                        readable_time = datetime.fromtimestamp(self.first_token_time).isoformat()
                        if self.llm_request_sent_time:
                            latency = self.first_token_time - self.llm_request_sent_time
                            self._latency_log(f"[LATENCY] LLM_FIRST_TOKEN: timestamp={self.first_token_time:.6f}, readable_time={readable_time}, latency_from_llm_request={latency:.6f}s")
                        else:
                            self._latency_log(f"[LATENCY] LLM_FIRST_TOKEN: timestamp={self.first_token_time:.6f}, readable_time={readable_time}")
                    
                    # Filter thinking tags if disabled (handle tags that span chunks)
                    if self.disable_thinking:
                        # Add token to buffer
                        thinking_buffer += token
                        
                        # Check for complete <think>...</think> tags
                        thinking_pattern = re.compile(r'<think>.*?</think>', re.DOTALL)
                        
                        # Remove complete tags from buffer
                        filtered_buffer = thinking_pattern.sub('', thinking_buffer)
                        
                        # Check if we have an incomplete opening tag at the end
                        # Look for <think> without a matching </think>
                        opening_tag = '<think>'
                        closing_tag = '</think>'
                        
                        # Find the last occurrence of opening tag
                        last_open_pos = filtered_buffer.rfind(opening_tag)
                        
                        if last_open_pos != -1:
                            # Check if there's a closing tag after this opening tag
                            remaining_text = filtered_buffer[last_open_pos:]
                            if closing_tag not in remaining_text:
                                # Incomplete tag - keep everything from the opening tag in buffer
                                # Yield only the safe portion (before the incomplete tag)
                                safe_to_yield = filtered_buffer[:last_open_pos]
                                thinking_buffer = filtered_buffer[last_open_pos:]
                                
                                # Clean up whitespace and yield
                                safe_to_yield = re.sub(r'\s+', ' ', safe_to_yield).strip()
                                if safe_to_yield:
                                    await self.publish_event({"event": "llm_token", "token": safe_to_yield})
                                    yield safe_to_yield
                            else:
                                # Complete tag found - yield everything and reset buffer
                                # Clean up whitespace
                                filtered_buffer = re.sub(r'\s+', ' ', filtered_buffer).strip()
                                if filtered_buffer:
                                    await self.publish_event({"event": "llm_token", "token": filtered_buffer})
                                    yield filtered_buffer
                                thinking_buffer = ""
                        else:
                            # No incomplete tags - yield everything and reset buffer
                            # Clean up whitespace
                            filtered_buffer = re.sub(r'\s+', ' ', filtered_buffer).strip()
                            if filtered_buffer:
                                await self.publish_event({"event": "llm_token", "token": filtered_buffer})
                                yield filtered_buffer
                            thinking_buffer = ""
                    else:
                        # Thinking not disabled - yield token directly
                        await self.publish_event({"event": "llm_token", "token": token})
                        yield token
            
            # After streaming completes, yield any remaining buffer content (if thinking is disabled)
            if self.disable_thinking and thinking_buffer:
                # Filter any remaining tags and yield
                thinking_pattern = re.compile(r'<think>.*?</think>', re.DOTALL)
                remaining = thinking_pattern.sub('', thinking_buffer)
                remaining = re.sub(r'\s+', ' ', remaining).strip()
                if remaining:
                    await self.publish_event({"event": "llm_token", "token": remaining})
                    yield remaining
            
            if self.cancel_event.is_set():
                await self.publish_event({"event": "llm_cancelled"})
                await self.set_activity(responding=False)
                logger.info("LLM streaming cancelled before completion")
                return
            
            # Log when full response is complete
            self.llm_full_response_time = time.time()
            readable_time = datetime.fromtimestamp(self.llm_full_response_time).isoformat()
            if self.llm_request_sent_time:
                latency = self.llm_full_response_time - self.llm_request_sent_time
                self._latency_log(f"[LATENCY] LLM_FULL_RESPONSE: timestamp={self.llm_full_response_time:.6f}, readable_time={readable_time}, latency_from_llm_request={latency:.6f}s")
            else:
                self._latency_log(f"[LATENCY] LLM_FULL_RESPONSE: timestamp={self.llm_full_response_time:.6f}, readable_time={readable_time}")
            
            await self.publish_event({"event": "llm_done"})
            await self.set_activity(responding=False)
        
        except Exception as e:
            # Handle provider-specific errors
            status_code, error_message = self.llm_provider.parse_error(e)
            logger.error(f"LLM error ({status_code}): {error_message}", exc_info=True)
        finally:
            # Always log and add to context, even if there was an error
            if full_response and not self.cancel_event.is_set():
                # Filter thinking tags from full response for logging and history
                if self.disable_thinking:
                    thinking_pattern = re.compile(r'<think>.*?</think>', re.DOTALL)
                    full_response = thinking_pattern.sub('', full_response)
                    full_response = re.sub(r'\s+', ' ', full_response).strip()
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
                        # Log first audio chunk received
                        if self.first_audio_received_time is None:
                            self.first_audio_received_time = time.time()
                            readable_time = datetime.fromtimestamp(self.first_audio_received_time).isoformat()
                            if self.first_tts_request_time:
                                latency = self.first_audio_received_time - self.first_tts_request_time
                                self._latency_log(f"[LATENCY] TTS_FIRST_AUDIO_RECEIVED: timestamp={self.first_audio_received_time:.6f}, readable_time={readable_time}, latency_from_tts_request={latency:.6f}s, chunk_size={len(message)}")
                            else:
                                self._latency_log(f"[LATENCY] TTS_FIRST_AUDIO_RECEIVED: timestamp={self.first_audio_received_time:.6f}, readable_time={readable_time}, chunk_size={len(message)}")
                        
                        # Log first audio chunk queued for playback
                        if self.first_audio_queued_time is None:
                            self.first_audio_queued_time = time.time()
                            readable_time = datetime.fromtimestamp(self.first_audio_queued_time).isoformat()
                            if self.speech_end_time:
                                e2e_latency = self.first_audio_queued_time - self.speech_end_time
                                self._latency_log(f"[LATENCY] TTS_FIRST_AUDIO_QUEUED: timestamp={self.first_audio_queued_time:.6f}, readable_time={readable_time}, e2e_latency_from_transcript={e2e_latency:.6f}s, chunk_size={len(message)}")
                            else:
                                self._latency_log(f"[LATENCY] TTS_FIRST_AUDIO_QUEUED: timestamp={self.first_audio_queued_time:.6f}, readable_time={readable_time}, chunk_size={len(message)}")
                        
                        # Audio chunk - queue it for playback
                        await self.audio_player.play_audio_chunk(message)
                    else:
                        # JSON message
                        try:
                            data = json.loads(message)
                            msg_type = data.get("type")
                            
                            if msg_type == "done":
                                logger.debug("TTS synthesis chunk complete")
                                if self.pending_tts_requests > 0:
                                    self.pending_tts_requests -= 1
                                else:
                                    logger.warning("Received TTS done with no pending requests")

                                # Only clear synthesizing when all pending requests are finished
                                if self.pending_tts_requests == 0:
                                    await self.set_activity(synthesizing=False)
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
                    await self.set_activity(synthesizing=False)
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
                    tts_websocket_url = get_config("services", "tts_websocket_url", default="ws://localhost:8003/synthesize/stream")
                    self.tts_websocket = await websockets.connect(
                        tts_websocket_url,
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

                # Send text chunk (TTS server will synthesize immediately)
                # Audio chunks will be received by the background receiver task
                await self.set_activity(synthesizing=True)
                await self.tts_websocket.send(json.dumps({
                    "type": "text",
                    "text": text,
                    "finalize": False
                }))

                # Track in-flight synthesis requests
                self.pending_tts_requests += 1
                
                # Log first TTS request sent
                if self.first_tts_request_time is None:
                    self.first_tts_request_time = time.time()
                    readable_time = datetime.fromtimestamp(self.first_tts_request_time).isoformat()
                    self._latency_log(f"[LATENCY] TTS_REQUEST_SENT: timestamp={self.first_tts_request_time:.6f}, readable_time={readable_time}, text={text[:50]}")
                
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
                if self.pending_tts_requests == 0:
                    await self.set_activity(synthesizing=True)
                await self.tts_websocket.send(json.dumps({
                    "type": "text",
                    "text": "",
                    "finalize": True
                }))

                # Track the finalize as an in-flight request (server will emit a final done)
                self.pending_tts_requests += 1
            except (websockets.exceptions.ConnectionClosed, ConnectionError, OSError) as e:
                logger.warning(f"TTS WebSocket connection closed during finalize: {e}")
                # Connection may have closed, but synthesis might still complete
                # The receiver loop will handle any remaining audio if connection is re-established
                return
        
        except Exception as e:
            logger.error(f"Error finalizing TTS: {e}", exc_info=True)
    
    async def stop_tts_stream(self):
        """Stop any active TTS playback and close the WebSocket."""
        self.tts_receiver_running = False
        # Reset pending synthesis tracking
        self.pending_tts_requests = 0
        
        if self.tts_receiver_task:
            self.tts_receiver_task.cancel()
            try:
                await self.tts_receiver_task
            except asyncio.CancelledError:
                pass
        
        if self.tts_websocket:
            try:
                await self.tts_websocket.close()
            except Exception:
                pass
            self.tts_websocket = None
        
        await self.audio_player.stop()
        await self.set_activity(synthesizing=False)
    
    async def cancel_current_interaction(self):
        """Signal cancellation to LLM/TTS and clear playback."""
        self.cancel_event.set()
        await self.publish_event({"event": "cancelled"})
        await self.stop_tts_stream()
    
    def get_history(self):
        """Get conversation history from ContextManager (single source of truth)."""
        return self.context_manager.conversation_history
    
    async def start(self):
        """Start the agent."""
        self.running = True
        await self.set_activity(
            listening=True,
            transcribing=False,
            responding=False,
            synthesizing=False,
            playing=False,
        )
        # Enable hot-reload for system prompt
        check_interval = float(get_config("orchestrator", "system_prompt_reload_interval", default=1.0))
        self.context_manager.enable_hot_reload(check_interval=check_interval)
        # Connect to STT server
        asyncio.create_task(self.stt_client.connect())
        logger.info("Voice Agent started")
    
    async def stop(self):
        """Stop the agent."""
        self.running = False
        # Disable hot-reload
        self.context_manager.disable_hot_reload()
        await self.stop_tts_stream()
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


@app.get("/ui/history")
async def get_history():
    """Get conversation history from ContextManager."""
    if not agent:
        return {"error": "Agent not initialized"}
    
    history = agent.get_history()
    return {"history": history}


@app.get("/ui")
async def ui_page():
    """Serve simple control panel UI."""
    static_file = Path(__file__).parent / "static" / "ui.html"
    if not static_file.exists():
        return {"error": "UI not found"}
    return FileResponse(static_file)


@app.websocket("/ui/events")
async def ui_events(websocket: WebSocket):
    """WebSocket that streams STT and LLM events to the browser UI."""
    await websocket.accept()
    
    if not agent:
        await websocket.close(code=1011)
        return
    
    queue = await agent.subscribe_events()
    try:
        while True:
            event = await queue.get()
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    finally:
        await agent.unsubscribe_events(queue)


@app.post("/ui/cancel")
async def ui_cancel():
    """Cancel current LLM/TTS interaction and stop playback."""
    if not agent:
        return {"error": "Agent not initialized"}
    
    await agent.cancel_current_interaction()
    return {"status": "cancelled"}


@app.get("/ui/system-prompt")
async def get_system_prompt():
    """Get current system prompt."""
    if not agent:
        return {"error": "Agent not initialized"}
    
    prompt = agent.context_manager.get_system_prompt()
    file_path = agent.context_manager.get_system_prompt_file_path()
    return {
        "prompt": prompt,
        "file_path": file_path
    }


class SystemPromptUpdate(BaseModel):
    """Request model for updating system prompt."""
    prompt: str


@app.post("/ui/system-prompt")
async def update_system_prompt(request: SystemPromptUpdate):
    """Update system prompt and save to file."""
    if not agent:
        return {"error": "Agent not initialized"}
    
    if not request.prompt:
        return {"error": "Prompt cannot be empty"}
    
    success = agent.context_manager.set_system_prompt(request.prompt)
    if success:
        return {
            "status": "success",
            "message": "System prompt updated",
            "file_path": agent.context_manager.get_system_prompt_file_path()
        }
    else:
        return {"error": "Failed to save system prompt"}


if __name__ == "__main__":
    orchestrator_host = get_config("orchestrator", "host", default="0.0.0.0")
    orchestrator_port = int(get_config("orchestrator", "port", default=8000))
    logger.info(f"Starting Orchestrator server on {orchestrator_host}:{orchestrator_port}...")
    uvicorn.run(app, host=orchestrator_host, port=orchestrator_port)

