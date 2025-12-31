import asyncio
import json
import numpy as np
import time
from typing import Dict, Set, Optional, Any
from dataclasses import dataclass, field
from fastapi import WebSocket, WebSocketDisconnect
from core.logging import get_logger
from core.settings import get_settings
from stt.providers import FasterWhisperProvider, FunASRProvider

logger = get_logger(__name__)


@dataclass
class ClientStreamingState:
    """State for a single client's streaming session."""
    asr_cache: Dict[str, Any] = field(default_factory=dict)
    vad_cache: Dict[str, Any] = field(default_factory=dict)
    audio_buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    asr_chunk_buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))  # Buffer for ASR chunks
    vad_chunk_buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))  # Buffer for VAD chunks
    silence_start_time: Optional[float] = None  # Timestamp when silence started (ms)
    current_transcript: str = ""  # Accumulated transcript text
    is_speaking: bool = False  # Whether currently detecting speech
    last_interim_time: float = 0.0  # Last time interim transcript was sent (for batch mode)

    # Queue infrastructure for decoupled network/processing
    audio_queue: Optional[asyncio.Queue] = None  # Queue for incoming audio chunks
    worker_task: Optional[asyncio.Task] = None  # Background worker task
    chunks_dropped: int = 0  # Count of dropped chunks due to queue overflow


class STTManager:
    """
    Manages STT transcription, client connections, and audio buffering.
    Decoupled from FastAPI WebSocket details.
    """

    # Queue configuration
    AUDIO_QUEUE_SIZE = 100  # 100 chunks at 100ms/chunk = 10 seconds buffer

    def __init__(self):
        # Configuration
        settings = get_settings()
        self.language_code = settings.stt.language_code
        self.sample_rate = settings.stt.sample_rate
        self.interim_min_samples = settings.stt.interim_transcript_min_samples

        # Provider setup
        self.provider_name = settings.stt.provider
        self.provider = self._load_provider()
        
        # Check if streaming mode is enabled
        self.streaming_enabled = False
        if self.provider_name == "funasr" and isinstance(self.provider, FunASRProvider):
            self.streaming_enabled = self.provider.is_streaming

        # State
        self.connected_clients: Set[WebSocket] = set()  # Legacy: both audio+transcript
        self.transcript_clients: Set[WebSocket] = set()  # Only receive transcripts
        self.clients_lock = asyncio.Lock()

        # Per-client streaming state
        self.client_states: Dict[WebSocket, ClientStreamingState] = {}
        self.client_states_lock = asyncio.Lock()
        
        # Silence threshold (use audio config or streaming config)
        if self.streaming_enabled:
            # Get streaming config from the FunASR provider
            streaming_config = self.provider.streaming_config if hasattr(self.provider, 'streaming_config') else None

            if streaming_config:
                self.silence_threshold_ms = streaming_config.silence_threshold_ms
                chunk_size = streaming_config.chunk_size
                vad_chunk_size_ms = streaming_config.vad_chunk_size_ms
                # ASR chunk size: chunk_size[1] * 960 samples = 600ms at 16kHz
                self.asr_chunk_samples = int(chunk_size[1] * 960) if len(chunk_size) > 1 else 9600
                # VAD chunk size: 200ms = 3200 samples at 16kHz
                self.vad_chunk_samples = int(vad_chunk_size_ms * self.sample_rate / 1000)
                logger.info(f"Streaming chunk sizes: ASR={self.asr_chunk_samples} samples ({self.asr_chunk_samples/self.sample_rate*1000:.0f}ms), VAD={self.vad_chunk_samples} samples ({vad_chunk_size_ms}ms)")
            else:
                self.silence_threshold_ms = settings.audio.silence_threshold_ms
                self.asr_chunk_samples = 0
                self.vad_chunk_samples = 0
        else:
            self.silence_threshold_ms = settings.audio.silence_threshold_ms
            self.asr_chunk_samples = 0
            self.vad_chunk_samples = 0
    
    def _load_provider(self):
        logger.info(f"Initializing STT provider: {self.provider_name}...")
        try:
            settings = get_settings()
            provider_config = settings.stt.get_provider_config()

            if self.provider_name == "faster-whisper":
                return FasterWhisperProvider(
                    model_path=provider_config.model_path,
                    device=provider_config.device,
                    compute_type=provider_config.compute_type
                )
            elif self.provider_name == "funasr":
                return FunASRProvider(
                    model_name=provider_config.model_name,
                    vad_model=provider_config.vad_model,
                    vad_kwargs=provider_config.vad_kwargs,
                    punc_model=provider_config.punc_model,
                    device=provider_config.device,
                    batch_size_s=provider_config.batch_size_s,
                    streaming_config=provider_config.streaming
                )
            else:
                raise ValueError(f"Unknown STT provider: {self.provider_name}")
        except Exception as e:
            logger.error(f"Error loading STT provider: {e}", exc_info=True)
            raise

    async def add_client(self, websocket: WebSocket):
        """Register a new client (LEGACY: both audio+transcript)."""
        async with self.clients_lock:
            self.connected_clients.add(websocket)

        # Initialize with queue and worker based on mode
        if self.streaming_enabled:
            await self._initialize_client_streaming(websocket)
        else:
            await self._initialize_client_batch(websocket)

        logger.info(f"Client connected (legacy). Total: {len(self.connected_clients)}")

    async def add_audio_client(self, websocket: WebSocket):
        """
        Register audio-only client (sends audio, doesn't receive transcripts).

        This prevents WebSocket backpressure from clients that don't consume responses.
        """
        # Initialize audio processing but DON'T add to broadcast list
        if self.streaming_enabled:
            await self._initialize_client_streaming(websocket)
        else:
            await self._initialize_client_batch(websocket)

        logger.info(f"Audio client connected (no broadcasts). Total audio processors: {len(self.client_states)}")

    async def add_transcript_client(self, websocket: WebSocket):
        """
        Register transcript-only client (receives transcripts, doesn't send audio).

        Transcripts are pushed to these clients via broadcast.
        """
        async with self.clients_lock:
            self.transcript_clients.add(websocket)

        logger.info(f"Transcript client connected. Total transcript receivers: {len(self.transcript_clients)}")

    async def remove_client(self, websocket: WebSocket):
        """Unregister a client."""
        # Clean up state (works for both streaming and batch modes)
        await self._cleanup_client_state(websocket)

        async with self.clients_lock:
            self.connected_clients.discard(websocket)
        logger.info(f"Client disconnected. Total: {len(self.connected_clients)}")

    async def remove_transcript_client(self, websocket: WebSocket):
        """Unregister a transcript-only client."""
        async with self.clients_lock:
            self.transcript_clients.discard(websocket)
        logger.info(f"Transcript client disconnected. Total: {len(self.transcript_clients)}")
    
    async def _initialize_client_streaming(self, websocket: WebSocket):
        """Initialize streaming state for a client with queue and worker."""
        if not isinstance(self.provider, FunASRProvider):
            return

        try:
            cache_dict = self.provider.initialize_streaming()

            # Create audio queue and worker task
            audio_queue = asyncio.Queue(maxsize=self.AUDIO_QUEUE_SIZE)
            worker_task = asyncio.create_task(
                self._streaming_worker(websocket, audio_queue)
            )

            async with self.client_states_lock:
                self.client_states[websocket] = ClientStreamingState(
                    asr_cache=cache_dict["asr_cache"],
                    vad_cache=cache_dict["vad_cache"],
                    audio_queue=audio_queue,
                    worker_task=worker_task
                )

            logger.info(f"Initialized streaming worker for client (queue size: {self.AUDIO_QUEUE_SIZE})")
        except Exception as e:
            logger.error(f"Error initializing streaming for client: {e}", exc_info=True)
    
    async def _initialize_client_batch(self, websocket: WebSocket):
        """Initialize batch mode state for a client with queue and worker."""
        try:
            # Create audio queue and worker task for batch mode
            audio_queue = asyncio.Queue(maxsize=self.AUDIO_QUEUE_SIZE)
            worker_task = asyncio.create_task(
                self._batch_worker(websocket, audio_queue)
            )

            async with self.client_states_lock:
                self.client_states[websocket] = ClientStreamingState(
                    audio_queue=audio_queue,
                    worker_task=worker_task
                )

            logger.info(f"Initialized batch mode worker for client (queue size: {self.AUDIO_QUEUE_SIZE})")
        except Exception as e:
            logger.error(f"Error initializing batch mode for client: {e}", exc_info=True)

    async def _cleanup_client_state(self, websocket: WebSocket):
        """Clean up client state (streaming or batch), including stopping worker."""
        # Get state and remove from dict FIRST (under lock)
        async with self.client_states_lock:
            state = self.client_states.pop(websocket, None)

        # Stop worker OUTSIDE lock to avoid blocking other operations
        if state:
            if state.worker_task and not state.worker_task.done():
                try:
                    # Send stop signal (non-blocking)
                    if state.audio_queue:
                        try:
                            state.audio_queue.put_nowait(None)
                        except asyncio.QueueFull:
                            # Queue full, cancel worker directly
                            state.worker_task.cancel()

                    # Wait for worker to finish (with timeout) - NOT holding lock!
                    await asyncio.wait_for(state.worker_task, timeout=2.0)
                    logger.info("Worker stopped gracefully")
                except asyncio.TimeoutError:
                    # Force cancel if worker doesn't stop in time
                    state.worker_task.cancel()
                    logger.warning("Worker didn't stop in time, forcing cancellation")
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error stopping worker: {e}")

            # Log dropped chunks statistics
            if state.chunks_dropped > 0:
                logger.warning(f"Client had {state.chunks_dropped} dropped chunks during session")

    async def _streaming_worker(self, websocket: WebSocket, queue: asyncio.Queue):
        """
        Background worker that processes audio chunks from queue.

        This runs independently from the network receive loop, allowing
        the TCP socket to drain at full speed while processing happens
        in the background.

        Args:
            websocket: Client WebSocket connection
            queue: Audio chunk queue to process from
        """
        client_id = id(websocket)
        logger.info(f"[Worker-{client_id}] Streaming worker started")
        chunk_count = 0

        try:
            while True:
                # Get chunk from queue (blocking wait)
                logger.debug(f"[Worker-{client_id}] Waiting for chunk from queue...")
                chunk = await queue.get()
                queue_depth = queue.qsize()

                # None is the stop signal
                if chunk is None:
                    logger.info(f"[Worker-{client_id}] Received stop signal, exiting worker")
                    break

                chunk_count += 1
                logger.debug(f"[Worker-{client_id}] Got chunk #{chunk_count} (queue_depth={queue_depth}, size={len(chunk)} bytes)")

                # Monitor queue depth
                if queue_depth > 50:  # >5 seconds lag
                    logger.warning(f"[Worker-{client_id}] PROCESSING LAG: {queue_depth} chunks queued ({queue_depth * 0.1:.1f}s behind)")

                # Process chunk using existing streaming logic
                try:
                    logger.debug(f"[Worker-{client_id}] Processing chunk #{chunk_count}...")
                    start_time = time.time()
                    await self._process_streaming_chunk_internal(websocket, chunk)
                    elapsed_ms = (time.time() - start_time) * 1000
                    logger.debug(f"[Worker-{client_id}] Chunk #{chunk_count} processed in {elapsed_ms:.1f}ms")
                except Exception as e:
                    logger.error(f"[Worker-{client_id}] ERROR processing chunk #{chunk_count}: {e}", exc_info=True)
                    # Continue processing next chunks - don't crash worker on single failure

                finally:
                    # Mark task as done
                    queue.task_done()

        except asyncio.CancelledError:
            logger.info(f"[Worker-{client_id}] Worker cancelled")
            raise
        except Exception as e:
            logger.error(f"[Worker-{client_id}] FATAL ERROR in streaming worker: {e}", exc_info=True)
        finally:
            logger.info(f"[Worker-{client_id}] Worker stopped (processed {chunk_count} chunks)")

    async def _batch_worker(self, websocket: WebSocket, queue: asyncio.Queue):
        """
        Background worker for batch mode processing.

        Similar to streaming worker but uses batch processing logic.

        Args:
            websocket: Client WebSocket connection
            queue: Audio chunk queue to process from
        """
        client_id = id(websocket)
        logger.info(f"Batch mode worker started for client {client_id}")

        try:
            while True:
                # Get chunk from queue (blocking wait)
                chunk = await queue.get()

                # None is the stop signal
                if chunk is None:
                    logger.info(f"Batch worker received stop signal for client {client_id}")
                    break

                # Monitor queue depth
                queue_depth = queue.qsize()
                if queue_depth > 50:  # >5 seconds lag
                    logger.warning(f"Processing lag (batch) for client {client_id}: {queue_depth} chunks queued")

                # Process chunk using batch logic
                try:
                    await self._process_batch_chunk_internal(websocket, chunk)
                except Exception as e:
                    logger.error(f"Error processing batch chunk for client {client_id}: {e}", exc_info=True)
                    # Continue processing next chunks - don't crash worker on single failure

                finally:
                    # Mark task as done
                    queue.task_done()

        except asyncio.CancelledError:
            logger.info(f"Batch worker cancelled for client {client_id}")
            raise
        except Exception as e:
            logger.error(f"Fatal error in batch worker for client {client_id}: {e}", exc_info=True)
        finally:
            logger.info(f"Batch worker stopped for client {client_id}")

    def _should_filter_transcript(self, text: str) -> bool:
        """
        Check if a transcript should be filtered out (just "嗯。").

        Args:
            text: The transcript text to check

        Returns:
            True if the transcript should be filtered out, False otherwise
        """
        if not text:
            return True

        # Just filter out "嗯。"
        if text.strip() == "嗯。":
            logger.debug(f"Filtering transcript: {repr(text)}")
            return True

        return False

    def _validate_audio_chunk(self, chunk: bytes) -> bool:
        """
        Validate audio chunk data for corruption.

        Args:
            chunk: Raw audio chunk bytes

        Returns:
            True if chunk is valid, False if corrupted
        """
        # Check if chunk is empty
        if not chunk:
            logger.warning("Received empty audio chunk")
            return False

        # Check if size is multiple of 4 (float32 = 4 bytes)
        if len(chunk) % 4 != 0:
            logger.warning(f"Audio chunk size {len(chunk)} is not multiple of 4 (float32)")
            return False

        # Convert to numpy array and check for invalid values
        try:
            audio_data = np.frombuffer(chunk, dtype=np.float32)

            # Check for NaN or Inf values
            if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
                logger.warning("Audio chunk contains NaN or Inf values")
                return False

            # Check if all values are zero (dead audio)
            if np.all(audio_data == 0):
                logger.debug("Audio chunk is all zeros (silence)")
                # This is valid, just note it

            # Sanity check: audio values should be in reasonable range [-1.0, 1.0] for normalized float32
            # Allow some headroom for non-normalized audio
            if np.any(np.abs(audio_data) > 10.0):
                logger.warning(f"Audio chunk has extreme values (max: {np.max(np.abs(audio_data))})")
                return False

            return True
        except Exception as e:
            logger.error(f"Error validating audio chunk: {e}")
            return False

    async def broadcast(self, message: dict):
        """
        Broadcast message to all connected clients (non-blocking).

        This method returns immediately without waiting for clients to receive.
        Slow clients won't block the processing pipeline.
        """
        message_str = json.dumps(message)

        # Create a background task to send to all clients
        # This ensures broadcast doesn't block the processing pipeline
        asyncio.create_task(self._broadcast_to_clients(message_str))

    async def _broadcast_to_clients(self, message_str: str):
        """
        Internal method that actually sends to transcript clients.

        Only sends to:
        - transcript_clients (registered via /ws/transcripts)
        - connected_clients (legacy /ws/transcribe endpoint)

        Does NOT send to audio-only clients to prevent backpressure.
        """
        # Quickly copy client lists under lock
        async with self.clients_lock:
            # Combine transcript clients and legacy clients
            all_recipients = list(self.transcript_clients | self.connected_clients)

        # Send to each client OUTSIDE lock (don't block other operations)
        disconnected_transcript = set()
        disconnected_legacy = set()

        for client in all_recipients:
            try:
                # Normal timeout - these clients should be consuming messages
                await asyncio.wait_for(client.send_text(message_str), timeout=0.5)
            except asyncio.TimeoutError:
                # Transcript client is slow - this shouldn't happen, log warning
                logger.warning(f"Transcript client {id(client)} slow to receive (timeout)")
                # Mark for cleanup if consistently slow
                if client in self.transcript_clients:
                    disconnected_transcript.add(client)
                else:
                    disconnected_legacy.add(client)
            except (ConnectionError, ConnectionResetError):
                # Connection is dead, mark for cleanup
                if client in self.transcript_clients:
                    disconnected_transcript.add(client)
                else:
                    disconnected_legacy.add(client)
            except Exception as e:
                # Log unexpected errors
                logger.warning(f"Error broadcasting to client {id(client)}: {e}")

        # Clean up disconnected clients (re-acquire lock)
        if disconnected_transcript or disconnected_legacy:
            async with self.clients_lock:
                if disconnected_transcript:
                    self.transcript_clients.difference_update(disconnected_transcript)
                    logger.info(f"Removed {len(disconnected_transcript)} slow/disconnected transcript clients")
                if disconnected_legacy:
                    self.connected_clients.difference_update(disconnected_legacy)
                    logger.info(f"Removed {len(disconnected_legacy)} slow/disconnected legacy clients")

    def transcribe_interim(self, audio_buffer: np.ndarray) -> Optional[str]:
        """Perform quick interim transcription without VAD."""
        # Skip interim for FunASR (too slow/non-streaming)
        if self.provider_name == "funasr":
            return None
            
        segments, _ = self.provider.transcribe(
            audio_buffer,
            language=self.language_code
        )
        return "".join([seg.text for seg in segments])

    async def process_audio_chunk(
        self,
        websocket: WebSocket,
        chunk: bytes
    ):
        """
        Enqueue audio chunk for processing (non-blocking).

        This method returns immediately after validation and enqueuing.
        The actual processing happens asynchronously in the worker task.

        Args:
            websocket: WebSocket connection for the client
            chunk: Audio chunk bytes
        """
        # Quick validation (cheap, minimal blocking)
        if not self._validate_audio_chunk(chunk):
            logger.warning("Received invalid audio chunk, skipping")
            return

        # Get client state
        async with self.client_states_lock:
            state = self.client_states.get(websocket)

        # Initialize if missing (MUST be outside lock to avoid deadlock)
        if not state:
            if self.streaming_enabled:
                await self._initialize_client_streaming(websocket)
            else:
                await self._initialize_client_batch(websocket)

            # Re-acquire lock to get the newly initialized state
            async with self.client_states_lock:
                state = self.client_states.get(websocket)
                if not state:
                    logger.error("Failed to initialize client state")
                    return

        # Enqueue chunk (non-blocking, fail-fast)
        if state.audio_queue:
            try:
                queue_depth_before = state.audio_queue.qsize()
                state.audio_queue.put_nowait(chunk)
                logger.debug(f"[Enqueue-{id(websocket)}] Chunk enqueued (queue: {queue_depth_before} → {queue_depth_before + 1})")
            except asyncio.QueueFull:
                # Queue overflow means server is 10+ seconds behind
                # This is a fatal error - disconnect client
                state.chunks_dropped += 1
                logger.critical(
                    f"[Enqueue-{id(websocket)}] ❌ QUEUE OVERFLOW! "
                    f"Server is {self.AUDIO_QUEUE_SIZE} chunks ({self.AUDIO_QUEUE_SIZE * 0.1}s) behind. "
                    f"Total dropped: {state.chunks_dropped}. DISCONNECTING CLIENT."
                )
                # Close WebSocket connection - better than garbled transcripts
                try:
                    await websocket.close(code=1008, reason="Server overloaded")
                except:
                    pass
        else:
            logger.error(f"[Enqueue-{id(websocket)}] ❌ No audio queue for client")
    
    async def _process_streaming_chunk_internal(self, websocket: WebSocket, chunk: bytes):
        """
        Internal method: Process audio chunk in streaming mode.

        Called by the worker task. Assumes chunk is already validated.

        Args:
            websocket: Client WebSocket connection
            chunk: Audio chunk bytes (already validated)
        """
        if not isinstance(self.provider, FunASRProvider):
            return

        client_id = id(websocket)

        async with self.client_states_lock:
            state = self.client_states.get(websocket)
            if not state:
                logger.error(f"[Process-{client_id}] No state for client in worker")
                return

        # Convert chunk to numpy array
        audio_chunk = np.frombuffer(chunk, dtype=np.float32)
        logger.debug(f"[Process-{client_id}] Audio chunk: {audio_chunk.shape} samples, "
                    f"asr_buffer={state.asr_chunk_buffer.size}, vad_buffer={state.vad_chunk_buffer.size}, "
                    f"is_speaking={state.is_speaking}, transcript_len={len(state.current_transcript)}")
        
        # Add to processing buffers
        state.asr_chunk_buffer = np.concatenate([state.asr_chunk_buffer, audio_chunk])
        state.vad_chunk_buffer = np.concatenate([state.vad_chunk_buffer, audio_chunk])
        
        current_time = time.time() * 1000
        
        # Process VAD when we have enough samples (200ms chunks)
        if state.vad_chunk_buffer.size >= self.vad_chunk_samples:
            vad_chunk = state.vad_chunk_buffer[:self.vad_chunk_samples].copy()
            state.vad_chunk_buffer = state.vad_chunk_buffer[self.vad_chunk_samples:]

            logger.debug(f"[Process-{client_id}] VAD processing {vad_chunk.size} samples (cache_size={len(state.vad_cache)})")

            # Run VAD in thread pool to avoid blocking event loop
            # This allows WebSocket to stay responsive during inference
            loop = asyncio.get_event_loop()
            vad_start = time.time()
            vad_result = await loop.run_in_executor(
                None,  # Use default ThreadPoolExecutor
                self.provider.process_vad_chunk,
                vad_chunk,
                state.vad_cache,
                False  # is_final
            )
            vad_elapsed_ms = (time.time() - vad_start) * 1000
            logger.debug(f"[Process-{client_id}] VAD completed in {vad_elapsed_ms:.1f}ms, result={vad_result}")

            # Handle VAD results
            if vad_result:
                # Check for speech start: [[beg, -1]]
                if len(vad_result) > 0 and len(vad_result[0]) == 2:
                    beg, end = vad_result[0]
                    if beg != -1 and end == -1:
                        # Speech start detected
                        logger.info(f"[Process-{client_id}] 🎤 SPEECH START detected (beg={beg}, end={end})")
                        # Only broadcast if we weren't already speaking (to avoid duplicate events)
                        if not state.is_speaking:
                            await self.broadcast({"type": "speech_start"})

                            # Only reset if we've already finalized a transcript
                            # This preserves interim transcripts when user interrupts themselves
                            if not state.current_transcript:
                                # No accumulated transcript - this is a clean new utterance
                                logger.info(f"[Process-{client_id}] 🆕 Starting NEW utterance (resetting caches and buffers)")
                                cache_dict = self.provider.initialize_streaming()
                                state.asr_cache = cache_dict["asr_cache"]
                                state.vad_cache = cache_dict["vad_cache"]
                                logger.debug(f"[Process-{client_id}] Cache reset: asr_cache={len(state.asr_cache)}, vad_cache={len(state.vad_cache)}")

                                # Clear buffers
                                state.asr_chunk_buffer = np.array([], dtype=np.float32)
                                state.vad_chunk_buffer = np.array([], dtype=np.float32)

                                # Preserve the VAD chunk that triggered detection
                                state.asr_chunk_buffer = vad_chunk.copy()
                                logger.debug(f"[Process-{client_id}] Preserved VAD chunk: {vad_chunk.size} samples")
                            else:
                                # User interrupted before finalization - preserve transcript
                                logger.info(f"[Process-{client_id}] 🔄 RESUMING speech (preserving {len(state.current_transcript)} chars of transcript)")
                                logger.debug(f"[Process-{client_id}] Current transcript: '{state.current_transcript[:100]}...'")
                                # Don't reset caches or transcript - continue accumulating
                        else:
                            logger.debug(f"[Process-{client_id}] Already speaking, ignoring duplicate speech_start")

                        state.is_speaking = True
                        state.silence_start_time = None
                    elif beg == -1 and end != -1:
                        # Speech end detected
                        logger.info(f"[Process-{client_id}] 🔇 SPEECH END detected (beg={beg}, end={end})")
                        state.is_speaking = False
                        state.silence_start_time = current_time
                        logger.debug(f"[Process-{client_id}] Silence started at {current_time:.0f}ms")
        
        # Process ASR when we have enough samples (600ms chunks)
        if state.asr_chunk_buffer.size >= self.asr_chunk_samples:
            asr_chunk = state.asr_chunk_buffer[:self.asr_chunk_samples].copy()
            state.asr_chunk_buffer = state.asr_chunk_buffer[self.asr_chunk_samples:]

            logger.debug(f"[Process-{client_id}] ASR processing {asr_chunk.size} samples (cache_size={len(state.asr_cache)})")

            # Run ASR in thread pool to avoid blocking event loop
            # This allows WebSocket to stay responsive during inference
            loop = asyncio.get_event_loop()
            asr_start = time.time()
            asr_text = await loop.run_in_executor(
                None,  # Use default ThreadPoolExecutor
                self.provider.process_streaming_chunk,
                asr_chunk,
                state.asr_cache,
                False  # is_final
            )
            asr_elapsed_ms = (time.time() - asr_start) * 1000
            logger.debug(f"[Process-{client_id}] ASR completed in {asr_elapsed_ms:.1f}ms, produced: '{asr_text}'")

            # Accumulate transcript if ASR produced text
            # ASR always returns text for this chunk only, so we always append it
            # BUT only if VAD has detected speech - ignore noise when silent
            if asr_text:
                if state.is_speaking:
                    old_len = len(state.current_transcript)
                    # Append new text to accumulated transcript (no space added - ASR handles spacing)
                    if state.current_transcript:
                        state.current_transcript += asr_text
                    else:
                        # First chunk - set it
                        state.current_transcript = asr_text

                    logger.info(f"[Process-{client_id}] 📝 Transcript updated: {old_len} → {len(state.current_transcript)} chars (+{len(asr_text)})")

                    # Send accumulated interim transcript
                    await self.broadcast({
                        "type": "interim",
                        "text": state.current_transcript
                    })
                    logger.debug(f"[Process-{client_id}] Interim broadcast: '{state.current_transcript[:100]}...'")
                else:
                    # ASR picked up noise during silence - ignore it
                    logger.debug(f"[Process-{client_id}] ASR produced text during silence (noise), ignoring: '{asr_text}'")
            else:
                logger.debug(f"[Process-{client_id}] ASR produced no text")
        
        # Check silence threshold
        if state.silence_start_time is not None:
            silence_duration = current_time - state.silence_start_time
            if silence_duration >= self.silence_threshold_ms:
                # Finalize transcript
                logger.info(f"[Process-{client_id}] ⏰ Silence threshold exceeded ({silence_duration:.0f}ms >= {self.silence_threshold_ms}ms), FINALIZING transcript")
                logger.info(f"[Process-{client_id}] Current state: transcript_len={len(state.current_transcript)}, "
                           f"asr_buffer={state.asr_chunk_buffer.size}, vad_buffer={state.vad_chunk_buffer.size}, "
                           f"queue_depth={state.audio_queue.qsize() if state.audio_queue else 0}")
                
                # Process any remaining buffered audio before finalizing
                # This ensures we capture all audio, not just complete chunks
                loop = asyncio.get_event_loop()
                remaining_chunks = 0
                while state.asr_chunk_buffer.size >= self.asr_chunk_samples:
                    remaining_chunks += 1
                    asr_chunk = state.asr_chunk_buffer[:self.asr_chunk_samples].copy()
                    state.asr_chunk_buffer = state.asr_chunk_buffer[self.asr_chunk_samples:]

                    logger.debug(f"[Process-{client_id}] Processing remaining chunk #{remaining_chunks} ({asr_chunk.size} samples)")

                    # Run in thread pool to avoid blocking
                    remaining_start = time.time()
                    remaining_text = await loop.run_in_executor(
                        None,
                        self.provider.process_streaming_chunk,
                        asr_chunk,
                        state.asr_cache,
                        False  # is_final
                    )
                    remaining_elapsed = (time.time() - remaining_start) * 1000
                    logger.debug(f"[Process-{client_id}] Remaining chunk #{remaining_chunks} done in {remaining_elapsed:.1f}ms: '{remaining_text}'")

                    if remaining_text:
                        # Accumulate remaining text (always append, no space)
                        if state.current_transcript:
                            state.current_transcript += remaining_text
                        else:
                            state.current_transcript = remaining_text
                        logger.debug(f"[Process-{client_id}] Transcript after remaining #{remaining_chunks}: {len(state.current_transcript)} chars")

                if remaining_chunks > 0:
                    logger.info(f"[Process-{client_id}] Processed {remaining_chunks} remaining chunks")
                
                # If there's still some audio left (less than a full chunk), process it
                if state.asr_chunk_buffer.size > 0:
                    logger.debug(f"[Process-{client_id}] Processing PARTIAL chunk ({state.asr_chunk_buffer.size} samples)")

                    # Run in thread pool to avoid blocking
                    partial_start = time.time()
                    remaining_text = await loop.run_in_executor(
                        None,
                        self.provider.process_streaming_chunk,
                        state.asr_chunk_buffer,
                        state.asr_cache,
                        False  # is_final
                    )
                    partial_elapsed = (time.time() - partial_start) * 1000
                    logger.debug(f"[Process-{client_id}] Partial chunk done in {partial_elapsed:.1f}ms: '{remaining_text}'")

                    if remaining_text:
                        # Accumulate remaining text (always append, no space)
                        if state.current_transcript:
                            state.current_transcript += remaining_text
                        else:
                            state.current_transcript = remaining_text
                        logger.info(f"[Process-{client_id}] Added partial text ({len(remaining_text)} chars), total: {len(state.current_transcript)} chars")

                # Finalize ASR with is_final=True to get any final text
                # Use empty chunk since cache contains all state
                logger.debug(f"[Process-{client_id}] Calling ASR with is_final=True to flush cache")
                final_chunk = np.array([], dtype=np.float32)

                # Run in thread pool to avoid blocking
                final_start = time.time()
                final_text = await loop.run_in_executor(
                    None,
                    self.provider.process_streaming_chunk,
                    final_chunk,
                    state.asr_cache,
                    True  # is_final
                )
                final_elapsed = (time.time() - final_start) * 1000
                logger.debug(f"[Process-{client_id}] Final ASR flush done in {final_elapsed:.1f}ms: '{final_text}'")
                
                # Accumulate final text if present (always append, no space)
                if final_text:
                    old_len = len(state.current_transcript)
                    if state.current_transcript:
                        state.current_transcript += final_text
                    else:
                        state.current_transcript = final_text
                    logger.info(f"[Process-{client_id}] Added final text ({len(final_text)} chars), total: {old_len} → {len(state.current_transcript)} chars")

                # Apply punctuation to the complete transcript
                logger.debug(f"[Process-{client_id}] Applying punctuation to transcript ({len(state.current_transcript)} chars)")
                if isinstance(self.provider, FunASRProvider):
                    # Run punctuation in thread pool to avoid blocking
                    punc_start = time.time()
                    transcript_to_send = await loop.run_in_executor(
                        None,
                        self.provider.apply_punctuation,
                        state.current_transcript
                    )
                    punc_elapsed = (time.time() - punc_start) * 1000
                    logger.debug(f"[Process-{client_id}] Punctuation done in {punc_elapsed:.1f}ms")
                else:
                    transcript_to_send = state.current_transcript

                logger.info(f"[Process-{client_id}] 📤 FINALIZING transcript: length={len(transcript_to_send)} chars, text='{transcript_to_send[:100]}...'")

                # Filter out short/noise transcripts
                if transcript_to_send and not self._should_filter_transcript(transcript_to_send):
                    await self.broadcast({
                        "type": "final",
                        "text": transcript_to_send
                    })
                    logger.info(f"[Process-{client_id}] ✅ Final transcript broadcast: '{transcript_to_send[:50]}...'")
                else:
                    # Filtered out or empty - don't send anything
                    if transcript_to_send:
                        logger.info(f"[Process-{client_id}] 🚫 Filtered out transcript: '{transcript_to_send[:50]}...'")
                    else:
                        logger.info(f"[Process-{client_id}] ⚠️ Empty transcript, not broadcasting")

                # Reset state (clear current_transcript after sending final)
                logger.info(f"[Process-{client_id}] 🔄 RESETTING state after finalization")
                cache_dict = self.provider.initialize_streaming()
                state.asr_cache = cache_dict["asr_cache"]
                state.vad_cache = cache_dict["vad_cache"]
                logger.debug(f"[Process-{client_id}] Caches reset: asr={len(state.asr_cache)}, vad={len(state.vad_cache)}")

                state.asr_chunk_buffer = np.array([], dtype=np.float32)
                state.vad_chunk_buffer = np.array([], dtype=np.float32)
                state.current_transcript = ""  # Clear after sending final
                state.silence_start_time = None
                state.is_speaking = False
                logger.debug(f"[Process-{client_id}] State cleared: buffers=0, transcript='', speaking=False")

                # CRITICAL: Drain queue of stale chunks
                # After finalization, any chunks remaining in queue are from the OLD utterance
                # Processing them with new (empty) caches causes hangs/corruption
                if state.audio_queue:
                    queue_depth_before = state.audio_queue.qsize()
                    logger.info(f"[Process-{client_id}] 🚮 DRAINING queue (depth before: {queue_depth_before})")

                    stale_chunks = 0
                    while not state.audio_queue.empty():
                        try:
                            _ = state.audio_queue.get_nowait()
                            state.audio_queue.task_done()
                            stale_chunks += 1
                        except asyncio.QueueEmpty:
                            break

                    if stale_chunks > 0:
                        logger.warning(f"[Process-{client_id}] ⚠️ Drained {stale_chunks} stale chunks from queue ({stale_chunks * 0.1:.1f}s of audio)")
                    else:
                        logger.debug(f"[Process-{client_id}] Queue was already empty")
    
    async def _process_batch_chunk_internal(self, websocket: WebSocket, chunk: bytes):
        """
        Internal method: Process audio chunk in batch mode.

        Called by the batch worker. Assumes chunk is already validated.

        Args:
            websocket: Client WebSocket connection
            chunk: Audio chunk bytes (already validated)
        """
        # Get state
        async with self.client_states_lock:
            state = self.client_states.get(websocket)
            if not state:
                logger.error("No state for client in batch worker")
                return

        # Add to buffer
        new_data = np.frombuffer(chunk, dtype=np.float32)
        state.audio_buffer = np.concatenate([state.audio_buffer, new_data])
        
        # Run interim transcription periodically (throttled)
        current_time = time.time() * 1000
        INTERIM_THROTTLE_MS = 500
        
        if (state.audio_buffer.size > self.interim_min_samples and 
            current_time - state.last_interim_time > INTERIM_THROTTLE_MS):
            
            interim_text = self.transcribe_interim(state.audio_buffer)
            if interim_text and interim_text.strip():
                await self.broadcast({
                    "type": "interim",
                    "text": interim_text
                })
                state.last_interim_time = current_time



