"""
Audio Player

Streaming audio playback using sounddevice.
"""
import asyncio
import numpy as np
import sounddevice as sd
from scipy import signal
from typing import Optional, Callable, Awaitable
from collections import deque
from pathlib import Path
import logging
import sys
import threading

from core.settings import get_settings
from core.logging import setup_logging, get_logger

# Set up logging
setup_logging()
logger = get_logger(__name__)


class AudioPlayer:
    """Streaming audio player."""

    def __init__(
        self,
        sample_rate: int = None,
        channels: int = None,
        output_sample_rate: Optional[int] = None,
        output_device: Optional[str] = None,
        on_play_state: Optional[Callable[[bool], Awaitable[None]]] = None,
    ):
        """
        Initialize audio player.

        Args:
            sample_rate: Audio sample rate (default: from config)
            channels: Number of audio channels (default: from config)
        """
        settings = get_settings()
        self.sample_rate = sample_rate or settings.audio.output.sample_rate
        self.channels = channels or settings.audio.output.channels
        self.playback_sample_rate = (
            output_sample_rate or settings.audio.output.sample_rate
        )
        self.output_device = output_device or settings.audio.output.device
        self.audio_queue: asyncio.Queue = asyncio.Queue()
        self.playing = False
        self.play_task: Optional[asyncio.Task] = None
        self.on_play_state = on_play_state
        self._audio_active = False
        
        # Reactive state management
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.get_event_loop()
        self._buffer_empty_event = asyncio.Event()
        self._buffer_empty_event.set()
        
        # Streaming buffer for continuous playback
        self._buffer_lock = threading.Lock()
        self._audio_buffer = deque()  # Queue of numpy arrays ready to play
        self._stream: Optional[sd.OutputStream] = None
        
        # Buffer size in frames (smaller = lower latency, but more overhead)
        # ~50ms buffer for low latency
        self._frames_per_chunk = int(self.playback_sample_rate * 0.05)  # 50ms chunks
    
    async def play_audio_chunk(self, audio_data: bytes, source_sample_rate: int):
        """
        Add audio chunk to playback queue.
        
        Args:
            audio_data: Audio data as bytes (float32 format, normalized to [-1, 1])
            source_sample_rate: Sample rate of the incoming audio
        """
        # Store source sample rate for this chunk if needed
        # We'll process it in the playback loop
        await self.audio_queue.put((audio_data, source_sample_rate))
        
        # Start playback task if not already running
        if not self.playing:
            self.playing = True
            self.play_task = asyncio.create_task(self._playback_loop())
    
    def _audio_callback(self, outdata, frames, time, status):
        """Callback function for sounddevice stream - called in audio thread."""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Get data from buffer
        with self._buffer_lock:
            if len(self._audio_buffer) == 0:
                # No data available - output silence
                outdata.fill(0)
                if self._audio_active:
                    # Signal that buffer is exhausted
                    self._loop.call_soon_threadsafe(self._buffer_empty_event.set)
                return
            
            # Collect enough frames from buffer
            collected = []
            total_frames = 0
            
            while total_frames < frames and len(self._audio_buffer) > 0:
                chunk = self._audio_buffer.popleft()
                collected.append(chunk)
                total_frames += len(chunk)
            
            if total_frames == 0:
                outdata.fill(0)
                return
            
            # Concatenate chunks
            if len(collected) == 1:
                audio_data = collected[0]
            else:
                audio_data = np.concatenate(collected, axis=0)
            
            # If we have more than needed, put excess back
            if total_frames > frames:
                excess = audio_data[frames:]
                self._audio_buffer.appendleft(excess)
                audio_data = audio_data[:frames]
            
            # Copy to output
            if len(audio_data) == frames:
                outdata[:] = audio_data
            else:
                # Pad with zeros if needed (shouldn't happen, but safety check)
                outdata.fill(0)
                outdata[:len(audio_data)] = audio_data
    
    async def _playback_loop(self):
        """Background task for audio playback - processes queue and feeds stream."""
        self._loop = asyncio.get_running_loop() 
        logger.info("Audio playback started")
        
        try:
            # Start the audio stream
            self._stream = sd.OutputStream(
                samplerate=self.playback_sample_rate,
                channels=self.channels,
                dtype=np.float32,
                device=self.output_device,
                blocksize=self._frames_per_chunk,
                callback=self._audio_callback,
                latency='low'  # Low latency mode
            )
            
            self._stream.start()
            
            # Process incoming audio chunks
            while self.playing:
                # Always wait for next chunk OR buffer empty
                get_task = asyncio.create_task(self.audio_queue.get())
                wait_empty_task = asyncio.create_task(self._buffer_empty_event.wait())
                
                # Initialize pending to avoid UnboundLocalError
                pending = {get_task, wait_empty_task}
                try:
                    done, pending = await asyncio.wait(
                        [get_task, wait_empty_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    if wait_empty_task in done:
                        # Buffer exhausted!
                        trigger_stop = False
                        with self._buffer_lock:
                            if self._audio_active:
                                self._audio_active = False
                                trigger_stop = True
                        if trigger_stop and self.on_play_state:
                            await self.on_play_state(False)
                    
                    if get_task in done:
                        item = get_task.result()
                    else:
                        # No new data yet, continue loop
                        continue
                        
                except asyncio.CancelledError:
                    # If cancelled, clean up and re-raise
                    for t in pending:
                        if not t.done():
                            t.cancel()
                    raise
                    
                except Exception as e:
                    logger.error(f"Error waiting for audio: {e}", exc_info=True)
                    # Clean up tasks
                    for t in pending:
                        if not t.done():
                            t.cancel()
                    continue
                    
                finally:
                    # Always cleanup pending tasks
                    for t in pending:
                        if not t.done():
                            t.cancel()

                # Process new item
                try:
                    if isinstance(item, tuple):
                        audio_bytes, source_rate = item
                    else:
                        audio_bytes = item
                        source_rate = self.sample_rate
                    
                    if not self.playing:
                        break
                    
                    # Convert bytes to numpy array
                    audio_float = np.frombuffer(audio_bytes, dtype=np.float32)
                    if self.channels == 1:
                        audio_float = audio_float.reshape(-1, 1)
                    else:
                        audio_float = audio_float.reshape(-1, self.channels)
                    
                    playback_audio = self._resample_audio(audio_float, source_rate)
                    
                    # Push to buffer
                    trigger_play = False
                    with self._buffer_lock:
                        self._audio_buffer.append(playback_audio)
                        self._buffer_empty_event.clear()
                        if not self._audio_active:
                            self._audio_active = True
                            trigger_play = True
                    
                    if trigger_play and self.on_play_state:
                        await self.on_play_state(True)
                    
                    logger.debug(f"Added audio chunk: {len(playback_audio)} frames to buffer")
                    
                except Exception as e:
                    logger.error(f"Audio processing error: {e}", exc_info=True)
        
        except Exception as e:
            logger.error(f"Playback loop error: {e}", exc_info=True)
        finally:
            # Stop and close stream
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as e:
                    logger.error(f"Error closing stream: {e}")
                finally:
                    self._stream = None
            
            self.playing = False
            logger.info("Audio playback stopped")
            
            if self.on_play_state and self._audio_active:
                self._audio_active = False
                await self.on_play_state(False)

    def _resample_audio(self, audio: np.ndarray, source_rate: int) -> np.ndarray:
        """Resample audio to the playback sample rate if needed."""
        if (
            audio.size == 0
            or self.playback_sample_rate == source_rate
        ):
            return audio

        src_len = audio.shape[0]
        target_len = max(
            1, int(round(src_len * self.playback_sample_rate / source_rate))
        )

        if target_len == src_len:
            return audio

        # Use scipy for high-quality resampling
        # Resample takes (data, num_samples)
        # audio is (samples, channels)
        resampled = signal.resample(audio, target_len, axis=0).astype(np.float32)
        return resampled
    
    async def stop(self):
        """Stop audio playback and clear all queued chunks."""
        # Set playing flag first to signal the playback loop to stop
        self.playing = False
        
        # Stop stream immediately
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception as e:
                logger.debug(f"Error stopping stream: {e}")
        
        # Clear buffer
        with self._buffer_lock:
            self._audio_buffer.clear()
        
        # Clear queue immediately to prevent any queued chunks from playing
        cleared_count = 0
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                cleared_count += 1
            except asyncio.QueueEmpty:
                break
        
        if cleared_count > 0:
            logger.info(f"Cleared {cleared_count} audio chunk(s) from queue")
        
        # Cancel the playback task
        if self.play_task:
            self.play_task.cancel()
            try:
                await self.play_task
            except asyncio.CancelledError:
                pass
        
        # Close stream
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception as e:
                logger.debug(f"Error closing stream: {e}")
            finally:
                self._stream = None
        
        # Update play state callback
        if self.on_play_state and self._audio_active:
            self._audio_active = False
            await self.on_play_state(False)

