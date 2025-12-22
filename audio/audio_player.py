"""
Audio Player

Streaming audio playback using sounddevice.
"""
import asyncio
import numpy as np
import sounddevice as sd
from typing import Optional, Callable, Awaitable
from collections import deque
from pathlib import Path
import logging
import sys

from core.config import get_config
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
        self.sample_rate = sample_rate or get_config("audio", "output", "sample_rate", default=16000)
        self.channels = channels or get_config("audio", "output", "channels", default=1)
        self.playback_sample_rate = (
            output_sample_rate
            or get_config("audio", "output", "sample_rate", default=self.sample_rate)
        )
        self.output_device = output_device or get_config("audio", "output", "device")
        self.audio_queue: asyncio.Queue = asyncio.Queue()
        self.playing = False
        self.play_task: Optional[asyncio.Task] = None
        self.on_play_state = on_play_state
        self._audio_active = False
    
    async def play_audio_chunk(self, audio_data: bytes):
        """
        Add audio chunk to playback queue.
        
        Args:
            audio_data: Audio data as bytes (int16 format)
        """
        await self.audio_queue.put(audio_data)
        
        # Start playback task if not already running
        if not self.playing:
            self.playing = True
            self.play_task = asyncio.create_task(self._playback_loop())
    
    async def _playback_loop(self):
        """Background task for audio playback."""
        logger.info("Audio playback started")
        try:
            while True:
                try:
                    # Get audio chunk from queue (with timeout)
                    audio_bytes = await asyncio.wait_for(self.audio_queue.get(), timeout=1.0)
                    
                    # Convert bytes to numpy array (int16)
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                    
                    # Convert to float32 and normalize
                    audio_float = audio_array.astype(np.float32) / 32767.0
                    
                    # Reshape if needed
                    if self.channels == 1:
                        audio_float = audio_float.reshape(-1, 1)
                    else:
                        audio_float = audio_float.reshape(-1, self.channels)
                    
                    # Calculate duration for logging
                    duration_ms = len(audio_float) / self.sample_rate * 1000
                    logger.info(f"Playing audio chunk: {len(audio_bytes)} bytes, {duration_ms:.1f}ms duration")
                    
                    # Resample if playback device requires a different rate
                    playback_audio = self._resample_audio(audio_float)

                    # Play audio
                    sd.play(
                        playback_audio,
                        samplerate=self.playback_sample_rate,
                        device=self.output_device,
                        blocking=False,
                    )
                    
                    if self.on_play_state and not self._audio_active:
                        self._audio_active = True
                        await self.on_play_state(True)
                    
                    # Wait for playback to finish, but allow stop() to interrupt via sd.stop()
                    await asyncio.to_thread(sd.wait)
                    
                    if self.audio_queue.empty() and self.on_play_state and self._audio_active:
                        self._audio_active = False
                        await self.on_play_state(False)
                
                except asyncio.TimeoutError:
                    # No audio for 1 second, check if we should continue
                    if self.audio_queue.empty():
                        # Wait a bit more before stopping
                        await asyncio.sleep(0.5)
                        if self.audio_queue.empty():
                            break
                except Exception as e:
                    logger.error(f"Audio playback error: {e}", exc_info=True)
        
        except Exception as e:
            logger.error(f"Playback loop error: {e}", exc_info=True)
        finally:
            self.playing = False
            logger.info("Audio playback stopped")
            if self.on_play_state and self._audio_active:
                self._audio_active = False
                await self.on_play_state(False)

    def _resample_audio(self, audio: np.ndarray) -> np.ndarray:
        """Resample audio to the playback sample rate if needed."""
        if (
            audio.size == 0
            or self.playback_sample_rate == self.sample_rate
        ):
            return audio

        src_len = audio.shape[0]
        target_len = max(
            1, int(round(src_len * self.playback_sample_rate / self.sample_rate))
        )

        if target_len == src_len:
            return audio

        x_old = np.linspace(0, src_len - 1, src_len)
        x_new = np.linspace(0, src_len - 1, target_len)

        if audio.ndim == 1 or audio.shape[1] == 1:
            resampled = np.interp(x_new, x_old, audio.reshape(-1)).astype(np.float32)
            return resampled.reshape(-1, 1)

        resampled = np.empty((target_len, audio.shape[1]), dtype=np.float32)
        for ch in range(audio.shape[1]):
            resampled[:, ch] = np.interp(x_new, x_old, audio[:, ch])
        return resampled
    
    async def stop(self):
        """Stop audio playback."""
        self.playing = False
        try:
            sd.stop()
        except Exception:
            pass
        if self.play_task:
            self.play_task.cancel()
            try:
                await self.play_task
            except asyncio.CancelledError:
                pass
        
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        if self.on_play_state and self._audio_active:
            self._audio_active = False
            await self.on_play_state(False)

