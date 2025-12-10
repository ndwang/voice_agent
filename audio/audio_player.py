"""
Audio Player

Streaming audio playback using sounddevice.
"""
import asyncio
import numpy as np
import sounddevice as sd
from typing import Optional
from collections import deque
from pathlib import Path
import logging
import sys

# Add project root to path to import config_loader
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config_loader import get_config

# Configure logging with time info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)


class AudioPlayer:
    """Streaming audio player."""
    
    def __init__(
        self,
        sample_rate: int = None,
        channels: int = None,
        output_sample_rate: Optional[int] = None,
        output_device: Optional[str] = None,
    ):
        """
        Initialize audio player.
        
        Args:
            sample_rate: Audio sample rate (default: from config)
            channels: Number of audio channels (default: from config)
        """
        self.sample_rate = sample_rate or get_config("audio", "sample_rate", default=16000)
        self.channels = channels or get_config("audio", "channels", default=1)
        self.playback_sample_rate = (
            output_sample_rate
            or get_config("audio", "output_sample_rate", default=self.sample_rate)
        )
        self.output_device = output_device or get_config("audio", "output_device")
        self.audio_queue: asyncio.Queue = asyncio.Queue()
        self.playing = False
        self.play_task: Optional[asyncio.Task] = None
    
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
                    )
                    sd.wait()  # Wait for playback to finish
                
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

