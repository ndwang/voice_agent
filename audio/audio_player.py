"""
Audio Player

Streaming audio playback using sounddevice.
"""
import asyncio
import numpy as np
import sounddevice as sd
from typing import Optional
from collections import deque
from orchestrator.config import Config


class AudioPlayer:
    """Streaming audio player."""
    
    def __init__(self, sample_rate: int = None, channels: int = None):
        """
        Initialize audio player.
        
        Args:
            sample_rate: Audio sample rate (default: from config)
            channels: Number of audio channels (default: from config)
        """
        self.sample_rate = sample_rate or Config.AUDIO_SAMPLE_RATE
        self.channels = channels or Config.AUDIO_CHANNELS
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
                    
                    # Play audio
                    sd.play(audio_float, samplerate=self.sample_rate)
                    sd.wait()  # Wait for playback to finish
                
                except asyncio.TimeoutError:
                    # No audio for 1 second, check if we should continue
                    if self.audio_queue.empty():
                        # Wait a bit more before stopping
                        await asyncio.sleep(0.5)
                        if self.audio_queue.empty():
                            break
                except Exception as e:
                    print(f"Audio playback error: {e}")
        
        except Exception as e:
            print(f"Playback loop error: {e}")
        finally:
            self.playing = False
    
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

