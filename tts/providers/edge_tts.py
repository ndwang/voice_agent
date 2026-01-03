"""
Edge TTS Provider

Microsoft Edge TTS provider implementation.
"""
import io
from typing import AsyncIterator, Optional
import numpy as np
from pydub import AudioSegment

from tts.base import TTSProvider


class EdgeTTSProvider(TTSProvider):
    """Microsoft Edge TTS provider implementation."""
    
    def __init__(self, default_voice: str = "zh-CN-XiaoxiaoNeural", 
                 default_rate: str = "+0%", 
                 default_pitch: str = "+0Hz"):
        """
        Initialize Edge TTS provider.
        
        Args:
            default_voice: Default voice to use
            default_rate: Default speech rate (e.g., "+0%", "-50%")
            default_pitch: Default pitch (e.g., "+0Hz", "+50Hz")
        """
        import edge_tts
        from edge_tts import VoicesManager
        
        self.edge_tts = edge_tts
        self.VoicesManager = VoicesManager
        self.default_voice = default_voice
        self.default_rate = default_rate
        self.default_pitch = default_pitch
    
    @property
    def native_sample_rate(self) -> int:
        """Edge TTS native rate varies, typically 24kHz."""
        return 24000

    async def synthesize_stream(self, text: str, **kwargs) -> AsyncIterator[bytes]:
        """
        Synthesize text using Edge TTS and stream audio chunks.
        
        Args:
            text: Text to synthesize
            **kwargs: Optional parameters:
                - voice: Voice name (default: self.default_voice)
                - rate: Speech rate (default: self.default_rate)
                - pitch: Pitch (default: self.default_pitch)
        
        Yields:
            Audio chunks as bytes (float32 PCM format, normalized to [-1, 1])
        """
        if not text or not text.strip():
            # Return empty audio for empty text
            return
        
        voice = kwargs.get("voice", self.default_voice)
        rate = kwargs.get("rate", self.default_rate)
        pitch = kwargs.get("pitch", self.default_pitch)
        
        # Create Communicate object (following edge-tts example pattern)
        communicate = self.edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate,
            pitch=pitch
        )
        
        # Stream and convert audio chunks immediately to reduce latency
        has_audio = False
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                # chunk["data"] contains MP3 audio bytes
                mp3_data = chunk["data"]
                if not mp3_data:
                    continue

                has_audio = True

                # Convert this MP3 chunk to PCM immediately
                try:
                    # Load MP3 audio from bytes
                    audio = AudioSegment.from_mp3(io.BytesIO(mp3_data))
                except Exception as e:
                    # Skip invalid chunks but continue processing
                    continue

                # Skip empty audio
                if len(audio) == 0:
                    continue

                # Convert to mono if needed
                if audio.channels > 1:
                    audio = audio.set_channels(1)

                # Convert to float32 PCM (normalized to [-1, 1])
                samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                # Normalize to [-1, 1] range (int16 range is [-32768, 32767])
                audio_float = samples / 32768.0
                audio_bytes = audio_float.tobytes()

                # Yield chunk immediately
                if len(audio_bytes) >= 4:  # At least one float32 sample
                    yield audio_bytes
            # Note: We ignore WordBoundary and SentenceBoundary chunks
            # Could be used for word-level subtitle timing in the future

        # Validate we received at least some audio
        if not has_audio:
            raise ValueError("No audio data received from Edge TTS")
    
    async def list_voices(self) -> list:
        """
        List all available Edge TTS voices.
        
        Can also use VoicesManager for dynamic voice selection.
        """
        voices = await self.edge_tts.list_voices()
        return [
            {
                "name": v["Name"],
                "locale": v["Locale"],
                "gender": v["Gender"],
                "short_name": v["ShortName"]
            }
            for v in voices
        ]
    
    async def find_voices(self, **filters) -> list:
        """
        Find voices by attributes (Gender, Language, Locale).
        
        Example:
            voices = await provider.find_voices(Gender="Female", Language="zh")
            voices = await provider.find_voices(Locale="zh-CN")
        
        Args:
            **filters: Voice attributes to filter by (Gender, Language, Locale)
        
        Returns:
            List of matching voice dictionaries
        """
        voices_manager = await self.VoicesManager.create()
        matching_voices = voices_manager.find(**filters)
        
        return [
            {
                "name": v["Name"],
                "locale": v["Locale"],
                "gender": v["Gender"],
                "short_name": v["ShortName"]
            }
            for v in matching_voices
        ]

