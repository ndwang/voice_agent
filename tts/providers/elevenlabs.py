"""
ElevenLabs TTS Provider

Implementation for ElevenLabs TTS provider with streaming support.
"""
import os
import asyncio
import numpy as np
from typing import AsyncIterator, Optional

from tts.base import TTSProvider
from core.logging import get_logger

logger = get_logger(__name__)

class ElevenLabsProvider(TTSProvider):
    """ElevenLabs TTS provider implementation."""
    
    def __init__(self, 
                 default_voice_id: str,
                 output_sample_rate: int = 16000,
                 stability: float = 0.5,
                 similarity_boost: float = 0.8,
                 style: float = 0.0,
                 use_speaker_boost: bool = True):
        """
        Initialize ElevenLabs provider.
        
        Args:
            default_voice_id: Default voice ID to use
            output_sample_rate: Target output sample rate in Hz
            stability: Voice stability setting (0.0 to 1.0)
            similarity_boost: Similarity boost setting (0.0 to 1.0)
            style: Style exaggeration setting (0.0 to 1.0)
            use_speaker_boost: Whether to use speaker boost
        """
        try:
            from elevenlabs.client import AsyncElevenLabs
            from elevenlabs import VoiceSettings
            self.AsyncElevenLabs = AsyncElevenLabs
            self.VoiceSettings = VoiceSettings
        except ImportError:
            logger.error("ElevenLabs SDK not installed. Please run 'pip install elevenlabs'")
            raise ImportError("ElevenLabs SDK not installed")

        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            error_msg = "ElevenLabs API key not provided. Set ELEVENLABS_API_KEY environment variable or provide api_key in config.yaml"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        self.client = self.AsyncElevenLabs(api_key=self.api_key)
        self.default_voice_id = default_voice_id
        self.output_sample_rate = output_sample_rate
        self.default_settings = {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style": style,
            "use_speaker_boost": use_speaker_boost
        }

    async def synthesize_stream(self, text: str, **kwargs) -> AsyncIterator[bytes]:
        """
        Synthesize text using ElevenLabs and stream audio chunks.
        
        Args:
            text: Text to synthesize
            **kwargs: Optional parameters:
                - voice_id: Voice ID (default: self.default_voice_id)
                - stability: Stability setting
                - similarity_boost: Similarity boost setting
                - style: Style setting
        
        Yields:
            Audio chunks as bytes (float32 PCM format, normalized to [-1, 1])
        """
        if not text or not text.strip():
            return

        voice_id = kwargs.get("voice_id", self.default_voice_id)
        
        # Merge settings
        settings = self.VoiceSettings(
            stability=kwargs.get("stability", self.default_settings["stability"]),
            similarity_boost=kwargs.get("similarity_boost", self.default_settings["similarity_boost"]),
            style=kwargs.get("style", self.default_settings["style"]),
            use_speaker_boost=kwargs.get("use_speaker_boost", self.default_settings["use_speaker_boost"])
        )

        try:
            audio_stream = self.client.text_to_speech.stream(
                text=text,
                voice_id=voice_id,
                model_id="eleven_multilingual_v2",
                voice_settings=settings,
                output_format="pcm_16000"
            )

            # Stream PCM chunks as they arrive (ElevenLabs returns int16 PCM, convert to float32)
            chunk_count = 0
            async for chunk in audio_stream:
                if chunk:
                    chunk_count += 1
                    # Convert int16 bytes to float32 normalized array
                    audio_int16 = np.frombuffer(chunk, dtype=np.int16)
                    # Normalize to [-1, 1] range
                    audio_float = audio_int16.astype(np.float32) / 32768.0
                    # Convert back to bytes (float32)
                    yield audio_float.tobytes()
            
            # Validate we received at least some audio data
            if chunk_count == 0:
                logger.warning("No audio chunks received from ElevenLabs")
                return

        except Exception as e:
            logger.error(f"ElevenLabs synthesis error: {e}", exc_info=True)
            raise

    async def list_voices(self) -> list:
        """
        List all available ElevenLabs voices.
        """
        try:
            response = await self.client.voices.get_all()
            return [
                {
                    "name": v.name,
                    "voice_id": v.voice_id,
                    "category": v.category,
                    "description": v.description,
                    "preview_url": v.preview_url
                }
                for v in response.voices
            ]
        except Exception as e:
            logger.error(f"Error listing ElevenLabs voices: {e}")
            return []

    def parse_error(self, exception: Exception) -> tuple[int, str]:
        """Parse ElevenLabs errors."""
        error_msg = str(exception)
        if "quota" in error_msg.lower():
            return 429, "ElevenLabs quota exceeded"
        elif "api key" in error_msg.lower():
            return 401, "Invalid ElevenLabs API key"
        return 500, f"ElevenLabs error: {error_msg}"


