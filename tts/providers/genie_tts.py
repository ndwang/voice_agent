"""
GenieTTS Provider implementation.
"""
import io
import asyncio
import logging
from typing import AsyncIterator, Optional, Union
from pathlib import Path

from tts.base import TTSProvider

logger = logging.getLogger(__name__)

class GenieTTSProvider(TTSProvider):
    """GenieTTS provider implementation."""
    
    def __init__(
        self,
        character_name: str,
        onnx_model_dir: str,
        language: str = "zh",
        reference_audio_path: Optional[str] = None,
        reference_audio_text: Optional[str] = None,
        source_sample_rate: int = 32000
    ):
        """
        Initialize GenieTTS provider.
        
        Args:
            character_name: Name of the character voice
            onnx_model_dir: Directory containing the ONNX model files
            language: Language code (default: "zh")
            reference_audio_path: Path to reference audio for cloning
            reference_audio_text: Text for the reference audio
            source_sample_rate: Sample rate of audio produced by GenieTTS
        """
        import genie_tts as genie
        self.genie = genie
        self.character_name = character_name
        self.onnx_model_dir = onnx_model_dir
        self.language = language
        self.reference_audio_path = reference_audio_path
        self.reference_audio_text = reference_audio_text
        self.source_sample_rate = source_sample_rate
        
        self._loaded = False
        self._current_ref_audio = (reference_audio_path, reference_audio_text)
        
        # Load character immediately
        self._load_character()

    @property
    def native_sample_rate(self) -> int:
        """Get native sample rate (from config)."""
        return self.source_sample_rate

    def _load_character(self):
        """Load character model and set initial reference audio."""
        try:
            logger.info(f"Loading GenieTTS character: {self.character_name} from {self.onnx_model_dir}")
            self.genie.load_character(
                character_name=self.character_name,
                onnx_model_dir=self.onnx_model_dir,
                language=self.language
            )
            
            if self.reference_audio_path and self.reference_audio_text:
                logger.info(f"Setting GenieTTS reference audio for {self.character_name}")
                self.genie.set_reference_audio(
                    character_name=self.character_name,
                    audio_path=self.reference_audio_path,
                    audio_text=self.reference_audio_text
                )
            
            self._loaded = True
        except Exception as e:
            logger.error(f"Failed to load GenieTTS character {self.character_name}: {e}")
            raise

    async def synthesize_stream(self, text: str, **kwargs) -> AsyncIterator[bytes]:
        """
        Synthesize text using GenieTTS and stream audio chunks.
        
        Args:
            text: Text to synthesize
            **kwargs: Optional overrides:
                - reference_audio_path: Override reference audio path
                - reference_audio_text: Override reference audio text
                - character_name: Override character name (must be loaded)
        
        Yields:
            Audio chunks as bytes (float32 PCM format, normalized to [-1, 1])
        """
        if not text or not text.strip():
            return

        if not self._loaded:
            self._load_character()

        char_name = kwargs.get("character_name", self.character_name)
        ref_path = kwargs.get("reference_audio_path", self.reference_audio_path)
        ref_text = kwargs.get("reference_audio_text", self.reference_audio_text)

        # Update reference audio if changed at runtime
        if (ref_path, ref_text) != self._current_ref_audio:
            if ref_path and ref_text:
                logger.info(f"Updating GenieTTS reference audio for {char_name}")
                self.genie.set_reference_audio(
                    character_name=char_name,
                    audio_path=ref_path,
                    audio_text=ref_text
                )
                self._current_ref_audio = (ref_path, ref_text)

        # Use pydub for conversion if needed, similar to EdgeTTS
        from pydub import AudioSegment
        import numpy as np

        audio_chunks = []
        try:
            async for chunk in self.genie.tts_async(
                character_name=char_name,
                text=text,
                play=False,
                split_sentence=kwargs.get("split_sentence", False)
            ):
                audio_chunks.append(chunk)
        except Exception as e:
            logger.error(f"GenieTTS synthesis error: {e}")
            raise

        if not audio_chunks:
            return

        # GenieTTS often yields raw PCM chunks instead of WAV chunks.
        # We check for the RIFF header to decide how to load it.
        wav_data = b"".join(audio_chunks)
        
        if not wav_data:
            return

        try:
            if wav_data.startswith(b"RIFF"):
                # Load WAV audio from bytes
                audio = AudioSegment.from_wav(io.BytesIO(wav_data))
            else:
                # Load raw PCM audio (assuming int16, mono)
                audio = AudioSegment.from_raw(
                    io.BytesIO(wav_data),
                    sample_width=2,
                    frame_rate=self.source_sample_rate,
                    channels=1
                )
            
            # Convert to mono if needed
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Convert to float32 PCM (normalized to [-1, 1])
            # Get samples as numpy array and normalize
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            # Normalize to [-1, 1] range (int16 range is [-32768, 32767])
            audio_float = samples / 32768.0
            # Convert to bytes
            audio_bytes = audio_float.tobytes()
            
            # Yield the processed audio
            yield audio_bytes
            
        except Exception as e:
            logger.error(f"Failed to process GenieTTS audio: {e}")
            raise

    async def list_voices(self) -> list:
        """List available voices (currently just the loaded character)."""
        return [
            {
                "name": self.character_name,
                "language": self.language,
                "model_dir": self.onnx_model_dir
            }
        ]

