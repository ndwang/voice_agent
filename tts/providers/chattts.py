"""
ChatTTS Provider

ChatTTS provider implementation for local model inference.
"""
from typing import AsyncIterator, Optional
import sys
from pathlib import Path
import numpy as np
import asyncio

from tts.base import TTSProvider


class ChatTTSProvider(TTSProvider):
    """ChatTTS provider implementation."""
    
    def __init__(self, 
                 model_source: str = "local",
                 device: Optional[str] = None):
        """
        Initialize ChatTTS provider.
        
        Args:
            model_source: Source for loading models ("local", "huggingface", "custom")
            device: Device to use ("cuda", "cpu", None for auto)
        """
        # Add ChatTTS to path if needed
        chattts_path = Path(__file__).parent.parent / "ChatTTS"
        if str(chattts_path) not in sys.path:
            sys.path.insert(0, str(chattts_path))
        
        # Import ChatTTS
        import ChatTTS
        
        self.ChatTTS = ChatTTS
        self.model_source = model_source
        self.device = device
        
        # Initialize ChatTTS instance (will be loaded on first use)
        self.chat: Optional[ChatTTS.Chat] = None
        self._loaded = False
    
    @property
    def native_sample_rate(self) -> int:
        """ChatTTS native sample rate is 24kHz."""
        return 24000

    def _ensure_loaded(self):
        """Ensure ChatTTS model is loaded."""
        if not self._loaded or self.chat is None:
            import logging
            logger = logging.getLogger("ChatTTS")
            
            self.chat = self.ChatTTS.Chat(logger)
            
            # Load models
            if self.chat.load(source=self.model_source):
                self._loaded = True
            else:
                raise RuntimeError("Failed to load ChatTTS model")
    
    async def synthesize_stream(self, text: str, **kwargs) -> AsyncIterator[bytes]:
        """
        Synthesize text using ChatTTS and stream audio chunks.
        
        Args:
            text: Text to synthesize (can be a list of sentences)
            **kwargs: Optional parameters:
                - speaker: Speaker embedding (default: random)
                - temperature: Temperature for inference (default: 0.3)
                - top_p: Top P for decoding (default: 0.7)
                - top_k: Top K for decoding (default: 20)
                - stream_speed: Stream speed (default: 1)
        
        Yields:
            Audio chunks as bytes (float32 PCM format, normalized to [-1, 1])
        """
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        
        # Ensure model is loaded
        self._ensure_loaded()
        
        # Prepare text (can be string or list)
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        # Get speaker embedding
        speaker = kwargs.get("speaker", None)
        if speaker is None:
            speaker = self.chat.sample_random_speaker()
        
        # Create inference parameters
        infer_params = self.ChatTTS.Chat.InferCodeParams(
            spk_emb=speaker,
            temperature=kwargs.get("temperature", 0.3),
            top_P=kwargs.get("top_p", 0.7),
            top_K=kwargs.get("top_k", 20),
            stream_speed=kwargs.get("stream_speed", 1),
        )
        
        # Run inference in executor (ChatTTS is synchronous)
        def run_inference():
            return self.chat.infer(
                texts,
                skip_refine_text=True,
                stream=True,
                params_infer_code=infer_params,
            )
        
        # Get streaming generator
        stream_generator = await loop.run_in_executor(None, run_inference)
        
        # Process stream chunks
        chat_sample_rate = 24000  # ChatTTS uses 24kHz
        
        for wav_chunk in stream_generator:
            # wav_chunk is numpy array with shape (n_texts, samples)
            # Convert to int16 PCM
            if len(wav_chunk.shape) == 2:
                # Multiple texts - take first one or concatenate
                wav = wav_chunk[0] if wav_chunk.shape[0] > 0 else wav_chunk.flatten()
            else:
                wav = wav_chunk
            
            # Ensure wav is 1D
            if len(wav.shape) > 1:
                wav = wav.flatten()
            
            # Skip empty chunks
            if len(wav) == 0:
                continue
            
            # Ensure wav is float32 and normalized to [-1, 1]
            if wav.dtype != np.float32:
                wav = wav.astype(np.float32)
            # Ensure values are in [-1, 1] range (ChatTTS should already be normalized)
            wav = np.clip(wav, -1.0, 1.0)
            
            # Convert float32 array to bytes
            audio_bytes = wav.tobytes()
            
            yield audio_bytes
    
    async def synthesize(self, text: str, **kwargs) -> bytes:
        """
        Synthesize text using ChatTTS and return complete audio (non-streaming).
        
        More efficient than streaming for complete audio generation.
        
        Args:
            text: Text to synthesize (can be a list of sentences)
            **kwargs: Optional parameters:
                - speaker: Speaker embedding (default: random)
                - temperature: Temperature for inference (default: 0.3)
                - top_p: Top P for decoding (default: 0.7)
                - top_k: Top K for decoding (default: 20)
        
        Returns:
            Complete audio as bytes (float32 PCM format, normalized to [-1, 1])
        """
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        
        # Ensure model is loaded
        self._ensure_loaded()
        
        # Prepare text (can be string or list)
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        # Get speaker embedding
        speaker = kwargs.get("speaker", None)
        if speaker is None:
            speaker = self.chat.sample_random_speaker()
        
        # Create inference parameters (no stream_speed for non-streaming)
        infer_params = self.ChatTTS.Chat.InferCodeParams(
            spk_emb=speaker,
            temperature=kwargs.get("temperature", 0.3),
            top_P=kwargs.get("top_p", 0.7),
            top_K=kwargs.get("top_k", 20),
        )
        
        # Run inference in executor (ChatTTS is synchronous)
        def run_inference():
            # Use non-streaming inference for efficiency
            wavs = self.chat.infer(
                texts,
                skip_refine_text=True,
                stream=False,  # Non-streaming
                params_infer_code=infer_params,
            )
            return wavs
        
        # Get complete audio
        wavs = await loop.run_in_executor(None, run_inference)
        
        # Process audio (wavs is a list of numpy arrays)
        chat_sample_rate = 24000  # ChatTTS uses 24kHz
        audio_chunks = []
        
        for wav in wavs:
            # Ensure wav is 1D
            if len(wav.shape) > 1:
                wav = wav.flatten()
            
            # Skip empty chunks
            if len(wav) == 0:
                continue
            
            # Ensure wav is float32 and normalized to [-1, 1]
            if wav.dtype != np.float32:
                wav = wav.astype(np.float32)
            # Ensure values are in [-1, 1] range (ChatTTS should already be normalized)
            wav = np.clip(wav, -1.0, 1.0)
            
            # Convert float32 array to bytes
            audio_chunks.append(wav.tobytes())
        
        # Combine all audio chunks
        return b"".join(audio_chunks)
    
    async def list_voices(self) -> list:
        """List available ChatTTS voices (speaker embeddings)."""
        self._ensure_loaded()
        
        # ChatTTS doesn't have predefined voices like Edge TTS
        # Instead, it uses speaker embeddings
        # Return a placeholder indicating random speaker selection
        return [
            {
                "name": "random",
                "description": "Random speaker embedding (default)",
                "type": "speaker_embedding"
            }
        ]

