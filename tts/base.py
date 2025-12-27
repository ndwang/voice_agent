"""
TTS Provider Base Class

Abstract base class for TTS providers with streaming support.
"""
from abc import ABC, abstractmethod
from typing import AsyncIterator


class TTSProvider(ABC):
    """Abstract base class for TTS providers."""
    
    @abstractmethod
    async def synthesize_stream(self, text: str, **kwargs) -> AsyncIterator[bytes]:
        """
        Synthesize text to speech and stream audio chunks.
        
        Args:
            text: Text to synthesize
            **kwargs: Provider-specific parameters (voice, rate, pitch, etc.)
            
        Yields:
            Audio chunks as bytes (float32 PCM format, normalized to [-1, 1])
        """
        pass
    
    async def synthesize(self, text: str, **kwargs) -> bytes:
        """
        Synthesize text to speech and return complete audio.
        
        Default implementation collects all chunks from synthesize_stream.
        Providers can override for more efficient non-streaming synthesis.
        
        Args:
            text: Text to synthesize
            **kwargs: Provider-specific parameters (voice, rate, pitch, etc.)
            
        Returns:
            Complete audio as bytes (float32 PCM format, normalized to [-1, 1])
        """
        # Default implementation: collect all stream chunks
        audio_chunks = []
        async for chunk in self.synthesize_stream(text, **kwargs):
            audio_chunks.append(chunk)
        return b"".join(audio_chunks)
    
    @abstractmethod
    async def list_voices(self) -> list:
        """
        List available voices for this provider.
        
        Returns:
            List of voice dictionaries with provider-specific information
        """
        pass
    
    def parse_error(self, exception: Exception) -> tuple[int, str]:
        """
        Parse provider-specific errors and return appropriate status code and message.
        
        Args:
            exception: The exception raised by the provider
            
        Returns:
            Tuple of (status_code, error_message)
        """
        # Default implementation
        return 500, f"TTS synthesis error: {str(exception)}"

