"""
STT Provider Base Class

Abstract base class for STT providers with transcription support.
"""
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any
import numpy as np


class Segment:
    """Segment-like object to match faster-whisper's segment format."""
    
    def __init__(self, text: str):
        self.text = text


class STTProvider(ABC):
    """Abstract base class for STT providers."""
    
    @abstractmethod
    def transcribe(
        self, 
        audio: np.ndarray, 
        language: str = "zh", 
        vad_filter: bool = False
    ) -> Tuple[List[Segment], Dict[str, Any]]:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio data as numpy array (float32, 16kHz mono)
            language: Language code (e.g., "zh", "en")
            vad_filter: Whether to apply VAD filtering
            
        Returns:
            Tuple of (segments, info):
            - segments: List of Segment objects with .text attribute
            - info: Dictionary with transcription metadata
        """
        pass
    
    def parse_error(self, exception: Exception) -> Tuple[int, str]:
        """
        Parse provider-specific errors and return appropriate status code and message.
        
        Args:
            exception: The exception raised by the provider
            
        Returns:
            Tuple of (status_code, error_message)
        """
        # Default implementation
        return 500, f"STT transcription error: {str(exception)}"

