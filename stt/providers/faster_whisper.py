"""
Faster-Whisper STT Provider

Faster-whisper provider implementation.
"""
import logging
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any
import numpy as np
import torch
from faster_whisper import WhisperModel

# Add project root to path to import config_loader
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from stt.base import STTProvider, Segment

logger = logging.getLogger(__name__)


class FasterWhisperProvider(STTProvider):
    """Faster-whisper STT provider implementation."""
    
    def __init__(
        self, 
        model_path: str = None,
        device: str = None,
        compute_type: str = None
    ):
        """
        Initialize Faster-Whisper provider.
        
        Args:
            model_path: Path to model directory (relative to stt/ or absolute)
            device: Device to use ("cuda" or "cpu")
            compute_type: Compute type ("int8", "float16", "float32", or "default")
        """
        # Default to local model if not specified
        if model_path is None:
            model_path = Path(__file__).parent.parent / "faster-whisper-small"
        else:
            # If relative path, make it relative to stt directory
            model_path = Path(model_path)
            if not model_path.is_absolute():
                model_path = Path(__file__).parent.parent / model_path
        
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Auto-detect compute type if not specified
        if compute_type is None:
            compute_type = "int8" if device == "cuda" else "default"
        
        self.model_path = str(model_path)
        self.device = device
        self.compute_type = compute_type
        
        # Load the model
        logger.info(f"Loading Faster-Whisper model from '{self.model_path}' on {self.device} ({self.compute_type})...")
        try:
            self.model = WhisperModel(self.model_path, device=self.device, compute_type=self.compute_type)
            logger.info("Faster-Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading Faster-Whisper model: {e}", exc_info=True)
            raise
    
    def transcribe(
        self, 
        audio: np.ndarray, 
        language: str = "zh", 
        vad_filter: bool = False
    ) -> Tuple[List[Segment], Dict[str, Any]]:
        """
        Transcribe audio using Faster-Whisper.
        
        Args:
            audio: Audio data as numpy array (float32, 16kHz mono)
            language: Language code (e.g., "zh", "en")
            vad_filter: Whether to apply VAD filtering
            
        Returns:
            Tuple of (segments, info):
            - segments: List of Segment objects with .text attribute
            - info: Dictionary with transcription metadata
        """
        segments, info = self.model.transcribe(
            audio,
            language=language,
            vad_filter=vad_filter
        )
        
        # Convert faster-whisper segments to our Segment format
        segment_list = [Segment(seg.text) for seg in segments]
        
        return segment_list, info

