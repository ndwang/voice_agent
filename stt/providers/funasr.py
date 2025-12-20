"""
FunASR STT Provider

FunASR provider implementation using Fun-ASR-Nano model.
"""
import logging
import sys
import tempfile
import os
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import scipy.io.wavfile as wavfile
import torch

# Add project root to path to import config_loader
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from stt.base import STTProvider, Segment

logger = logging.getLogger(__name__)


class FunASRProvider(STTProvider):
    """FunASR STT provider implementation."""
    
    def __init__(
        self,
        model_name: str = "FunAudioLLM/Fun-ASR-Nano-2512",
        vad_model: str = "fsmn-vad",
        vad_kwargs: Optional[Dict[str, Any]] = None,
        device: str = None,
        batch_size_s: int = 0
    ):
        """
        Initialize FunASR provider.
        
        Args:
            model_name: Model name or path (e.g., "FunAudioLLM/Fun-ASR-Nano-2512")
            vad_model: VAD model name (default: "fsmn-vad")
            vad_kwargs: VAD configuration kwargs (e.g., {"max_single_segment_time": 30000})
            device: Device to use ("cuda:0", "cuda", or "cpu")
            batch_size_s: Batch size for processing (default: 0)
        """
        try:
            from funasr import AutoModel
        except ImportError:
            raise ImportError(
                "funasr package is not installed. Please install it with: pip install funasr"
            )
        
        self.AutoModel = AutoModel
        self.model_name = model_name
        self.vad_model = vad_model
        self.vad_kwargs = vad_kwargs or {"max_single_segment_time": 30000}
        
        # Auto-detect device if not specified
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size_s = batch_size_s
        
        # Load the model
        logger.info(f"Loading FunASR model '{self.model_name}' on {self.device}...")
        try:
            self.model = AutoModel(
                model=self.model_name,
                vad_model=self.vad_model,
                vad_kwargs=self.vad_kwargs,
                device=self.device
            )
            logger.info("FunASR model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading FunASR model: {e}", exc_info=True)
            raise
    
    def _save_audio_to_temp_file(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Save numpy array audio to temporary WAV file.
        
        Args:
            audio: Audio data as numpy array (float32)
            sample_rate: Sample rate in Hz (default: 16000)
            
        Returns:
            Path to temporary WAV file
        """
        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav', prefix='funasr_')
        # Close the file descriptor immediately - wavfile.write will open the file by path
        os.close(temp_fd)
        
        try:
            # Convert float32 to int16 if needed
            if audio.dtype == np.float32:
                # Normalize to [-1, 1] range and convert to int16
                audio_int16 = (audio * 32767.0).astype(np.int16)
            elif audio.dtype == np.int16:
                audio_int16 = audio
            else:
                # Convert to float32 first, then to int16
                audio_float32 = audio.astype(np.float32)
                if audio_float32.max() > 1.0 or audio_float32.min() < -1.0:
                    # Normalize if needed
                    audio_float32 = audio_float32 / max(abs(audio_float32.max()), abs(audio_float32.min()))
                audio_int16 = (audio_float32 * 32767.0).astype(np.int16)
            
            # Write WAV file
            wavfile.write(temp_path, sample_rate, audio_int16)
            return temp_path
        except Exception as e:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except:
                pass
            raise RuntimeError(f"Failed to save audio to temp file: {e}") from e
    
    def transcribe(
        self, 
        audio: np.ndarray, 
        language: str = "zh", 
        vad_filter: bool = False
    ) -> Tuple[List[Segment], Dict[str, Any]]:
        """
        Transcribe audio using FunASR.
        
        Args:
            audio: Audio data as numpy array (float32, 16kHz mono)
            language: Language code (e.g., "zh", "en") - Note: FunASR may auto-detect
            vad_filter: Whether to apply VAD filtering - Note: FunASR uses VAD model by default
            
        Returns:
            Tuple of (segments, info):
            - segments: List of Segment objects with .text attribute
            - info: Dictionary with transcription metadata
        """
        # Save audio to temporary WAV file
        temp_path = None
        try:
            temp_path = self._save_audio_to_temp_file(audio, sample_rate=16000)
            
            # Run transcription
            res = self.model.generate(
                input=[temp_path],
                cache={},
                batch_size_s=self.batch_size_s
            )
            
            # Extract text from result
            # res is a list of dicts, each with "text" key
            if res and len(res) > 0:
                text = res[0].get("text", "")
            else:
                text = ""
            
            # Create segment object
            segments = [Segment(text)] if text else []
            
            # Create info dict (minimal metadata)
            info = {
                "language": language,
                "language_probability": 1.0,  # FunASR doesn't provide this
            }
            
            return segments, info
            
        except Exception as e:
            logger.error(f"Error during FunASR transcription: {e}", exc_info=True)
            raise
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {temp_path}: {e}")

