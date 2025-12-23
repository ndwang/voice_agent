"""
FunASR STT Provider

FunASR provider implementation using Fun-ASR-Nano model.
Supports both batch and streaming modes.
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

from stt.base import STTProvider, Segment

logger = logging.getLogger(__name__)


class FunASRProvider(STTProvider):
    """FunASR STT provider implementation."""
    
    def __init__(
        self,
        model_name: str = "FunAudioLLM/Fun-ASR-Nano-2512",
        vad_model: str = "fsmn-vad",
        vad_kwargs: Optional[Dict[str, Any]] = None,
        punc_model: Optional[str] = None,
        device: str = None,
        batch_size_s: int = 0,
        streaming_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize FunASR provider.
        
        Args:
            model_name: Model name or path (e.g., "FunAudioLLM/Fun-ASR-Nano-2512")
            vad_model: VAD model name (default: "fsmn-vad")
            vad_kwargs: VAD configuration kwargs (e.g., {"max_single_segment_time": 30000})
            device: Device to use ("cuda:0", "cuda", or "cpu")
            batch_size_s: Batch size for processing (default: 0)
            streaming_config: Streaming configuration dict with chunk_size, encoder_chunk_look_back, etc.
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
        self.punc_model = punc_model
        self.punc_model_instance = None
        
        # Auto-detect device if not specified
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size_s = batch_size_s
        
        # Streaming configuration
        self.streaming_config = streaming_config or {}
        self.is_streaming = self.streaming_config.get("enabled", False) and "streaming" in model_name.lower()
        
        # Streaming parameters
        self.chunk_size = self.streaming_config.get("chunk_size", [0, 10, 5])
        self.encoder_chunk_look_back = self.streaming_config.get("encoder_chunk_look_back", 4)
        self.decoder_chunk_look_back = self.streaming_config.get("decoder_chunk_look_back", 1)
        self.vad_chunk_size_ms = self.streaming_config.get("vad_chunk_size_ms", 200)
        
        # Load the ASR model
        logger.info(f"Loading FunASR model '{self.model_name}' on {self.device}...")
        try:
            if self.is_streaming:
                # For streaming, load ASR model without VAD (VAD is separate)
                self.model = AutoModel(
                    model=self.model_name,
                    device=self.device,
                    disable_update=True,
                    disable_pbar=True
                )
                # Load separate VAD model for streaming
                logger.info(f"Loading streaming VAD model '{self.vad_model}'...")
                self.vad_model_instance = AutoModel(
                    model=self.vad_model,
                    device=self.device,
                    disable_update=True,
                    disable_pbar=True
                )
                # Load separate punctuation model if specified
                if self.punc_model:
                    logger.info(f"Loading punctuation model '{self.punc_model}'...")
                    self.punc_model_instance = AutoModel(
                        model=self.punc_model,
                        device=self.device,
                        disable_update=True,
                        disable_pbar=True
                    )
            else:
                # For batch mode, load with VAD integrated
                self.model = AutoModel(
                    model=self.model_name,
                    vad_model=self.vad_model,
                    vad_kwargs=self.vad_kwargs,
                    device=self.device,
                    disable_update=True,
                    disable_pbar=True
                )
                self.vad_model_instance = None
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
                batch_size_s=self.batch_size_s,
                disable_pbar=True
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
    
    def initialize_streaming(self) -> Dict[str, Any]:
        """
        Initialize streaming mode and return cache dictionaries.
        
        Returns:
            Dict with 'asr_cache' and 'vad_cache' keys
        """
        if not self.is_streaming:
            raise RuntimeError("Streaming mode not enabled or model is not streaming-capable")
        
        return {
            "asr_cache": {},
            "vad_cache": {}
        }
    
    def process_streaming_chunk(
        self,
        audio_chunk: np.ndarray,
        cache: Dict[str, Any],
        is_final: bool = False
    ) -> Optional[str]:
        """
        Process a single audio chunk with streaming ASR.
        
        Args:
            audio_chunk: Audio data as numpy array (float32, 16kHz mono)
            cache: ASR cache dict (maintains state between chunks)
            is_final: Whether this is the final chunk (triggers final output)
            
        Returns:
            Text string if available, None otherwise
        """
        if not self.is_streaming:
            raise RuntimeError("Streaming mode not enabled")
        
        try:
            # Ensure audio is float32 and 1D
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)
            if len(audio_chunk.shape) > 1:
                audio_chunk = audio_chunk.flatten()
            
            # Log chunk info for debugging
            logger.debug(f"Processing ASR chunk: {len(audio_chunk)} samples, dtype={audio_chunk.dtype}, shape={audio_chunk.shape}")
            
            # Process chunk with streaming ASR
            # FunASR expects input as numpy array or list
            res = self.model.generate(
                input=audio_chunk,
                cache=cache,
                is_final=is_final,
                chunk_size=self.chunk_size,
                encoder_chunk_look_back=self.encoder_chunk_look_back,
                decoder_chunk_look_back=self.decoder_chunk_look_back,
                disable_pbar=True
            )
            
            # Log result for debugging
            logger.debug(f"ASR result: {res}")
            
            # Extract text from result
            if res and len(res) > 0:
                text = res[0].get("text", "")
                if text:
                    logger.debug(f"ASR produced text: {text[:50]}...")
                return text if text else None
            return None
            
        except Exception as e:
            logger.error(f"Error processing streaming ASR chunk: {e}", exc_info=True)
            return None
    
    def apply_punctuation(self, text: str) -> str:
        """
        Apply punctuation to text using the punctuation model.
        
        Args:
            text: Text without punctuation
            
        Returns:
            Text with punctuation applied
        """
        if not self.punc_model_instance or not text:
            return text
        
        try:
            res = self.punc_model_instance.generate(input=text, disable_pbar=True)
            if res and len(res) > 0:
                punctuated_text = res[0].get("text", "")
                if punctuated_text:
                    logger.debug(f"Punctuation applied: {text[:50]}... -> {punctuated_text[:50]}...")
                    return punctuated_text
            return text
        except Exception as e:
            logger.error(f"Error applying punctuation: {e}", exc_info=True)
            return text
    
    def process_vad_chunk(
        self,
        audio_chunk: np.ndarray,
        vad_cache: Dict[str, Any],
        is_final: bool = False
    ) -> Optional[List[List[int]]]:
        """
        Process a single audio chunk with streaming VAD.
        
        Args:
            audio_chunk: Audio data as numpy array (float32, 16kHz mono)
            vad_cache: VAD cache dict (maintains state between chunks)
            is_final: Whether this is the final chunk
            
        Returns:
            VAD result: [[beg, -1]] for speech start, [[-1, end]] for speech end, or None if no event
        """
        if not self.is_streaming or self.vad_model_instance is None:
            raise RuntimeError("Streaming VAD not available")
        
        try:
            # Ensure audio is float32 and 1D
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)
            if len(audio_chunk.shape) > 1:
                audio_chunk = audio_chunk.flatten()
            
            # Log chunk info for debugging
            logger.debug(f"Processing VAD chunk: {len(audio_chunk)} samples, dtype={audio_chunk.dtype}")
            
            # Process chunk with streaming VAD
            res = self.vad_model_instance.generate(
                input=audio_chunk,
                cache=vad_cache,
                is_final=is_final,
                chunk_size=self.vad_chunk_size_ms,
                disable_pbar=True
            )
            
            # Log result for debugging
            logger.debug(f"VAD result: {res}")
            
            # Extract VAD result
            # Format: [[beg, -1]] for speech start, [[-1, end]] for speech end
            if res and len(res) > 0:
                vad_result = res[0].get("value", [])
                # Only return if there's a speech event (start or end)
                if vad_result and len(vad_result) > 0:
                    logger.debug(f"VAD detected event: {vad_result}")
                    return vad_result
            return None
            
        except Exception as e:
            logger.error(f"Error processing VAD chunk: {e}", exc_info=True)
            return None

