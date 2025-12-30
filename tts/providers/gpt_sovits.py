"""
GPT-SoVITS Provider

GPT-SoVITS provider implementation with reference audio support.
"""
import asyncio
import io
import logging
import os
import numpy as np
from typing import AsyncIterator, Optional, Dict, Any
import aiohttp
from pydub import AudioSegment

from tts.base import TTSProvider

logger = logging.getLogger(__name__)


class GPTSoVITSProvider(TTSProvider):
    """GPT-SoVITS provider implementation."""
    
    def __init__(
        self,
        server_url: str = "http://127.0.0.1:9880",
        default_reference: str = "default",
        references: Optional[Dict[str, Dict[str, Any]]] = None,
        default_text_lang: str = "zh",
        gpt_weights_path: Optional[str] = None,
        sovits_weights_path: Optional[str] = None,
        streaming_mode: int = 2,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 15,
        speed_factor: float = 1.0,
        timeout: float = 30.0
    ):
        """
        Initialize GPT-SoVITS provider.
        """
        self.server_url = server_url.rstrip('/')
        self.default_reference = default_reference
        self.default_text_lang = default_text_lang
        self.gpt_weights_path = gpt_weights_path
        self.sovits_weights_path = sovits_weights_path
        self.streaming_mode = streaming_mode
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.speed_factor = speed_factor
        self.timeout = timeout
        
        # Reference configurations - always use registry pattern
        self.references = references or {}
        if not self.references:
            raise ValueError("At least one reference must be configured in 'references' dict")
        
        if self.default_reference not in self.references:
            raise ValueError(f"Default reference '{self.default_reference}' not found in references")
        
        self._current_reference = None
        self._session: Optional[aiohttp.ClientSession] = None
    
    @property
    def native_sample_rate(self) -> int:
        """GPT-SoVITS typically outputs at 32kHz."""
        return 32000
    
    def _ensure_absolute_path(self, path: str) -> str:
        """Convert relative path to absolute path with forward slashes for GPT-SoVITS server."""
        if not os.path.isabs(path):
            abs_path = os.path.abspath(path)
            # Convert Windows backslashes to forward slashes for GPT-SoVITS server
            abs_path = abs_path.replace('\\', '/')
            logger.debug(f"Converted relative path '{path}' to absolute path '{abs_path}'")
            return abs_path
        else:
            # Already absolute, just ensure forward slashes
            return path.replace('\\', '/')
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session
    
    
    def _get_reference_config(self, reference_name: Optional[str] = None) -> Dict[str, Any]:
        """Get reference configuration by name with absolute paths."""
        ref_name = reference_name or self.default_reference
        
        if ref_name in self.references:
            ref_config = self.references[ref_name].copy()
            # Convert relative path to absolute path
            ref_config["ref_audio_path"] = self._ensure_absolute_path(ref_config["ref_audio_path"])
            return ref_config
        else:
            raise ValueError(f"Reference '{ref_name}' not found in references")
    
    def load_reference(
        self,
        reference_name: str,
        ref_audio_path: str,
        prompt_text: str,
        prompt_lang: str = "zh"
    ):
        """
        Load a reference configuration.
        
        Args:
            reference_name: Name for this reference
            ref_audio_path: Path to reference audio file
            prompt_text: Prompt text describing the reference audio
            prompt_lang: Language of prompt text
        """
        self.references[reference_name] = {
            "ref_audio_path": ref_audio_path,
            "prompt_text": prompt_text,
            "prompt_lang": prompt_lang
        }
        logger.info(f"Loaded GPT-SoVITS reference: {reference_name}")
    
    async def synthesize_stream(self, text: str, **kwargs) -> AsyncIterator[bytes]:
        """
        Synthesize text using GPT-SoVITS and stream audio chunks.
        
        Args:
            text: Text to synthesize
            **kwargs: Optional parameters:
                - reference_name: Reference to use
                - ref_audio_path: Override reference audio path
                - prompt_text: Override prompt text
                - prompt_lang: Override prompt language
                - text_lang: Override text language
                - temperature: Override temperature
                - top_p: Override top_p
                - top_k: Override top_k
                - speed_factor: Override speed factor
                - streaming_mode: Override streaming mode
        
        Yields:
            Audio chunks as bytes (float32 PCM format, normalized to [-1, 1])
        """
        if not text or not text.strip():
            return
        
        # Get reference configuration
        ref_name = kwargs.get("reference_name", self.default_reference)
        
        # Allow direct parameter override or use reference config
        if "ref_audio_path" in kwargs and "prompt_text" in kwargs:
            ref_config = {
                "ref_audio_path": kwargs["ref_audio_path"],
                "prompt_text": kwargs["prompt_text"],
                "prompt_lang": kwargs.get("prompt_lang", "zh")
            }
        else:
            ref_config = self._get_reference_config(ref_name)
        
        # Prepare request parameters
        request_data = {
            "text": text,
            "text_lang": kwargs.get("text_lang", self.default_text_lang),  # Language of TTS input text
            "ref_audio_path": ref_config["ref_audio_path"],
            "prompt_text": ref_config["prompt_text"],
            "prompt_lang": ref_config["prompt_lang"],  # Language of reference prompt
            "streaming_mode": kwargs.get("streaming_mode", self.streaming_mode),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "top_k": kwargs.get("top_k", self.top_k),
            "speed_factor": kwargs.get("speed_factor", self.speed_factor),
            "media_type": "wav",  # Use WAV for compatibility
            "text_split_method": "cut5",
            "batch_size": 1,
        }
        
        logger.info(f"GPT-SoVITS request:, text_lang={request_data['text_lang']}, ref={ref_name}, streaming_mode={request_data['streaming_mode']}")
        
        session = await self._get_session()
        
        try:
            async with session.post(f"{self.server_url}/tts", json=request_data) as response:
                if response.status != 200:
                    try:
                        error_text = await response.text()
                    except:
                        error_text = f"HTTP {response.status}"
                    raise RuntimeError(f"GPT-SoVITS API error {response.status}: {error_text}")
                
                if request_data["streaming_mode"] in [2, 3, True]:
                    # Streaming mode - process chunks as they arrive
                    try:
                        audio_buffer = b""
                        first_chunk = True
                        has_audio_data = False
                        
                        async for chunk in response.content.iter_chunked(8192):
                            if not chunk:
                                continue
                            
                            audio_buffer += chunk
                            has_audio_data = True
                            
                            # For first chunk, skip WAV header if present
                            if first_chunk and audio_buffer.startswith(b"RIFF"):
                                first_chunk = False
                                # Find data chunk
                                data_start = audio_buffer.find(b"data")
                                if data_start != -1:
                                    data_start += 8  # Skip "data" + size
                                    audio_buffer = audio_buffer[data_start:]
                            
                            # Process available audio data in smaller chunks to avoid blocking
                            while len(audio_buffer) >= 512:  # Smaller chunks for better streaming
                                chunk_to_process = audio_buffer[:512]
                                audio_buffer = audio_buffer[512:]
                                
                                processed_chunk = await self._process_audio_chunk(chunk_to_process)
                                if processed_chunk:
                                    yield processed_chunk
                        
                        # Process remaining buffer
                        if audio_buffer:
                            processed_chunk = await self._process_audio_chunk(audio_buffer)
                            if processed_chunk:
                                yield processed_chunk
                                
                        # If no audio data received, it might be an error response
                        if not has_audio_data:
                            raise RuntimeError("No audio data received from GPT-SoVITS API")
                            
                    except Exception as stream_error:
                        # If streaming fails, try to get error message from response
                        if not has_audio_data:
                            try:
                                # Try to read any error message
                                remaining_content = await response.read()
                                if remaining_content:
                                    error_msg = remaining_content.decode('utf-8', errors='ignore')
                                    raise RuntimeError(f"GPT-SoVITS streaming error: {error_msg}")
                            except:
                                pass
                        raise stream_error
                
                else:
                    # Non-streaming mode - get complete audio
                    audio_data = await response.read()
                    if not audio_data:
                        raise RuntimeError("No audio data received from GPT-SoVITS API")
                    processed_audio = await self._process_complete_audio(audio_data)
                    if processed_audio:
                        yield processed_audio
                        
        except Exception as e:
            logger.error(f"GPT-SoVITS synthesis error: {e}")
            raise
    
    async def _process_audio_chunk(self, audio_data: bytes) -> Optional[bytes]:
        """Process raw audio chunk to normalized float32."""
        try:
            if len(audio_data) < 2:  # Need at least one sample
                return None
            
            # Assume 16-bit PCM from GPT-SoVITS
            # Convert to float32 normalized to [-1, 1]
            audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
            if len(audio_int16) == 0:
                return None
            
            # Normalize to [-1, 1] range
            audio_float = audio_int16.astype(np.float32) / 32768.0
            audio_float = np.clip(audio_float, -1.0, 1.0)
            
            return audio_float.tobytes()
            
        except Exception as e:
            logger.warning(f"Error processing audio chunk: {e}")
            return None
    
    async def _process_complete_audio(self, audio_data: bytes) -> Optional[bytes]:
        """Process complete audio file to normalized float32."""
        try:
            if not audio_data:
                return None
            
            # Use pydub to handle WAV format properly
            audio_segment = AudioSegment.from_wav(io.BytesIO(audio_data))
            
            # Convert to mono if needed
            if audio_segment.channels > 1:
                audio_segment = audio_segment.set_channels(1)
            
            # Get raw audio samples and convert to float32
            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            
            # Normalize to [-1, 1] range
            audio_float = samples / 32768.0
            audio_float = np.clip(audio_float, -1.0, 1.0)
            
            return audio_float.tobytes()
            
        except Exception as e:
            logger.error(f"Error processing complete audio: {e}")
            return None
    
    async def list_voices(self) -> list:
        """List available references."""
        voices = []
        for ref_name, ref_config in self.references.items():
            voices.append({
                "name": ref_name,
                "ref_audio_path": ref_config["ref_audio_path"],
                "prompt_text": ref_config["prompt_text"],
                "prompt_lang": ref_config["prompt_lang"]
            })
        return voices
    
    def parse_error(self, exception: Exception) -> tuple[int, str]:
        """Parse GPT-SoVITS specific errors."""
        if isinstance(exception, aiohttp.ClientError):
            return 503, f"GPT-SoVITS connection error: {str(exception)}"
        elif isinstance(exception, asyncio.TimeoutError):
            return 504, "GPT-SoVITS request timeout"
        elif isinstance(exception, ValueError) and "Reference" in str(exception):
            return 400, str(exception)
        else:
            return 500, f"GPT-SoVITS synthesis error: {str(exception)}"
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup session."""
        if self._session and not self._session.closed:
            await self._session.close()