from pydantic import BaseModel, Field
from typing import Literal, Optional, Any


class FasterWhisperConfig(BaseModel):
    """Configuration for Faster-Whisper provider"""
    model_path: str = "faster-whisper-small"
    device: Optional[str] = None
    compute_type: Optional[str] = None


class FunASRStreamingConfig(BaseModel):
    """Streaming configuration for FunASR"""
    enabled: bool = True
    chunk_size: list[int] = Field(default_factory=lambda: [0, 8, 4])
    encoder_chunk_look_back: int = 4
    decoder_chunk_look_back: int = 1
    vad_chunk_size_ms: int = 100
    silence_threshold_ms: int = 300


class FunASRVADKwargs(BaseModel):
    """VAD kwargs for FunASR"""
    max_single_segment_time: int = 30000


class FunASRConfig(BaseModel):
    """Configuration for FunASR provider"""
    model_name: str = "paraformer-zh-streaming"
    vad_model: str = "fsmn-vad"
    vad_kwargs: FunASRVADKwargs = Field(default_factory=FunASRVADKwargs)
    punc_model: str = "ct-punc"
    device: str = "cpu"
    batch_size_s: int = 0
    streaming: FunASRStreamingConfig = Field(default_factory=FunASRStreamingConfig)


class STTSettings(BaseModel):
    """STT configuration settings"""
    host: str = "0.0.0.0"
    port: int = 8001
    provider: Literal["faster-whisper", "funasr"] = "faster-whisper"
    language_code: str = "zh"
    sample_rate: int = 16000
    interim_transcript_min_samples: int = 16000
    log_level: str = "INFO"
    log_file: Optional[str] = None
    providers: dict[str, Any] = Field(default_factory=dict)

    def get_provider_config(self) -> FasterWhisperConfig | FunASRConfig:
        """
        Get configuration for the active provider.

        Returns:
            Typed provider config
        """
        provider_data = self.providers.get(self.provider, {})

        # Handle None values from YAML
        if provider_data is None:
            provider_data = {}

        if self.provider == "faster-whisper":
            return FasterWhisperConfig(**provider_data)
        elif self.provider == "funasr":
            # Handle nested configs
            if "streaming" in provider_data and provider_data["streaming"]:
                provider_data = provider_data.copy()
                provider_data["streaming"] = FunASRStreamingConfig(**provider_data["streaming"])
            if "vad_kwargs" in provider_data and provider_data["vad_kwargs"]:
                provider_data = provider_data.copy()
                provider_data["vad_kwargs"] = FunASRVADKwargs(**provider_data["vad_kwargs"])
            return FunASRConfig(**provider_data)
        else:
            raise ValueError(f"Unknown STT provider: {self.provider}")

    @classmethod
    def from_dict(cls, data: dict) -> "STTSettings":
        """Create STTSettings from config dict"""
        return cls(**data)
