from pydantic import BaseModel, Field
from typing import Literal, Optional, Any


class EdgeTTSConfig(BaseModel):
    """Configuration for Edge-TTS provider"""
    voice: str = "zh-CN-XiaoyiNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"


class ChatTTSConfig(BaseModel):
    """Configuration for ChatTTS provider"""
    model_source: str = "local"
    device: Optional[str] = None


class ElevenLabsConfig(BaseModel):
    """Configuration for ElevenLabs provider"""
    voice_id: str
    stability: float = 0.5
    similarity_boost: float = 0.8
    style: float = 0.0


class GenieTTSConfig(BaseModel):
    """Configuration for Genie-TTS provider"""
    character_name: str = "ema"
    onnx_model_dir: str
    language: str = "jp"
    reference_audio_path: str
    reference_audio_text: str
    source_sample_rate: int = 32000


class GPTSoVITSReference(BaseModel):
    """Reference configuration for GPT-SoVITS"""
    ref_audio_path: str
    prompt_text: str
    prompt_lang: str


class GPTSoVITSConfig(BaseModel):
    """Configuration for GPT-SoVITS provider"""
    server_url: str = "http://127.0.0.1:9880"
    default_reference: str = "normal"
    default_text_lang: str = "ja"
    gpt_weights_path: str
    sovits_weights_path: str
    streaming_mode: int = 2
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 15
    speed_factor: float = 1.0
    timeout: float = 30.0
    references: dict[str, GPTSoVITSReference] = Field(default_factory=dict)


class TTSSettings(BaseModel):
    """TTS configuration settings"""
    host: str = "0.0.0.0"
    port: int = 8003
    provider: Literal["edge-tts", "chattts", "elevenlabs", "genie-tts", "gpt-sovits"] = "edge-tts"
    providers: dict[str, Any] = Field(default_factory=dict)

    def get_provider_config(self) -> EdgeTTSConfig | ChatTTSConfig | ElevenLabsConfig | GenieTTSConfig | GPTSoVITSConfig:
        """
        Get configuration for the active provider.

        Returns:
            Typed provider config
        """
        provider_data = self.providers.get(self.provider, {})

        # Handle None values from YAML
        if provider_data is None:
            provider_data = {}

        if self.provider == "edge-tts":
            return EdgeTTSConfig(**provider_data)
        elif self.provider == "chattts":
            return ChatTTSConfig(**provider_data)
        elif self.provider == "elevenlabs":
            return ElevenLabsConfig(**provider_data)
        elif self.provider == "genie-tts":
            return GenieTTSConfig(**provider_data)
        elif self.provider == "gpt-sovits":
            # Handle references specially
            if "references" in provider_data:
                refs = provider_data["references"]
                if refs:
                    provider_data = provider_data.copy()
                    provider_data["references"] = {
                        name: GPTSoVITSReference(**ref_data)
                        for name, ref_data in refs.items()
                    }
            return GPTSoVITSConfig(**provider_data)
        else:
            raise ValueError(f"Unknown TTS provider: {self.provider}")

    @classmethod
    def from_dict(cls, data: dict) -> "TTSSettings":
        """Create TTSSettings from config dict"""
        return cls(**data)
