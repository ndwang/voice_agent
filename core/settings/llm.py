from pydantic import BaseModel, Field
from typing import Literal, Optional, Any


class GeminiConfig(BaseModel):
    """Configuration for Gemini provider"""
    model: str = "gemini-2.5-flash"
    api_key: str = ""
    generation_config: Optional[dict] = Field(default_factory=dict)


class OllamaConfig(BaseModel):
    """Configuration for Ollama provider"""
    model: str = "qwen2.5:32b"
    base_url: str = "http://localhost:11434"
    timeout: float = 300.0
    disable_thinking: bool = False
    generation_config: Optional[dict] = Field(default_factory=dict)


class LLMSettings(BaseModel):
    """LLM configuration settings"""
    provider: Literal["gemini", "ollama"] = "ollama"
    providers: dict[str, Any] = Field(default_factory=dict)

    def get_provider_config(self) -> GeminiConfig | OllamaConfig:
        """
        Get configuration for the active provider.

        Returns:
            Typed provider config (GeminiConfig or OllamaConfig)
        """
        provider_data = self.providers.get(self.provider, {})

        # Handle None values from YAML (e.g., empty generation_config section)
        if provider_data.get("generation_config") is None:
            provider_data = provider_data.copy()
            provider_data["generation_config"] = {}

        if self.provider == "gemini":
            return GeminiConfig(**provider_data)
        elif self.provider == "ollama":
            return OllamaConfig(**provider_data)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    @classmethod
    def from_dict(cls, data: dict) -> "LLMSettings":
        """Create LLMSettings from config dict"""
        return cls(**data)
