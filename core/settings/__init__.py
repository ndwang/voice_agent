"""
Typed configuration system using Pydantic.

Usage:
    # Production - use global singleton
    from core.settings import get_settings
    settings = get_settings()
    print(settings.orchestrator["port"])  # Type-safe access (will be typed in later steps)

    # Testing - inject mock settings
    from core.settings import AppSettings
    test_settings = AppSettings(orchestrator={"port": 9999})
    manager = InteractionManager(event_bus, settings=test_settings)
"""

from typing import Optional
from .base import AppSettings
from .llm import LLMSettings, GeminiConfig, OllamaConfig
from .orchestrator import OrchestratorSettings
from .tts import (
    TTSSettings, EdgeTTSConfig, ChatTTSConfig,
    ElevenLabsConfig, GenieTTSConfig, GPTSoVITSConfig, GPTSoVITSReference
)
from .stt import (
    STTSettings, FasterWhisperConfig, FunASRConfig,
    FunASRStreamingConfig, FunASRVADKwargs
)
from .audio import AudioSettings, AudioInputSettings, AudioOutputSettings
from .ocr import OCRSettings
from .obs import OBSSettings, OBSWebsocketSettings
from .bilibili import BilibiliSettings
from .services import ServicesSettings

# Global singleton
_settings: Optional[AppSettings] = None


def get_settings() -> AppSettings:
    """
    Get global settings instance (lazy-loaded singleton).

    Returns:
        Global AppSettings instance
    """
    global _settings
    if _settings is None:
        _settings = AppSettings.from_yaml()
    return _settings


def set_settings(settings: AppSettings):
    """
    Set global settings instance.
    Used for testing and initialization.

    Args:
        settings: AppSettings instance to set as global
    """
    global _settings
    _settings = settings


def reload_settings(path: str = "config.yaml") -> AppSettings:
    """
    Reload settings from YAML file.

    Args:
        path: Path to config.yaml

    Returns:
        Reloaded AppSettings instance
    """
    global _settings
    _settings = AppSettings.from_yaml(path)
    return _settings


def update_settings(updates: dict, persist: bool = True) -> AppSettings:
    """
    Update settings with new values.

    Args:
        updates: Dict with updates (e.g., {"llm": {"provider": "gemini"}})
        persist: If True, save changes to YAML file

    Returns:
        Updated AppSettings instance
    """
    global _settings
    current = get_settings()
    _settings = current.update_from_dict(updates)

    if persist:
        _settings.to_yaml()

    # Notify listeners of changes
    _settings.notify_listeners(updates)

    return _settings


__all__ = [
    'AppSettings',
    'LLMSettings',
    'GeminiConfig',
    'OllamaConfig',
    'OrchestratorSettings',
    'TTSSettings',
    'EdgeTTSConfig',
    'ChatTTSConfig',
    'ElevenLabsConfig',
    'GenieTTSConfig',
    'GPTSoVITSConfig',
    'GPTSoVITSReference',
    'STTSettings',
    'FasterWhisperConfig',
    'FunASRConfig',
    'FunASRStreamingConfig',
    'FunASRVADKwargs',
    'AudioSettings',
    'AudioInputSettings',
    'AudioOutputSettings',
    'OCRSettings',
    'OBSSettings',
    'OBSWebsocketSettings',
    'BilibiliSettings',
    'ServicesSettings',
    'get_settings',
    'set_settings',
    'reload_settings',
    'update_settings'
]
