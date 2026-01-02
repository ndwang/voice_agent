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

from typing import Optional, List, Tuple, TYPE_CHECKING
from .base import AppSettings

if TYPE_CHECKING:
    from .reload_coordinator import ReloadCoordinator
    from .reload_result import ReloadResult
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


def update_settings(
    updates: dict,
    persist: bool = True,
    reload_coordinator: Optional["ReloadCoordinator"] = None
) -> Tuple[AppSettings, Optional[List["ReloadResult"]]]:
    """
    Update settings with new values and optionally coordinate reload.

    Args:
        updates: Dict with updates (e.g., {"llm": {"provider": "gemini"}})
        persist: If True, save changes to YAML file
        reload_coordinator: Optional ReloadCoordinator for hot-reload support

    Returns:
        Tuple of (updated_settings, reload_results or None)
    """
    global _settings
    current = get_settings()
    _settings = current.update_from_dict(updates)

    if persist:
        _settings.to_yaml()

    # Notify listeners of changes (legacy support)
    _settings.notify_listeners(updates)

    # Coordinate reload if provided
    reload_results = None
    if reload_coordinator:
        reload_results = reload_coordinator.reload_config(updates)

    return _settings, reload_results


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
