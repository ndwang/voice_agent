from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional, Any, Callable, List
import yaml

from .llm import LLMSettings
from .orchestrator import OrchestratorSettings
from .tts import TTSSettings
from .stt import STTSettings
from .audio import AudioSettings
from .ocr import OCRSettings
from .obs import OBSSettings
from .bilibili import BilibiliSettings
from .services import ServicesSettings

# Module-level storage for change listeners
_change_listeners: List[Callable[[dict], None]] = []


class AppSettings(BaseModel):
    """
    Root settings for entire application.
    Loaded from config.yaml at startup.
    """
    # All settings are now fully typed
    orchestrator: OrchestratorSettings = Field(default_factory=OrchestratorSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    stt: STTSettings = Field(default_factory=STTSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    ocr: OCRSettings = Field(default_factory=OCRSettings)
    obs: OBSSettings = Field(default_factory=OBSSettings)
    bilibili: BilibiliSettings = Field(default_factory=BilibiliSettings)
    services: ServicesSettings = Field(default_factory=ServicesSettings)

    # Change notification
    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "AppSettings":
        """
        Load settings from YAML file.

        Args:
            path: Path to config.yaml

        Returns:
            AppSettings instance
        """
        config_path = Path(path)
        if not config_path.exists():
            # Fall back to project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yaml"

        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)

    def to_yaml(self, path: str = "config.yaml"):
        """
        Save settings to YAML file.

        Args:
            path: Path to config.yaml
        """
        config_path = Path(path)
        if not config_path.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / path

        # Convert to dict (preserves nested structure)
        data = self.model_dump()

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    def update_from_dict(self, updates: dict) -> "AppSettings":
        """
        Create new settings instance with updates applied.
        Immutable update pattern.

        Args:
            updates: Dict with partial updates (e.g., {"llm": {"provider": "gemini"}})

        Returns:
            New AppSettings instance with updates applied
        """
        current_dict = self.model_dump()
        merged = self._deep_merge(current_dict, updates)
        return AppSettings(**merged)

    @staticmethod
    def _deep_merge(base: dict, updates: dict) -> dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        for key, value in updates.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = AppSettings._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @classmethod
    def add_change_listener(cls, callback: Callable[[dict], None]):
        """Register callback for config changes"""
        global _change_listeners
        _change_listeners.append(callback)

    def notify_listeners(self, changes: dict):
        """Notify all listeners of config changes"""
        global _change_listeners
        for listener in _change_listeners:
            try:
                listener(changes)
            except Exception as e:
                import logging
                logging.error(f"Error in config change listener: {e}")
