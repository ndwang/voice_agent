"""
Shared Configuration Loader

Loads configuration from a YAML file for all submodules.
"""
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads and manages configuration from YAML file."""
    
    _instance: Optional['ConfigLoader'] = None
    _config: Optional[Dict[str, Any]] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._config = {}
    
    def load(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file. If None, looks for config.yaml in project root.
            
        Returns:
            Configuration dictionary
        """
        if self._config:
            return self._config
        
        if config_path is None:
            # Look for config.yaml in project root
            # Assume core/config.py -> core/ -> project_root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config.yaml"
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file not found at {config_path}. Using defaults.")
            self._config = self._get_default_config()
            return self._config
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
            
            logger.info(f"Configuration loaded from {config_path}")
            return self._config
        except Exception as e:
            logger.error(f"Error loading config file: {e}. Using defaults.")
            self._config = self._get_default_config()
            return self._config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "orchestrator": {
                "host": "0.0.0.0",
                "port": 8000,
                "stt_websocket_path": "/ws/stt",
                "log_level": "INFO"
            },
            "stt": {
                "host": "0.0.0.0",
                "port": 8001,
                "provider": "faster-whisper",
                "language_code": "zh",
                "sample_rate": 16000,
                "interim_transcript_min_samples": 4800,
                "providers": {
                    "faster-whisper": {
                        "model_path": "faster-whisper-small",
                        "device": None,
                        "compute_type": None
                    },
                    "funasr": {
                        "model_name": "FunAudioLLM/Fun-ASR-Nano-2512",
                        "vad_model": "fsmn-vad",
                        "vad_kwargs": {
                            "max_single_segment_time": 30000
                        },
                        "device": None,
                        "batch_size_s": 0
                    }
                }
            },
            "llm": {
                "host": "0.0.0.0",
                "port": 8002,
                "provider": "gemini",
                "providers": {
                    "gemini": {
                        "model": "gemini-2.5-flash",
                        "api_key": ""
                    },
                    "llamacpp": {
                        "model_path": "",
                        "n_ctx": 4096,
                        "n_threads": 0,
                        "n_gpu_layers": -1
                    }
                }
            },
            "tts": {
                "host": "0.0.0.0",
                "port": 8003,
                "provider": "edge-tts",
                "providers": {
                    "edge-tts": {
                        "voice": "zh-CN-XiaoxiaoNeural",
                        "rate": "+0%",
                        "pitch": "+0Hz"
                    },
                    "chattts": {
                        "model_source": "local",
                        "device": None
                    },
                    "genie-tts": {
                        "character_name": "default",
                        "onnx_model_dir": "",
                        "language": "zh",
                        "reference_audio_path": "",
                        "reference_audio_text": "",
                        "source_sample_rate": 32000
                    }
                }
            },
            "ocr": {
                "host": "0.0.0.0",
                "port": 8004,
                "language": "ch",
                "interval_ms": 1000,
                "texts_storage_file_prefix": "ocr_detected_texts"
            },
            "audio": {
                "input": {
                    "sample_rate": 16000,
                    "channels": 1,
                    "device": None
                },
                "output": {
                    "sample_rate": 16000,
                    "channels": 1,
                    "device": None
                },
                "dtype": "float32",
                "block_size_ms": 100,
                "silence_threshold_ms": 500,
                "vad_min_speech_prob": 0.5,
                "plot_window_seconds": 2,
                "plot_update_interval_ms": 50
            },
            "obs": {
                "websocket": {
                    "host": "localhost",
                    "port": 4455,
                    "password": ""
                }
            },
            "services": {
                "stt_websocket_url": "ws://localhost:8001/ws/transcribe",
                "llm_base_url": "http://localhost:8002",
                "tts_websocket_url": "ws://localhost:8003/synthesize/stream",
                "ocr_websocket_url": "ws://localhost:8004/monitor/stream",
                "ocr_base_url": "http://localhost:8004"
            }
        }
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            *keys: Keys to traverse (e.g., 'llm', 'provider')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        if not self._config:
            self.load()
        
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def reload(self, config_path: Optional[str] = None):
        """Reload configuration from file."""
        self._config = None
        return self.load(config_path)

    def save(self, config_path: Optional[str] = None) -> bool:
        """
        Save current configuration to YAML file.
        
        Args:
            config_path: Path to config file. If None, uses the path from which config was loaded.
            
        Returns:
            True if successful, False otherwise.
        """
        if config_path is None:
            # Look for config.yaml in project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config.yaml"
        else:
            config_path = Path(config_path)
            
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
            logger.info(f"Configuration saved to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving config file: {e}")
            return False


# Global instance
_loader = ConfigLoader()


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file."""
    return _loader.load(config_path)


def save_config(config_path: Optional[str] = None) -> bool:
    """Save current configuration to file."""
    return _loader.save(config_path)


def get_full_config() -> Dict[str, Any]:
    """Get the entire configuration dictionary."""
    if not _loader._config:
        _loader.load()
    return _loader._config


def get_config(*keys: str, default: Any = None) -> Any:
    """Get configuration value using dot notation."""
    return _loader.get(*keys, default=default)


