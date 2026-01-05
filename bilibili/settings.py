"""
Configuration loader for Bilibili service.
Loads service-specific configuration from bilibili/config.yaml.
"""

from pathlib import Path
from pydantic import BaseModel
import yaml


class ServiceSettings(BaseModel):
    """Service-level settings (host, port, logging)"""
    host: str = "0.0.0.0"
    port: int = 8002
    log_level: str = "INFO"


class BilibiliSettings(BaseModel):
    """Bilibili connection settings"""
    room_id: int
    sessdata: str = ""
    danmaku_ttl_seconds: int = 60
    enabled: bool = False
    danmaku_enabled_default: bool = True
    superchat_enabled_default: bool = True


class DashboardSettings(BaseModel):
    """Dashboard display settings"""
    default_theme: str = "dark"
    default_max_messages: int = 20
    default_font_size: int = 16


class BilibiliServiceConfig(BaseModel):
    """Complete Bilibili service configuration"""
    service: ServiceSettings
    bilibili: BilibiliSettings
    dashboard: DashboardSettings


def load_config() -> BilibiliServiceConfig:
    """Load configuration from bilibili/config.yaml"""
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return BilibiliServiceConfig(**data)


_config: BilibiliServiceConfig | None = None


def get_config() -> BilibiliServiceConfig:
    """Get the loaded configuration (singleton pattern)"""
    global _config
    if _config is None:
        _config = load_config()
    return _config
