from pydantic import BaseModel
from typing import Optional


class AudioInputSettings(BaseModel):
    """Audio input device configuration"""
    sample_rate: int = 16000
    channels: int = 1
    device: Optional[int] = None  # Device index or None for default


class AudioOutputSettings(BaseModel):
    """Audio output device configuration"""
    sample_rate: int = 32000
    channels: int = 1
    device: Optional[int] = None  # Device index or None for default


class AudioSettings(BaseModel):
    """Audio driver configuration"""
    input: AudioInputSettings
    output: AudioOutputSettings
    dtype: str = "float32"
    block_size_ms: int = 100
    silence_threshold_ms: int = 500
    listening_status_poll_interval: float = 1.0

    @classmethod
    def from_dict(cls, data: dict) -> "AudioSettings":
        """Create AudioSettings from config dict"""
        # Handle nested configs
        data = data.copy()
        if "input" in data:
            data["input"] = AudioInputSettings(**data["input"])
        if "output" in data:
            data["output"] = AudioOutputSettings(**data["output"])
        return cls(**data)
