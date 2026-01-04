from pydantic import BaseModel


class BilibiliSettings(BaseModel):
    """Bilibili livestream integration configuration (DEPRECATED - moved to bilibili/config.yaml)"""
    enabled: bool = False
    room_id: int = 0  # Default to 0 (service uses its own config now)
    sessdata: str = ""
    danmaku_ttl_seconds: int = 60

    @classmethod
    def from_dict(cls, data: dict) -> "BilibiliSettings":
        """Create BilibiliSettings from config dict"""
        return cls(**data)
