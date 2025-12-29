from pydantic import BaseModel


class BilibiliSettings(BaseModel):
    """Bilibili livestream integration configuration"""
    enabled: bool = False
    room_id: int
    sessdata: str = ""
    danmaku_ttl_seconds: int = 60

    @classmethod
    def from_dict(cls, data: dict) -> "BilibiliSettings":
        """Create BilibiliSettings from config dict"""
        return cls(**data)
