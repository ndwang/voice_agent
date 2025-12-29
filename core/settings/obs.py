from pydantic import BaseModel
from typing import Optional


class OBSWebsocketSettings(BaseModel):
    """OBS WebSocket connection settings"""
    host: str = "localhost"
    port: int = 4455
    password: str


class OBSSettings(BaseModel):
    """OBS integration configuration"""
    websocket: OBSWebsocketSettings
    subtitle_source: str = "subtitle"
    subtitle_ttl_seconds: int = 10
    visibility_source: Optional[str] = None
    appear_filter_name: Optional[str] = None
    clear_filter_name: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "OBSSettings":
        """Create OBSSettings from config dict"""
        data = data.copy()
        if "websocket" in data:
            data["websocket"] = OBSWebsocketSettings(**data["websocket"])
        return cls(**data)
