"""Pydantic models for API requests/responses."""
from pydantic import BaseModel


class SystemPromptUpdate(BaseModel):
    """Request model for updating system prompt."""
    prompt: str


class ListeningSetRequest(BaseModel):
    """Request model for setting listening state."""
    enabled: bool


class HotkeyUpdateRequest(BaseModel):
    """Request model for updating hotkey."""
    hotkey: str


class ConfigUpdate(BaseModel):
    """Request model for updating full config."""
    config: dict


class BilibiliDanmakuSetRequest(BaseModel):
    """Request model for setting bilibili danmaku state."""
    enabled: bool


class BilibiliSuperChatSetRequest(BaseModel):
    """Request model for setting bilibili superchat state."""
    enabled: bool

