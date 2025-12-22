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

