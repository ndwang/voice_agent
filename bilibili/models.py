"""
Pydantic models for Bilibili service API requests and responses.
"""

from typing import List, Dict, Any
from pydantic import BaseModel


# Message models
class DanmakuMessage(BaseModel):
    """Danmaku (chat) message"""
    id: str
    user: str
    content: str
    timestamp: float


class SuperChatMessage(BaseModel):
    """SuperChat (paid message) model"""
    id: str
    user: str
    content: str
    timestamp: float
    amount: float


# State models
class ServiceState(BaseModel):
    """Service state response"""
    connected: bool
    running: bool
    danmaku_enabled: bool
    superchat_enabled: bool
    room_id: int


class ServiceStats(BaseModel):
    """Service statistics"""
    danmaku_count: int
    superchat_count: int
    danmaku_buffer_size: int
    superchat_buffer_size: int
    uptime_seconds: float
    total_danmaku_received: int
    total_superchat_received: int


# Chat snapshot models
class ChatSnapshot(BaseModel):
    """Complete chat snapshot (danmaku + superchat)"""
    connected: bool
    danmaku_enabled: bool
    superchat_enabled: bool
    danmaku: List[DanmakuMessage]
    superchat: List[SuperChatMessage]


class DanmakuSnapshot(BaseModel):
    """Danmaku-only snapshot"""
    danmaku: List[DanmakuMessage]


class SuperChatSnapshot(BaseModel):
    """SuperChat-only snapshot"""
    superchat: List[SuperChatMessage]


# WebSocket message models
class WSSubscribeMessage(BaseModel):
    """WebSocket subscribe message"""
    type: str = "subscribe"
    channels: List[str]


class WSUnsubscribeMessage(BaseModel):
    """WebSocket unsubscribe message"""
    type: str = "unsubscribe"
    channels: List[str]


class WSPingMessage(BaseModel):
    """WebSocket ping message"""
    type: str = "ping"


class WSDanmakuMessage(BaseModel):
    """WebSocket danmaku broadcast"""
    type: str = "danmaku"
    data: DanmakuMessage


class WSSuperChatMessage(BaseModel):
    """WebSocket superchat broadcast"""
    type: str = "superchat"
    data: SuperChatMessage


class WSStateChangedMessage(BaseModel):
    """WebSocket state change notification"""
    type: str = "state_changed"
    data: Dict[str, Any]


class WSErrorMessage(BaseModel):
    """WebSocket error message"""
    type: str = "error"
    message: str


class WSPongMessage(BaseModel):
    """WebSocket pong response"""
    type: str = "pong"


# API response models
class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    connected: bool
    uptime: float
    buffer_health: Dict[str, int]
    client_health: Dict[str, int]


class EnableStateResponse(BaseModel):
    """Enable/disable state response"""
    success: bool
    danmaku_enabled: bool | None = None
    superchat_enabled: bool | None = None
