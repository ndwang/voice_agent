"""
Pydantic models for Bilibili service API requests and responses.
"""

from typing import Optional
from pydantic import BaseModel


# =============================================================================
# Message models
# =============================================================================

class MedalInfo(BaseModel):
    """Fan medal info"""
    level: int = 0
    name: str = ""


class DanmakuMessage(BaseModel):
    """Danmaku (chat) message"""
    id: str
    user: str
    uid: int = 0
    face: str = ""
    content: str
    timestamp: float
    admin: bool = False
    guard_level: int = 0
    medal: Optional[MedalInfo] = None
    dm_type: int = 0  # 0=text, 1=emoticon, 2=voice
    emoticon_url: str = ""
    # For mixed text+emoticon messages (dm_type=0). Each item is:
    # - {"type": "text", "text": "..."}
    # - {"type": "emoticon", "text": "[...]", "url": "https://...", "width": 0, "height": 0}
    segments: list[dict] = []
    color: int = 0xFFFFFF
    wealth_level: int = 0
    privilege_type: int = 0


class SuperChatMessage(BaseModel):
    """SuperChat (paid message) model"""
    id: str
    bili_id: int = 0
    user: str
    uid: int = 0
    face: str = ""
    content: str
    timestamp: float
    amount: float
    guard_level: int = 0
    medal: Optional[MedalInfo] = None
    background_color: str = ""
    background_bottom_color: str = ""
    background_price_color: str = ""
    start_time: int = 0
    end_time: int = 0
    message_trans: str = ""


class GiftMessage(BaseModel):
    """Gift message"""
    id: str
    user: str
    uid: int = 0
    face: str = ""
    timestamp: float
    gift_name: str
    gift_id: int = 0
    num: int = 1
    coin_type: str = ""  # "silver" or "gold"
    total_coin: int = 0
    price: int = 0
    action: str = ""
    gift_icon: str = ""
    guard_level: int = 0
    medal: Optional[MedalInfo] = None
    combo_id: str = ""


class GuardMessage(BaseModel):
    """Guard/Fleet purchase message"""
    id: str
    user: str
    uid: int = 0
    timestamp: float
    guard_level: int = 0  # 1=总督, 2=提督, 3=舰长
    price: int = 0
    num: int = 1
    unit: str = ""
    gift_name: str = ""
    toast_msg: str = ""


# =============================================================================
# State models
# =============================================================================

class ServiceState(BaseModel):
    """Service state response"""
    connected: bool
    running: bool
    danmaku_enabled: bool
    paid_enabled: bool
    room_id: int


class ServiceStats(BaseModel):
    """Service statistics"""
    danmaku_buffer_size: int
    paid_buffer_size: int = 0
    uptime_seconds: float
    total_danmaku_received: int
    total_paid_received: int = 0
    total_gift_coins: int = 0


# =============================================================================
# Chat snapshot models
# =============================================================================

class ChatSnapshot(BaseModel):
    """Complete chat snapshot (all message types)"""
    connected: bool
    danmaku_enabled: bool
    paid_enabled: bool
    danmaku: list[DanmakuMessage]
    paid: list[dict] = []


class DanmakuSnapshot(BaseModel):
    """Danmaku-only snapshot"""
    danmaku: list[DanmakuMessage]


class PaidSnapshot(BaseModel):
    """Paid messages snapshot (superchat + gift + guard)"""
    paid: list[dict]


# =============================================================================
# API response models
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    connected: bool
    uptime: float
    buffer_health: dict[str, int]
    client_health: dict[str, int]


class EnableStateResponse(BaseModel):
    """Enable/disable state response"""
    success: bool
    danmaku_enabled: bool | None = None
    paid_enabled: bool | None = None
