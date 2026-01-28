"""
BilibiliManager - Core business logic for Bilibili live chat service.
Manages client lifecycle, message buffering, and WebSocket broadcasting.
"""

import asyncio
import re
import time
from collections import deque
from itertools import count
from typing import Optional, Set, Dict, List, Any, Deque
from fastapi import WebSocket
from core.logging import get_logger
from bilibili.bilibili_client import BilibiliClient
from bilibili.settings import BilibiliServiceConfig

logger = get_logger(__name__)

GUARD_LEVEL_NAMES = {1: "总督", 2: "提督", 3: "舰长"}

VALID_CHANNELS = {"danmaku", "paid"}


def _normalize_external_url(url: Any) -> str:
    """
    Normalize Bilibili-provided URLs for browser rendering.

    - Converts protocol-relative URLs ("//...") to https.
    - Upgrades http to https to avoid mixed-content blocks when the dashboard is served via https.
    """
    if not url:
        return ""
    s = str(url).strip()
    if not s:
        return ""
    if s.startswith("//"):
        return f"https:{s}"
    if s.startswith("http://"):
        return f"https://{s[len('http://'):]}"
    return s


def _build_danmaku_segments(message: Any) -> list[dict]:
    """
    Build rich segments for mixed text+emoticon danmaku.

    Bilibili (via blivedm) exposes inline emoticons in `message.extra_dict["emots"]`
    for dm_type==0 messages that contain placeholders like "[doge]".
    """
    try:
        msg = getattr(message, "msg", "") or ""
        if not isinstance(msg, str) or not msg:
            return []

        extra = getattr(message, "extra_dict", None)
        if callable(extra):
            extra = extra()
        if not isinstance(extra, dict):
            return []

        emots = extra.get("emots")
        if not isinstance(emots, dict) or not emots:
            return []

        token_map: dict[str, dict] = {}
        for token, info in emots.items():
            if not isinstance(token, str) or not token:
                continue
            if not isinstance(info, dict):
                continue
            url = (
                info.get("url")
                or info.get("url_v2")
                or info.get("url_v3")
                or info.get("gif_url")
            )
            url = _normalize_external_url(url)
            if not url:
                continue
            token_map[token] = {
                "type": "emoticon",
                "text": token,
                "url": url,
                "width": int(info.get("width") or 0),
                "height": int(info.get("height") or 0),
            }

        if not token_map:
            return []

        # Longest tokens first to avoid partial matches
        tokens_sorted = sorted(token_map.keys(), key=len, reverse=True)
        pattern = re.compile("|".join(re.escape(t) for t in tokens_sorted))

        segments: list[dict] = []
        last = 0
        for m in pattern.finditer(msg):
            if m.start() > last:
                segments.append({"type": "text", "text": msg[last:m.start()]})
            segments.append(token_map[m.group(0)])
            last = m.end()
        if last < len(msg):
            segments.append({"type": "text", "text": msg[last:]})

        # If nothing changed, avoid sending redundant segments
        if len(segments) == 1 and segments[0].get("type") == "text":
            return []
        return segments
    except Exception:
        return []


class BilibiliManager:
    """Manages Bilibili client, message buffers, and WebSocket broadcasting"""

    def __init__(self, config: BilibiliServiceConfig):
        self.config = config

        # Core components
        self.client: Optional[BilibiliClient] = None

        # Monotonic ID counter to avoid collisions
        self._id_counter = count()

        # State flags
        self.connected: bool = False
        self.running: bool = False
        self.danmaku_enabled: bool = config.bilibili.danmaku_enabled_default
        self.paid_enabled: bool = config.bilibili.paid_enabled_default

        # Message buffers
        self.danmaku_buffer: Deque[Dict[str, Any]] = deque(
            maxlen=config.bilibili.danmaku_max_buffer
        )
        self.paid_buffer: Deque[Dict[str, Any]] = deque(
            maxlen=config.bilibili.paid_max_buffer
        )

        # Statistics
        self.total_danmaku_received: int = 0
        self.total_paid_received: int = 0
        self.total_gift_coins: int = 0
        self.start_time: float = time.time()

        # WebSocket clients for real-time streaming
        # Maps WebSocket -> set of subscribed channel names
        self.ws_clients: Dict[WebSocket, Set[str]] = {}

        # Reconnection state
        self._reconnect_task: Optional[asyncio.Task] = None
        self._intentional_disconnect: bool = False

    async def start(self):
        """Initialize manager and start cleanup task"""
        if self.running:
            return

        self.running = True
        logger.info("BilibiliManager started")

        # Auto-connect if enabled
        if self.config.bilibili.enabled:
            await self.connect()

    async def stop(self):
        """Shutdown manager and cleanup resources"""
        if not self.running:
            return

        self.running = False

        # Cancel reconnect task
        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
            self._reconnect_task = None

        # Disconnect client
        self._intentional_disconnect = True
        await self.disconnect()

        logger.info("BilibiliManager stopped")

    async def connect(self):
        """Connect to Bilibili room"""
        if self.connected:
            logger.warning("Already connected to Bilibili")
            return

        self._intentional_disconnect = False

        try:
            self.client = BilibiliClient(
                room_id=self.config.bilibili.room_id,
                sessdata=self.config.bilibili.sessdata
            )
            self.client.set_handlers(
                on_danmaku=self._on_danmaku,
                on_super_chat=self._on_super_chat,
                on_gift=self._on_gift,
                on_guard=self._on_guard,
                on_super_chat_delete=self._on_super_chat_delete,
                on_client_stopped=self._on_client_stopped,
            )

            await self.client.start()
            self.connected = True
            logger.info(f"Connected to Bilibili room {self.config.bilibili.room_id}")

            # Broadcast state change
            await self._broadcast_state_change()

        except Exception as e:
            logger.error(f"Failed to connect to Bilibili: {e}")
            self.connected = False
            raise

    async def disconnect(self):
        """Disconnect from Bilibili room"""
        if not self.connected:
            return

        self._intentional_disconnect = True

        try:
            if self.client:
                await self.client.stop()
                self.client = None

            self.connected = False
            logger.info("Disconnected from Bilibili")

            # Broadcast state change
            await self._broadcast_state_change()

        except Exception as e:
            logger.error(f"Error disconnecting from Bilibili: {e}")

    async def set_danmaku_enabled(self, enabled: bool):
        """Toggle danmaku processing and broadcast state change"""
        self.danmaku_enabled = enabled
        logger.info(f"Danmaku processing {'enabled' if enabled else 'disabled'}")
        await self._broadcast_state_change()

    async def set_paid_enabled(self, enabled: bool):
        """Toggle paid message processing (superchat/gift/guard) and broadcast state change"""
        self.paid_enabled = enabled
        logger.info(f"Paid message processing {'enabled' if enabled else 'disabled'}")
        await self._broadcast_state_change()

    # =========================================================================
    # Message handlers
    # =========================================================================

    async def _on_danmaku(self, message):
        """Handle incoming danmaku from Bilibili client"""
        if not self.danmaku_enabled:
            return

        medal = None
        if message.medal_name:
            medal = {"level": message.medal_level, "name": message.medal_name}

        emoticon_url = ""
        if message.dm_type == 1 and isinstance(message.emoticon_options, dict):
            emoticon_url = _normalize_external_url(message.emoticon_options.get("url", ""))

        segments = []
        if message.dm_type == 0:
            segments = _build_danmaku_segments(message)

        normalized = {
            "id": f"dm_{next(self._id_counter)}_{message.uid}",
            "user": message.uname,
            "uid": message.uid,
            "face": _normalize_external_url(getattr(message, "face", "")),
            "content": message.msg,
            "timestamp": time.time(),
            "admin": bool(message.admin),
            "guard_level": message.privilege_type,
            "medal": medal,
            "dm_type": message.dm_type,
            "emoticon_url": emoticon_url,
            "segments": segments,
            "color": message.color,
            "wealth_level": getattr(message, "wealth_level", 0),
            "privilege_type": message.privilege_type,
        }

        self.danmaku_buffer.append(normalized)
        self.total_danmaku_received += 1

        await self._broadcast_message({
            "type": "danmaku",
            "data": normalized
        }, channel="danmaku")

        logger.debug(f"Danmaku: {normalized['user']}: {normalized['content']}")

    async def _on_super_chat(self, message):
        """Handle incoming superchat from Bilibili client"""
        if not self.paid_enabled:
            return

        medal = None
        if getattr(message, "medal_name", ""):
            medal = {
                "level": getattr(message, "medal_level", 0),
                "name": getattr(message, "medal_name", ""),
            }

        normalized = {
            "id": f"sc_{next(self._id_counter)}_{message.uid}",
            "paid_type": "superchat",
            "bili_id": message.id,
            "user": message.uname,
            "uid": message.uid,
            "face": _normalize_external_url(getattr(message, "face", "")),
            "content": message.message,
            "timestamp": time.time(),
            "amount": message.price,
            "guard_level": getattr(message, "guard_level", 0),
            "medal": medal,
            "background_color": getattr(message, "background_color", ""),
            "background_bottom_color": getattr(message, "background_bottom_color", ""),
            "background_price_color": getattr(message, "background_price_color", ""),
            "start_time": getattr(message, "start_time", 0),
            "end_time": getattr(message, "end_time", 0),
            "message_trans": getattr(message, "message_trans", ""),
        }

        self.paid_buffer.append(normalized)
        self.total_paid_received += 1

        await self._broadcast_message({
            "type": "paid",
            "data": normalized
        }, channel="paid")

        logger.info(f"SuperChat: {normalized['user']} (¥{normalized['amount']}): {normalized['content']}")

    async def _on_gift(self, message):
        """Handle incoming gift from Bilibili client"""
        if not self.paid_enabled:
            return

        medal = None
        if getattr(message, "medal_name", ""):
            medal = {
                "level": getattr(message, "medal_level", 0),
                "name": getattr(message, "medal_name", ""),
            }

        normalized = {
            "id": f"gift_{next(self._id_counter)}_{message.uid}",
            "paid_type": "gift",
            "user": message.uname,
            "uid": message.uid,
            "face": _normalize_external_url(getattr(message, "face", "")),
            "timestamp": time.time(),
            "gift_name": message.gift_name,
            "gift_id": message.gift_id,
            "num": message.num,
            "coin_type": message.coin_type,
            "total_coin": message.total_coin,
            "price": message.price,
            "action": message.action,
            "gift_icon": _normalize_external_url(getattr(message, "gift_img_basic", "")),
            "guard_level": getattr(message, "guard_level", 0),
            "medal": medal,
            "combo_id": getattr(message, "tid", ""),
        }

        self.paid_buffer.append(normalized)
        self.total_paid_received += 1
        self.total_gift_coins += message.total_coin

        await self._broadcast_message({
            "type": "paid",
            "data": normalized
        }, channel="paid")

        logger.debug(
            f"Gift: {normalized['user']} {normalized['action']} "
            f"{normalized['gift_name']}x{normalized['num']} "
            f"({normalized['coin_type']}: {normalized['total_coin']})"
        )

    async def _on_guard(self, message):
        """Handle guard/fleet purchase (UserToastV2)"""
        if not self.paid_enabled:
            return

        guard_name = GUARD_LEVEL_NAMES.get(message.guard_level, f"guard_{message.guard_level}")

        normalized = {
            "id": f"guard_{next(self._id_counter)}_{message.uid}",
            "paid_type": "guard",
            "user": message.username,
            "uid": message.uid,
            "timestamp": time.time(),
            "guard_level": message.guard_level,
            "price": message.price,
            "num": message.num,
            "unit": getattr(message, "unit", ""),
            "gift_name": getattr(message, "gift_name", guard_name),
            "toast_msg": getattr(message, "toast_msg", ""),
        }

        self.paid_buffer.append(normalized)
        self.total_paid_received += 1

        await self._broadcast_message({
            "type": "paid",
            "data": normalized
        }, channel="paid")

        logger.info(
            f"Guard: {normalized['user']} purchased {guard_name} "
            f"(level {normalized['guard_level']}, ¥{normalized['price']})"
        )

    async def _on_super_chat_delete(self, message):
        """Handle superchat deletion"""
        deleted_ids = message.ids
        # Remove matching superchats from paid buffer
        before_len = len(self.paid_buffer)
        self.paid_buffer = deque(
            (m for m in self.paid_buffer
             if not (m.get("paid_type") == "superchat" and m.get("bili_id") in deleted_ids)),
            maxlen=self.config.bilibili.paid_max_buffer,
        )
        removed = before_len - len(self.paid_buffer)

        await self._broadcast_message({
            "type": "superchat_delete",
            "data": {"ids": deleted_ids}
        }, channel="paid")

        if removed:
            logger.info(f"Superchat delete: removed {removed} superchats (ids: {deleted_ids})")

    def _on_client_stopped(self, client, exception):
        """Called when the blivedm client stops (connection lost)."""
        if exception:
            logger.warning(f"Bilibili client stopped with exception: {exception}")
        else:
            logger.info("Bilibili client stopped")

        self.connected = False

        # Schedule reconnect if not intentional
        if not self._intentional_disconnect and self.running:
            logger.info("Scheduling auto-reconnect...")
            asyncio.create_task(self._broadcast_state_change())
            self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self):
        """Auto-reconnect with exponential backoff"""
        delay = self.config.bilibili.reconnect_delay_seconds
        max_delay = self.config.bilibili.reconnect_max_delay_seconds

        while self.running and not self.connected:
            logger.info(f"Attempting reconnect in {delay:.1f}s...")
            await asyncio.sleep(delay)

            if not self.running or self.connected:
                break

            try:
                # Cleanup old client
                if self.client:
                    try:
                        await self.client.stop()
                    except Exception:
                        pass
                    self.client = None

                await self.connect()
                logger.info("Reconnected successfully")
                return
            except Exception as e:
                logger.warning(f"Reconnect failed: {e}")
                delay = min(delay * 2, max_delay)

    # =========================================================================
    # Broadcasting
    # =========================================================================

    async def _broadcast_message(self, message: dict, channel: str | None = None):
        """Broadcast message to subscribed WebSocket clients.

        Args:
            message: The message dict to send.
            channel: If set, only send to clients subscribed to this channel.
                     If None, send to all clients (used for state_changed, etc).
        """
        if not self.ws_clients:
            return

        disconnected = []
        for client, subscribed in self.ws_clients.items():
            if channel is not None and channel not in subscribed:
                continue
            try:
                await client.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket client: {e}")
                disconnected.append(client)

        for client in disconnected:
            self.ws_clients.pop(client, None)

    async def _broadcast_state_change(self):
        """Broadcast state change to all WebSocket clients"""
        await self._broadcast_message({
            "type": "state_changed",
            "data": {
                "connected": self.connected,
                "danmaku_enabled": self.danmaku_enabled,
                "paid_enabled": self.paid_enabled,
            }
        })

    # =========================================================================
    # Public API methods for REST endpoints
    # =========================================================================

    def get_chat_snapshot(self) -> Dict[str, Any]:
        """Get current chat snapshot (all message types)"""

        return {
            "connected": self.connected,
            "danmaku_enabled": self.danmaku_enabled,
            "paid_enabled": self.paid_enabled,
            "danmaku": list(self.danmaku_buffer),
            "paid": list(self.paid_buffer),
        }

    def get_danmaku_snapshot(self) -> List[Dict[str, Any]]:
        """Get danmaku-only snapshot"""

        return list(self.danmaku_buffer)

    def get_paid_snapshot(self) -> List[Dict[str, Any]]:
        """Get paid messages snapshot (superchat/gift/guard)"""

        return list(self.paid_buffer)

    def get_state(self) -> Dict[str, Any]:
        """Get current service state"""
        return {
            "connected": self.connected,
            "running": self.running,
            "danmaku_enabled": self.danmaku_enabled,
            "paid_enabled": self.paid_enabled,
            "room_id": self.config.bilibili.room_id,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            "danmaku_buffer_size": len(self.danmaku_buffer),
            "paid_buffer_size": len(self.paid_buffer),
            "uptime_seconds": time.time() - self.start_time,
            "total_danmaku_received": self.total_danmaku_received,
            "total_paid_received": self.total_paid_received,
            "total_gift_coins": self.total_gift_coins,
        }

    def get_health(self) -> Dict[str, Any]:
        """Get detailed health status"""
        return {
            "status": "healthy" if self.connected else "disconnected",
            "connected": self.connected,
            "uptime": time.time() - self.start_time,
            "buffer_health": {
                "danmaku_size": len(self.danmaku_buffer),
                "paid_size": len(self.paid_buffer),
            },
            "client_health": {
                "ws_clients": len(self.ws_clients),
            }
        }

    # WebSocket client management
    def add_ws_client(self, client: WebSocket):
        """Add a WebSocket client for broadcasting (starts with no subscriptions)"""
        self.ws_clients[client] = set()
        logger.debug(f"WebSocket client added. Total clients: {len(self.ws_clients)}")

    def remove_ws_client(self, client: WebSocket):
        """Remove a WebSocket client"""
        self.ws_clients.pop(client, None)
        logger.debug(f"WebSocket client removed. Total clients: {len(self.ws_clients)}")

    def subscribe_ws_client(self, client: WebSocket, channels: list[str]):
        """Subscribe a WebSocket client to channels"""
        if client in self.ws_clients:
            valid = set(channels) & VALID_CHANNELS
            self.ws_clients[client].update(valid)

    def unsubscribe_ws_client(self, client: WebSocket, channels: list[str]):
        """Unsubscribe a WebSocket client from channels"""
        if client in self.ws_clients:
            self.ws_clients[client] -= set(channels)
