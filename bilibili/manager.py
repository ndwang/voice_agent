"""
BilibiliManager - Core business logic for Bilibili live chat service.
Manages client lifecycle, message buffering, and WebSocket broadcasting.
"""

import asyncio
import time
from collections import deque
from typing import Optional, Set, Dict, List, Any, Deque
from fastapi import WebSocket
from core.logging import get_logger
from bilibili.bilibili_client import BilibiliClient
from bilibili.settings import BilibiliServiceConfig

logger = get_logger(__name__)


class BilibiliManager:
    """Manages Bilibili client, message buffers, and WebSocket broadcasting"""

    def __init__(self, config: BilibiliServiceConfig):
        self.config = config

        # Core components
        self.client: Optional[BilibiliClient] = None

        # State flags
        self.connected: bool = False
        self.running: bool = False
        self.danmaku_enabled: bool = config.bilibili.danmaku_enabled_default
        self.superchat_enabled: bool = config.bilibili.superchat_enabled_default

        # Message buffers (same as current implementation)
        self.danmaku_buffer: Deque[Dict[str, Any]] = deque()
        self.superchat_buffer: Deque[Dict[str, Any]] = deque()

        # Statistics
        self.total_danmaku_received: int = 0
        self.total_superchat_received: int = 0
        self.start_time: float = time.time()

        # WebSocket clients for real-time streaming
        self.ws_clients: Set[WebSocket] = set()

        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self):
        """Initialize manager and start cleanup task"""
        if self.running:
            return

        self.running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("BilibiliManager started")

        # Auto-connect if enabled
        if self.config.bilibili.enabled:
            await self.connect()

    async def stop(self):
        """Shutdown manager and cleanup resources"""
        if not self.running:
            return

        self.running = False

        # Stop cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Disconnect client
        await self.disconnect()

        logger.info("BilibiliManager stopped")

    async def connect(self):
        """Connect to Bilibili room"""
        if self.connected:
            logger.warning("Already connected to Bilibili")
            return

        try:
            self.client = BilibiliClient(
                room_id=self.config.bilibili.room_id,
                sessdata=self.config.bilibili.sessdata
            )
            self.client.set_handlers(
                on_danmaku=self._on_danmaku,
                on_super_chat=self._on_super_chat
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

    def set_danmaku_enabled(self, enabled: bool):
        """Toggle danmaku processing"""
        self.danmaku_enabled = enabled
        logger.info(f"Danmaku processing {'enabled' if enabled else 'disabled'}")

    def set_superchat_enabled(self, enabled: bool):
        """Toggle superchat processing"""
        self.superchat_enabled = enabled
        logger.info(f"Superchat processing {'enabled' if enabled else 'disabled'}")

    async def _on_danmaku(self, message):
        """Handle incoming danmaku from Bilibili client"""
        if not self.danmaku_enabled:
            return

        normalized = {
            "id": f"dm_{int(time.time() * 1000)}_{message.uid}",
            "user": message.uname,
            "content": message.msg,
            "timestamp": time.time()
        }

        self.danmaku_buffer.append(normalized)
        self.total_danmaku_received += 1

        # Broadcast to all WebSocket clients
        await self._broadcast_message({
            "type": "danmaku",
            "data": normalized
        })

        logger.debug(f"Danmaku: {normalized['user']}: {normalized['content']}")

    async def _on_super_chat(self, message):
        """Handle incoming superchat from Bilibili client"""
        if not self.superchat_enabled:
            return

        normalized = {
            "id": f"sc_{int(time.time() * 1000)}_{message.uid}",
            "user": message.uname,
            "content": message.message,
            "timestamp": time.time(),
            "amount": message.price
        }

        self.superchat_buffer.append(normalized)
        self.total_superchat_received += 1

        # Broadcast to all WebSocket clients
        await self._broadcast_message({
            "type": "superchat",
            "data": normalized
        })

        logger.info(f"SuperChat: {normalized['user']} (¥{normalized['amount']}): {normalized['content']}")

    async def _cleanup_loop(self):
        """Periodically remove expired danmaku from buffer"""
        while self.running:
            try:
                await asyncio.sleep(1.0)  # Check every second
                self._prune_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    def _prune_buffer(self):
        """Prune expired danmaku from the front of the deque"""
        now = time.time()
        ttl = self.config.bilibili.danmaku_ttl_seconds

        # Since danmaku are added in time order, oldest are at the front
        while self.danmaku_buffer and now - self.danmaku_buffer[0]["timestamp"] >= ttl:
            self.danmaku_buffer.popleft()

    async def _broadcast_message(self, message: dict):
        """Broadcast message to all connected WebSocket clients"""
        if not self.ws_clients:
            return

        # Send to all clients, remove disconnected ones
        disconnected = set()
        for client in self.ws_clients:
            try:
                await client.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket client: {e}")
                disconnected.add(client)

        # Remove disconnected clients
        self.ws_clients -= disconnected

    async def _broadcast_state_change(self):
        """Broadcast state change to all WebSocket clients"""
        await self._broadcast_message({
            "type": "state_changed",
            "data": {
                "connected": self.connected,
                "danmaku_enabled": self.danmaku_enabled,
                "superchat_enabled": self.superchat_enabled
            }
        })

    # Public API methods for REST endpoints
    def get_chat_snapshot(self) -> Dict[str, Any]:
        """Get current chat snapshot (danmaku + superchat)"""
        self._prune_buffer()  # Ensure buffer is up-to-date
        return {
            "connected": self.connected,
            "danmaku_enabled": self.danmaku_enabled,
            "superchat_enabled": self.superchat_enabled,
            "danmaku": list(self.danmaku_buffer),
            "superchat": list(self.superchat_buffer)
        }

    def get_danmaku_snapshot(self) -> List[Dict[str, Any]]:
        """Get danmaku-only snapshot"""
        self._prune_buffer()
        return list(self.danmaku_buffer)

    def get_superchat_snapshot(self) -> List[Dict[str, Any]]:
        """Get superchat-only snapshot"""
        return list(self.superchat_buffer)

    def get_state(self) -> Dict[str, Any]:
        """Get current service state"""
        return {
            "connected": self.connected,
            "running": self.running,
            "danmaku_enabled": self.danmaku_enabled,
            "superchat_enabled": self.superchat_enabled,
            "room_id": self.config.bilibili.room_id
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        self._prune_buffer()
        uptime = time.time() - self.start_time

        return {
            "danmaku_count": len(self.danmaku_buffer),
            "superchat_count": len(self.superchat_buffer),
            "danmaku_buffer_size": len(self.danmaku_buffer),
            "superchat_buffer_size": len(self.superchat_buffer),
            "uptime_seconds": uptime,
            "total_danmaku_received": self.total_danmaku_received,
            "total_superchat_received": self.total_superchat_received
        }

    def get_health(self) -> Dict[str, Any]:
        """Get detailed health status"""
        return {
            "status": "healthy" if self.connected else "disconnected",
            "connected": self.connected,
            "uptime": time.time() - self.start_time,
            "buffer_health": {
                "danmaku_size": len(self.danmaku_buffer),
                "superchat_size": len(self.superchat_buffer)
            },
            "client_health": {
                "ws_clients": len(self.ws_clients)
            }
        }

    # WebSocket client management
    def add_ws_client(self, client: WebSocket):
        """Add a WebSocket client for broadcasting"""
        self.ws_clients.add(client)
        logger.debug(f"WebSocket client added. Total clients: {len(self.ws_clients)}")

    def remove_ws_client(self, client: WebSocket):
        """Remove a WebSocket client"""
        self.ws_clients.discard(client)
        logger.debug(f"WebSocket client removed. Total clients: {len(self.ws_clients)}")
