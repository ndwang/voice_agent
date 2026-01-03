"""
Message buffer for chat messages.

Extracted from orchestrator/sources/bilibili_source.py to be reusable
without event bus dependency.
"""
import asyncio
import time
from collections import deque
from typing import List, Dict, Optional, Any, Deque
from core.logging import get_logger
from bilibili import BilibiliClient

logger = get_logger(__name__)


class ChatMessageBuffer:
    """
    Manages buffering of Bilibili chat messages (danmaku and superchat).

    Reuses the buffering logic from BilibiliSource but without event bus dependency.
    """

    def __init__(self, room_id: int, sessdata: Optional[str] = None, ttl_seconds: int = 300):
        """
        Initialize message buffer.

        Args:
            room_id: Bilibili room ID to connect to
            sessdata: Optional SESSDATA for authentication
            ttl_seconds: Time-to-live for danmaku messages (default 5 minutes)
        """
        self.room_id = room_id
        self.ttl = ttl_seconds

        # Components
        self.client = BilibiliClient(room_id=room_id, sessdata=sessdata)
        self.client.set_handlers(
            on_danmaku=self._on_danmaku,
            on_super_chat=self._on_super_chat
        )

        # State
        self.danmaku_buffer: Deque[Dict[str, Any]] = deque()
        self.superchat_buffer: Deque[Dict[str, Any]] = deque()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = asyncio.Lock()  # For thread-safe access

    async def start(self):
        """Start the Bilibili connection and cleanup task."""
        if self._running:
            return

        self._running = True
        await self.client.start()
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info(f"ChatMessageBuffer started for room {self.room_id}")

    async def stop(self):
        """Stop the Bilibili connection and cleanup task."""
        if not self._running:
            return

        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        await self.client.stop()
        logger.info("ChatMessageBuffer stopped")

    async def _on_danmaku(self, message):
        """Handle incoming danmaku (internal callback)."""
        normalized = {
            "id": f"dm_{int(time.time() * 1000)}_{message.uid}",
            "type": "danmaku",
            "user": message.uname,
            "content": message.msg,
            "timestamp": time.time()
        }

        async with self._lock:
            self.danmaku_buffer.append(normalized)

        logger.debug(f"Danmaku: {normalized['user']}: {normalized['content']}")

    async def _on_super_chat(self, message):
        """Handle incoming superchat (internal callback)."""
        normalized = {
            "id": f"sc_{int(time.time() * 1000)}_{message.uid}",
            "type": "superchat",
            "user": message.uname,
            "content": message.message,
            "timestamp": time.time(),
            "amount": message.price
        }

        async with self._lock:
            self.superchat_buffer.append(normalized)

        logger.info(f"SuperChat: {normalized['user']} (¥{normalized['amount']}): {normalized['content']}")

    async def _cleanup_loop(self):
        """Periodically remove expired danmaku from the buffer."""
        while self._running:
            try:
                await asyncio.sleep(1.0)
                await self._prune_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def _prune_buffer(self):
        """Prune expired danmaku from the front of the deque."""
        async with self._lock:
            now = time.time()
            while self.danmaku_buffer and now - self.danmaku_buffer[0]["timestamp"] >= self.ttl:
                self.danmaku_buffer.popleft()

    async def get_all_messages(
        self,
        max_messages: Optional[int] = None,
        time_window_seconds: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent messages (both danmaku and superchat combined).

        Args:
            max_messages: Maximum number of messages to return (most recent)
            time_window_seconds: Only include messages from last N seconds

        Returns:
            List of message dicts sorted by timestamp
        """
        await self._prune_buffer()

        async with self._lock:
            # Combine both buffers
            all_messages = list(self.danmaku_buffer) + list(self.superchat_buffer)

        # Sort by timestamp
        all_messages.sort(key=lambda m: m["timestamp"])

        # Filter by time window if specified
        if time_window_seconds:
            cutoff = time.time() - time_window_seconds
            all_messages = [m for m in all_messages if m["timestamp"] >= cutoff]

        # Limit to max_messages (most recent)
        if max_messages:
            all_messages = all_messages[-max_messages:]

        return all_messages

    async def get_buffer_stats(self) -> Dict[str, Any]:
        """Get current buffer statistics."""
        await self._prune_buffer()

        async with self._lock:
            danmaku_count = len(self.danmaku_buffer)
            superchat_count = len(self.superchat_buffer)
            all_messages = list(self.danmaku_buffer) + list(self.superchat_buffer)

        if all_messages:
            all_messages.sort(key=lambda m: m["timestamp"])
            oldest = all_messages[0]["timestamp"]
            newest = all_messages[-1]["timestamp"]
        else:
            oldest = None
            newest = None

        return {
            "danmaku_count": danmaku_count,
            "superchat_count": superchat_count,
            "total_count": danmaku_count + superchat_count,
            "oldest_timestamp": oldest,
            "newest_timestamp": newest
        }
