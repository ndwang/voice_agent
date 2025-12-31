import asyncio
import time
from collections import deque
from typing import List, Dict, Optional, Any, Deque
from core.event_bus import EventBus, Event
from core.logging import get_logger
from core.settings import get_settings
from orchestrator.sources.base import BaseSource
from orchestrator.events import EventType
from bilibili import BilibiliClient

logger = get_logger(__name__)

class BilibiliSource(BaseSource):
    """
    Source that connects to Bilibili live stream and manages danmaku and superChat.
    """

    def __init__(self, event_bus: EventBus):
        super().__init__(event_bus)

        # Config
        settings = get_settings()
        self.room_id = settings.bilibili.room_id
        self.sessdata = settings.bilibili.sessdata
        self.ttl = settings.bilibili.danmaku_ttl_seconds
        
        # Components
        self.client = BilibiliClient(room_id=self.room_id, sessdata=self.sessdata)
        self.client.set_handlers(
            on_danmaku=self._on_danmaku,
            on_super_chat=self._on_super_chat
        )
        
        # State
        self.danmaku_buffer: Deque[Dict[str, Any]] = deque()
        self.superchat_buffer: Deque[Dict[str, Any]] = deque()  # Persistent storage for UI
        self.superchat_queue: asyncio.Queue = asyncio.Queue()
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the Bilibili source."""
        if self.running or self.room_id == 0:
            if self.room_id == 0:
                logger.warning("Bilibili room_id not configured, source will not start.")
            return
            
        self.running = True
        await self.client.start()
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info(f"Bilibili source started for room {self.room_id}")

    async def stop(self):
        """Stop the Bilibili source."""
        if not self.running:
            return
            
        self.running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        await self.client.stop()
        logger.info("Bilibili source stopped")

    async def _on_danmaku(self, message):
        """Handle incoming danmaku."""
        normalized = {
            "id": f"dm_{int(time.time() * 1000)}_{message.uid}",
            "user": message.uname,
            "content": message.msg,
            "timestamp": time.time()
        }
        self.danmaku_buffer.append(normalized)

        # Publish event for UI
        await self.event_bus.publish(Event(
            EventType.BILIBILI_DANMAKU.value,
            normalized
        ))

        logger.debug(f"Danmaku added: {normalized['user']}: {normalized['content']}")

    async def _on_super_chat(self, message):
        """Handle incoming superChat."""
        normalized = {
            "id": f"sc_{int(time.time() * 1000)}_{message.uid}",
            "user": message.uname,
            "content": message.message,
            "timestamp": time.time(),
            "amount": message.price
        }
        self.superchat_buffer.append(normalized)  # Store for UI snapshot
        await self.superchat_queue.put(normalized)

        # Publish event for UI
        await self.event_bus.publish(Event(
            EventType.BILIBILI_SUPERCHAT.value,
            normalized
        ))

        logger.info(f"SuperChat added: {normalized['user']} (¥{normalized['amount']}): {normalized['content']}")

    async def _cleanup_loop(self):
        """Periodically remove expired danmaku from the buffer."""
        while self.running:
            try:
                await asyncio.sleep(1.0) # Check every second
                self._prune_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in Bilibili source cleanup loop: {e}")

    def _prune_buffer(self):
        """Prune expired danmaku from the front of the deque."""
        now = time.time()
        # Since danmaku are added in time order, the oldest are always at the front
        while self.danmaku_buffer and now - self.danmaku_buffer[0]["timestamp"] >= self.ttl:
            self.danmaku_buffer.popleft()

    def get_danmaku_snapshot(self) -> List[Dict[str, Any]]:
        """Return a snapshot of current non-expired danmaku."""
        self._prune_buffer()
        return list(self.danmaku_buffer)

    def get_danmaku_latest(self) -> Dict[str, Any]:
        """Return the latest danmaku from the buffer."""
        self._prune_buffer()
        return self.danmaku_buffer[-1] if self.danmaku_buffer else {}

    async def get_superchat(self) -> Dict[str, Any]:
        """Wait for and return the next superChat from the queue."""
        return await self.superchat_queue.get()

    def get_danmaku_count(self) -> int:
        """Return count of current non-expired danmaku."""
        self._prune_buffer()
        return len(self.danmaku_buffer)

    def get_superchat_count(self) -> int:
        """Return the number of superChats currently in the queue."""
        return self.superchat_queue.qsize()

    def get_superchat_snapshot(self) -> List[Dict[str, Any]]:
        """Return a snapshot of all superChats (no expiration)."""
        return list(self.superchat_buffer)

