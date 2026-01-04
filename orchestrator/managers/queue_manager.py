"""
Queue Manager

Manages the priority queue for all input sources (voice, bilibili, ocr, game).
Subscribes to input events and enqueues them with appropriate priorities.
"""
from typing import Optional
from core.event_bus import EventBus, Event
from core.logging import get_logger
from core.settings import get_settings
from orchestrator.events import EventType
from orchestrator.managers.base import BaseManager
from core.priority_queue import AsyncPriorityQueue, InputItem

logger = get_logger(__name__)


class QueueManager(BaseManager):
    """
    Manages priority queue for all input sources.

    Subscribes to input events (TRANSCRIPT_FINAL, etc.) and enqueues them
    with appropriate priorities.

    Priorities (lower = higher priority):
    - P0: CRITICAL - Always triggers interruption (reserved)
    - P1: Voice (user speech)
    - P2: Reserved for future high-priority
    - P3: Bilibili single danmaku
    - P4: Bilibili batch snapshot
    - P5+: OCR, game events, etc.
    """

    def __init__(self, event_bus: EventBus):
        self.queue = AsyncPriorityQueue()
        self.settings = get_settings()

        self.priorities = {
            "critical": 0,
            "voice": 1,
            "reserved": 2,
            "bilibili_single": 3,
            "bilibili_batch": 4,
            "ocr": 5,
            "game": 6
        }

        super().__init__(event_bus)

    def _register_handlers(self):
        """Subscribe to input events from various sources."""
        self.event_bus.subscribe(EventType.TRANSCRIPT_FINAL.value, self.on_transcript_final)
        self.event_bus.subscribe(EventType.BILIBILI_DANMAKU.value, self.on_bilibili_danmaku)
        self.event_bus.subscribe(EventType.BILIBILI_SUPERCHAT.value, self.on_bilibili_superchat)
        # Future: Subscribe to ocr events, game events

    async def on_bilibili_danmaku(self, event: Event):
        """Handle incoming danmaku."""
        from orchestrator.core.activity_state import get_activity_state

        # Check toggle state first
        if not get_activity_state().state.bilibili_danmaku_enabled:
            return

        danmaku = event.data

        # Existing '!' filter
        if not danmaku['content'].startswith('!'):
            return

        item = InputItem(
            priority=self.priorities["bilibili_single"],
            source_type="bilibili_single",
            get_data={"user": danmaku['user'], "content": danmaku['content'][1:]}
        )

        self.logger.info(f"Enqueued bilibili_single: {danmaku['user']}: {danmaku['content']}")
        await self._enqueue(item)

    async def on_bilibili_superchat(self, event: Event):
        """Handle incoming SuperChat."""
        from orchestrator.core.activity_state import get_activity_state

        # Check toggle state
        if not get_activity_state().state.bilibili_superchat_enabled:
            return

        superchat = event.data

        item = InputItem(
            priority=self.priorities["bilibili_single"],  # P3
            source_type="bilibili_superchat",
            get_data={
                "user": superchat['user'],
                "content": superchat['content'],
                "amount": superchat['amount']
            }
        )

        self.logger.info(f"Enqueued bilibili_superchat: {superchat['user']} (¥{superchat['amount']}): {superchat['content']}")
        await self._enqueue(item)

    async def on_transcript_final(self, event: Event):
        """
        Handle voice transcript - enqueue as P1 by default.

        Sources can specify priority in event.data["priority"] to override default.
        Priority 0 is reserved for critical interrupting messages.
        """
        text = event.data.get("text", "")
        if not text.strip():
            return

        # Allow source to override priority, otherwise use default (P1 for voice)
        priority = event.data.get("priority", self.priorities["voice"])

        # Store any additional metadata from source
        source_metadata = event.data.get("metadata", {})

        item = InputItem(
            priority=priority,
            source_type="voice",
            get_data={"text": text},
            metadata=source_metadata
        )
        await self._enqueue(item)

    async def _enqueue(self, item: InputItem):
        """
        Internal method to enqueue an input item.

        Args:
            item: InputItem with priority and source_type set
        """
        # P0 (Critical): Publish CRITICAL_INPUT event to trigger interruption
        if item.priority == 0:
            self.logger.info(f"Critical input (P0) detected: {item.source_type}")
            await self.event_bus.publish(Event(EventType.CRITICAL_INPUT.value))

        await self.queue.put(item, priority=item.priority)
        self.logger.debug(f"Enqueued {item.source_type} (P{item.priority})")

        # Notify consumer that item was added
        await self.event_bus.publish(Event("queue.item_added", {
            "priority": item.priority,
            "source_type": item.source_type
        }))

    def has_items(self) -> bool:
        """Check if queue has any items."""
        return not self.queue.empty()

    async def dequeue(self) -> Optional[InputItem]:
        """Dequeue the highest priority item."""
        if self.queue.empty():
            return None

        return await self.queue.get()

    def clear(self):
        """Clear all items from the queue."""
        self.queue.clear()
        self.logger.debug("Queue cleared")

    def get_size(self) -> int:
        """Get current queue size."""
        return self.queue.qsize()

    def peek_priority(self) -> Optional[int]:
        """Get priority of next item without dequeuing."""
        return self.queue.peek_priority()
