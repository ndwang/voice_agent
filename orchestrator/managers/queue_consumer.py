"""
Queue Consumer

Autonomous consumer loop that processes items from the priority queue.
Monitors agent state and applies cooldown logic for non-voice inputs.
"""
import asyncio
from typing import Optional
from core.event_bus import EventBus, Event
from core.logging import get_logger
from core.settings import get_settings
from orchestrator.events import EventType
from orchestrator.managers.base import BaseManager
from orchestrator.core.activity_state import get_activity_state
from core.priority_queue import InputItem

logger = get_logger(__name__)


class QueueConsumer(BaseManager):
    """
    Consumes items from the priority queue and processes them.

    Agent is considered "busy" when: transcribing, responding, synthesizing, or playing
    Agent is considered "idle" when: none of the above

    Priority handling:
    - P0 (Critical): Always triggers interruption if agent is busy, no cooldown
    - P1 (Voice): No cooldown, processes immediately when idle
    - P2+: Wait for cooldown after TURN_ENDED before processing next
    """

    def __init__(self, event_bus: EventBus, queue_manager):
        self.queue_manager = queue_manager
        self.settings = get_settings()

        # Access centralized activity state
        self.activity_state = get_activity_state()

        # Cooldown configuration
        self.cooldown_seconds = self.settings.orchestrator.queue_cooldown_seconds
        self._in_cooldown = False
        self._cooldown_task: Optional[asyncio.Task] = None

        # Temporary holding for P2+ item during cooldown
        # This preserves FIFO ordering when we can't peek
        self._held_item: Optional[InputItem] = None

        # Consumer loop task
        self._consumer_task: Optional[asyncio.Task] = None
        self._running = False

        super().__init__(event_bus)

    def _register_handlers(self):
        """Subscribe to events."""
        self.event_bus.subscribe(EventType.TURN_ENDED.value, self.on_turn_ended)

    async def start(self):
        """Start the consumer loop."""
        if self._running:
            return

        self._running = True
        self._consumer_task = asyncio.create_task(self._consumer_loop())
        self.logger.info("Queue consumer started")

    async def stop(self):
        """Stop the consumer loop."""
        if not self._running:
            return

        self._running = False

        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass

        if self._cooldown_task:
            self._cooldown_task.cancel()
            try:
                await self._cooldown_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Queue consumer stopped")

    async def on_turn_ended(self, event: Event):
        """
        Start cooldown timer after turn ends.
        Only affects processing of non-voice items from queue.
        """
        # Only start cooldown if we're not already in cooldown
        if not self._in_cooldown:
            self._cooldown_task = asyncio.create_task(self._cooldown_timer())

    async def _cooldown_timer(self):
        """Cooldown timer - blocks processing of non-voice queue items."""
        try:
            self._in_cooldown = True
            self.logger.debug(f"Cooldown started ({self.cooldown_seconds}s)")
            await asyncio.sleep(self.cooldown_seconds)
            self._in_cooldown = False
            self.logger.debug("Cooldown ended")
        except asyncio.CancelledError:
            self._in_cooldown = False
            raise

    def _is_agent_busy(self) -> bool:
        """Check if agent is currently busy processing."""
        return self.activity_state.is_busy()

    async def _consumer_loop(self):
        """
        Main consumer loop.

        Continuously monitors the queue and processes items when:
        1. Agent is idle (not busy)
        2. Cooldown has elapsed (for non-voice items)

        Uses a held_item mechanism to avoid breaking FIFO ordering:
        - When we dequeue a P2+ item during cooldown, we hold it temporarily
        - We never put items back into the queue
        - This preserves FIFO ordering for same-priority items
        """
        self.logger.info("Consumer loop started")

        while self._running:
            try:
                # Check if agent is busy
                if self._is_agent_busy():
                    await asyncio.sleep(0.1)
                    continue

                # First, check if we have a held item from previous cooldown
                if self._held_item:
                    # We have a held P2+ item
                    if self._in_cooldown:
                        # Still in cooldown, wait
                        await asyncio.sleep(0.2)
                        continue
                    else:
                        # Cooldown ended, process the held item
                        self.logger.debug(f"Processing held item: {self._held_item.source_type} (P{self._held_item.priority})")
                        await self._process_item(self._held_item)
                        self._held_item = None
                        continue

                # No held item, check queue
                if not self.queue_manager.has_items():
                    await asyncio.sleep(0.1)
                    continue

                # Dequeue next item
                item = await self.queue_manager.dequeue()
                if not item:
                    await asyncio.sleep(0.1)
                    continue

                # P0 (Critical) or P1 (Voice): Process immediately (bypasses cooldown)
                if item.priority <= 1:
                    self.logger.debug(f"Processing priority {item.priority} input: {item.source_type}")
                    await self._process_item(item)
                    continue

                # P2+ items: Check cooldown
                if self._in_cooldown:
                    # Hold this item temporarily instead of putting back in queue
                    self.logger.debug(f"Cooldown active, holding {item.source_type} (P{item.priority})")
                    self._held_item = item
                    await asyncio.sleep(0.2)
                    continue

                # Process P2+ item (no cooldown)
                self.logger.debug(f"Processing {item.source_type} (P{item.priority})")
                await self._process_item(item)

                # Small delay to prevent tight loop
                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                self.logger.info("Consumer loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in consumer loop: {e}", exc_info=True)
                await asyncio.sleep(1.0)

        self.logger.info("Consumer loop stopped")

    async def _process_item(self, item: InputItem):
        """
        Process a queue item by fetching its data and publishing INPUT_RECEIVED event.

        Args:
            item: InputItem to process
        """
        try:
            # Fetch data (calls get_danmaku_snapshot() if callable)
            data = item.fetch_data()

            # Extract text based on source type
            if item.source_type == "voice":
                text = data.get("text", "")
            elif item.source_type == "bilibili_single":
                # Format single danmaku: "username: message"
                user = data.get("user", "Unknown")
                content = data.get("content", "")
                text = f"来自{user}的留言: {content}"
            elif item.source_type == "bilibili_batch":
                # Format batch: list of danmaku
                if isinstance(data, list):
                    # Format as "user1: msg1\nuser2: msg2"
                    formatted = "\n".join([
                        f"{d.get('user', 'Unknown')}: {d.get('content', '')}"
                        for d in data
                    ])
                    text = f"Recent chat messages:\n{formatted}"
                else:
                    text = str(data)
            else:
                # Future: OCR, game, etc.
                text = str(data)

            if not text.strip():
                self.logger.debug(f"Empty text from {item.source_type}, skipping")
                return

            # Publish INPUT_RECEIVED event for InteractionManager
            await self.event_bus.publish(Event(
                EventType.INPUT_RECEIVED.value,
                {
                    "text": text,
                    "source": item.source_type,
                    "priority": item.priority,
                    "metadata": item.metadata or {}
                }
            ))

            self.logger.info(f"Processed {item.source_type}: {text[:50]}...")

        except Exception as e:
            self.logger.error(f"Error processing item {item.source_type}: {e}", exc_info=True)
