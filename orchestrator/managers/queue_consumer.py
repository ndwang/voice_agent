"""
Queue Consumer

Event-driven consumer that processes items from the priority queue.
Uses a state machine to manage flow and prevent race conditions.
"""
import asyncio
from enum import Enum
from typing import Optional
from core.event_bus import EventBus, Event
from core.logging import get_logger
from core.settings import get_settings
from orchestrator.events import EventType
from orchestrator.managers.base import BaseManager
from orchestrator.core.activity_state import get_activity_state
from core.priority_queue import InputItem

logger = get_logger(__name__)


class ConsumerState(Enum):
    """Consumer state machine states."""
    IDLE = "idle"                  # Ready to process next item
    PROCESSING = "processing"      # Currently processing an item
    COOLDOWN = "cooldown"          # Waiting after turn (blocks P2+, P0/P1 can bypass)


class QueueConsumer(BaseManager):
    """
    Event-driven consumer that processes items from the priority queue.

    Uses a state machine to manage processing flow:
    - IDLE: Ready to process next item
    - PROCESSING: Agent is responding/synthesizing/playing
    - COOLDOWN: Waiting after turn ends (P2+ blocked, P0/P1 bypass)

    Priority handling:
    - P0 (Critical): Bypasses cooldown, triggers CRITICAL_INTERRUPT if agent busy
    - P1 (Voice): Bypasses cooldown, triggers VOICE_INTERRUPT if agent busy
    - P2+: Must wait for cooldown after TURN_ENDED
    """

    def __init__(self, event_bus: EventBus, queue_manager):
        self.queue_manager = queue_manager
        self.settings = get_settings()

        # Access centralized activity state
        self.activity_state = get_activity_state()

        # State machine
        self.state = ConsumerState.IDLE
        self._state_lock = asyncio.Lock()

        # Cooldown configuration
        self.cooldown_seconds = self.settings.orchestrator.queue_cooldown_seconds
        self._cooldown_task: Optional[asyncio.Task] = None

        # Track when voice interrupt is pending (between VOICE_INTERRUPT and voice processing)
        self._pending_voice_interrupt = False
        self._interrupt_timeout_task: Optional[asyncio.Task] = None

        super().__init__(event_bus)

    def _register_handlers(self):
        """Subscribe to state change events instead of polling."""
        self.event_bus.subscribe(EventType.TURN_ENDED.value, self.on_turn_ended)
        self.event_bus.subscribe(EventType.VOICE_INTERRUPT.value, self.on_voice_interrupt)
        self.event_bus.subscribe(EventType.CRITICAL_INTERRUPT.value, self.on_critical_interrupt)
        self.event_bus.subscribe(EventType.INPUT_RECEIVED.value, self.on_input_started)
        self.event_bus.subscribe("queue.item_added", self.on_queue_item_added)
        self.event_bus.subscribe("queue.cooldown_expired", self.on_cooldown_expired)

    # === Event Handlers ===

    async def on_queue_item_added(self, event: Event):
        """New item in queue - try processing."""
        await self._try_process_next()

    async def on_turn_ended(self, event: Event):
        """
        Agent finished speaking - always start cooldown.

        Cooldown prevents rapid-fire processing. P0/P1 items can bypass
        the cooldown in _try_process_next(), but P2+ items must wait.
        """
        async with self._state_lock:
            if self.state != ConsumerState.PROCESSING:
                return  # Ignore duplicates

            # Always transition to cooldown after turn ends
            self.state = ConsumerState.COOLDOWN
            self._start_cooldown()
            self.logger.debug(f"Turn ended → COOLDOWN ({self.cooldown_seconds}s)")

        await self._try_process_next()

    async def on_voice_interrupt(self, event: Event):
        """
        Voice interruption - cancel cooldown, reset to IDLE, block P2+ items.

        This event signals both:
        1. Cancel current activity (like old LLM_CANCELLED)
        2. Voice input is pending (block P2+ until voice processed)
        """
        async with self._state_lock:
            # Cancel cooldown timer
            if self._cooldown_task and not self._cooldown_task.done():
                self._cooldown_task.cancel()
                self._cooldown_task = None

            # Reset to IDLE
            old_state = self.state
            self.state = ConsumerState.IDLE

            # Block P2+ items until voice input is processed
            self._pending_voice_interrupt = True
            self.logger.info(f"Voice interrupt: {old_state} → IDLE, blocking P2+ items")

            # Safety: Clear flag after timeout in case no transcript arrives
            if self._interrupt_timeout_task:
                self._interrupt_timeout_task.cancel()
            self._interrupt_timeout_task = asyncio.create_task(
                self._clear_interrupt_flag_after_timeout()
            )

        await self._try_process_next()

    async def on_critical_interrupt(self, event: Event):
        """
        Critical P0 interruption - cancel cooldown, reset to IDLE.

        Unlike voice interrupt, this doesn't block P2+ items because
        the P0 item is already in the queue and will be processed immediately.
        """
        async with self._state_lock:
            # Cancel cooldown timer
            if self._cooldown_task and not self._cooldown_task.done():
                self._cooldown_task.cancel()
                self._cooldown_task = None

            # Reset to IDLE
            old_state = self.state
            self.state = ConsumerState.IDLE
            self.logger.info(f"Critical interrupt: {old_state} → IDLE")

        await self._try_process_next()

    async def on_input_started(self, event: Event):
        """INPUT_RECEIVED published - mark as PROCESSING."""
        async with self._state_lock:
            self.state = ConsumerState.PROCESSING
            priority = event.data.get("priority")

            # Clear voice interrupt flag when P0/P1 is processed
            if priority is not None and priority <= 1:
                self._pending_voice_interrupt = False
                # Cancel timeout task
                if self._interrupt_timeout_task:
                    self._interrupt_timeout_task.cancel()
                    self._interrupt_timeout_task = None
                self.logger.debug("Voice/critical input processed - P2+ items unblocked")

    async def on_cooldown_expired(self, event: Event):
        """Cooldown timer expired - try processing next."""
        await self._try_process_next()

    # === Core Processing Logic ===

    async def _try_process_next(self):
        """
        Attempt to dequeue and process next item.

        Runs within EventBus sequential context - no race conditions.
        """
        async with self._state_lock:
            # Guard: Check if ready
            if not self._can_process():
                return

            # Guard: Queue empty
            if not self.queue_manager.has_items():
                return

            # Check priority before dequeuing
            next_priority = self.queue_manager.peek_priority()
            if next_priority is None:
                return

            # P2+ blocked during cooldown (P0/P1 bypass cooldown)
            if next_priority >= 2 and self.state == ConsumerState.COOLDOWN:
                self.logger.debug(f"P{next_priority} blocked by cooldown")
                return

            # P2+ blocked when voice interrupt is pending
            # Covers both: user currently speaking AND gap between transcribing=False and voice enqueued
            if next_priority >= 2 and (self.activity_state.state.transcribing or self._pending_voice_interrupt):
                self.logger.debug(f"P{next_priority} blocked by voice interrupt/transcription")
                return

            # ATOMIC: Dequeue + state transition
            item = await self.queue_manager.dequeue()
            self.state = ConsumerState.PROCESSING
            self.logger.debug(f"Dequeued {item.source_type} (P{item.priority}) → PROCESSING")

        # Process outside lock (state already set)
        await self._process_item(item)

    def _can_process(self) -> bool:
        """
        Check if ready to process. Called within lock.

        Note: This checks general readiness. Priority-specific checks
        (cooldown, transcribing) are done in _try_process_next().
        """
        if self.activity_state.is_busy():
            return False
        return self.state == ConsumerState.IDLE

    async def _process_item(self, item: InputItem):
        """
        Process item by publishing INPUT_RECEIVED.

        No semaphore needed - state machine prevents concurrent calls.
        No 50ms delay needed - state already set to PROCESSING.
        """
        try:
            # Fetch data (calls get_danmaku_snapshot() if callable)
            data = item.fetch_data()

            # Extract text based on source type
            text = self._format_text(item, data)

            if not text.strip():
                self.logger.debug(f"Empty text from {item.source_type}, skipping")
                async with self._state_lock:
                    self.state = ConsumerState.IDLE
                await self._try_process_next()
                return

            # Publish INPUT_RECEIVED event for InteractionManager
            # on_input_started handler runs in same event cycle
            await self.event_bus.publish(Event(
                EventType.INPUT_RECEIVED.value,
                {
                    "text": text,
                    "source": item.source_type,
                    "priority": item.priority,
                    "metadata": item.metadata or {}
                }
            ))

            # No delay needed - state already updated to PROCESSING in same event cycle

            self.logger.info(f"Processed {item.source_type} (P{item.priority}): {text[:50]}...")

        except Exception as e:
            self.logger.error(f"Error processing item {item.source_type}: {e}", exc_info=True)
            async with self._state_lock:
                self.state = ConsumerState.IDLE
            await self._try_process_next()

    def _format_text(self, item: InputItem, data: dict) -> str:
        """Format text based on source type."""
        if item.source_type == "voice":
            return data.get("text", "")
        elif item.source_type == "bilibili_single":
            # Format single danmaku: "username: message"
            user = data.get("user", "Unknown")
            content = data.get("content", "")
            return f"来自{user}的留言: {content}"
        elif item.source_type == "bilibili_batch":
            # Format batch: list of danmaku
            if isinstance(data, list):
                # Format as "user1: msg1\nuser2: msg2"
                formatted = "\n".join([
                    f"{d.get('user', 'Unknown')}: {d.get('content', '')}"
                    for d in data
                ])
                return f"Recent chat messages:\n{formatted}"
            else:
                return str(data)
        else:
            # Future: OCR, game, etc.
            return str(data)

    # === Cooldown Management ===

    def _start_cooldown(self):
        """Start cooldown timer. Called within _state_lock."""
        if self._cooldown_task and not self._cooldown_task.done():
            self._cooldown_task.cancel()
        self._cooldown_task = asyncio.create_task(self._cooldown_timer())

    async def _cooldown_timer(self):
        """Cooldown timer - publishes event when expired."""
        try:
            await asyncio.sleep(self.cooldown_seconds)

            async with self._state_lock:
                if self.state == ConsumerState.COOLDOWN:
                    self.state = ConsumerState.IDLE
                    self.logger.debug("Cooldown expired → IDLE")

            await self.event_bus.publish(Event("queue.cooldown_expired", {}))

        except asyncio.CancelledError:
            self.logger.debug("Cooldown cancelled")
            raise

    async def _clear_interrupt_flag_after_timeout(self):
        """Safety timeout: Clear interrupt flag if no voice input arrives within 10s."""
        try:
            await asyncio.sleep(10.0)
            async with self._state_lock:
                if self._pending_voice_interrupt:
                    self.logger.warning(
                        "Voice interrupt timeout - no transcript received, clearing flag"
                    )
                    self._pending_voice_interrupt = False
            await self._try_process_next()
        except asyncio.CancelledError:
            pass  # Cancelled when voice is processed normally
