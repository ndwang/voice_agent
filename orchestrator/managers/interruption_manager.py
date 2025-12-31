"""
Interruption Manager

Manages conversation state transitions and user barge-in logic.
"""
from core.event_bus import EventBus, Event
from core.logging import get_logger
from orchestrator.events import EventType
from orchestrator.managers.context_manager import ContextManager
from orchestrator.core.activity_state import get_activity_state

logger = get_logger(__name__)


class InterruptionManager:
    """Handles user interruption and message concatenation logic."""

    def __init__(self, event_bus: EventBus):
        """
        Initialize the interruption manager.

        Args:
            event_bus: Event bus for publishing cancellation events
        """
        self.event_bus = event_bus
        self._interrupted_before_finished = False

        # Access centralized activity state
        self.activity_state = get_activity_state()

        # Subscribe to all interruption triggers and cancellation
        self.event_bus.subscribe(EventType.SPEECH_START.value, self.on_speech_start)
        self.event_bus.subscribe(EventType.CRITICAL_INPUT.value, self.on_critical_input)
        self.event_bus.subscribe(EventType.LLM_CANCELLED.value, self.on_cancelled)

    async def on_speech_start(self, event: Event):
        """
        Handle SPEECH_START - check if busy and cancel if needed.

        Args:
            event: SPEECH_START event
        """
        if self.activity_state.is_busy():
            await self.event_bus.publish(Event(EventType.LLM_CANCELLED.value))

    async def on_critical_input(self, event: Event):
        """
        Handle critical input (P0) - trigger interruption immediately.

        Args:
            event: CRITICAL_INPUT event
        """
        logger.info("Critical input detected - triggering interruption")
        await self.event_bus.publish(Event(EventType.LLM_CANCELLED.value))

    async def on_cancelled(self, event: Event):
        """
        Handle LLM_CANCELLED - mark if we were interrupted during response generation.

        Args:
            event: LLM_CANCELLED event
        """
        # Check if we were still generating a response when cancelled
        if self.activity_state.state.responding:
            logger.info("Interrupted during LLM generation - will concatenate next message")
            self._interrupted_before_finished = True

    def was_interrupted(self) -> bool:
        """
        Check if previous turn was interrupted before finishing.

        Returns:
            True if interrupted before LLM finished
        """
        return self._interrupted_before_finished

    def clear_interrupted(self):
        """Clear the interrupted flag."""
        self._interrupted_before_finished = False
