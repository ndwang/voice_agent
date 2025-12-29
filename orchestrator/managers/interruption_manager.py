"""
Interruption Manager

Manages conversation state transitions and user barge-in logic.
"""
from core.event_bus import EventBus, Event
from core.logging import get_logger
from orchestrator.events import EventType
from orchestrator.managers.context_manager import ContextManager
from orchestrator.core.models import SystemState

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

    async def check_and_cancel(self, current_state: SystemState):
        """
        Check if system is busy and publish cancellation event if needed.

        Args:
            current_state: Current system activity state
        """
        # Only publish cancel if we are currently active (responding, synthesizing, or playing)
        is_busy = (
            current_state.responding or
            current_state.synthesizing or
            current_state.playing
        )

        if is_busy:
            await self.event_bus.publish(Event(EventType.LLM_CANCELLED.value))

    def mark_interrupted(self, is_responding: bool):
        """
        Mark that an interruption occurred during LLM generation.

        Args:
            is_responding: Whether the system was still generating a response
        """
        if is_responding:
            logger.info("Interrupted during LLM generation - will concatenate next message")
            self._interrupted_before_finished = True

    def handle_transcript_history(self, text: str, context: ContextManager) -> str:
        """
        Handle transcript history with interruption concatenation logic.

        If previous turn was interrupted before LLM finished, concatenates
        the new transcript with the previous user message.

        Args:
            text: New transcript text
            context: Context manager for conversation history

        Returns:
            Final text to be processed (either original or concatenated)
        """
        # Concatenate if previous turn was interrupted before LLM finished
        if self._interrupted_before_finished:
            last_msg = context.get_last_message()
            if last_msg and last_msg["role"] == "user":
                logger.info(f"Concatenating interrupted message: '{last_msg['content']}' + '{text}'")
                combined_text = f"{last_msg['content']} [interrupted] {text}"
                context.update_last_message(combined_text, role="user")
                self._interrupted_before_finished = False
                return combined_text
            else:
                context.add_user_message(text)
                self._interrupted_before_finished = False
                return text
        else:
            context.add_user_message(text)
            return text
