"""
Context manager for maintaining dialogue history during chapter processing.
"""
from typing import List
from collections import deque

from core.logging import get_logger
from vn_commentary.models import Dialogue

logger = get_logger(__name__)


class ContextManager:
    """
    Manages dialogue context for LLM analysis.

    Maintains a sliding window of recent dialogues to provide context
    for commentary decisions.
    """

    def __init__(self, max_context_size: int = 20):
        """
        Initialize context manager.

        Args:
            max_context_size: Maximum number of dialogues to keep in context
        """
        self.max_context_size = max_context_size
        self._context: deque[Dialogue] = deque(maxlen=max_context_size)

    def add_dialogue(self, dialogue: Dialogue):
        """
        Add a dialogue to the context.

        Args:
            dialogue: Dialogue to add
        """
        self._context.append(dialogue)
        logger.debug(f"Added dialogue to context: {dialogue.dialogue_id} (context size: {len(self._context)})")

    def get_context(self) -> List[Dialogue]:
        """
        Get current context as a list.

        Returns:
            List of dialogues in current context (oldest first)
        """
        return list(self._context)

    def format_context_for_llm(self) -> str:
        """
        Format context as a string for LLM prompt.

        Returns:
            Formatted context string with recent dialogues
        """
        if not self._context:
            return "No previous context."

        lines = ["Recent dialogue context:"]
        for dialogue in self._context:
            lines.append(f"- {dialogue.format_for_llm()}")

        return "\n".join(lines)

    def clear(self):
        """Clear all context (e.g., when starting a new chapter)."""
        self._context.clear()
        logger.info("Context cleared")

    def __len__(self) -> int:
        """Get current context size."""
        return len(self._context)
