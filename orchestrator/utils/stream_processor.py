"""
Stream Processor Utility

Handles parsing of LLM token streams and routing to appropriate destinations.
"""
import asyncio
from typing import AsyncIterator
from core.event_bus import EventBus, Event
from orchestrator.events import EventType
from orchestrator.utils.text_processing import (
    filter_thinking_tags_final,
    LLMStreamParser
)


class StreamProcessor:
    """Processes LLM token streams with tag-based routing."""

    def __init__(self, event_bus: EventBus):
        """
        Initialize the stream processor.

        Args:
            event_bus: Event bus for publishing events
        """
        self.event_bus = event_bus
        self.current_cancel_event: asyncio.Event = None

        # Cache parsers for reuse
        self._parser = None
        self.disable_thinking = True

    async def _default_callback(self, text: str):
        """Handle untagged content - no operation"""
        # if not self.current_cancel_event.is_set():
        #     await self.event_bus.publish(Event(EventType.TTS_REQUEST.value, {"text": text}))
        #     await self.event_bus.publish(Event(EventType.SUBTITLE_REQUEST.value, {"text": text}))

    async def _jp_callback(self, text: str):
        """Handle <jp> tag content - send to TTS."""
        if not self.current_cancel_event.is_set():
            await self.event_bus.publish(Event(EventType.TTS_REQUEST.value, {"text": text}))

    async def _zh_callback(self, text: str):
        """Handle <zh> tag content - send to OBS subtitles."""
        if not self.current_cancel_event.is_set():
            await self.event_bus.publish(Event(EventType.SUBTITLE_REQUEST.value, {"text": text}))

    def _get_or_create_parser(self, disable_thinking: bool) -> LLMStreamParser:
        """Get cached parser or create a new one if needed."""
        # Configure parser
        tag_configs = []
        if disable_thinking:
            # Discard redacted_reasoning tags (no callback)
            tag_configs.append({"name": "think"})

        # Add jp and zh tag handlers
        tag_configs.append({"name": "jp", "callback": self._jp_callback})
        tag_configs.append({"name": "zh", "callback": self._zh_callback})

        # Check cache
        if self._parser is None:
            self._parser = LLMStreamParser(
                tag_configs, default_callback=self._default_callback
            )
        elif self.disable_thinking != disable_thinking:
            self._parser = LLMStreamParser(
                tag_configs, default_callback=self._default_callback
            )
            self.disable_thinking = disable_thinking

        return self._parser

    async def process_response(
        self,
        stream: AsyncIterator,
        cancel_event: asyncio.Event,
        disable_thinking: bool
    ) -> dict:
        """
        Process an LLM token stream, routing content by tags.

        Args:
            stream: Async iterator yielding tokens (str) or tool calls (dict) from LLM
            cancel_event: Event that signals cancellation
            disable_thinking: Whether to filter thinking tags

        Returns:
            Dict with:
            - "text": Full response text for history (with thinking tags filtered if needed)
            - "has_tool_calls": Boolean indicating if tool calls were detected
            - "tool_calls": List of tool call dicts (if any)
        """
        # Set current cancel event for callback access
        self.current_cancel_event = cancel_event

        # Get or create cached parser
        parser = self._get_or_create_parser(disable_thinking)
        parser.reset()

        full_response = ""
        tool_calls = []

        # Process stream
        async for item in stream:
            if cancel_event.is_set():
                break

            if not item:
                continue

            # Check if this is a tool call
            if isinstance(item, dict) and item.get("type") == "tool_call":
                tool_calls.append(item)
                # Don't publish token event for tool calls
                continue

            # Normal text token
            full_response += item

            # Publish token for UI display (timestamp added automatically by EventBus)
            await self.event_bus.publish(Event(EventType.LLM_TOKEN.value, {"token": item}))

            # Process token through parser
            await parser.process_token(item)

        # Final flush of parser buffers
        await parser.finalize()

        # Filter for history if needed
        history_response = full_response
        if disable_thinking:
            history_response = filter_thinking_tags_final(full_response)

        return {
            "text": history_response,
            "has_tool_calls": len(tool_calls) > 0,
            "tool_calls": tool_calls
        }
