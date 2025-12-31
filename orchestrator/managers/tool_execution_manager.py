"""
Tool Execution Manager

Manages tool execution lifecycle:
- Receives tool call requests from LLM
- Executes tool handlers via ToolRegistry
- Publishes results and triggers interpretation
- Handles cancellation during execution
"""
import asyncio
from typing import Dict, Any, List
from core.event_bus import EventBus, Event
from core.logging import get_logger
from orchestrator.events import EventType
from orchestrator.managers.base import BaseManager
from orchestrator.core.activity_state import get_activity_state
from orchestrator.tools.registry import ToolRegistry

logger = get_logger(__name__)


class ToolExecutionManager(BaseManager):
    """Manages tool execution and result interpretation."""

    def __init__(self, event_bus: EventBus, tool_registry: ToolRegistry):
        """
        Initialize ToolExecutionManager.

        Args:
            event_bus: Event bus for publishing and subscribing
            tool_registry: Registry of available tools
        """
        self.event_bus = event_bus
        self.tool_registry = tool_registry
        self.activity_state = get_activity_state()
        self.cancel_event = asyncio.Event()
        super().__init__(event_bus)

    def _register_handlers(self):
        """Register event handlers."""
        self.event_bus.subscribe(
            EventType.TOOL_CALL_REQUESTED.value,
            self.on_tool_call_requested
        )
        self.event_bus.subscribe(
            EventType.LLM_CANCELLED.value,
            self.on_cancel
        )

    async def on_cancel(self, event: Event):
        """
        Handle cancellation during tool execution.

        Sets cancel flag but lets tools complete (safer than force-stop).
        """
        self.cancel_event.set()
        logger.info("Tool execution cancellation requested")

    async def on_tool_call_requested(self, event: Event):
        """
        Handle tool call request from LLM.

        Args:
            event: Event with data containing:
                - tool_calls: List of {"id": str, "name": str, "arguments": dict}
        """
        tool_calls = event.data.get("tool_calls", [])

        if not tool_calls:
            logger.warning("Received TOOL_CALL_REQUESTED with no tool_calls")
            return

        logger.info(f"Executing {len(tool_calls)} tool(s): {[tc['name'] for tc in tool_calls]}")

        # Reset cancel state for this execution
        self.cancel_event.clear()

        # Update activity state
        await self.activity_state.update({"executing_tools": True})

        # Execute all tools in parallel (let them complete even if cancelled - safer)
        results = await asyncio.gather(
            *[self._execute_tool(tc) for tc in tool_calls],
            return_exceptions=True
        )

        # Update activity state
        await self.activity_state.update({"executing_tools": False})

        # Publish results
        for tool_call, result in zip(tool_calls, results):
            if isinstance(result, Exception):
                # Tool execution failed
                error_msg = str(result)
                logger.error(
                    f"Tool '{tool_call['name']}' failed: {error_msg}",
                    exc_info=result
                )
                await self.event_bus.publish(Event(
                    EventType.TOOL_ERROR.value,
                    {
                        "tool_call_id": tool_call["id"],
                        "name": tool_call["name"],
                        "error": error_msg
                    }
                ))
            else:
                # Tool execution succeeded
                logger.info(f"Tool '{tool_call['name']}' completed successfully")
                await self.event_bus.publish(Event(
                    EventType.TOOL_RESULT.value,
                    {
                        "tool_call_id": tool_call["id"],
                        "name": tool_call["name"],
                        "result": result
                    }
                ))

        # Only request interpretation if not cancelled
        if not self.cancel_event.is_set():
            logger.info("Requesting LLM interpretation of tool results")
            await self.event_bus.publish(Event(
                EventType.TOOL_INTERPRETATION_REQUEST.value,
                {
                    "tool_calls": tool_calls,
                    "tool_results": results
                }
            ))
        else:
            logger.info("Tool execution cancelled - skipping interpretation")

    async def _execute_tool(self, tool_call: Dict[str, Any]) -> Any:
        """
        Execute a single tool call.

        Args:
            tool_call: Dict with "id", "name", and "arguments"

        Returns:
            Tool execution result (any type)

        Raises:
            Exception: If tool execution fails
        """
        tool_name = tool_call["name"]
        tool_args = tool_call.get("arguments", {})

        # Check if tool is enabled
        if not self.tool_registry.is_tool_enabled(tool_name):
            error_msg = f"Tool '{tool_name}' is currently disabled"
            logger.warning(error_msg)
            raise ValueError(error_msg)

        # Publish executing event
        await self.event_bus.publish(Event(
            EventType.TOOL_EXECUTING.value,
            {
                "name": tool_name,
                "arguments": tool_args
            }
        ))

        # Execute tool via registry
        logger.debug(f"Executing tool '{tool_name}' with args: {tool_args}")
        result = await self.tool_registry.execute(tool_name, tool_args)

        logger.debug(f"Tool '{tool_name}' returned: {result}")
        return result
