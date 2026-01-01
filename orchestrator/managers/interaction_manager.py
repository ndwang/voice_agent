import asyncio
from typing import Optional
from core.event_bus import EventBus, Event
from core.logging import get_logger
from core.settings import get_settings, LLMSettings, OrchestratorSettings
from core.settings.reload_result import ReloadResult
from orchestrator.events import EventType
from orchestrator.managers.base import BaseManager
from orchestrator.managers.context_manager import ContextManager
from orchestrator.managers.interruption_manager import InterruptionManager
from orchestrator.core.activity_state import get_activity_state
from orchestrator.utils.event_helpers import publish_history_updated
from orchestrator.utils.llm_factory import create_provider
from orchestrator.utils.stream_processor import StreamProcessor

logger = get_logger(__name__)

class InteractionManager(BaseManager):
    """
    Orchestrates the conversation flow:
    Event(Transcript) -> Context -> LLM -> Event(Token) -> Buffer -> TTS -> Event(Audio)
    """

    def __init__(
        self,
        event_bus: EventBus,
        tool_registry = None,
        llm_settings: Optional[LLMSettings] = None,
        orch_settings: Optional[OrchestratorSettings] = None
    ):
        # Use injected settings if provided, otherwise use global
        app_settings = get_settings()
        self.llm_settings = llm_settings or app_settings.llm
        self.orch_settings = orch_settings or app_settings.orchestrator

        # Type-safe access with IDE autocomplete!
        self.disable_thinking = self.llm_settings.get_provider_config().disable_thinking

        # Tool registry and tool execution manager
        self.tool_registry = tool_registry
        if self.tool_registry:
            from orchestrator.managers.tool_execution_manager import ToolExecutionManager
            self.tool_execution_manager = ToolExecutionManager(event_bus, tool_registry)

        # Components
        self.interruption_manager = InterruptionManager(event_bus)
        self.context_manager = ContextManager(
            system_prompt_file=self.orch_settings.system_prompt_file
        )
        self.llm_provider = create_provider(self.llm_settings)
        self.stream_processor = StreamProcessor(event_bus)
        self.cancel_event = asyncio.Event()

        # Access centralized activity state
        self.activity_state = get_activity_state()

        # Register for config changes
        from core.settings import AppSettings
        AppSettings.add_change_listener(self.on_config_changed)

        super().__init__(event_bus)

    def on_config_changed(self, changes: dict) -> ReloadResult:
        """
        React to configuration changes.

        Args:
            changes: Dict with changed config sections

        Returns:
            ReloadResult with status and details
        """
        result = ReloadResult(handler_name="InteractionManager", success=True)

        try:
            if "llm" in changes:
                # LLM config changed - reload provider
                old_provider = self.llm_settings.provider
                self.llm_settings = get_settings().llm
                new_provider = self.llm_settings.provider
                self.disable_thinking = self.llm_settings.get_provider_config().disable_thinking
                self.llm_provider = create_provider(self.llm_settings)

                result.changes_applied.append(f"llm.provider: {old_provider} -> {new_provider}")
                logger.info(f"LLM provider reloaded: {new_provider}")

            if "orchestrator" in changes:
                # Orchestrator config changed
                orch_changes = changes.get("orchestrator", {})

                # Hot-reloadable: system_prompt_file
                if "system_prompt_file" in orch_changes:
                    old_file = self.orch_settings.system_prompt_file
                    self.orch_settings = get_settings().orchestrator
                    new_file = self.orch_settings.system_prompt_file

                    # Reload system prompt
                    self.context_manager.system_prompt_file = new_file
                    self.context_manager.reload_system_prompt()

                    result.changes_applied.append(f"orchestrator.system_prompt_file: {old_file} -> {new_file}")
                    logger.info(f"System prompt reloaded: {new_file}")
                else:
                    # Other orchestrator settings
                    self.orch_settings = get_settings().orchestrator
                    logger.info("Orchestrator settings updated")

                # Restart-required: host/port
                if "host" in orch_changes or "port" in orch_changes:
                    result.restart_required.append("orchestrator.host/port requires Orchestrator restart")

        except Exception as e:
            result.success = False
            result.errors.append(f"Failed to reload config: {str(e)}")
            logger.error(f"Config reload error: {e}", exc_info=True)

        return result

    def _register_handlers(self):
        self.event_bus.subscribe(EventType.INPUT_RECEIVED.value, self.on_input_received)
        self.event_bus.subscribe(EventType.LLM_CANCELLED.value, self.on_cancel)
        if self.tool_registry:
            self.event_bus.subscribe(EventType.TOOL_INTERPRETATION_REQUEST.value, self.on_tool_interpretation_request)

    async def on_cancel(self, event: Event):
        """Handle cancellation from any source (speech start, hotkey, UI)."""
        self.cancel_event.set()

    async def on_input_received(self, event: Event):
        """
        Handle input from queue consumer: User/Chat -> LLM -> TTS.

        This replaces the old on_transcript handler and works with all input sources.
        """
        raw_text = event.data.get("text")
        source = event.data.get("source", "unknown")
        priority = event.data.get("priority", 999)
        images = event.data.get("images")  # Optional images field

        if not raw_text:
            return

        if not self.activity_state.state.listening and source == "voice":
            self.logger.info("Not listening, skipping voice")
            return

        # Check interruption state and format input accordingly
        was_interrupted = self.interruption_manager.was_interrupted()
        text = self.context_manager.format_input(raw_text, source, images=images, was_interrupted=was_interrupted)

        # Clear interruption flag after handling
        if was_interrupted:
            self.interruption_manager.clear_interrupted()

        self.logger.info(f"{source.capitalize()}: {text}")
        self.cancel_event.clear()

        # Update activity: responding (transcribing already done for voice)
        await self.activity_state.update({"transcribing": False, "responding": True})

        # Publish history update for user message
        await publish_history_updated(self.event_bus)

        await self.event_bus.publish(Event(EventType.LLM_REQUEST.value))

        # Prepare context (includes images for current message)
        context = self.context_manager.format_context_for_llm(text, images=images)

        try:
            # Get tools schema if tool registry is available
            tools_schema = None
            if self.tool_registry:
                tools_schema = self.tool_registry.get_tools_schema()

            # Generate stream from LLM
            stream = self.llm_provider.generate_stream(
                messages=context["messages"],
                system_prompt=context.get("system_prompt"),
                tools=tools_schema  # Always pass (None or empty list if no tools)
            )

            # Process stream with tag routing
            result = await self.stream_processor.process_response(
                stream,
                self.cancel_event,
                self.disable_thinking
            )

            if not self.cancel_event.is_set():
                # Check if result contains tool calls
                if result.get("has_tool_calls"):
                    # Tool calls detected - publish event and skip history update
                    self.logger.info(f"Tool calls detected: {[tc for tc in result['tool_calls']]}")
                    await self.activity_state.update({"responding": False})
                    await self.event_bus.publish(Event(
                        EventType.TOOL_CALL_REQUESTED.value,
                        {"tool_calls": result["tool_calls"]}
                    ))
                    # Don't add to history yet - wait for interpretation
                else:
                    # Normal response without tool calls
                    await self.activity_state.update({"responding": False})

                    # Save to history
                    self.context_manager.add_assistant_message(result["text"])
                    await self.event_bus.publish(Event(EventType.LLM_RESPONSE_DONE.value, self.llm_provider.last_token_count))

                    # Publish history update for assistant message
                    await publish_history_updated(self.event_bus)
            else:
                # Cancelled - reset activity states
                await self.activity_state.update({"responding": False, "synthesizing": False, "playing": False})

        except Exception as e:
            self.logger.error(f"LLM Error: {e}", exc_info=True)
            # Reset activity states on error
            await self.activity_state.update({"responding": False, "synthesizing": False, "playing": False, "executing_tools": False})

    async def on_tool_interpretation_request(self, event: Event):
        """
        Handle tool interpretation request.

        Send tool results back to LLM for interpretation and natural language response.

        Args:
            event: Event with data containing:
                - tool_calls: List of tool call dicts
                - tool_results: List of tool results
        """
        tool_calls = event.data.get("tool_calls", [])
        tool_results = event.data.get("tool_results", [])

        self.logger.info("Requesting LLM interpretation of tool results")

        # Clear cancel event for new LLM call
        self.cancel_event.clear()

        # Update activity state
        await self.activity_state.update({"responding": True})

        try:
            # Get messages with tool results formatted for LLM
            messages = self.context_manager.get_messages_with_tool_results(
                tool_calls, tool_results
            )

            # Request LLM interpretation (no tools in this call)
            stream = self.llm_provider.generate_stream(
                messages=messages,
                system_prompt=None  # System prompt already in messages
            )

            # Process interpretation stream
            result = await self.stream_processor.process_response(
                stream,
                self.cancel_event,
                self.disable_thinking
            )

            if not self.cancel_event.is_set():
                # Update activity state
                await self.activity_state.update({"responding": False})

                # Add interpretation to history
                self.context_manager.add_assistant_message(result["text"])

                # Publish completion event
                await self.event_bus.publish(Event(
                    EventType.LLM_RESPONSE_DONE.value,
                    self.llm_provider.last_token_count
                ))

                # Publish history update
                await publish_history_updated(self.event_bus)

                self.logger.info("Tool interpretation completed")
            else:
                # Cancelled during interpretation
                await self.activity_state.update({"responding": False, "synthesizing": False, "playing": False})
                self.logger.info("Tool interpretation cancelled")

        except Exception as e:
            self.logger.error(f"Tool interpretation error: {e}", exc_info=True)
            # Reset activity states on error
            await self.activity_state.update({"responding": False, "synthesizing": False, "playing": False})

