import asyncio
from typing import Optional
from core.event_bus import EventBus, Event
from core.logging import get_logger
from core.settings import get_settings, LLMSettings, OrchestratorSettings
from orchestrator.events import EventType
from orchestrator.managers.base import BaseManager
from orchestrator.managers.context_manager import ContextManager
from orchestrator.managers.interruption_manager import InterruptionManager
from orchestrator.core.models import SystemState
from orchestrator.utils.event_helpers import (
    publish_activity,
    publish_history_updated
)
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
        llm_settings: Optional[LLMSettings] = None,
        orch_settings: Optional[OrchestratorSettings] = None
    ):
        # Use injected settings if provided, otherwise use global
        app_settings = get_settings()
        self.llm_settings = llm_settings or app_settings.llm
        self.orch_settings = orch_settings or app_settings.orchestrator

        # Type-safe access with IDE autocomplete!
        self.disable_thinking = self.llm_settings.get_provider_config().disable_thinking

        # Components
        self.context_manager = ContextManager(
            system_prompt_file=self.orch_settings.system_prompt_file
        )
        self.llm_provider = create_provider(self.llm_settings)
        self.interruption_manager = InterruptionManager(event_bus)
        self.stream_processor = StreamProcessor(event_bus)
        self.cancel_event = asyncio.Event()

        # Systematic State Tracking
        self.activity_state = SystemState()

        # Register for config changes
        from core.settings import AppSettings
        AppSettings.add_change_listener(self.on_config_changed)

        super().__init__(event_bus)

    def on_config_changed(self, changes: dict):
        """
        React to configuration changes.

        Args:
            changes: Dict with changed config sections
        """
        if "llm" in changes:
            # LLM config changed - reload provider
            self.llm_settings = get_settings().llm
            self.disable_thinking = self.llm_settings.get_provider_config().disable_thinking
            self.llm_provider = create_provider(self.llm_settings)
            logger.info(f"LLM provider reloaded: {self.llm_settings.provider}")

        if "orchestrator" in changes:
            # Orchestrator config changed
            self.orch_settings = get_settings().orchestrator
            logger.info("Orchestrator settings updated")

    def _register_handlers(self):
        self.event_bus.subscribe(EventType.TRANSCRIPT_FINAL.value, self.on_transcript)
        self.event_bus.subscribe(EventType.SPEECH_START.value, self.on_interruption)
        self.event_bus.subscribe(EventType.LLM_CANCELLED.value, self.on_cancel)
        # Unified state tracking
        self.event_bus.subscribe(EventType.STATE_CHANGED.value, self.on_state_changed)
        self.event_bus.subscribe(EventType.LISTENING_STATE_CHANGED.value, self.on_listening_state_changed)

    async def on_state_changed(self, event: Event):
        """Update local state snapshot from system state changes."""
        new_state = event.data.get("state", {})
        self.activity_state.update(new_state)

    async def on_listening_state_changed(self, event: Event):
        """Update local listening state."""
        self.activity_state.listening = event.data.get("enabled", True)

    async def on_interruption(self, event: Event):
        """User started speaking, publish cancel event if busy."""
        await self.interruption_manager.check_and_cancel(self.activity_state)

    async def on_cancel(self, event: Event):
        """Handle cancellation from any source (speech start, hotkey, UI)."""
        self.cancel_event.set()

        # Point 1: If we finished generating but are still playing, mark as interrupted
        # if not self.activity_state.responding and self.activity_state.playing:
        #     last_msg = self.context_manager.get_last_message()
        #     if last_msg and last_msg["role"] == "assistant":
        #         if "[interrupted]" not in last_msg["content"]:
        #             self.logger.info("Marking assistant response as interrupted in history")
        #             new_content = last_msg["content"].strip() + " [interrupted]"
        #             self.context_manager.update_last_message(new_content, role="assistant")
        #             await publish_history_updated(self.event_bus)

        # Point 2: If we are still responding, the next transcript should be concatenated
        self.interruption_manager.mark_interrupted(self.activity_state.responding)

    async def on_transcript(self, event: Event):
        """Handle final transcript: User -> LLM -> TTS."""
        if not self.activity_state.listening:
            return

        raw_text = event.data.get("text")
        if not raw_text:
            return

        # Handle interruption concatenation and add to context
        text = self.interruption_manager.handle_transcript_history(raw_text, self.context_manager)

        self.logger.info(f"User: {text}")
        self.cancel_event.clear()

        # Publish activity: transcribing is done, now responding
        await publish_activity(self.event_bus, {"transcribing": False, "responding": True})

        # Publish history update for user message
        await publish_history_updated(self.event_bus)

        await self.event_bus.publish(Event(EventType.LLM_REQUEST.value))

        # Prepare context
        context = self.context_manager.format_context_for_llm(text)

        try:
            # Generate stream from LLM
            stream = self.llm_provider.generate_stream(
                messages=context["messages"],
                system_prompt=context.get("system_prompt")
            )

            # Process stream with tag routing
            history_response = await self.stream_processor.process_response(
                stream,
                self.cancel_event,
                self.disable_thinking
            )

            if not self.cancel_event.is_set():
                # Publish activity: responding done (synthesizing will be set by TTS handler)
                await publish_activity(self.event_bus, {"responding": False})

                # Save to history
                self.context_manager.add_assistant_message(history_response)
                await self.event_bus.publish(Event(EventType.LLM_RESPONSE_DONE.value, self.llm_provider.last_token_count))

                # Publish history update for assistant message
                await publish_history_updated(self.event_bus)
            else:
                # Cancelled - reset activity states
                await publish_activity(self.event_bus, {"responding": False, "synthesizing": False, "playing": False})

        except Exception as e:
            self.logger.error(f"LLM Error: {e}", exc_info=True)
            # Reset activity states on error
            await publish_activity(self.event_bus, {"responding": False, "synthesizing": False, "playing": False})

