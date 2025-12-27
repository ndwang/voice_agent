import asyncio
from typing import Optional, List, Dict
from core.event_bus import EventBus, Event
from core.logging import get_logger
from core.config import get_config
from orchestrator.events import EventType
from orchestrator.managers.base import BaseManager
from orchestrator.managers.context_manager import ContextManager
from orchestrator.core.constants import SENTENCE_END_PATTERN
from orchestrator.utils.event_helpers import (
    publish_activity,
    publish_history_updated
)
from orchestrator.utils.text_processing import (
    filter_thinking_tags,
    filter_thinking_tags_final,
    LLMStreamParser
)
from llm.providers import GeminiProvider, OllamaProvider

logger = get_logger(__name__)

class InteractionManager(BaseManager):
    """
    Orchestrates the conversation flow:
    Event(Transcript) -> Context -> LLM -> Event(Token) -> Buffer -> TTS -> Event(Audio)
    """

    def __init__(self, event_bus: EventBus):
        # State
        self.listening_enabled = True
        
        # Thinking filter state
        provider_name = get_config("llm", "provider", default="ollama")
        self.disable_thinking = get_config("llm", "providers", provider_name, "disable_thinking", default=False)
        
        # Get system prompt file from config
        system_prompt_file = get_config("orchestrator", "system_prompt_file", default=None)
        
        # Components
        self.context_manager = ContextManager(system_prompt_file=system_prompt_file)
        self.llm_provider = self._init_llm()
        self.cancel_event = asyncio.Event()
        
        super().__init__(event_bus)

    def _init_llm(self):
        provider = get_config("llm", "provider", default="ollama")
        if provider == "gemini":
            generation_config = get_config("llm", "providers", "gemini", "generation_config", default={})
            return GeminiProvider(
                model=get_config("llm", "providers", "gemini", "model"),
                api_key=get_config("llm", "providers", "gemini", "api_key"),
                generation_config=generation_config if generation_config else None
            )
        else:
            generation_config = get_config("llm", "providers", "ollama", "generation_config", default={})
            return OllamaProvider(
                model=get_config("llm", "providers", "ollama", "model"),
                base_url=get_config("llm", "providers", "ollama", "base_url"),
                timeout=float(get_config("llm", "providers", "ollama", "timeout", default=300)),
                disable_thinking=get_config("llm", "providers", "ollama", "disable_thinking", default=False),
                generation_config=generation_config if generation_config else None
            )

    def _register_handlers(self):
        self.event_bus.subscribe(EventType.TRANSCRIPT_FINAL.value, self.on_transcript)
        self.event_bus.subscribe(EventType.SPEECH_START.value, self.on_interruption)

    async def on_interruption(self, event: Event):
        """User started speaking, cancel current generation."""
        self.cancel_event.set()
        # Publish cancelled event so TTS/Audio can stop
        await self.event_bus.publish(Event(EventType.LLM_CANCELLED.value))

    async def on_transcript(self, event: Event):
        """Handle final transcript: User -> LLM -> TTS."""
        if not self.listening_enabled:
            return
            
        text = event.data.get("text")
        if not text:
            return

        self.logger.info(f"User: {text}")
        self.context_manager.add_user_message(text)
        self.cancel_event.clear()
        
        # Publish activity: transcribing is done, now responding
        await publish_activity(self.event_bus, {"transcribing": False, "responding": True})
        
        # Publish history update for user message
        await publish_history_updated(self.event_bus)
        
        await self.event_bus.publish(Event(EventType.LLM_REQUEST.value))
        
        # Prepare context
        context = self.context_manager.format_context_for_llm(text)
        
        # Stream response state
        full_response = ""
        
        # Setup parser callbacks
        async def default_callback(text: str):
            """Handle untagged content - send to TTS."""
            if not self.cancel_event.is_set():
                await self.event_bus.publish(Event(EventType.TTS_REQUEST.value, {"text": text}))
        
        # Configure parser
        tag_configs = []
        if self.disable_thinking:
            # Discard redacted_reasoning tags (no callback)
            tag_configs.append({"name": "redacted_reasoning"})
        
        # Create parser
        parser = LLMStreamParser(tag_configs, default_callback=default_callback)
        
        try:
            async for token in self.llm_provider.generate_stream(
                messages=context["messages"],
                system_prompt=context.get("system_prompt")
            ):
                if self.cancel_event.is_set():
                    break
                
                if not token:
                    continue

                full_response += token
                
                # Publish token for UI display
                await self.event_bus.publish(Event(EventType.LLM_TOKEN.value, {"token": token}))
                
                # Process token through parser
                await parser.process_token(token)

            # Final flush of parser buffers
            await parser.finalize()
            
            if not self.cancel_event.is_set():
                # Publish activity: responding done (synthesizing will be set by TTS handler)
                await publish_activity(self.event_bus, {"responding": False})

            # Save to history
            if not self.cancel_event.is_set():
                # Filter full_response for history if needed
                history_response = full_response
                if self.disable_thinking:
                    history_response = filter_thinking_tags_final(full_response)
                
                self.context_manager.add_assistant_message(history_response)
                await self.event_bus.publish(Event(EventType.LLM_RESPONSE_DONE.value))
                
                # Publish history update for assistant message
                await publish_history_updated(self.event_bus)
            else:
                # Cancelled - reset activity states
                await publish_activity(self.event_bus, {"responding": False, "synthesizing": False, "playing": False})

        except Exception as e:
            self.logger.error(f"LLM Error: {e}", exc_info=True)
            # Reset activity states on error
            await publish_activity(self.event_bus, {"responding": False, "synthesizing": False, "playing": False})

    async def _process_tokens(self, tokens: str, current_sentence: str) -> str:
        """
        Processes a string of tokens, publishing them and updating sentence buffer.
        Returns the updated sentence buffer.
        """
        await self.event_bus.publish(Event(EventType.LLM_TOKEN.value, {"token": tokens}))
        
        current_sentence += tokens
        
        # Check for sentence completion
        match = SENTENCE_END_PATTERN.search(current_sentence)
        while match:
            end_pos = match.end()
            sentence_to_tts = current_sentence[:end_pos].strip()
            
            if sentence_to_tts:
                await self.event_bus.publish(Event(EventType.TTS_REQUEST.value, {"text": sentence_to_tts}))
            
            current_sentence = current_sentence[end_pos:]
            match = SENTENCE_END_PATTERN.search(current_sentence)
            
        return current_sentence

