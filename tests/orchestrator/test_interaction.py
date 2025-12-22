import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from core.event_bus import EventBus, Event
from orchestrator.events import EventType
from orchestrator.managers.interaction_manager import InteractionManager

@pytest.fixture
def event_bus():
    return EventBus()

@pytest.fixture
def mock_config():
    with patch("orchestrator.managers.interaction_manager.get_config") as mock:
        # Smart mock that returns reasonable defaults based on the key/path
        def side_effect(*args, **kwargs):
            if "provider" in args: return "ollama"
            if "timeout" in args: return 300.0
            if "disable_thinking" in args: return False
            if "model" in args: return "llama3"
            if "base_url" in args: return "http://localhost:11434"
            return kwargs.get('default')
            
        mock.side_effect = side_effect
        yield mock

@pytest.fixture
def mock_llm_provider():
    with patch("orchestrator.managers.interaction_manager.OllamaProvider") as mock_cls:
        provider = AsyncMock()
        # Mock streaming response
        async def mock_stream(**kwargs):
            yield "Hello "
            yield "world"
            yield "."
        provider.generate_stream = mock_stream
        mock_cls.return_value = provider
        yield provider

@pytest.mark.asyncio
async def test_interaction_flow(event_bus, mock_config, mock_llm_provider):
    manager = InteractionManager(event_bus)
    
    # Mock publishing
    published_events = []
    original_publish = event_bus.publish
    
    async def mock_publish(event):
        published_events.append(event)
        await original_publish(event)
        
    event_bus.publish = mock_publish
    
    # Simulate User Transcript
    transcript_event = Event(EventType.TRANSCRIPT_FINAL.value, {"text": "Hi"})
    await event_bus.publish(transcript_event)
    
    # Allow async tasks to run
    await asyncio.sleep(0.1)
    
    # Verification
    event_names = [e.name for e in published_events]
    
    # 1. Should trigger LLM Request
    assert EventType.LLM_REQUEST.value in event_names
    
    # 2. Should receive Tokens
    assert EventType.LLM_TOKEN.value in event_names
    
    # 3. Should trigger TTS Request (since "Hello world." is a sentence)
    assert EventType.TTS_REQUEST.value in event_names
    
    # 4. Should finish
    assert EventType.LLM_RESPONSE_DONE.value in event_names

@pytest.mark.asyncio
async def test_thinking_tag_filtering(event_bus, mock_config):
    # Mock config to disable thinking
    def side_effect(*args, **kwargs):
        if "disable_thinking" in args: return True
        if "provider" in args: return "ollama"
        if "timeout" in args: return 300.0
        if "model" in args: return "llama3"
        if "base_url" in args: return "http://localhost:11434"
        return kwargs.get('default')
    mock_config.side_effect = side_effect
    
    # Mock LLM provider to return a response with <think> tags
    with patch("orchestrator.managers.interaction_manager.OllamaProvider") as mock_cls:
        provider = AsyncMock()
        async def mock_stream(**kwargs):
            yield "Hello "
            yield "<think>internal reasoning</think>"
            yield "world."
        provider.generate_stream = mock_stream
        mock_cls.return_value = provider
        
        manager = InteractionManager(event_bus)
        
        # Track published events
        tokens = []
        tts_requests = []
        
        async def mock_publish(event):
            if event.name == EventType.LLM_TOKEN.value:
                tokens.append(event.data.get("token"))
            elif event.name == EventType.TTS_REQUEST.value:
                tts_requests.append(event.data.get("text"))
                
        event_bus.publish = mock_publish
        
        # Simulate User Transcript
        await manager.on_transcript(Event(EventType.TRANSCRIPT_FINAL.value, {"text": "Hi"}))
        
        # Verification
        # 1. "internal reasoning" should NOT be in tokens
        full_tokens = "".join(tokens)
        assert "internal reasoning" not in full_tokens
        assert "<think>" not in full_tokens
        assert "</think>" not in full_tokens
        
        # 2. TTS request should not contain thinking
        full_tts = " ".join(tts_requests)
        assert "internal reasoning" not in full_tts
        assert "Hello world." in full_tts

