import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch
from tts.engine import TTSEngine

@pytest.fixture
def mock_config():
    with patch("tts.engine.get_config") as mock:
        mock.side_effect = lambda section, key, *args, **kwargs: {
            "output_sample_rate": 16000,
            "provider": "edge-tts",
            "providers": {
                "edge-tts": {"voice": "test-voice", "rate": "+0%", "pitch": "+0Hz"},
                "chattts": {}
            }
        }.get(key, kwargs.get("default"))
        yield mock

@pytest.fixture
def mock_provider():
    with patch("tts.engine.EdgeTTSProvider") as mock_cls:
        provider = AsyncMock()
        provider.list_voices.return_value = [{"Name": "Test Voice"}]
        # Mock streaming synthesis
        async def mock_stream(text, **kwargs):
            yield b"chunk1"
            yield b"chunk2"
        provider.synthesize_stream = mock_stream
        
        mock_cls.return_value = provider
        yield provider

@pytest.mark.asyncio
async def test_tts_manager_init(mock_config, mock_provider):
    manager = TTSEngine()
    assert manager.provider_name == "edge-tts"
    assert manager.output_sample_rate == 16000

@pytest.mark.asyncio
async def test_websocket_text_handling(mock_config, mock_provider):
    manager = TTSEngine()
    mock_ws = AsyncMock()
    
    # Simulate a text message
    text_data = {"type": "text", "text": "Hello world"}
    
    # Mock send_bytes to verify streaming
    mock_ws.send_bytes = AsyncMock()
    
    await manager._handle_text_message(
        mock_ws, text_data, [], {}
    )
    
    # Verify status ack was sent
    mock_ws.send_text.assert_any_call(json.dumps({"type": "status", "message": "received"}))
    
    # Verify chunks were sent
    assert mock_ws.send_bytes.call_count == 2  # chunk1, chunk2
    
    # Verify done message
    mock_ws.send_text.assert_any_call(json.dumps({"type": "done"}))

@pytest.mark.asyncio
async def test_websocket_config_update(mock_config, mock_provider):
    manager = TTSEngine()
    params = {"voice": "old-voice"}
    
    data = {"type": "config", "voice": "new-voice", "rate": "+50%"}
    manager._update_params(data, params)
    
    assert params["voice"] == "new-voice"
    assert params["rate"] == "+50%"


