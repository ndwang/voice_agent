import pytest
import asyncio
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from stt.manager import STTManager

class MockSegment:
    def __init__(self, text):
        self.text = text

@pytest.fixture
def mock_config():
    with patch("stt.manager.get_config") as mock:
        # Default config values
        mock.side_effect = lambda section, key, *args, **kwargs: {
            "language_code": "zh",
            "sample_rate": 16000,
            "interim_transcript_min_samples": 100,
            "provider": "faster-whisper",
            "providers": {
                "faster-whisper": {"model_path": "test", "device": "cpu"},
                "funasr": {}
            }
        }.get(key, kwargs.get("default"))
        yield mock

@pytest.fixture
def mock_provider():
    with patch("stt.manager.FasterWhisperProvider") as mock_cls:
        provider = MagicMock()
        # Mock transcribe return value
        provider.transcribe.return_value = ([MockSegment("test transcript")], {})
        mock_cls.return_value = provider
        yield provider

@pytest.mark.asyncio
async def test_stt_manager_init(mock_config, mock_provider):
    manager = STTManager()
    assert manager.provider is not None
    assert manager.language_code == "zh"

@pytest.mark.asyncio
async def test_client_management(mock_config, mock_provider):
    manager = STTManager()
    mock_ws = AsyncMock()
    
    await manager.add_client(mock_ws)
    assert len(manager.connected_clients) == 1
    
    await manager.remove_client(mock_ws)
    assert len(manager.connected_clients) == 0

@pytest.mark.asyncio
async def test_broadcast(mock_config, mock_provider):
    manager = STTManager()
    mock_ws = AsyncMock()
    await manager.add_client(mock_ws)
    
    msg = {"type": "test"}
    await manager.broadcast(msg)
    
    mock_ws.send_text.assert_called_once()
    assert '"type": "test"' in mock_ws.send_text.call_args[0][0]

@pytest.mark.asyncio
async def test_process_audio_chunk_accumulate(mock_config, mock_provider):
    manager = STTManager()
    
    audio_buffer = np.array([], dtype=np.float32)
    chunk = np.zeros(100, dtype=np.float32).tobytes()
    last_time = 0
    
    new_buffer, new_time = await manager.process_audio_chunk(
        chunk, audio_buffer, last_time
    )
    
    assert new_buffer.size == 100
    # Provider should NOT be called for small chunk
    manager.provider.transcribe.assert_not_called()


