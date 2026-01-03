import pytest
import asyncio
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from stt.manager import STTManager

class MockSegment:
    def __init__(self, text):
        self.text = text

@pytest.fixture
def mock_settings():
    """Mock the settings for STT tests."""
    with patch("stt.manager.get_settings") as mock:
        # Create mock settings object
        settings_mock = MagicMock()
        settings_mock.stt.provider = "faster-whisper"
        settings_mock.stt.language_code = "zh"
        settings_mock.stt.sample_rate = 16000
        settings_mock.stt.get_provider_config.return_value = MagicMock(
            model_path="test",
            device="cpu"
        )
        mock.return_value = settings_mock
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
async def test_stt_manager_init(mock_settings, mock_provider):
    """Test that STTManager initializes correctly with settings."""
    manager = STTManager()
    assert manager.provider is not None

@pytest.mark.asyncio
async def test_client_management(mock_settings, mock_provider):
    """Test adding and removing WebSocket clients."""
    manager = STTManager()
    mock_ws = AsyncMock()

    await manager.add_client(mock_ws)
    assert mock_ws in manager.connected_clients

    await manager.remove_client(mock_ws)
    assert mock_ws not in manager.connected_clients

@pytest.mark.asyncio
async def test_broadcast(mock_settings, mock_provider):
    """Test broadcasting messages to connected clients."""
    manager = STTManager()
    mock_ws = AsyncMock()
    await manager.add_client(mock_ws)

    msg = {"type": "test"}
    await manager.broadcast(msg)

    # Give background task time to execute
    await asyncio.sleep(0.1)

    mock_ws.send_text.assert_called_once()
    assert '"type": "test"' in mock_ws.send_text.call_args[0][0]

@pytest.mark.asyncio
async def test_process_audio_chunk_enqueue(mock_settings, mock_provider):
    """Test that process_audio_chunk validates and enqueues audio chunks."""
    manager = STTManager()
    mock_ws = AsyncMock()

    # Add client
    await manager.add_client(mock_ws)

    # Create valid audio chunk (float32 format, 1 channel)
    chunk = np.zeros(100, dtype=np.float32).tobytes()

    # Process chunk (should enqueue without error)
    await manager.process_audio_chunk(mock_ws, chunk)

    # Verify chunk was enqueued (state should exist)
    async with manager.client_states_lock:
        state = manager.client_states.get(mock_ws)
        assert state is not None


