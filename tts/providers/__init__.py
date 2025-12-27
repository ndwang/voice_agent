"""
TTS Providers

Export all available TTS providers.
"""
from tts.providers.edge_tts import EdgeTTSProvider
from tts.providers.chattts import ChatTTSProvider
from tts.providers.elevenlabs import ElevenLabsProvider
from tts.providers.genie_tts import GenieTTSProvider

__all__ = ["EdgeTTSProvider", "ChatTTSProvider", "ElevenLabsProvider", "GenieTTSProvider"]

