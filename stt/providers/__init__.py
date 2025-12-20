"""
STT Providers

Provider implementations for different STT backends.
"""
from stt.providers.faster_whisper import FasterWhisperProvider
from stt.providers.funasr import FunASRProvider

__all__ = ["FasterWhisperProvider", "FunASRProvider"]

