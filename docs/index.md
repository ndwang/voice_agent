# Voice Agent

Welcome to the Voice Agent documentation. This is an event-driven, multi-service voice agent system that integrates speech-to-text (STT), language models (LLM), and text-to-speech (TTS) to create an intelligent voice assistant.

## Features

- **Microservices Architecture**: Decoupled services for STT, TTS, and Orchestration.
- **Event-Driven**: Internal asynchronous event bus for low-latency interaction.
- **Real-Time Streaming**: Streaming support for STT transcripts, LLM tokens, and TTS audio.
- **Provider System**: Extensible support for multiple AI providers (Ollama, Gemini, FunASR, Edge-TTS, etc.).
- **Voice Activity Detection (VAD)**: Smart microphone capture that only streams when you speak.
- **Interruption Support**: Seamlessly interrupt the agent while it's speaking.

## Quick Start

### 1. Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended for dependency management)
- CUDA-capable GPU (recommended)
- `ffmpeg` (required for Edge TTS)

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/ndwang/voice_agent.git
cd voice_agent

# Install dependencies
uv sync
```

*Note: The default setup assumes CUDA 12.6. Optional components like **blivedm**, **ChatTTS**, **Genie TTS**, and **Edge-TTS** require separate installation or system setup. See the [Installation Guide](guide/installation.md) for details.*

### 3. Run All Services

The easiest way to start the system is using the provided script:

```bash
uv run scripts/start_services.py
```

This will launch:
1. **STT Service** (Port 8001)
2. **TTS Service** (Port 8003)
3. **Orchestrator** (Port 8000)

### 4. Talk to the Agent

Once the services are running:
- **Speak**: The agent will listen and transcribe your voice.
- **Toggle Listening**: Use `Ctrl+Shift+L` to enable/disable the microphone.
- **Web UI**: Access the control panel at `http://localhost:8000/ui`.

---

See the [Architecture](architecture/overview.md) section for a deep dive into how it works.

