# Voice Agent

An event-driven, multi-service voice agent system that integrates speech-to-text (STT), language models (LLM), text-to-speech (TTS), and optical character recognition (OCR) to create an intelligent voice assistant capable of understanding speech, generating contextual responses, and reading screen content.

## Overview

The Voice Agent is a modular, event-driven system that orchestrates multiple AI services to enable natural voice interactions. It can:

- **Listen** to speech input via microphone with Voice Activity Detection (VAD)
- **Transcribe** speech to text in real-time with interim and final transcripts
- **Capture** context from screen content (OCR) for more informed responses
- **Generate** responses using language models with streaming support
- **Speak** responses using text-to-speech synthesis with real-time audio streaming
- **Manage** conversation history and context automatically
- **Control** via hotkeys and web UI

The system is designed with a microservices architecture, where each component runs independently and communicates via WebSocket and HTTP protocols. The orchestrator uses an internal event bus for decoupled, asynchronous communication between components.

## Architecture

The system consists of five main services plus an orchestrator:

### Services

1. **STT Service** (Port 8001)
   - Speech-to-Text transcription with multiple provider support
   - Real-time audio processing with WebSocket streaming
   - Supports interim and final transcripts
   - Broadcasts transcripts to all connected clients
   - **Providers**: Faster Whisper, FunASR

2. **TTS Service** (Port 8003)
   - Text-to-Speech synthesis with streaming and non-streaming support
   - Real-time audio generation via WebSocket streaming
   - Complete audio generation via REST API
   - **Providers**: Edge TTS, ChatTTS

3. **OCR Service** (Port 8004)
   - Optical Character Recognition for screen monitoring
   - Captures and extracts text from screen regions
   - Provides context to the LLM for more informed responses
   - On-demand text retrieval via HTTP API

4. **Orchestrator** (Port 8000)
   - Main coordinator that connects all services
   - Event-driven architecture with internal event bus
   - Manages conversation context and history
   - Handles the complete voice interaction pipeline
   - Provides REST API and WebSocket endpoints for control and monitoring
   - **LLM Providers**: Gemini, Ollama (integrated directly, no separate service)

5. **Audio Driver**
   - Captures microphone input with VAD
   - Streams audio to STT service
   - Manages audio playback
   - Polls orchestrator for listening state

### Event-Driven Architecture

The orchestrator uses an internal event bus for decoupled communication:

```
┌─────────────┐
│Audio Driver │───WebSocket (audio)───▶┌─────────────┐
└─────────────┘                        │ STT Service │
                                       │  (Port 8001) │
                                       └──────┬───────┘
                                              │
                                              │ WebSocket (broadcast transcripts)
                                              ▼
                                       ┌──────────────┐
                                       │ Orchestrator │
                                       │  (Port 8000) │
                                       └──────┬───────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    │                         │                         │
                    ▼                         ▼                         ▼
            ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
            │ LLM Provider │          │ TTS Service  │          │ OCR Service  │
            │ (Gemini/     │          │  (Port 8003) │          │  (Port 8004) │
            │  Ollama)     │          └──────────────┘          └──────────────┘
            └──────────────┘
```

**Internal Event Flow:**
- `STTSource` receives transcripts → publishes `TRANSCRIPT_FINAL` events
- `InteractionManager` subscribes to transcripts → generates LLM response → publishes `LLM_TOKEN` events
- `TTSManager` subscribes to `TTS_REQUEST` events → streams to TTS service → receives audio → publishes `TTS_AUDIO_CHUNK` events
- `AudioPlayer` plays audio chunks → publishes `AUDIO_PLAYING`/`AUDIO_STOPPED` events
- `SubtitleManager` and `LatencyTracker` monitor events for UI/metrics

### Key Components

**Orchestrator Managers:**
- **InteractionManager**: Handles conversation flow (transcript → LLM → TTS)
- **TTSManager**: Manages TTS service connection and audio playback
- **SubtitleManager**: Manages subtitle display (OBS integration)
- **LatencyTracker**: Tracks latency metrics for performance monitoring
- **ContextManager**: Manages conversation history, OCR context, and system prompts

**Orchestrator Sources:**
- **STTSource**: WebSocket client that receives transcripts from STT service

**Core Infrastructure:**
- **EventBus**: Asynchronous pub/sub event system
- **ConfigLoader**: Centralized YAML-based configuration
- **ToolRegistry**: Extensible tool/function calling system

## Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended for optimal performance)
- Microphone for voice input
- Speakers/headphones for audio output
- `ffmpeg` (required for Edge TTS audio format conversion)

## Installation

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd voice_agent
   ```

2. **Install dependencies using uv**:
   ```bash
   uv sync
   ```

   This will:
   - Create a virtual environment (`.venv`)
   - Install all required dependencies
   - Configure PyTorch with CUDA support (on Linux/Windows)

3. **Activate the virtual environment**:
   - Windows (PowerShell):
     ```powershell
     .venv\Scripts\activate
     ```
   - Linux/macOS:
     ```bash
     source .venv/bin/activate
     ```

4. **Configure the system**:
   Edit `config.yaml` to configure services, providers, and settings. See [Configuration](#configuration) section for details.

## Quick Start

### Start All Services

The easiest way to start all services is using the provided script:

```bash
python scripts/start_services.py
```

This will start all services in separate windows/terminals:
- STT Service (Port 8001)
- TTS Service (Port 8003)
- Orchestrator (Port 8000)
- Audio Driver

### Start Services Individually

You can also start services individually:

```bash
# Terminal 1: STT Service
python -m stt.server

# Terminal 2: TTS Service
python -m tts.server

# Terminal 3: Orchestrator
python -m orchestrator.server

# Terminal 4: Audio Driver
python -m audio.audio_driver
```

### Verify Services

Check that all services are running:

```bash
python scripts/check_services.py
```

Or manually check the orchestrator health endpoint:

```bash
curl http://localhost:8000/health
```

### Stop All Services

```bash
python scripts/stop_services.py
```

## Usage

Once all services are running:

1. **Start speaking** - The audio driver will capture your voice (VAD-enabled)
2. **See interim transcripts** - The STT service provides real-time interim transcripts
3. **Receive final transcript** - When speech ends, a final transcript is generated
4. **Get LLM response** - The orchestrator processes your input through the LLM and generates a streaming response
5. **Hear the response** - The TTS service synthesizes and plays the audio response in real-time

The system maintains conversation history and can incorporate OCR context from your screen for more contextual responses.

### Web UI

Access the web-based control panel:

```
http://localhost:8000/ui
```

The UI provides:
- Real-time transcript display
- LLM token streaming visualization
- Listening state toggle
- Conversation history
- System prompt editing
- Hotkey configuration
- Activity status indicators

### Hotkeys

Default hotkey to toggle listening: `Ctrl+Shift+L`

Configure hotkeys via the web UI or `config.yaml`.

### OCR Context

The OCR service monitors screen regions and extracts text. This text is automatically included in the conversation context, allowing the agent to respond to questions about what's on your screen.

To fetch OCR texts manually:

```bash
curl http://localhost:8000/ocr/texts
```

## Configuration

All configuration is managed through `config.yaml`. The system uses a centralized configuration loader that all services access.

### Service Configuration

```yaml
orchestrator:
  host: "0.0.0.0"
  port: 8000
  hotkeys:
    toggle_listening: "ctrl+shift+l"

services:
  stt_websocket_url: "ws://localhost:8001/ws/transcribe"
  tts_websocket_url: "ws://localhost:8003/synthesize/stream"
  ocr_base_url: "http://localhost:8004"
```

### STT Configuration

```yaml
stt:
  provider: "funasr"  # Options: faster-whisper, funasr
  language_code: "zh"
  sample_rate: 16000
  providers:
    faster-whisper:
      model_path: "faster-whisper-small"
      device: null  # null = auto-detect
    funasr:
      model_name: "FunAudioLLM/Fun-ASR-Nano-2512"
      device: null
```

### LLM Configuration

```yaml
llm:
  provider: "ollama"  # Options: gemini, ollama
  providers:
    gemini:
      model: "gemini-2.5-flash"
      api_key: ""  # Leave empty to use GEMINI_API_KEY env var
    ollama:
      model: "Qwen3-8B-Q4-8kcontext"
      base_url: "http://localhost:11434"
      disable_thinking: true  # Filter thinking tags from responses
```

**Note:** For Gemini, set the `GEMINI_API_KEY` environment variable or configure it in `config.yaml`.

### TTS Configuration

```yaml
tts:
  provider: "edge-tts"  # Options: edge-tts, chattts
  providers:
    edge-tts:
      voice: "zh-CN-XiaoyiNeural"
      rate: "+0%"
      pitch: "+0Hz"
    chattts:
      model_source: "local"  # Options: local, huggingface, custom
      device: null  # null = auto-detect
```

**Note:** Edge TTS requires `ffmpeg` to be installed on your system for audio format conversion.

### Audio Driver Configuration

```yaml
audio:
  input:
    sample_rate: 16000
    channels: 1
    device: null  # null = default device
  output:
    sample_rate: 16000
    channels: 1
    device: null
  vad_min_speech_prob: 0.5
  silence_threshold_ms: 1000
```

### OCR Configuration

```yaml
ocr:
  host: "0.0.0.0"
  port: 8004
  language: "ch"
  interval_ms: 1000
```

## Development

### Project Structure

```
voice_agent/
├── audio/               # Audio capture and playback
│   ├── audio_driver.py  # Microphone input and VAD
│   └── audio_player.py  # Audio playback
├── core/                # Core infrastructure
│   ├── config.py        # Configuration loader
│   ├── event_bus.py     # Event system
│   ├── logging.py       # Logging setup
│   └── server.py        # FastAPI app factory
├── llm/                 # Language model providers
│   ├── base.py          # LLM provider base class
│   └── providers/       # Provider implementations
│       ├── gemini.py    # Google Gemini provider
│       └── ollama.py    # Ollama provider
├── ocr/                 # Optical character recognition service
├── orchestrator/       # Main orchestration logic
│   ├── api.py           # REST API endpoints
│   ├── context_manager.py  # Conversation history and context
│   ├── events.py        # Event type definitions
│   ├── managers/        # Component managers
│   │   ├── interaction_manager.py  # Conversation flow
│   │   ├── tts_manager.py         # TTS service client
│   │   ├── subtitle_manager.py    # Subtitle display
│   │   └── latency_manager.py     # Performance tracking
│   ├── sources/         # External service sources
│   │   └── stt_source.py          # STT service client
│   ├── tools/           # Tool/function registry
│   ├── server.py        # Orchestrator server
│   └── static/          # Web UI files
├── stt/                 # Speech-to-text service
│   ├── base.py          # STT provider base class
│   ├── manager.py       # STT session management
│   ├── server.py        # STT service server
│   └── providers/       # Provider implementations
│       ├── faster_whisper.py
│       └── funasr.py
├── tts/                 # Text-to-speech service
│   ├── base.py          # TTS provider base class
│   ├── manager.py       # TTS session management
│   ├── server.py        # TTS service server
│   └── providers/       # Provider implementations
│       ├── edge_tts.py
│       └── chattts.py
├── scripts/             # Utility scripts
├── config.yaml          # Centralized configuration
└── pyproject.toml       # Project dependencies
```

### Key Components

- **`orchestrator/server.py`**: Main orchestrator server that initializes all managers and sources
- **`orchestrator/managers/interaction_manager.py`**: Handles conversation flow from transcript to LLM to TTS
- **`orchestrator/context_manager.py`**: Manages conversation history, OCR context, and system prompts with hot-reload
- **`orchestrator/sources/stt_source.py`**: WebSocket client that receives transcripts from STT service
- **`core/event_bus.py`**: Asynchronous event bus for internal communication
- **`core/config.py`**: Centralized YAML configuration loader

### Event System

The orchestrator uses an event-driven architecture. Key events:

- `input.transcript.final`: Final transcript from STT
- `input.transcript.interim`: Interim transcript from STT
- `input.speech_start`: User started speaking (interruption)
- `llm.request`: LLM generation started
- `llm.token`: LLM token generated (streaming)
- `llm.response_done`: LLM response complete
- `llm.cancelled`: LLM generation cancelled
- `tts.request`: TTS synthesis requested
- `tts.audio_chunk`: Audio chunk received from TTS
- `tts.done`: TTS synthesis complete
- `audio.playing`: Audio playback started
- `audio.stopped`: Audio playback stopped

### Testing

Individual components can be tested using scripts in the `scripts/` directory:

```bash
# Test audio player
python scripts/test_audio_player.py

# Test TTS
python scripts/test_tts.py

# Verify GPU setup
python scripts/verify_gpu_setup.py

# Check services
python scripts/check_services.py
```

## API Endpoints

### Orchestrator API

- `GET /`: Service status
- `GET /health`: Health check
- `GET /ui`: Web UI control panel
- `GET /ui/history`: Get conversation history
- `WebSocket /ui/events`: Real-time event stream for UI
- `POST /ui/cancel`: Cancel current interaction
- `GET /ui/listening/status`: Get listening state
- `POST /ui/listening/toggle`: Toggle listening state
- `POST /ui/listening/set`: Set listening state
- `GET /ui/system-prompt`: Get system prompt
- `POST /ui/system-prompt`: Update system prompt
- `GET /ui/hotkeys`: List registered hotkeys
- `GET /ui/hotkeys/{hotkey_id}`: Get hotkey configuration
- `POST /ui/hotkeys/{hotkey_id}`: Update hotkey
- `DELETE /ui/hotkeys/{hotkey_id}`: Delete hotkey
- `GET /ocr/texts`: Fetch all OCR texts

### STT Service

- `WebSocket /ws/transcribe`: Audio streaming endpoint
  - Receives: Audio bytes (float32 PCM)
  - Sends: JSON messages with `type` ("final", "interim", "speech_start") and `text`

### TTS Service

- `GET /`: Service status and provider information
- `GET /health`: Health check
- `GET /voices`: List available voices for the current provider
- `POST /synthesize`: Non-streaming synthesis (returns complete audio as binary PCM)
- `WebSocket /synthesize/stream`: Streaming synthesis
  - Receives: JSON messages with `type: "text"`, `text: "..."`, `finalize: bool`
  - Sends: Audio chunks as bytes (int16 PCM) or JSON `{"type": "done"}`

### OCR Service

- `WebSocket /monitor/stream`: Screen monitoring endpoint
- `GET /texts`: Fetch extracted texts
- `GET /texts/get`: Fetch all detected texts (used by orchestrator)

## Features

### Event-Driven Architecture
- Decoupled components communicate via events
- Easy to extend with new managers or sources
- Asynchronous, non-blocking operations

### Provider Pattern
- Extensible provider system for STT, TTS, and LLM
- Easy to add new providers
- Consistent interface across providers

### Real-Time Streaming
- STT: Interim and final transcripts
- LLM: Token-by-token streaming
- TTS: Chunk-by-chunk audio streaming

### Context Management
- Automatic conversation history management
- OCR context integration
- System prompt hot-reload
- Configurable history length

### Performance Monitoring
- Latency tracking for all stages
- Activity state tracking
- Real-time metrics via WebSocket

### User Interface
- Web-based control panel
- Real-time event visualization
- Hotkey configuration
- System prompt editing

## Troubleshooting

### Services Won't Start
- Ensure virtual environment is activated
- Check that ports are not already in use
- Verify all dependencies are installed: `uv sync`

### Audio Issues
- Check audio device configuration in `config.yaml`
- Verify microphone permissions
- Test audio devices: `python scripts/list_output_devices.py`

### LLM Not Responding
- For Ollama: Ensure Ollama server is running and model is available
- For Gemini: Verify API key is set (environment variable or config)
- Check logs for error messages

### TTS Not Working
- For Edge TTS: Ensure `ffmpeg` is installed
- Check TTS service logs
- Verify WebSocket connection to TTS service

### High Latency
- Enable GPU acceleration (CUDA)
- Reduce model size if using local models
- Check network latency for API-based providers
- Review latency metrics in the web UI

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
