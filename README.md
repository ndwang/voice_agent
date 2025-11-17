# Voice Agent

A multi-service voice agent system that integrates speech-to-text, language models, text-to-speech, and optical character recognition to create an intelligent voice assistant capable of understanding speech, generating contextual responses, and reading screen content.

## Overview

The Voice Agent is a modular system that orchestrates multiple AI services to enable natural voice interactions. It can:

- **Listen** to speech input via microphone
- **Transcribe** speech to text in real-time
- **Capture** context from screen content (OCR)
- **Generate** responses using language models
- **Speak** responses using text-to-speech synthesis

The system is designed with a microservices architecture, where each component runs independently and communicates via WebSocket and HTTP protocols.

## Architecture

The system consists of six main components:

### Services

1. **STT Service** (Port 8001)
   - Speech-to-Text transcription using Whisper
   - Real-time audio processing with WebSocket streaming
   - Supports interim and final transcripts

2. **LLM Service** (Port 8002)
   - Language model inference with streaming support
   - Generates contextual responses based on conversation history and OCR context
   - Supports multiple LLM providers

3. **TTS Service** (Port 8003)
   - Text-to-Speech synthesis with WebSocket streaming
   - Real-time audio generation and playback
   - Supports multiple TTS backends

4. **OCR Service** (Port 8004)
   - Optical Character Recognition for screen monitoring
   - Captures and extracts text from screen regions
   - Provides context to the LLM for more informed responses

5. **Orchestrator** (Port 8000)
   - Main coordinator that connects all services
   - Manages conversation context and history
   - Handles the complete voice interaction pipeline

6. **Audio Driver**
   - Captures microphone input
   - Streams audio to STT service
   - Manages audio playback

## Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended for optimal performance)
- Microphone for voice input
- Speakers/headphones for audio output

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

4. **Configure environment variables** (optional):
   Create a `.env` file or set environment variables:
   ```bash
   # LLM Configuration
   LLM_PROVIDER=gemini
   LLM_MODEL=gemini-2.5-flash
   GEMINI_API_KEY=your_api_key_here

   # TTS Configuration
   TTS_BACKEND=cosyvoice
   COSYVOICE_MODEL_DIR=path/to/model
   COSYVOICE_SPEAKER=中文女

   # Service URLs (defaults shown)
   STT_WEBSOCKET_URL=ws://localhost:8001/ws/transcribe
   LLM_BASE_URL=http://localhost:8002
   TTS_WEBSOCKET_URL=ws://localhost:8003/synthesize/stream
   OCR_BASE_URL=http://localhost:8004
   ```

## Quick Start

### Start All Services

The easiest way to start all services is using the provided script:

```bash
python scripts/start_services.py
```

This will start all services in separate windows/terminals:
- STT Service (Port 8001)
- LLM Service (Port 8002)
- TTS Service (Port 8003)
- OCR Service (Port 8004)
- Orchestrator (Port 8000)
- Audio Driver

### Start Services Individually

You can also start services individually:

```bash
# Terminal 1: STT Service
python -m stt.stt_server
# Terminal 2: LLM Service
python -m llm.llm_server
# Terminal 3: TTS Service
python -m tts.tts_server
# Terminal 4: OCR Service
python -m ocr.ocr_server
# Terminal 5: Orchestrator
python -m orchestrator.agent
# Terminal 6: Audio Driver
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

1. **Start speaking** - The audio driver will capture your voice
2. **Wait for transcription** - The STT service will transcribe your speech
3. **Receive response** - The orchestrator processes your input through the LLM and generates a response
4. **Hear the response** - The TTS service synthesizes and plays the audio response

The system maintains conversation history and can incorporate OCR context from your screen for more contextual responses.

### OCR Context

The OCR service monitors screen regions and extracts text. This text is automatically included in the conversation context, allowing the agent to respond to questions about what's on your screen.

To fetch OCR texts manually:

```bash
curl http://localhost:8000/ocr/texts
```

## Configuration

### Service Configuration

Service URLs and settings can be configured via environment variables or by modifying `orchestrator/config.py`:

- `STT_WEBSOCKET_URL`: WebSocket URL for STT service
- `LLM_BASE_URL`: Base URL for LLM service
- `TTS_WEBSOCKET_URL`: WebSocket URL for TTS service
- `OCR_BASE_URL`: Base URL for OCR service
- `ORCHESTRATOR_HOST`: Host for orchestrator server (default: `0.0.0.0`)
- `ORCHESTRATOR_PORT`: Port for orchestrator server (default: `8000`)

### STT Configuration

Edit `stt/stt_server.py` to configure:
- Model size (default: `small`)
- Language code (default: `zh` for Chinese)
- Sample rate (default: `16000`)

### LLM Configuration

Set environment variables:
- `LLM_PROVIDER`: LLM provider to use (e.g., `gemini`)
- `LLM_MODEL`: Model name
- `GEMINI_API_KEY`: API key for Gemini provider

### TTS Configuration

Set environment variables:
- `TTS_BACKEND`: TTS backend to use
- `COSYVOICE_MODEL_DIR`: Path to TTS model directory
- `COSYVOICE_SPEAKER`: Speaker name/voice

### OCR Configuration

Configure OCR settings in `ocr/ocr_server.py`:
- Screen monitoring regions
- Update frequency
- Text extraction parameters

## Development

### Project Structure

```
voice_agent/
├── audio/               # Audio capture and playback
├── llm/                 # Language model service
├── ocr/                 # Optical character recognition service
├── orchestrator/        # Main orchestration logic
├── stt/                 # Speech-to-text service
├── tts/                 # Text-to-speech service
├── scripts/             # Utility scripts
└── pyproject.toml       # Project dependencies
```

### Key Components

- **`orchestrator/agent.py`**: Main agent that coordinates all services
- **`orchestrator/context_manager.py`**: Manages conversation history and OCR context
- **`orchestrator/config.py`**: Centralized configuration
- **`orchestrator/stt_client.py`**: STT service client
- **`orchestrator/ocr_client.py`**: OCR service client

### Testing

Individual components can be tested using scripts in the `scripts/` directory:

```bash
# Test audio player
python scripts/test_audio_player.py

# Test LLM streaming
python scripts/test_llm_stream.py "Your test prompt here"

# Verify GPU setup
python scripts/verify_gpu_setup.py
```

## API Endpoints

### Orchestrator API

- `GET /`: Service status
- `GET /health`: Health check
- `GET /ocr/texts`: Fetch all OCR texts

### STT Service

- `WebSocket /ws/transcribe`: Audio streaming endpoint

### LLM Service

- `POST /generate`: Non-streaming generation
- `POST /generate/stream`: Streaming generation (SSE)

### TTS Service

- `WebSocket /synthesize/stream`: Text streaming endpoint

### OCR Service

- `WebSocket /monitor/stream`: Screen monitoring endpoint
- `GET /texts`: Fetch extracted texts
