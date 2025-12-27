# Installation Guide

The Voice Agent is designed to be easy to set up using `uv` or standard `pip`.

## Prerequisites

### 1. Python Environment
- Python 3.10 or higher is required.
- We recommend using [uv](https://github.com/astral-sh/uv) for fast and reliable dependency management.

### 2. System Dependencies
- **PortAudio**: Required for microphone capture.
    - Linux: `sudo apt install libportaudio2`
    - Windows/macOS: Usually included with Python audio libraries

### 3. Hardware Requirements
- **GPU**: A CUDA-capable NVIDIA GPU is highly recommended for running local models (STT/LLM) with low latency. The more VRAM the better.

## Setup Steps

### 1. Clone the Repository
```bash
git clone https://github.com/ndwang/voice_agent.git
cd voice_agent
```

### 2. Install Dependencies

The Voice Agent uses optional dependency groups, allowing you to install only the providers you need. This reduces installation size and avoids unnecessary dependencies.

#### Minimal Installation (Core Only)
For core functionality without any providers:
```bash
uv sync
```

#### Install All Providers
To install all optional providers and features:
```bash
uv sync --extra all
```

#### Install Specific Providers
Install only the providers you plan to use:

**Example: FunASR STT + Ollama LLM + Edge TTS**
```bash
uv sync --extra stt-funasr --extra llm-ollama --extra tts-edge
```

#### Available Dependency Groups

**STT Providers:**
- `stt-faster-whisper` - Faster-Whisper STT provider
- `stt-funasr` - FunASR STT provider
- `stt-all` - All STT providers

**LLM Providers:**
- `llm-gemini` - Google Gemini API provider
- `llm-ollama` - Ollama local LLM provider
- `llm-llama-cpp` - llama.cpp provider
- `llm-all` - All LLM providers

**TTS Providers:**
- `tts-edge` - Edge TTS (Microsoft Edge voices)
- `tts-elevenlabs` - ElevenLabs TTS provider
- `tts-genie` - Genie TTS provider
- `tts-all` - All TTS providers

**Other:**
- `ocr` - OCR dependencies (PaddleOCR, etc.)
- `all` - JUST GIVE ME EVERYTHING (STT, LLM, TTS, OCR)

#### GPU and CUDA Configuration

The default installation assumes **CUDA 12.6**. If you have a different CUDA version or wish to run without a GPU, you may need to modify `pyproject.toml`.

**Customizing CUDA Version**

If your system uses a different CUDA version (e.g., 12.4), update the `pytorch-cuda` index in `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cuda"
# For CUDA 12.4, use: https://download.pytorch.org/whl/cu124
# For CUDA 11.8, use: https://download.pytorch.org/whl/cu118
url = "https://download.pytorch.org/whl/cu126" 
explicit = true
```

After modifying the file, run `uv sync` again.

**CPU-Only Installation**

If you do not have an NVIDIA GPU, you can remove or comment out the `[tool.uv.sources]` and `[[tool.uv.index]]` sections in `pyproject.toml` to install the standard CPU versions of Torch.

#### Using pip (Alternative)

If you prefer pip, you can install with optional dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[all]"  # Install all providers
# Or install specific providers:
pip install -e ".[stt-faster-whisper,llm-ollama,tts-edge]"
```

### 3. LLM Setup (Local)
If you plan to use **Ollama**:
1. Install Ollama from [ollama.com](https://ollama.com/).
2. Pull your desired model:
   ```bash
   ollama pull Qwen3-8B  # or any other supported model
   ```

### 4. Verify Installation
Run the GPU verification script:
```bash
python scripts/verify_gpu_setup.py
```

Check if audio devices are detected:
```bash
python scripts/list_output_devices.py
```

## Additional Setup

### Bilibili Integration (blivedm)
To use the Bilibili live stream source, you must install the `blivedm` library. **Note:** The version on PyPI is abandoned and incompatible; you must install it directly from the source.

- **Installation (Remote)**: `uv pip install git+https://github.com/xfgryujk/blivedm.git`
- **Project Page**: [xfgryujk/blivedm](https://github.com/xfgryujk/blivedm)

### ChatTTS
To use the `chattts` provider for TTS, it must be installed separately (not available as an optional dependency):
- **Project Page**: [2noise/ChatTTS](https://github.com/2noise/ChatTTS)

### Provider-Specific Notes

**Genie TTS:**
- Requires character-specific ONNX models. See the [Genie TTS](https://github.com/High-Logic/Genie-TTS) page for details on model preparation.

**Edge-TTS:**
- Requires `ffmpeg` installed on your system path:
    - Windows: `choco install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org/).
    - Linux: `sudo apt install ffmpeg`
    - macOS: `brew install ffmpeg`
- Project Page: [rany2/edge-tts](https://github.com/rany2/edge-tts)


