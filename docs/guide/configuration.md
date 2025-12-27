# Configuration Reference

The system is configured via `config.yaml` in the root directory. All services share this file, allowing for centralized management.

## Service Ports

| Service | Port | Description |
| --- | --- | --- |
| Orchestrator | 8000 | Main coordinator and Web UI |
| STT Service | 8001 | Speech-to-Text WebSocket API |
| LLM Service | 8002 | Language Model API |
| TTS Service | 8003 | Text-to-Speech WebSocket API |

## Core Configuration

### Orchestrator
```yaml
orchestrator:
  host: "0.0.0.0"
  port: 8000
  stt_websocket_path: "/ws/stt"
  log_level: "INFO"
  enable_latency_tracking: true
  system_prompt_file: "orchestrator/system_prompt.txt"
  hotkeys:
    toggle_listening: "ctrl+shift+l"
    cancel_speech: "ctrl+shift+c"
```

### LLM Providers
Supported: `ollama`, `gemini`, `llamacpp`.

```yaml
llm:
  provider: "ollama"  # Options: gemini, ollama
  providers:
    gemini:
      model: "gemini-2.5-flash"
      api_key: ""  # Optional: Override GEMINI_API_KEY env var
      generation_config:
        thinking_budget: 0  # 0 = disabled, -1 = dynamic (Gemini 2.5), 2.5 pro 128-32768, 2.5 flash 0-24576
        # thinking_level: "low"  # Gemini 3 Pro low/high, Flash low/high/medium/minimal
        temperature: 1.0
        top_p: 0.95
        top_k: 40
        max_output_tokens: 8192
    ollama:
      model: "Qwen3-8B-Q4-8kcontext"
      base_url: "http://localhost:11434"
      timeout: 300.0
      disable_thinking: true
      generation_config:
        temperature: 0.7
        top_p: 0.9
        top_k: 40
        num_predict: 2048
    llamacpp:
      model_path: ""
      n_ctx: 4096
      n_threads: 0
      n_gpu_layers: -1
```

### STT Configuration
Supported: `funasr` (recommended), `faster-whisper`.

```yaml
stt:
  host: "0.0.0.0"
  port: 8001
  provider: "funasr"
  language_code: "zh"
  sample_rate: 16000
  interim_transcript_min_samples: 16000
  providers:
    faster-whisper:
      model_path: "faster-whisper-small"
      device: null  # null = auto-detect
      compute_type: null  # null = auto-detect
    funasr:
      model_name: "paraformer-zh-streaming"
      vad_model: "fsmn-vad"
      punc_model: "ct-punc"  # Optional punctuation model
      vad_kwargs:
        max_single_segment_time: 30000
      device: null
      batch_size_s: 0
      streaming:
        enabled: true
        chunk_size: [0, 8, 4]
        encoder_chunk_look_back: 4
        decoder_chunk_look_back: 1
        vad_chunk_size_ms: 100
        silence_threshold_ms: 300
```

### TTS Configuration
Supported: `genie-tts`, `edge-tts`, `chattts`, `elevenlabs`.

```yaml
tts:
  host: "0.0.0.0"
  port: 8003
  provider: "genie-tts"
  providers:
    edge-tts:
      voice: "zh-CN-XiaoyiNeural"
      rate: "+0%"
      pitch: "+0Hz"
    chattts:
      model_source: "local"  # Options: local, huggingface, custom
      device: null
    elevenlabs:
      voice_id: ""
      stability: 0.5
      similarity_boost: 0.8
      style: 0.0
    genie-tts:
      character_name: "ema"
      onnx_model_dir: "path/to/model"
      language: "jp"  # Options: zh, en, jp
      reference_audio_path: ""
      reference_audio_text: ""
      source_sample_rate: 32000
```

### Audio Configuration
```yaml
audio:
  input:
    sample_rate: 16000
    channels: 1
    device: null  # null = default device
  output:
    sample_rate: 32000
    channels: 1
    device: null  # null = default device
  dtype: "float32"
  block_size_ms: 100
  silence_threshold_ms: 500  # Used as fallback for STT providers
  listening_status_poll_interval: 1.0
```

### OBS Configuration
```yaml
obs:
  websocket:
    host: "localhost"
    port: 4455
    password: ""
  subtitle_source: "subtitle"
  subtitle_ttl_seconds: 10
  visibility_source: ""  # Optional
  appear_filter_name: ""  # Optional
  clear_filter_name: ""  # Optional
```

### Bilibili Configuration
```yaml
bilibili:
  enabled: true
  room_id: 31232063
  sessdata: ""  # Optional SESSDATA cookie
  danmaku_ttl_seconds: 60
```

### Service URLs
Configuration for inter-service communication (used by orchestrator).

```yaml
services:
  orchestrator_base_url: "http://localhost:8000"
  stt_websocket_url: "ws://localhost:8001/ws/transcribe"
  tts_websocket_url: "ws://localhost:8003/synthesize/stream"
  ocr_websocket_url: "ws://localhost:8004/monitor/stream"
  ocr_base_url: "http://localhost:8004"
```

## Hot-Reloading
The Orchestrator monitors the `system_prompt_file` for changes. Updating the text file will automatically update the agent's persona without restarting.

