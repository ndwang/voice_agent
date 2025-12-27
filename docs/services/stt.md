# STT Service (Speech-to-Text)

The STT Service provides real-time transcription of audio streams. It is designed to run as a standalone service (on port 8001) that multiple clients can connect to via WebSocket.

## Supported Providers

### 1. FunASR (Recommended)
Uses Alibaba's FunASR framework. It is highly optimized for Chinese and English, supporting:
- **Streaming Paraformer**: Low-latency transcription.
- **VAD (Voice Activity Detection)**: Built-in model to detect speech boundaries.
- **Punctuation**: Automatic punctuation restoration.

### 2. Faster-Whisper
An efficient implementation of OpenAI's Whisper model using CTranslate2. 
- Best for high-accuracy multilingual support.
- Supports different model sizes (tiny, base, small, medium, large).

## WebSocket Protocol

### Input: Audio Streaming
The client sends raw PCM audio bytes (typically 16kHz, mono, float32) directly over the WebSocket.

### Output: JSON Transcripts
The service broadcasts transcripts to all connected clients:

```json
// Interim (Partial) Result
{
  "type": "interim",
  "text": "Hello world"
}

// Final (Sentence) Result
{
  "type": "final",
  "text": "Hello world."
}

// Speech Boundary Detection
{
  "type": "speech_start"
}
```

## Internal Workflow

1. **VAD Chunking**: Incoming audio is processed by a VAD model to identify speech segments.
2. **Model Inference**: Speech chunks are sent to the ASR model.
3. **Post-Processing**: (Optional) Punctuation and text formatting are applied.
4. **Broadcast**: The resulting text is sent back through the WebSocket.


