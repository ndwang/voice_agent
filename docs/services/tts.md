# TTS Service (Text-to-Speech)

The TTS Service (port 8003) converts text into natural-sounding speech. It supports both high-quality offline models and fast online providers.

## Supported Providers

### 1. Genie-TTS (High Quality)
A high-performance CPU inference engine for GPT-SoVITS models. 
- **Character Consistency**: Uses pre-trained character models.
- **Language Support**: Optimized for Japanese, Chinese, and English.
- **Latency**: GENIE optimizes the original model for outstanding CPU performance.

### 2. Edge-TTS (Fast & Easy)
Uses Microsoft Edge's online TTS engine.
- **No GPU Required**: Processing happens on Microsoft servers.
- **Wide Voice Selection**: Access to dozens of neural voices.
- **Requirement**: `ffmpeg` must be installed locally for format conversion.

### 3. ChatTTS
Optimized for conversational speech, including fillers like [laugh] and [um].

## WebSocket Streaming Protocol

To minimize "Time to First Byte" (TTFB), the TTS Service supports streaming synthesis:

1. **Client Request**:
   ```json
   {
     "type": "text",
     "text": "Hello, how are you?",
     "finalize": false
   }
   ```
2. **Audio Streaming**: The service begins generating audio and sends it back as raw binary PCM chunks.
3. **Finalization**: When `finalize: true` is sent, the service finishes the current buffer and closes the stream segment.

## Latency Optimization

- **Pre-connection**: The Orchestrator's `TTSManager` opens a WebSocket connection as soon as the LLM *starts* generating, reducing the handshake overhead when the first sentence is ready.
- **Sentence-Level Chunking**: The system doesn't wait for the full LLM response. As soon as a sentence is complete (detected by punctuation or tags), it is sent to the TTS service.


