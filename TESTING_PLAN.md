# Voice Agent Manual Testing Plan

This document provides a comprehensive step-by-step plan to manually test all components of the voice agent system.

## System Architecture Overview

- **STT Service** (Port 8001): Speech-to-Text transcription
  - WebSocket endpoint: `/ws/transcribe` - accepts audio and broadcasts transcripts to all connected clients
- **LLM Service** (Port 8002): Language model for generating responses
  - HTTP endpoint: `/generate/stream` - SSE streaming for token generation
- **TTS Service** (Port 8003): Text-to-Speech synthesis
  - WebSocket endpoint: `/synthesize/stream` - streams text and receives audio chunks
- **OCR Service** (Port 8004): Optical Character Recognition for screen monitoring
  - HTTP endpoint: `/texts/get` - fetches detected texts on demand
- **Orchestrator** (Port 8000): Main coordinator that connects all services
  - Connects to STT server WebSocket as a listener to receive broadcast transcripts
  - Makes HTTP requests to LLM service for streaming responses
  - Connects to TTS service WebSocket to stream text and receive audio
  - Makes HTTP requests to OCR service to fetch texts on demand
- **Audio Driver**: Microphone input capture
  - Connects directly to STT server WebSocket to send audio data

### Connection Flow

```
┌─────────────┐
│Audio Driver │
│ (Microphone)│
└──────┬──────┘
       │ WebSocket (audio bytes)
       │
       ▼
┌─────────────┐         ┌──────────────┐
│ STT Server  │────────▶│ Orchestrator │
│  (Port 8001)│broadcast│  (Port 8000) │
└─────────────┘         │              │
                        │  ┌─────────┐ │
                        │  │STTClient│ │ (listens)
                        │  └─────────┘ │
                        │              │
                        │  ┌─────────┐ │
                        │  │Context  │ │
                        │  │Manager  │ │
                        │  └─────────┘ │
                        │              │
                        │  ┌─────────┐ │
                        │  │  Audio  │ │
                        │  │ Player  │ │
                        │  └─────────┘ │
                        └──────┬───────┘
                               │
                    ┌──────────┼──────────┐
                    │          │          │
                    ▼          ▼          ▼
            ┌──────────┐ ┌──────────┐ ┌──────────┐
            │   LLM    │ │   TTS    │ │   OCR    │
            │ Service  │ │ Service  │ │ Service  │
            │(Port 8002)│ │(Port 8003)│ │(Port 8004)│
            └──────────┘ └──────────┘ └──────────┘
               ▲            ▲            ▲
               │            │            │
               │ HTTP SSE   │ WebSocket  │ HTTP GET
               │ (tokens)   │ (text→audio│ (on-demand)
               │            │  chunks)   │
               └────────────┴────────────┘
                        │
                        │
                  Orchestrator
              (coordinates all flows)
```

**Flow Description:**

1. **Audio Input**: Audio Driver captures microphone input and sends audio bytes to STT Server via WebSocket
2. **Speech Recognition**: STT Server transcribes audio and broadcasts transcripts to all connected clients (including Orchestrator's STTClient)
3. **Context Building**: Orchestrator's ContextManager formats the prompt, optionally fetching OCR texts on-demand
4. **LLM Generation**: Orchestrator sends prompt to LLM Service via HTTP SSE, receives streaming tokens
5. **Speech Synthesis**: Orchestrator streams tokens to TTS Service via WebSocket, receives audio chunks
6. **Audio Output**: Orchestrator's Audio Player plays the received audio chunks

**Key Connection Patterns**:
- **STT**: Broadcast pattern - STT server broadcasts transcripts to all connected WebSocket clients (both audio driver and orchestrator can be connected simultaneously)
- **LLM**: HTTP SSE streaming - Orchestrator makes HTTP POST requests, receives streaming tokens via Server-Sent Events
- **TTS**: Bidirectional WebSocket - Orchestrator sends text chunks, receives audio chunks in real-time
- **OCR**: On-demand HTTP - Orchestrator fetches OCR texts via HTTP GET when building context (not in main voice flow)
- **Audio Player**: Internal component of Orchestrator - receives audio chunks from TTS and plays them

---

## Prerequisites

1. **Environment Setup**
   - Ensure virtual environment is activated: `.venv\Scripts\activate` (Windows)
   - All dependencies installed: `uv sync`
   - Required models/services configured:
     - STT: Whisper model (small)
     - LLM: Gemini/other LLM provider configured
     - TTS: Edge TTS (default) or ChatTTS model configured
     - OCR: PaddleOCR initialized

2. **Hardware Requirements**
   - Microphone connected and working
   - Speakers/headphones for audio output
   - Screen with visible text for OCR testing

---

## Phase 1: Individual Component Testing

### Test 1.1: STT Service (Standalone)

**Objective**: Verify STT service can transcribe audio correctly.

**Steps**:
1. Start STT service:
   ```powershell
   uv run python -m stt.stt_server
   ```
2. In a new terminal, start the audio driver:
   ```powershell
   uv run python -m audio.audio_driver
   ```
3. **Test Cases**:
   - **Test 1.1a**: Speak clearly in Chinese for 3-5 seconds, pause. Verify transcript appears.
   - **Test 1.1b**: Speak multiple sentences with pauses. Verify each sentence is transcribed separately.
   - **Test 1.1c**: Speak quietly. Verify VAD detects speech or handles silence appropriately.
   - **Test 1.1d**: Speak with background noise. Verify transcription quality.

**Expected Results**:
- STT service accepts WebSocket connections
- Transcripts appear in real-time (interim) and finalize after silence
- Chinese text is correctly transcribed
- No crashes or errors

**Verification**:
- Check STT server console for "Client connected" and transcript messages
- Check audio driver console for "FINAL:" transcripts

---

### Test 1.2: LLM Service (Standalone)

**Objective**: Verify LLM service generates responses correctly.

**Steps**:
1. Ensure LLM service dependencies are configured (Ollama running, API keys set, etc.)
2. Start LLM service:
   ```powershell
   uv run python -m llm.llm_server
   ```
3. **Test Cases**:
   - **Test 1.2a**: Test non-streaming endpoint:
     ```powershell
     curl -X POST http://localhost:8002/generate -H "Content-Type: application/json" -d "{\"prompt\": \"你好，请介绍一下你自己\"}"
     ```
   
   - **Test 1.2b**: Test streaming endpoint (using Python script - RECOMMENDED):
     ```powershell
     uv run python scripts/test_llm_stream.py "用一句话解释什么是人工智能"
     ```
     This script will show tokens arriving incrementally and verify streaming works.
   
   - **Test 1.2b-alt**: Test streaming endpoint using curl (may buffer):
     ```powershell
     curl -N -X POST http://localhost:8002/generate/stream -H "Content-Type: application/json" -d "{\"prompt\": \"用一句话解释什么是人工智能\"}"
     ```
     Note: The `-N` flag disables buffering. You should see SSE format output with `event: token` and `data: {...}` lines.
   
   - **Test 1.2c**: Test with temperature parameter:
     ```powershell
     curl -X POST http://localhost:8002/generate -H "Content-Type: application/json" -d "{\"prompt\": \"用一句话解释什么是人工智能\", \"temperature\": 0.7}"
     ```

**Expected Results**:
- `/generate` returns complete response in JSON format: `{"response": "..."}`
- `/generate/stream` returns SSE stream with tokens arriving incrementally
- SSE format: `event: token` followed by `data: {"token": "..."}` lines
- Final event: `event: done` with `data: {"status": "complete"}`
- Temperature parameter is respected when provided
- No errors or timeouts

**Verification**:
- **For streaming test script**: Tokens should appear one by one on screen (not all at once)
- **For curl**: You should see multiple `event: token` lines appearing over time
- Check LLM server console for request logs
- Verify response quality and relevance
- **Key indicator**: If streaming works, you'll see tokens appearing gradually (not all at once)
- **If buffered**: All tokens appear simultaneously after a delay

---

### Test 1.3: TTS Service (Standalone)

**Objective**: Verify TTS service synthesizes speech correctly with both streaming and non-streaming methods.

**Steps**:
1. Start TTS service:
   ```powershell
   uv run python -m tts.tts_server
   ```
   Note: Default provider is Edge TTS. To use ChatTTS, set `TTS_PROVIDER=chattts` environment variable.

2. **Test Cases**:
   
   **Recommended: Use the comprehensive `test_tts.py` script**:
   
   The `test_tts.py` script tests all TTS endpoints and automatically converts PCM to WAV:
   - Voice listing (GET /voices)
   - REST endpoint (POST /synthesize) - non-streaming synthesis
   - WebSocket endpoint (WS /synthesize/stream) - streaming synthesis
   
   ```powershell
   # Run all three tests (RECOMMENDED)
   uv run python scripts/test_tts.py --all "测试文本"
   
   # Run individual tests
   uv run python scripts/test_tts.py --test-voices  # Voice listing
   uv run python scripts/test_tts.py --test-rest "测试"  # REST endpoint
   uv run python scripts/test_tts.py "测试"  # WebSocket (default)
   
   # Options
   uv run python scripts/test_tts.py "测试" --voice "zh-CN-XiaoxiaoNeural"  # Custom voice
   uv run python scripts/test_tts.py "测试" --output "my_audio"  # Custom output file
   uv run python scripts/test_tts.py "测试" --test-multiple  # Test multiple chunks
   ```
   
   **Individual Test Cases (Alternative Methods)**:
   
   - **Test 1.3a**: Check service status:
     ```powershell
     curl http://localhost:8003/
     ```
     Should return provider information and sample rate.
   
   - **Test 1.3b**: List available voices:
     ```powershell
     uv run python scripts/test_tts.py --test-voices
     # Or: curl http://localhost:8003/voices
     ```
     For Edge TTS: Returns list of available voices with locale, gender, etc.
     For ChatTTS: Returns speaker embedding information.
   
   - **Test 1.3c**: Test non-streaming synthesis (REST API):
     ```powershell
     uv run python scripts/test_tts.py --test-rest "你好，这是一个测试"
     # Or: curl -X POST http://localhost:8003/synthesize -H "Content-Type: application/json" -d "{\"text\": \"你好，这是一个测试\", \"voice\": \"zh-CN-XiaoxiaoNeural\"}" --output audio.pcm
     ```
     For Edge TTS, you can specify `voice`, `rate`, `pitch` parameters.
     For ChatTTS, you can specify `speaker`, `temperature`, `top_p`, `top_k` parameters.
     Note: The script automatically converts PCM to WAV format for easy playback.
   
   - **Test 1.3d**: Test WebSocket streaming:
     ```powershell
     uv run python scripts/test_tts.py "测试"
     # Or: uv run python scripts/test_tts.py --test-websocket "测试"
     ```
     
     Alternatively, use a WebSocket client like `websocat`:
     ```bash
     echo '{"type": "text", "text": "这是流式语音合成测试", "finalize": true, "voice": "zh-CN-XiaoxiaoNeural"}' | websocat ws://localhost:8003/synthesize/stream
     ```
   
   - **Test 1.3e**: Test provider switching:
     - Test with Edge TTS: `TTS_PROVIDER=edge-tts`
     - Test with ChatTTS: `TTS_PROVIDER=chattts` (requires ChatTTS models)
     - Verify both providers work correctly

**Expected Results**:
- Service status endpoint returns provider information
- `/voices` endpoint returns available voices
- `/synthesize` returns complete audio as binary PCM data
- WebSocket streams audio chunks as bytes
- Audio quality is clear and natural
- Chinese text is correctly synthesized
- Both providers work correctly

**Verification**:
- Play back generated audio files (automatically saved as WAV format by the test script)
- Check TTS server console for synthesis logs
- Verify audio chunks are received in streaming mode
- Verify provider-specific parameters are respected
- Check that audio format is int16 PCM, 16kHz, mono (converted to WAV for playback)
- Review test script output for detailed test results and statistics

---

### Test 1.4: Audio Player (Standalone)

**Objective**: Verify audio player can receive and play audio chunks correctly.

**Steps**:
1. Run the audio player test script:
   ```powershell
   uv run python scripts/test_audio_player.py
   ```
   
   This script includes multiple test cases:
   - Basic playback (440 Hz tone)
   - Streaming chunks (multiple chunks)
   - Stop functionality
   - Invalid data handling
   - Multiple tones sequence

2. **Test Cases**:
   - **Test 1.4a**: Basic playback test
     - Run the test script above
     - Verify you hear a 440 Hz tone for 1 second
     - Check console for no errors
   
   - **Test 1.4b**: Streaming chunks test
     - Modify test to send multiple small chunks
     - Verify audio plays continuously without gaps
     - Check queue handling works correctly
   
   - **Test 1.4c**: Stop functionality test
     - Start playback
     - Call `stop()` mid-playback
     - Verify playback stops and queue is cleared
   
   - **Test 1.4d**: Invalid audio data handling
     - Send empty bytes
     - Send malformed audio data
     - Verify graceful error handling (no crashes)

**Expected Results**:
- Audio player initializes correctly
- Audio chunks are queued and played
- Playback is smooth without gaps between chunks
- Stop functionality works correctly
- Invalid data is handled gracefully
- No crashes or audio device errors

**Verification**:
- Listen for audio output (test tone should be audible)
- Check console for error messages
- Verify playback completes without hanging
- Test with different audio formats and chunk sizes

---

### Test 1.5: OCR Service (Standalone)

**Objective**: Verify OCR service can detect and extract text from screen regions.

**Steps**:
1. Start OCR service:
   ```powershell
   uv run python -m ocr.ocr_server
   ```
2. **Test Cases**:
   - **Test 1.5a**: Test single region OCR:
     ```powershell
     curl -X POST http://localhost:8004/region/set -H "Content-Type: application/json" -d "{\"region\": [100, 100, 400, 200]}"
     ```
     Open a window with visible Chinese text in that region.
   
   - **Test 1.5b**: Start monitoring:
     ```powershell
     curl -X POST http://localhost:8004/monitor/start -H "Content-Type: application/json" -d "{\"region\": [100, 100, 400, 200], \"interval_ms\": 1000, \"clear_texts\": true}"
     ```
     Change text in the monitored region and verify detection.
   
   - **Test 1.5c**: Get detected texts:
     ```powershell
     curl http://localhost:8004/texts/get
     ```
   
   - **Test 1.5d**: Test WebSocket streaming:
     Connect to `ws://localhost:8004/monitor/stream` and send:
     ```json
     {"type": "get_texts"}
     ```

**Expected Results**:
- OCR correctly extracts text from specified regions
- Monitoring detects text changes
- Detected texts are stored and retrievable
- WebSocket streams updates correctly

**Verification**:
- Check OCR server console for detection logs
- Verify extracted text matches screen content
- Check text storage files are created

---

## Phase 2: Service Integration Testing

### Test 2.1: STT → Orchestrator Connection

**Objective**: Verify orchestrator connects to STT service and receives broadcast transcripts.

**Steps**:
1. Start STT service:
   ```powershell
   uv run python -m stt.stt_server
   ```
2. Start Orchestrator:
   ```powershell
   uv run python -m orchestrator.agent
   ```
3. Wait for connection message: "STT Client: Connected to STT server"
4. Start audio driver:
   ```powershell
   uv run python -m audio.audio_driver
   ```
5. Speak a test phrase and verify:
   - Audio driver sends audio to STT server
   - STT server broadcasts transcripts to all connected clients (including orchestrator)
   - Orchestrator receives and logs the transcript

**Expected Results**:
- Orchestrator's STTClient connects to STT server WebSocket as a listener
- STT server broadcasts transcripts to all connected clients (broadcast pattern)
- Orchestrator logs "Transcript received: [text]"
- Both audio driver and orchestrator can be connected simultaneously

**Verification**:
- Check orchestrator console for "STT Client: Connected to STT server" message
- Check orchestrator console for "Transcript received: [text]" messages
- Check STT server console for "Client connected" and "Broadcast final transcript" messages
- Verify that transcripts appear in both audio driver and orchestrator consoles

---

### Test 2.2: LLM → Orchestrator Integration

**Objective**: Verify orchestrator can call LLM service and receive responses.

**Steps**:
1. Start LLM service:
   ```powershell
   uv run python -m llm.llm_server
   ```
2. Start Orchestrator:
   ```powershell
   uv run python -m orchestrator.agent
   ```
3. Send a test request directly to orchestrator's LLM flow (this happens automatically when STT sends a transcript, but you can also test via context manager):
   - Use STT → Orchestrator flow from Test 2.1
   - Speak: "你好，请说一句话"
   - Verify orchestrator calls LLM and receives streaming tokens

**Expected Results**:
- Orchestrator successfully calls LLM streaming endpoint
- Tokens are received and processed
- Response is added to conversation history

**Verification**:
- Check orchestrator console for LLM streaming logs
- Check LLM server console for request logs
- Verify no connection errors

---

### Test 2.3: TTS → Orchestrator Integration

**Objective**: Verify orchestrator can stream text to TTS and receive audio.

**Steps**:
1. Start TTS service:
   ```powershell
   uv run python -m tts.tts_server
   ```
2. Start Orchestrator:
   ```powershell
   uv run python -m orchestrator.agent
   ```
3. Trigger TTS via full flow:
   - Start STT service and audio driver
   - Speak: "请说：测试成功"
   - Verify audio is played back

**Expected Results**:
- Orchestrator connects to TTS WebSocket
- Text chunks are sent to TTS
- Audio chunks are received and played
- Audio playback is clear and synchronized

**Verification**:
- Listen for audio output (should be clear and natural)
- Check orchestrator console for TTS connection logs
- Check TTS server console for synthesis logs
- Verify no audio dropouts or errors
- **Audio Player**: Verify audio chunks are received and played smoothly
  - Audio should start playing before full response completes (streaming)
  - No gaps or stuttering between chunks
  - Audio quality is clear

---

### Test 2.4: OCR → Orchestrator Integration

**Objective**: Verify orchestrator can fetch OCR texts from OCR service on demand.

**Steps**:
1. Start OCR service:
   ```powershell
   uv run python -m ocr.ocr_server
   ```
2. Start Orchestrator:
   ```powershell
   uv run python -m orchestrator.agent
   ```
3. Start OCR monitoring:
   ```powershell
   curl -X POST http://localhost:8004/monitor/start -H "Content-Type: application/json" -d "{\"region\": [100, 100, 400, 200], \"interval_ms\": 1000}"
   ```
4. Display some text in the monitored region
5. Wait for OCR to detect the text
6. Test orchestrator's OCR endpoint:
   ```powershell
   curl http://localhost:8000/ocr/texts
   ```
7. Verify orchestrator can fetch OCR texts

**Expected Results**:
- Orchestrator's OCRClient makes HTTP requests to OCR service on demand
- OCR service stores detected texts
- Orchestrator can fetch all OCR texts via HTTP endpoint
- OCR texts are included in LLM context when processing user input

**Verification**:
- Check orchestrator console for OCR fetch operations (when LLM processes input)
- Check OCR server console for detection logs
- Verify `/ocr/texts` endpoint returns detected texts
- Verify OCR context is included in LLM prompts (check context manager)

---

## Phase 3: End-to-End Testing

### Test 3.1: Full Voice Conversation Flow

**Objective**: Test complete voice interaction: Speech → STT → LLM → TTS → Audio.

**Steps**:
1. Start all services using the startup script:
   ```powershell
   uv run python scripts/start_services.py
   ```
   Or start manually in order:
   - STT Service (must start before orchestrator and audio driver)
   - LLM Service  
   - TTS Service
   - OCR Service
   - Orchestrator (connects to STT, LLM, TTS, OCR)
   - Audio Driver (connects to STT)

2. Wait for all services to initialize (check each console window)

3. Verify service health:
   ```powershell
   uv run python scripts/check_services.py
   ```
   Or manually:
   ```powershell
   curl http://localhost:8000/health
   ```

4. **Test Conversation**:
   - Speak: "你好"
   - Wait for response audio
   - Speak: "请介绍一下你自己"
   - Wait for response
   - Speak: "谢谢"
   - Wait for response

**Expected Results**:
- All services start successfully
- Health check shows all services connected
- Speech is transcribed correctly
- LLM generates appropriate responses
- TTS synthesizes and plays audio
- Conversation context is maintained
- No errors or crashes

**Verification**:
- Listen for clear audio responses
- Check orchestrator console for complete flow logs
- Verify conversation history is maintained
- Check all service consoles for errors

---

### Test 3.2: OCR Context Integration

**Objective**: Test that OCR context influences LLM responses.

**Steps**:
1. Start all services (as in Test 3.1)
2. Set up OCR monitoring on a region with visible text:
   ```powershell
   curl -X POST http://localhost:8004/monitor/start -H "Content-Type: application/json" -d "{\"region\": [100, 100, 500, 300], \"interval_ms\": 1000}"
   ```
3. Display some Chinese text in the monitored region (e.g., "当前状态：运行中")
4. Wait for OCR to detect the text
5. Speak: "屏幕上显示的是什么？"
6. Verify LLM response references the OCR-detected text

**Expected Results**:
- OCR detects text from screen
- OCR context is added to LLM context
- LLM response mentions or references the OCR text
- Response is accurate and relevant

**Verification**:
- Check OCR service for detected text
- Check orchestrator console for OCR updates
- Verify LLM response includes OCR context
- Listen to TTS response mentioning OCR content

---

### Test 3.3: Multi-Turn Conversation with Context

**Objective**: Test that conversation history is maintained across multiple turns.

**Steps**:
1. Start all services
2. Have a multi-turn conversation:
   - Turn 1: "我的名字是张三"
   - Turn 2: "我刚才说了什么名字？"
   - Turn 3: "请用那个名字造一个句子"
3. Verify LLM remembers previous context

**Expected Results**:
- Each turn is processed correctly
- LLM remembers information from earlier turns
- Responses are contextually appropriate
- Conversation history grows correctly

**Verification**:
- Check orchestrator context manager logs
- Verify LLM responses reference previous conversation
- Check conversation history structure

---

## Phase 4: Error Handling & Edge Cases

### Test 4.1: Service Disconnection Recovery

**Objective**: Test system behavior when services disconnect.

**Steps**:
1. Start all services
2. Stop LLM service abruptly (Ctrl+C)
3. Try to speak and trigger a response
4. Restart LLM service
5. Try speaking again

**Expected Results**:
- System handles disconnection gracefully
- Error messages are logged
- System recovers when service restarts
- No crashes

**Verification**:
- Check orchestrator error logs
- Verify reconnection attempts
- Test that system works after restart

---

### Test 4.2: Invalid Input Handling

**Objective**: Test system behavior with edge case inputs.

**Steps**:
1. Start all services
2. **Test Cases**:
   - Speak very quietly (below VAD threshold)
   - Speak very long sentences (30+ seconds)
   - Speak with long pauses mid-sentence
   - Speak in English (if system expects Chinese)
   - Send empty audio chunks

**Expected Results**:
- System handles edge cases without crashing
- Appropriate error messages or timeouts
- System continues to function normally

**Verification**:
- Check service logs for error handling
- Verify no crashes or hangs
- System recovers to normal operation

---

### Test 4.3: Concurrent Requests

**Objective**: Test system under concurrent load.

**Steps**:
1. Start all services
2. Rapidly speak multiple phrases without waiting for responses
3. Trigger multiple OCR region changes simultaneously
4. Monitor system behavior

**Expected Results**:
- System handles concurrent requests
- No race conditions or deadlocks
- Responses are queued appropriately
- System remains stable

**Verification**:
- Check for error messages
- Verify all requests are processed
- Check system resource usage

---

## Phase 5: Performance Testing

### Test 5.1: Latency Measurement

**Objective**: Measure end-to-end latency.

**Steps**:
1. Start all services
2. Measure time from speech end to audio response start
3. Record multiple measurements
4. Calculate average latency

**Expected Results**:
- Latency is acceptable (< 3 seconds for typical responses)
- Latency is consistent
- No significant delays

**Verification**:
- Record timestamps at each stage
- Calculate latency breakdown:
  - STT processing time
  - LLM generation time
  - TTS synthesis time
  - Network overhead

---

### Test 5.2: Resource Usage

**Objective**: Monitor system resource consumption.

**Steps**:
1. Start all services
2. Monitor CPU, memory, GPU usage
3. Run extended conversation (10+ minutes)
4. Check for memory leaks or resource exhaustion

**Expected Results**:
- Resource usage is reasonable
- No memory leaks
- GPU usage is appropriate (if available)
- System remains responsive

**Verification**:
- Use Task Manager or `htop` to monitor resources
- Check for gradual memory increase
- Verify GPU utilization (if CUDA available)

---

## Test Checklist Summary

### Pre-Testing Setup
- [x] Virtual environment activated
- [x] Dependencies installed (`uv sync`)
- [x] Models configured (STT, TTS, LLM)
- [x] Microphone and speakers working
- [x] Test screen region identified for OCR

### Individual Components
- [x] STT Service: Transcription works
- [x] LLM Service: Generation works (streaming and non-streaming)
- [x] TTS Service: Synthesis works (streaming and non-streaming)
  - [x] Non-streaming REST endpoint (`POST /synthesize`)
  - [x] Streaming WebSocket endpoint (`WS /synthesize/stream`)
  - [x] Voice listing (`GET /voices`)
  - [x] Edge TTS provider works
  - [ ] ChatTTS provider works (if configured)
- [x] Audio Player: Playback works (chunk queuing, streaming, stop)
- [x] OCR Service: Text detection works

### Integration
- [x] STT → Orchestrator connection
- [x] LLM → Orchestrator integration
- [x] TTS → Orchestrator integration
- [ ] OCR → Orchestrator integration

### End-to-End
- [x] Full voice conversation flow
- [ ] OCR context integration
- [x] Multi-turn conversation with context

### Error Handling
- [ ] Service disconnection recovery
- [ ] Invalid input handling
- [ ] Concurrent request handling

### Performance
- [ ] Latency measurement
- [ ] Resource usage monitoring

---

## Troubleshooting Common Issues

### Services Won't Start
- Check if ports are already in use: `netstat -ano | findstr :8000`
- Verify virtual environment is activated
- Check for missing dependencies

### STT Not Transcribing
- Verify microphone permissions
- Check audio device selection in driver
- Verify Whisper model is loaded
- Check STT server logs for errors

### LLM Not Responding
- Verify LLM provider is configured (Ollama running, API keys set)
- Check LLM service logs
- Test LLM service directly with curl

### TTS Not Playing Audio
- Verify audio output device
- Check TTS provider is configured correctly (`TTS_PROVIDER` environment variable)
- For Edge TTS: Verify `ffmpeg` is installed (required for audio conversion)
- For ChatTTS: Verify models are loaded (check console for model loading messages)
- Verify WebSocket connection
- Check audio player initialization
- Test TTS service directly: `curl -X POST http://localhost:8003/synthesize ...`
- Verify audio format (should be int16 PCM, 16kHz, mono)

### Audio Player Not Playing
- Verify speakers/headphones are connected and working
- Check audio device permissions
- Test audio player standalone: `uv run python scripts/test_audio_player.py`
- Verify audio format (should be int16 bytes, 16kHz sample rate)
- Check for audio device errors in console
- Try different audio output device if available

### OCR Not Detecting Text
- Verify screen region coordinates
- Check OCR model is loaded
- Verify text is visible and clear
- Check OCR service logs

### Orchestrator Not Connecting
- Verify all services are running (especially STT service must be running before orchestrator)
- Check WebSocket URLs in config (`orchestrator/config.py`)
- Verify network connectivity
- Check orchestrator logs for connection errors
- For STT connection: Verify orchestrator's STTClient connects to STT server WebSocket
- For OCR: Verify orchestrator makes HTTP requests (not WebSocket) to OCR service

---

## Quick Test Commands Reference

```powershell
# Start all services
uv run python scripts/start_services.py

# Check service status
uv run python scripts/check_services.py

# Stop all services
uv run python scripts/stop_services.py

# Health checks
curl http://localhost:8000/health  # Orchestrator
curl http://localhost:8001/        # STT
curl http://localhost:8002/health  # LLM
curl http://localhost:8003/health  # TTS
curl http://localhost:8004/health  # OCR

# Test LLM
curl -X POST http://localhost:8002/generate -H "Content-Type: application/json" -d "{\"prompt\": \"你好\"}"

# Test TTS
curl http://localhost:8003/  # Service status
# Comprehensive TTS testing (covers all endpoints)
uv run python scripts/test_tts.py --all "测试文本"
# Individual tests
uv run python scripts/test_tts.py --test-voices  # List voices
uv run python scripts/test_tts.py --test-rest "测试"  # REST endpoint
uv run python scripts/test_tts.py "测试"  # WebSocket streaming (default)

# Test Audio Player
uv run python scripts/test_audio_player.py

# Start OCR monitoring
curl -X POST http://localhost:8004/monitor/start -H "Content-Type: application/json" -d "{\"region\": [100, 100, 400, 200], \"interval_ms\": 1000}"
```

---

## Notes

- All tests assume Chinese language input/output
- Adjust test phrases based on your language requirements
- Some tests require manual interaction (speaking, screen changes)
- Keep service console windows visible to monitor logs
- Save test results and logs for debugging

