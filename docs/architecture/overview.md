# System Architecture Overview

The Voice Agent is built on a **Microservices Architecture** using an **Event-Driven** core. This design ensures that each component (STT, TTS, LLM) can operate independently and asynchronously, providing a responsive experience.

## High-Level Diagram

```mermaid
graph TD
    subgraph Client [Client Side]
        AD[Audio Driver]
        UI[Web UI]
    end

    subgraph Orchestrator [Orchestrator - Port 8000]
        EB[Event Bus]
        IM[Interaction Manager]
        CM[Context Manager]
        STT_S[STT Source]
        TTS_M[TTS Manager]
    end

    subgraph Services [Back-end Services]
        STT_SVC[STT Service - Port 8001]
        TTS_SVC[TTS Service - Port 8003]
        LLM[LLM Provider - Ollama/Gemini]
    end

    AD -- Audio Stream --> STT_SVC
    STT_SVC -- Transcripts --> STT_S
    STT_S -- Event --> EB
    EB -- Event --> IM
    IM -- Request --> LLM
    LLM -- Stream --> IM
    IM -- Event --> EB
    EB -- Event --> TTS_M
    TTS_M -- Text Stream --> TTS_SVC
    TTS_SVC -- Audio Chunks --> TTS_M
    TTS_M -- Play Audio --> AD
```

## Core Components

### 1. Orchestrator
The "brain" of the system. It doesn't perform heavy AI processing itself but coordinates between services. It maintains the **Event Bus**, manages conversation context, and handles the logic for starting/stopping interactions.

### 2. Event Bus
A simple, internal pub/sub system. Every major action (transcript received, user spoke, LLM generated a token, TTS finished) is an **Event**. This allows components like the `SubtitleManager` or `LatencyTracker` to "listen" to the system without being tightly coupled to the main logic.

### 3. Audio Driver
A local component that handles hardware interaction. It uses **Voice Activity Detection (VAD)** to capture microphone input only when speech is detected and handles the real-time playback of audio chunks received from the TTS service.

### 4. AI Services (STT & TTS)
Stateless services that wrap specific AI models. They expose WebSocket APIs for real-time streaming, allowing the Orchestrator to pipe text or audio through them with minimal overhead.


