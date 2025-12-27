# Interaction Flow & Logic

The `InteractionManager` is responsible for the main loop of the voice agent. It translates user speech into agent responses by orchestrating the flow between STT, LLM, and TTS.

## The Interaction Pipeline

The standard flow follows these steps:

1. **Transcript Finalized**: `STTSource` publishes `TRANSCRIPT_FINAL`.
2. **Context Preparation**: `InteractionManager` asks `ContextManager` for the conversation history + system prompt.
3. **LLM Generation**: The prompt is sent to the LLM provider. Tokens are streamed back.
4. **Stream Parsing**: Tokens are passed through a parser.
5. **TTS Request**: When a complete sentence or tagged block is identified, a `TTS_REQUEST` is published.
6. **Audio Playback**: `TTSManager` receives the request, streams it to the TTS Service, and sends the resulting audio chunks to the `AudioPlayer`.

### Standard Flow Sequence Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant AD as Audio Driver
    participant STT_SVC as STT Service
    participant STT_SRC as STT Source
    participant EB as Event Bus
    participant IM as Interaction Manager
    participant CM as Context Manager
    participant LLM as LLM Provider
    participant Parser as Stream Parser
    participant TTS_M as TTS Manager
    participant TTS_SVC as TTS Service
    participant AP as Audio Player

    U->>AD: Speaks
    AD->>STT_SVC: Audio stream (WebSocket)
    STT_SVC->>STT_SRC: Transcript
    STT_SRC->>EB: Event: TRANSCRIPT_FINAL
    EB->>IM: Event: TRANSCRIPT_FINAL
    
    IM->>CM: Get conversation history + system prompt
    CM-->>IM: Context (messages + system prompt)
    
    IM->>LLM: Generate stream (messages, system_prompt)
    
    loop For each token
        LLM-->>IM: Stream token
        IM->>EB: Event: LLM_TOKEN (for UI)
        IM->>Parser: Process token
        Parser->>Parser: Buffer until sentence complete
        
        opt When sentence or tagged block complete
            Parser->>IM: Sentence ready
            IM->>EB: Event: TTS_REQUEST (text)
            EB->>TTS_M: Event: TTS_REQUEST
            
            TTS_M->>TTS_SVC: Text chunk (WebSocket)
            TTS_SVC-->>TTS_M: Audio chunk
            TTS_M->>AP: Play audio chunk
            AP->>U: Audio output
        end
    end
    
    LLM-->>IM: Stream complete
    IM->>EB: Event: LLM_RESPONSE_DONE
    EB->>TTS_M: Event: LLM_RESPONSE_DONE
    TTS_M->>TTS_SVC: Finalize stream
    TTS_SVC-->>TTS_M: Final audio chunks
    TTS_M->>AP: Play remaining audio
    AP->>U: Audio output
```

## Interruption Logic

One of the most complex parts of the system is handling interruptions (when the user starts speaking while the agent is still responding).

```mermaid
sequenceDiagram
    participant U as User
    participant AD as Audio Driver
    participant STT as STT Service
    participant IM as Interaction Manager
    participant TTS as TTS Manager

    IM->>TTS: Playing response...
    U->>AD: (User starts speaking)
    AD->>STT: Audio stream
    STT-->>IM: Event: SPEECH_START
    IM->>IM: Cancel current LLM generation
    IM->>TTS: Stop playback & Clear queue
    STT-->>IM: Event: TRANSCRIPT_FINAL
    IM->>IM: Start new interaction
```

### Concatenation Logic
If the user interrupts the agent *before* it finishes its sentence, the system can be configured to concatenate the interrupted prompt with the next user input, ensuring the agent understands the context of the unfinished thought. Currently this is commented out.

