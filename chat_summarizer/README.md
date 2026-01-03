# Chat Summarizer Service

Standalone service for analyzing Bilibili live chat messages using LLM.

## Features

- **Direct Bilibili Integration**: Connects to Bilibili live room and collects messages in real-time
- **Message Buffering**: Maintains a time-based buffer of recent danmaku and superchat messages
- **LLM Analysis**: Uses configured LLM (Gemini or Ollama) to analyze chat sentiment and identify interesting messages
- **HTTP API**: Simple REST API for triggering analysis and checking status

## Architecture

This service is completely standalone and does not depend on the orchestrator. It:

1. Connects directly to Bilibili using the existing `BilibiliClient`
2. Buffers messages with TTL-based cleanup (extracted from `BilibiliSource` logic)
3. Provides HTTP endpoints to trigger LLM-based analysis
4. Reuses the project's LLM providers via `llm_factory`

## Configuration

The service uses existing configuration from `config.yaml`:

```yaml
bilibili:
  room_id: 123456  # Your Bilibili live room ID
  sessdata: "your_sessdata_here"  # Optional authentication
  danmaku_ttl_seconds: 300  # How long to keep messages in buffer

llm:
  provider: ollama  # or gemini
  providers:
    ollama:
      model: Qwen3-8B-Q4-4kcontext
      # ... other LLM settings
```

No additional configuration needed!

## Usage

### Start the Service

```bash
python -m chat_summarizer.server
```

The service will start on `http://localhost:8005` and begin collecting messages.

### Automatic Periodic Summarization

To automatically trigger summarization at regular intervals, use the auto-summarizer script:

```bash
# Summarize every 30 seconds (analyzes ALL messages in buffer)
python scripts/auto_summarize.py --interval 30

# Or use the batch script on Windows
scripts\start_auto_summarize.bat
```

**Options:**
- `--interval` / `-i`: Seconds between summarizations (default: 30)
- `--max-messages` / `-m`: Limit number of messages to analyze (default: all messages)
- `--time-window` / `-w`: Only analyze messages from last N seconds (optional)
- `--url` / `-u`: Service URL (default: http://localhost:8005)

**Examples:**
```bash
# Analyze all messages every 60 seconds (default behavior)
python scripts/auto_summarize.py -i 60

# Limit to 100 most recent messages
python scripts/auto_summarize.py -i 30 -m 100

# Only analyze messages from last 5 minutes
python scripts/auto_summarize.py -i 30 -w 300
```

The script will continuously run and print summaries to the console. Press Ctrl+C to stop.

### Analyze Recent Chat

**Request:**
```bash
# Analyze all messages in buffer
curl -X POST http://localhost:8005/summarize \
  -H "Content-Type: application/json" \
  -d '{}'

# Or limit to 50 most recent messages
curl -X POST http://localhost:8005/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "max_messages": 50
  }'
```

**Parameters:**
- `max_messages` (optional): Number of recent messages to analyze (default: all messages)
- `time_window_seconds` (optional): Only analyze messages from last N seconds

**Response:**
```json
{
  "overall_sentiment": "Chat is excited about the new game feature, lots of questions about mechanics",
  "most_interesting_message": {
    "id": "dm_1234567890_12345",
    "type": "danmaku",
    "user": "游戏玩家123",
    "content": "这个新机制会不会影响PVP平衡？",
    "timestamp": 1234567890.0,
    "amount": null
  },
  "reasoning": "This message asks a thoughtful question about game balance that would make for interesting discussion",
  "messages_analyzed": 50,
  "timestamp": "2024-01-15T10:30:00"
}
```

### Check Buffer Status

```bash
curl http://localhost:8005/buffer
```

**Response:**
```json
{
  "danmaku_count": 120,
  "superchat_count": 3,
  "total_count": 123,
  "oldest_timestamp": 1234567800.0,
  "newest_timestamp": 1234567890.0,
  "sample_messages": [
    {
      "type": "danmaku",
      "user": "用户123",
      "content": "666"
    },
    {
      "type": "superchat",
      "user": "土豪456",
      "content": "主播加油！",
      "amount": 50.0
    }
  ]
}
```

### Health Check

```bash
curl http://localhost:8005/health
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service information |
| `/health` | GET | Health check (auto-provided by core.server) |
| `/summarize` | POST | Analyze recent chat messages with LLM |
| `/buffer` | GET | Get buffer statistics and sample messages |

## Code Structure

```
chat_summarizer/
├── __init__.py           # Package initialization
├── message_buffer.py     # ChatMessageBuffer - handles Bilibili connection and buffering
├── models.py             # Pydantic models for API requests/responses
├── summarizer.py         # ChatSummarizer - LLM analysis logic
├── api.py                # FastAPI routes
└── server.py             # Main entry point with lifespan management
```

## Components Reused

This service maximally reuses existing code:

- `bilibili/bilibili_client.py` - Bilibili connection management
- `llm/providers/*` - LLM provider implementations (Gemini, Ollama)
- `orchestrator/utils/llm_factory.py` - LLM provider factory
- `core/server.py` - FastAPI app creation with health endpoint
- `core/settings.py` - Configuration management
- `core/logging.py` - Logging utilities

The buffering logic is extracted from `orchestrator/sources/bilibili_source.py` but simplified to remove event bus dependency.

## How It Works

1. **Startup**: Service connects to Bilibili live room and starts collecting messages
2. **Collection**: Messages (danmaku and superchat) are stored in a time-based buffer
3. **Cleanup**: Background task removes expired messages based on TTL
4. **Analysis**: When `/summarize` is called:
   - Recent messages are retrieved from buffer
   - Messages are formatted and sent to LLM
   - LLM analyzes sentiment and selects most interesting message
   - Results are returned via API and logged

## LLM Prompt Strategy

The service sends messages to the LLM with instructions to:
1. Summarize overall sentiment and topics (1-2 sentences)
2. Identify the most interesting message that deserves a response
3. Provide reasoning for the selection

The LLM looks for:
- Thoughtful questions
- Insightful comments
- Funny or entertaining content
- Meaningful contributions
- Superchats that deserve acknowledgment

## Future Enhancements

Possible improvements (not currently implemented):

- [ ] Add service to `scripts/start_services.py` for unified management
- [x] Automatic periodic summarization (timer-based) - **DONE** via `scripts/auto_summarize.py`
- [ ] Publish summaries to event bus for orchestrator integration
- [ ] Persistent storage of summaries
- [ ] Web UI dashboard for viewing summaries
- [ ] Support for multiple Bilibili rooms
- [ ] Configurable port in config.yaml

## Dependencies

All dependencies already exist in the project:
- `blivedm` - Bilibili live client library
- `google.genai` - Gemini LLM API
- `ollama` - Local LLM support
- `fastapi` - Web framework
- `pydantic` - Data validation

## Testing

Manual testing steps:

1. Ensure `bilibili.room_id` is configured in `config.yaml`
2. Start the service: `python -m chat_summarizer.server`
3. Verify Bilibili connection in logs
4. Wait for some messages to arrive (check `/buffer` endpoint)
5. Trigger summarization: `POST /summarize`
6. Check logs for LLM analysis results

## Notes

- The service runs independently and does not require the orchestrator
- Messages are stored in memory (not persisted to disk)
- TTL cleanup ensures the buffer doesn't grow indefinitely
- The LLM may occasionally fail to return valid JSON - the service handles this gracefully
