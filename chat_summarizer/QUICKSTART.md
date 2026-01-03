# Chat Summarizer - Quick Start Guide

## Step 1: Configure Bilibili Connection

Make sure your `config.yaml` has the Bilibili room configured:

```yaml
bilibili:
  room_id: 123456  # Your Bilibili live room ID
  sessdata: "your_sessdata_here"  # Optional
  danmaku_ttl_seconds: 300
```

## Step 2: Start the Chat Summarizer Service

Open a terminal and run:

```bash
python -m chat_summarizer.server
```

You should see:
```
============================================================
Chat Summarizer Service
============================================================
Starting server on 0.0.0.0:8005...
✓ Chat Summarizer Service started successfully
  - Bilibili room: 123456
  - LLM provider: ollama
  - Message TTL: 300s
```

The service is now collecting messages from the Bilibili live room!

## Step 3: Choose Your Workflow

### Option A: Manual Summarization (API Calls)

Trigger summarization manually when you want:

```bash
curl -X POST http://localhost:8005/summarize \
  -H "Content-Type: application/json" \
  -d '{"max_messages": 50}'
```

### Option B: Automatic Periodic Summarization (Recommended)

Open a **second terminal** and run:

```bash
# Analyze ALL messages every 30 seconds
python scripts/auto_summarize.py --interval 30
```

Or on Windows, just double-click:
```
scripts\start_auto_summarize.bat
```

This will automatically trigger summarization every 30 seconds and print results to the console:

```
================================================================================
📊 CHAT SUMMARY - 2024-01-15T10:30:00
================================================================================
Messages analyzed: 45

Overall Sentiment:
  Chat is excited about the new game feature, lots of questions about mechanics

Most Interesting Message: [danmaku]
  User: 游戏玩家123
  Content: 这个新机制会不会影响PVP平衡？

Reasoning:
  This message asks a thoughtful question about game balance that would make
  for interesting discussion
================================================================================
```

## Step 4: Check Service Status

At any time, check the buffer status:

```bash
curl http://localhost:8005/buffer
```

## Common Use Cases

### Use Case 1: Monitor Live Stream Chat During Broadcast

```bash
# Terminal 1: Start service
python -m chat_summarizer.server

# Terminal 2: Auto-summarize every minute (all messages)
python scripts/auto_summarize.py -i 60
```

### Use Case 2: Quick Check on Demand

```bash
# Start service in background
python -m chat_summarizer.server &

# Manually trigger when needed (analyze all messages)
curl -X POST http://localhost:8005/summarize -H "Content-Type: application/json" -d '{}'
```

### Use Case 3: Recent Activity Only

```bash
# Only analyze messages from last 5 minutes
python scripts/auto_summarize.py -i 30 -w 300
```

## Stopping the Services

- **Chat Summarizer Service**: Press `Ctrl+C` in the terminal running the service
- **Auto Summarizer Script**: Press `Ctrl+C` in the terminal running the script

## Troubleshooting

### "Service not running" error

Make sure the chat summarizer service is started first:
```bash
python -m chat_summarizer.server
```

### "No messages to analyze"

Wait for some messages to arrive in the Bilibili room, or check:
```bash
curl http://localhost:8005/buffer
```

### LLM errors

Check that your LLM provider is configured correctly in `config.yaml`:
```yaml
llm:
  provider: ollama  # or gemini
  providers:
    ollama:
      model: Qwen3-8B-Q4-4kcontext
      base_url: http://localhost:11434
```

For Gemini, make sure `GEMINI_API_KEY` environment variable is set.

## Next Steps

- Integrate summaries into your streaming workflow
- Use the API to build custom automation
- Adjust the interval and message count based on chat activity
- See `README.md` for full documentation

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  Bilibili Live Room                                     │
└────────────────┬────────────────────────────────────────┘
                 │ (danmaku & superchat messages)
                 ▼
┌─────────────────────────────────────────────────────────┐
│  Chat Summarizer Service (port 8005)                    │
│  ┌──────────────────────────────────────────────────┐   │
│  │  ChatMessageBuffer                               │   │
│  │  - Collects messages in real-time                │   │
│  │  - TTL-based expiration (5 min default)          │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │  HTTP API                                        │   │
│  │  - POST /summarize                               │   │
│  │  - GET /buffer                                   │   │
│  │  - GET /health                                   │   │
│  └──────────────────────────────────────────────────┘   │
└────────────────┬────────────────────────────────────────┘
                 │ HTTP requests
                 ▼
┌─────────────────────────────────────────────────────────┐
│  Auto Summarizer Script (optional)                      │
│  - Triggers /summarize every X seconds                  │
│  - Prints results to console                            │
└─────────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  LLM Provider (Gemini or Ollama)                        │
│  - Analyzes sentiment                                   │
│  - Selects interesting messages                         │
└─────────────────────────────────────────────────────────┘
```

Enjoy using the Chat Summarizer! 🎉
