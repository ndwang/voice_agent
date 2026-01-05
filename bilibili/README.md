# Bilibili Live Chat Service

A standalone service for streaming Bilibili live chat messages (danmaku and superchat) via WebSocket. Provides real-time message access for the orchestrator, OBS overlay dashboards, and custom LLM clients.

## Features

- **Standalone Operation**: Run independently of the main voice agent
- **WebSocket Streaming**: Real-time push of danmaku and superchat messages
- **Multiple Clients**: Support multiple simultaneous WebSocket connections
- **In-Memory Buffers**: Danmaku with 60s TTL, persistent superchat buffer
- **OBS Integration**: Transparent overlay for streaming with customizable styling
- **Full Dashboard**: Web control panel with live chat viewer and statistics
- **REST API**: Full control over service state and configuration
- **Auto-Reconnection**: Resilient connection handling with exponential backoff

## Quick Start

### 1. Configuration

Create or edit `bilibili/config.yaml`:

```yaml
service:
  host: 0.0.0.0
  port: 8002
  log_level: INFO

bilibili:
  room_id: 21379697  # Your Bilibili room ID
  sessdata: "your_sessdata_cookie"  # Get from browser cookies
  danmaku_ttl_seconds: 60
  enabled: true  # Auto-connect on startup
  danmaku_enabled_default: true
  superchat_enabled_default: true

dashboard:
  default_theme: dark
  default_max_messages: 20
  default_font_size: 16
```

**Getting your SESSDATA cookie:**
1. Log in to bilibili.com in your browser
2. Open Developer Tools (F12) → Application → Cookies
3. Find the `SESSDATA` cookie and copy its value
4. Paste into `bilibili/config.yaml`

### 2. Start Service

```bash
# Standalone
uv run python -m bilibili.server

# Or with all services
uv run python scripts/start_services.py
```

### 3. Verify Service

```bash
# Check health
curl http://localhost:8002/health

# Get current messages
curl http://localhost:8002/chat

# Open dashboard
# Browser: http://localhost:8002/dashboard.html
```

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    Bilibili Service (8002)                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ BilibiliManager                                          │  │
│  │  ├── BilibiliClient (wraps blivedm library)              │  │
│  │  ├── danmaku_buffer (Deque, 60s TTL)                     │  │
│  │  ├── superchat_buffer (Deque, persistent)                │  │
│  │  └── ws_clients (Set[WebSocket])                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  REST API              WebSocket API          Static Files     │
│  ├── GET /health       /ws/stream             /dashboard.html  │
│  ├── GET /chat                                /obs.html        │
│  ├── GET /state                                                │
│  └── POST /state/*                                             │
└────────────────────────────────────────────────────────────────┘
                              ↓ WebSocket
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   Orchestrator          Dashboard (OBS)    LLM Summarizer
   (WebSocket client)    (Browser source)   (WebSocket client)
```

## REST API Reference

### Health & Info

#### `GET /`
Service info and health check.

**Response:**
```json
{
  "service": "bilibili",
  "status": "ok",
  "version": "1.0.0"
}
```

#### `GET /health`
Detailed health status.

**Response:**
```json
{
  "service": "bilibili",
  "status": "healthy",
  "connected": true,
  "running": true,
  "danmaku_enabled": true,
  "superchat_enabled": true,
  "message_counts": {
    "danmaku": 142,
    "superchat": 5
  },
  "uptime_seconds": 3600.5
}
```

### Messages

#### `GET /chat`
Get current message buffers (both danmaku and superchat).

**Response:**
```json
{
  "danmaku": [
    {
      "id": "dm_1234567890_123456",
      "user": "username",
      "content": "message text",
      "timestamp": 1234567890.123
    }
  ],
  "superchat": [
    {
      "id": "sc_1234567890_123456",
      "user": "username",
      "content": "message text",
      "timestamp": 1234567890.123,
      "amount": 30.0
    }
  ]
}
```

#### `GET /chat/danmaku`
Get danmaku messages only.

#### `GET /chat/superchat`
Get superchat messages only.

### State Management

#### `GET /state`
Get current service state.

**Response:**
```json
{
  "connected": true,
  "running": true,
  "danmaku_enabled": true,
  "superchat_enabled": true
}
```

#### `POST /state/danmaku/enable`
Enable danmaku message processing.

#### `POST /state/danmaku/disable`
Disable danmaku message processing.

#### `POST /state/superchat/enable`
Enable superchat message processing.

#### `POST /state/superchat/disable`
Disable superchat message processing.

### Connection Control

#### `POST /connect`
Connect to Bilibili room (if not already connected).

#### `POST /disconnect`
Disconnect from Bilibili room.

### Configuration

#### `GET /config`
Get current configuration.

**Response:**
```json
{
  "room_id": 21379697,
  "danmaku_ttl_seconds": 60,
  "danmaku_enabled_default": true,
  "superchat_enabled_default": true
}
```

#### `POST /config`
Update configuration (requires service restart for some settings).

**Request:**
```json
{
  "room_id": 21379697,
  "danmaku_ttl_seconds": 90
}
```

### Statistics

#### `GET /stats`
Get service statistics.

**Response:**
```json
{
  "total_danmaku": 1420,
  "total_superchat": 42,
  "current_danmaku": 23,
  "current_superchat": 5,
  "uptime_seconds": 7200.5,
  "connected_clients": 3,
  "messages_per_second": {
    "danmaku": 2.3,
    "superchat": 0.01
  }
}
```

## WebSocket API

### Endpoint: `ws://localhost:8002/ws/stream`

Real-time message streaming with channel-based subscriptions.

### Client → Server Messages

#### Subscribe to Channels
```json
{
  "type": "subscribe",
  "channels": ["danmaku", "superchat"]
}
```

Available channels: `danmaku`, `superchat`

#### Unsubscribe from Channels
```json
{
  "type": "unsubscribe",
  "channels": ["danmaku"]
}
```

#### Heartbeat
```json
{
  "type": "ping"
}
```

### Server → Client Messages

#### Danmaku Message
```json
{
  "type": "danmaku",
  "data": {
    "id": "dm_1234567890_123456",
    "user": "username",
    "content": "message text",
    "timestamp": 1234567890.123
  }
}
```

#### Superchat Message
```json
{
  "type": "superchat",
  "data": {
    "id": "sc_1234567890_123456",
    "user": "username",
    "content": "message text",
    "timestamp": 1234567890.123,
    "amount": 30.0
  }
}
```

#### State Change Notification
```json
{
  "type": "state_changed",
  "data": {
    "danmaku_enabled": true,
    "superchat_enabled": true
  }
}
```

#### Heartbeat Response
```json
{
  "type": "pong"
}
```

#### Error
```json
{
  "type": "error",
  "message": "Error details"
}
```

## Dashboards

### Full Dashboard

**URL:** `http://localhost:8002/dashboard.html`

Full-featured control panel with:
- Live chat viewer with auto-scroll
- Connection status indicator (connected/disconnected)
- Enable/disable toggles for danmaku and superchat
- Statistics panel (message counts, rates, uptime)
- Connect/disconnect buttons
- Configuration viewer

Perfect for:
- Testing and debugging
- Manual service control
- Monitoring message flow

### OBS Overlay

**URL:** `http://localhost:8002/obs.html`

Minimal, transparent overlay for OBS browser source:
- Transparent background (for chroma key)
- Auto-scrolling chat messages
- Highlighted superchat with amount
- Fade-in animations for new messages
- No controls or UI elements

**URL Parameters:**
- `theme=dark|light` - Color theme (default: dark)
- `show_danmaku=true|false` - Show danmaku messages (default: true)
- `show_superchat=true|false` - Show superchat messages (default: true)
- `font_size=16` - Font size in pixels (default: 16)
- `max_messages=20` - Max messages to display (default: 20)

**Example URLs:**
```
http://localhost:8002/obs.html?theme=dark&max_messages=20
http://localhost:8002/obs.html?show_danmaku=true&show_superchat=false&font_size=18
```

### OBS Setup Guide

1. **Add Browser Source:**
   - In OBS, add a new "Browser" source
   - Set URL to `http://localhost:8002/obs.html?theme=dark&max_messages=20`
   - Set width/height to match your layout (e.g., 600x800)

2. **Configure Display:**
   - Adjust font size and max messages via URL parameters
   - Position on your stream layout
   - Use "Shutdown source when not visible" for performance

3. **Styling Tips:**
   - Use dark theme for dark overlays, light for bright backgrounds
   - Adjust font size based on stream resolution
   - Limit max messages to prevent clutter (15-25 recommended)

4. **Performance:**
   - Enable "Shutdown source when not visible"
   - Use CSS hardware acceleration (already enabled)
   - Minimize browser source resolution

## Client Examples

### Python WebSocket Client

Example client for LLM chat summarization:

```python
import asyncio
import aiohttp
import json

class BilibiliLLMClient:
    """Example client that consumes Bilibili chat messages."""

    def __init__(self, service_url: str = "ws://localhost:8002/ws/stream"):
        self.service_url = service_url
        self.ws = None
        self.running = False

    async def start(self):
        """Connect and start listening."""
        self.running = True
        await self._connect_and_listen()

    async def stop(self):
        """Disconnect."""
        self.running = False
        if self.ws:
            await self.ws.close()

    async def _connect_and_listen(self):
        """Connect to WebSocket and process messages."""
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(self.service_url) as ws:
                self.ws = ws
                print("Connected to Bilibili service")

                # Subscribe to both channels
                await ws.send_json({
                    "type": "subscribe",
                    "channels": ["danmaku", "superchat"]
                })

                # Listen for messages
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        await self._handle_message(data)

    async def _handle_message(self, data: dict):
        """Process incoming message."""
        msg_type = data.get("type")

        if msg_type == "danmaku":
            message = data["data"]
            print(f"[Danmaku] {message['user']}: {message['content']}")
            # TODO: Add to summarization buffer

        elif msg_type == "superchat":
            message = data["data"]
            print(f"[Superchat ¥{message['amount']}] {message['user']}: {message['content']}")
            # TODO: Prioritize in summarization

if __name__ == "__main__":
    client = BilibiliLLMClient()
    try:
        asyncio.run(client.start())
    except KeyboardInterrupt:
        print("Shutting down...")
```

### JavaScript WebSocket Client

```javascript
const ws = new WebSocket('ws://localhost:8002/ws/stream');

ws.onopen = () => {
  console.log('Connected to Bilibili service');

  // Subscribe to both channels
  ws.send(JSON.stringify({
    type: 'subscribe',
    channels: ['danmaku', 'superchat']
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === 'danmaku') {
    console.log(`[Danmaku] ${data.data.user}: ${data.data.content}`);
    // Handle danmaku message
  } else if (data.type === 'superchat') {
    console.log(`[Superchat ¥${data.data.amount}] ${data.data.user}: ${data.data.content}`);
    // Handle superchat message
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Disconnected from Bilibili service');
  // Implement reconnection logic
};
```

## Integration with Orchestrator

The orchestrator connects to the Bilibili service via `BilibiliWebSocketSource` (see `orchestrator/sources/bilibili_websocket_source.py`):

1. **Connection:** Opens WebSocket to `ws://localhost:8002/ws/stream`
2. **Subscription:** Subscribes to `danmaku` and `superchat` channels
3. **Event Publishing:** Publishes received messages to orchestrator EventBus as:
   - `BILIBILI_DANMAKU` events (for danmaku)
   - `BILIBILI_SUPERCHAT` events (for superchat)
4. **Auto-Reconnection:** Handles connection loss with exponential backoff

The orchestrator's `QueueManager` subscribes to these events and enqueues messages with appropriate priorities (P2 for danmaku, P2 for superchat).

## Message Buffer Management

### Danmaku Buffer

- **Type:** `collections.deque` with maxlen
- **TTL:** 60 seconds (configurable via `danmaku_ttl_seconds`)
- **Pruning:** Background cleanup task removes expired messages every 5 seconds
- **Purpose:** Keep recent chat context without unbounded memory growth

### Superchat Buffer

- **Type:** `collections.deque` with maxlen
- **TTL:** None (persistent until service restart)
- **Purpose:** Superchat is important and should persist longer

Both buffers are in-memory only. Messages are not persisted to disk.

## Troubleshooting

### Service Won't Start

**Issue:** `Config not found` error

**Solution:** Ensure `bilibili/config.yaml` exists with valid configuration. Copy from `bilibili/config.example.yaml` if needed.

---

**Issue:** Port 8002 already in use

**Solution:** Change port in `bilibili/config.yaml` or kill existing process:
```bash
# Windows
netstat -ano | findstr :8002
taskkill /PID <pid> /F

# Linux/Mac
lsof -ti:8002 | xargs kill
```

### No Messages Received

**Issue:** Connected but no danmaku/superchat appears

**Solutions:**
1. Check if Bilibili room is live: `curl http://localhost:8002/state`
2. Verify `danmaku_enabled` and `superchat_enabled` are true
3. Check if SESSDATA cookie is valid (may expire after ~6 months)
4. Verify room_id is correct: `curl http://localhost:8002/config`

### WebSocket Connection Fails

**Issue:** Client can't connect to `ws://localhost:8002/ws/stream`

**Solutions:**
1. Verify service is running: `curl http://localhost:8002/health`
2. Check firewall settings (Windows may block local WebSocket)
3. Try connecting from browser DevTools console:
   ```javascript
   const ws = new WebSocket('ws://localhost:8002/ws/stream');
   ws.onopen = () => console.log('Connected');
   ws.onerror = (e) => console.error('Error:', e);
   ```

### OBS Overlay Not Displaying

**Issue:** OBS browser source shows blank

**Solutions:**
1. Verify service is running and dashboard loads in regular browser
2. Check OBS browser source settings:
   - URL is correct (with http://, not ws://)
   - Width/height are set
   - "Shutdown source when not visible" may prevent initial load
3. Check OBS logs for browser errors
4. Try opening URL directly in browser first to verify it works

### High Memory Usage

**Issue:** Service memory grows over time

**Solutions:**
1. Reduce `danmaku_ttl_seconds` in config (default: 60s)
2. Reduce max_messages in OBS overlay URL parameters
3. Check for WebSocket client leaks (clients not properly disconnecting)
4. Restart service periodically if running 24/7

## Development

### Running Tests

```bash
# Unit tests
uv run pytest tests/bilibili/

# Integration tests
uv run pytest tests/integration/test_bilibili.py
```

### Adding New Features

1. **New REST endpoint:** Add to `bilibili/api.py`
2. **New WebSocket message type:** Update protocol in `bilibili/manager.py`
3. **Dashboard changes:** Edit `bilibili/static/dashboard.html` or `obs.html`
4. **Configuration changes:** Update `bilibili/settings.py` and `config.yaml`

### Code Structure

```
bilibili/
├── __init__.py           # Package init
├── bilibili_client.py    # BilibiliClient (wraps blivedm)
├── blivedm/             # Bundled Bilibili live chat library
├── config.yaml          # Service configuration
├── settings.py          # Configuration loader (Pydantic models)
├── server.py            # Service entry point
├── manager.py           # BilibiliManager (core business logic)
├── api.py               # FastAPI routes (REST + WebSocket)
├── models.py            # Pydantic request/response models
├── static/              # Dashboard files
│   ├── dashboard.html   # Full dashboard
│   ├── obs.html        # OBS overlay
│   └── style.css       # Shared styles
└── README.md           # This file
```

## License

Part of the Voice Agent project. See main project LICENSE for details.

## Support

For issues and questions:
- Check existing issues in the main project repository
- Review CLAUDE.md for architecture details
- See `orchestrator/sources/bilibili_websocket_source.py` for integration example
