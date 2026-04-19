# Bilibili Live Chat Service

A standalone service for streaming Bilibili live room events via WebSocket. Captures danmaku, superchat, gifts, and guard purchases in real-time, exposing them through REST and WebSocket APIs for the orchestrator, OBS overlays, and custom clients.

## Features

- **Message Coverage**: Danmaku (chat), superchat (paid messages), gifts, guard/fleet purchases
- **Rich Message Data**: User avatars, fan medals, guard badges, admin flags, emoticon images, superchat background colors, gift icons, coin totals
- **Superchat Lifecycle**: Tracks superchat creation and deletion; deleted superchats are removed from the buffer and broadcast to clients
- **WebSocket Streaming**: Real-time push with channel-based subscriptions (`danmaku`, `superchat`, `gift`)
- **Channel Filtering**: Clients only receive messages for channels they subscribe to; `state_changed` events are always sent
- **Multiple Clients**: Supports multiple simultaneous WebSocket connections
- **Configurable Buffers**: Separate max buffer sizes for danmaku, superchat, and gift messages (bounded deques)
- **Auto-Reconnection**: Exponential backoff reconnection on Bilibili WebSocket disconnect (configurable delay/max)
- **OBS Integration**: Transparent overlay with gift display, guard announcements, emoticon rendering, badge display, superchat colors
- **Full Dashboard**: Web control panel with live chat viewer, avatars, badges, scroll-pause, and statistics
- **REST API**: Endpoints for messages, service state, statistics, and configuration

## Quick Start

### 1. Configuration

Create or edit `bilibili/config.yaml`:

```yaml
service:
  host: 0.0.0.0
  port: 8002
  log_level: INFO

bilibili:
  room_id: 21379697            # Your Bilibili room ID
  sessdata: "your_sessdata"    # Get from browser cookies
  danmaku_max_buffer: 60       # Max danmaku messages to retain
  superchat_max_buffer: 100    # Max superchat messages to retain
  gift_max_buffer: 100         # Max gift/guard messages to retain
  enabled: true                # Auto-connect on startup
  reconnect_delay_seconds: 1.0      # Initial reconnect delay
  reconnect_max_delay_seconds: 30.0 # Max reconnect delay (exponential backoff)

dashboard:
  default_theme: dark
  default_max_messages: 20
  default_font_size: 16
```

**Getting your SESSDATA cookie:**
1. Log in to bilibili.com in your browser
2. Open Developer Tools (F12) -> Application -> Cookies
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
+--------------------------------------------------------------------+
|                    Bilibili Service (8002)                          |
|  +--------------------------------------------------------------+  |
|  | BilibiliManager                                               |  |
|  |  +-- BilibiliClient (wraps blivedm library)                   |  |
|  |  |   +-- on_danmaku, on_super_chat, on_gift                  |  |
|  |  |   +-- on_guard (user_toast_v2), on_super_chat_delete       |  |
|  |  |   +-- on_client_stopped (triggers auto-reconnect)          |  |
|  |  +-- danmaku_buffer (Deque, bounded)                          |  |
|  |  +-- superchat_buffer (Deque, bounded)                        |  |
|  |  +-- gift_buffer (Deque, bounded)                             |  |
|  |  +-- ws_clients (Dict[WebSocket, Set[channel]])               |  |
|  +--------------------------------------------------------------+  |
|                                                                     |
|  REST API              WebSocket API          Static Files          |
|  +-- GET /health       /ws/stream             /dashboard.html       |
|  +-- GET /chat                                /obs.html             |
|  +-- GET /chat/danmaku                                              |
|  +-- GET /chat/superchat                                            |
|  +-- GET /chat/gift                                                 |
|  +-- GET /state                                                     |
|  +-- GET /stats                                                     |
+--------------------------------------------------------------------+
                              | WebSocket
        +---------------------+---------------------+
        |                     |                     |
   Orchestrator          Dashboard (OBS)    Custom Client
   (WebSocket client)    (Browser source)   (WebSocket client)
```

## Message Types

Superchats use the `superchat` channel and buffer. Gifts and guards share the `gift` channel and buffer. Each message includes a `paid_type` field to distinguish its subtype.

### Danmaku (Chat Messages)

Standard chat messages from viewers.

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique message ID |
| `user` | string | Username |
| `uid` | int | User ID |
| `face` | string | Avatar URL |
| `content` | string | Message text |
| `timestamp` | float | Unix timestamp |
| `admin` | bool | Room administrator flag |
| `guard_level` | int | Guard level (0=none, 1=governor, 2=admiral, 3=captain) |
| `medal` | object | Fan medal (`{level, name}`) or null |
| `dm_type` | int | 0=text, 1=emoticon, 2=voice |
| `emoticon_url` | string | Emoticon image URL (when dm_type=1) |
| `color` | int | Danmaku color |
| `wealth_level` | int | User wealth/honor level |
| `privilege_type` | int | Guard privilege type |

### SuperChat (Paid Messages)

Paid highlighted messages. Broadcast with `paid_type: "superchat"`.

| Field | Type | Description |
|---|---|---|
| `id` | string | Internal message ID |
| `paid_type` | string | Always `"superchat"` |
| `bili_id` | int | Bilibili superchat ID (used for deletion tracking) |
| `user` | string | Username |
| `uid` | int | User ID |
| `face` | string | Avatar URL |
| `content` | string | Message text |
| `timestamp` | float | Unix timestamp |
| `amount` | float | Price in RMB |
| `guard_level` | int | Guard level |
| `medal` | object | Fan medal or null |
| `background_color` | string | Header background color |
| `background_bottom_color` | string | Body background color |
| `background_price_color` | string | Price area color |
| `start_time` | int | Display start timestamp |
| `end_time` | int | Display end timestamp |
| `message_trans` | string | Translation (if available) |

### Gifts

Gift events from viewers. Broadcast with `paid_type: "gift"`.

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique gift event ID |
| `paid_type` | string | Always `"gift"` |
| `user` | string | Username |
| `uid` | int | User ID |
| `face` | string | Avatar URL |
| `timestamp` | float | Unix timestamp |
| `gift_name` | string | Gift name |
| `gift_id` | int | Gift type ID |
| `num` | int | Quantity |
| `coin_type` | string | `"silver"` or `"gold"` |
| `total_coin` | int | Total coin value |
| `price` | int | Unit price in coins |
| `action` | string | Action text (e.g., "赠送") |
| `gift_icon` | string | Gift icon URL |
| `guard_level` | int | Guard level |
| `medal` | object | Fan medal or null |
| `combo_id` | string | Combo/transaction ID |

### Guard (Fleet Purchases)

Guard/fleet membership purchases. Broadcast with `paid_type: "guard"`.

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique event ID |
| `paid_type` | string | Always `"guard"` |
| `user` | string | Username |
| `uid` | int | User ID |
| `timestamp` | float | Unix timestamp |
| `guard_level` | int | 1=总督 (Governor), 2=提督 (Admiral), 3=舰长 (Captain) |
| `price` | int | Price in gold coins |
| `num` | int | Quantity |
| `unit` | string | Duration unit (e.g., "月") |
| `gift_name` | string | Guard gift name |
| `toast_msg` | string | Toast notification message |

## REST API Reference

### Health & Info

#### `GET /`
Service info and health check.

```json
{
  "service": "Bilibili Live Chat Service",
  "version": "1.0.0",
  "status": "running",
  "connected": true,
  "room_id": 21379697
}
```

#### `GET /health`
Detailed health status with buffer sizes.

```json
{
  "status": "healthy",
  "connected": true,
  "uptime": 3600.5,
  "buffer_health": {
    "danmaku_size": 42,
    "superchat_size": 3,
    "gift_size": 5
  },
  "client_health": {
    "ws_clients": 3
  }
}
```

### Messages

#### `GET /chat`
Get all message buffers at once.

```json
{
  "connected": true,
  "danmaku": [ ... ],
  "superchat": [ ... ],
  "gift": [ ... ]
}
```

#### `GET /chat/danmaku`
Danmaku messages only.

#### `GET /chat/superchat`
Superchat messages only.

#### `GET /chat/gift`
Gift and guard messages only.

### State Management

#### `GET /state`
```json
{
  "connected": true,
  "running": true,
  "room_id": 21379697
}
```

### Connection Control

#### `POST /connect`
Connect to the configured Bilibili room.

#### `POST /disconnect`
Disconnect from the Bilibili room.

### Configuration

#### `GET /config`
Returns the service configuration (sessdata excluded).

```json
{
  "service": { "host": "0.0.0.0", "port": 8002, "log_level": "INFO" },
  "bilibili": {
    "room_id": 21379697,
    "danmaku_max_buffer": 60,
    "superchat_max_buffer": 100,
    "gift_max_buffer": 100,
    "enabled": true,
    "reconnect_delay_seconds": 1.0,
    "reconnect_max_delay_seconds": 30.0
  },
  "dashboard": { "default_theme": "dark", "default_max_messages": 20, "default_font_size": 16 }
}
```

### Statistics

#### `GET /stats`

```json
{
  "danmaku_buffer_size": 42,
  "superchat_buffer_size": 3,
  "gift_buffer_size": 5,
  "uptime_seconds": 7200.5,
  "total_danmaku_received": 1420,
  "total_superchat_received": 15,
  "total_gift_received": 27,
  "total_gift_coins": 2850000
}
```

## WebSocket API

### Endpoint: `ws://localhost:8002/ws/stream`

Real-time message streaming with channel-based subscriptions. Clients must subscribe to channels to receive messages. `state_changed` events are always sent to all connected clients regardless of subscription.

### Available Channels

| Channel | Messages |
|---|---|
| `danmaku` | Danmaku chat messages |
| `superchat` | Superchat messages and superchat deletion events |
| `gift` | Gift and guard purchase messages |

### Client -> Server Messages

#### Subscribe to Channels
```json
{
  "type": "subscribe",
  "channels": ["danmaku", "superchat", "gift"]
}
```

Only valid channel names (`danmaku`, `superchat`, `gift`) are accepted; invalid names are silently ignored.

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

### Server -> Client Messages

#### Danmaku Message
```json
{
  "type": "danmaku",
  "data": {
    "id": "dm_0_123456",
    "user": "viewer_name",
    "uid": 123456,
    "face": "https://i0.hdslb.com/bfs/face/xxx.jpg",
    "content": "message text",
    "timestamp": 1234567890.123,
    "admin": false,
    "guard_level": 0,
    "medal": {"level": 15, "name": "medal_name"},
    "dm_type": 0,
    "emoticon_url": "",
    "color": 16777215,
    "wealth_level": 0,
    "privilege_type": 0
  }
}
```

#### Superchat Message
```json
{
  "type": "superchat",
  "data": {
    "id": "sc_1_123456",
    "paid_type": "superchat",
    "bili_id": 9876543,
    "user": "supporter",
    "uid": 123456,
    "face": "https://i0.hdslb.com/bfs/face/xxx.jpg",
    "content": "Great stream!",
    "timestamp": 1234567890.123,
    "amount": 30.0,
    "guard_level": 3,
    "medal": {"level": 20, "name": "medal_name"},
    "background_color": "#e69138",
    "background_bottom_color": "#d5831b",
    "background_price_color": "#e69138",
    "start_time": 1234567890,
    "end_time": 1234567950,
    "message_trans": ""
  }
}
```

#### Gift Message
```json
{
  "type": "gift",
  "data": {
    "id": "gift_2_123456",
    "paid_type": "gift",
    "user": "gifter",
    "uid": 123456,
    "face": "https://i0.hdslb.com/bfs/face/xxx.jpg",
    "timestamp": 1234567890.123,
    "gift_name": "小花花",
    "gift_id": 31036,
    "num": 5,
    "coin_type": "gold",
    "total_coin": 5000,
    "price": 1000,
    "action": "赠送",
    "gift_icon": "https://s1.hdslb.com/bfs/live/xxx.png",
    "guard_level": 0,
    "medal": null,
    "combo_id": "abc123"
  }
}
```

#### Guard Message
```json
{
  "type": "guard",
  "data": {
    "id": "guard_3_123456",
    "paid_type": "guard",
    "user": "captain_buyer",
    "uid": 123456,
    "timestamp": 1234567890.123,
    "guard_level": 3,
    "price": 198000,
    "num": 1,
    "unit": "月",
    "gift_name": "舰长",
    "toast_msg": "captain_buyer 在主播的直播间开通了舰长"
  }
}
```

#### Superchat Deletion
```json
{
  "type": "superchat_delete",
  "data": {
    "ids": [9876543, 9876544]
  }
}
```

#### State Change Notification
Sent to all connected clients regardless of channel subscription.

```json
{
  "type": "state_changed",
  "data": {
    "connected": true
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
- **Live chat viewer** displaying danmaku, superchat, gifts, and guard purchases
- **User avatars** next to each message
- **Badge display**: admin badges, guard level badges (Governor/Admiral/Captain), and fan medal badges with level
- **Superchat rendering** with background colors, translation text, and visual strikethrough for deleted superchats
- **Gift display** with gift icons, action text, quantity, and coin value
- **Guard announcements** with distinct styling
- **Emoticon rendering** for emoticon-type danmaku (dm_type=1)
- **Scroll-pause**: auto-scroll pauses when scrolled up, with a click-to-resume banner
- **Statistics panel**: total danmaku, superchat, and gift messages received, total gift coin value, uptime
- **Connection controls**: connect/disconnect buttons
- **Connection status indicator**

### OBS Overlay

**URL:** `http://localhost:8002/obs.html`

Minimal, transparent overlay for OBS browser source:
- Transparent background for compositing
- Danmaku, superchat, gifts, guard announcements
- User avatars and badge display
- Superchat with background colors and translation text
- Gift display with icons and quantity
- Guard purchase announcements with pulse animation
- Emoticon image rendering
- Fade-in/fade-out animations
- Deleted superchat handling (fades out removed superchats)

**URL Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `show_danmaku` | `true` | Show danmaku messages |
| `show_superchat` | `true` | Show superchat messages |
| `show_gift` | `true` | Show gift and guard messages |
| `font_size` | `16` | Font size in pixels |
| `max_messages` | `20` | Max messages to display |

**Example URLs:**
```
http://localhost:8002/obs.html?max_messages=20
http://localhost:8002/obs.html?show_gift=false&font_size=18
http://localhost:8002/obs.html?show_danmaku=true&show_superchat=true&show_gift=true&max_messages=15
```

### OBS Setup Guide

1. **Add Browser Source:**
   - In OBS, add a new "Browser" source
   - Set URL to `http://localhost:8002/obs.html?max_messages=20`
   - Set width/height to match your layout (e.g., 600x800)

2. **Configure Display:**
   - Adjust which message types to show via URL parameters
   - Adjust font size and max messages for your resolution

3. **Performance:**
   - Enable "Shutdown source when not visible"
   - Minimize browser source resolution

## Message Buffer Management

All buffers are in-memory only. Messages are not persisted to disk.

| Buffer | Max Size | Pruning |
|---|---|---|
| Danmaku | 60 (configurable via `danmaku_max_buffer`) | Bounded deque auto-evicts oldest |
| Superchat | 100 (configurable via `superchat_max_buffer`) | Bounded deque auto-evicts oldest + superchat deletion events |
| Gift | 100 (configurable via `gift_max_buffer`) | Bounded deque auto-evicts oldest |

## Auto-Reconnection

If the Bilibili WebSocket connection drops, the service detects it via the `on_client_stopped` callback and automatically reconnects:

- Initial delay: `reconnect_delay_seconds` (default: 1.0s)
- Delay doubles on each failed attempt (exponential backoff)
- Caps at `reconnect_max_delay_seconds` (default: 30.0s)
- Reconnection stops if the service is intentionally shut down or manually disconnected
- State change events are broadcast to WebSocket clients during disconnection and reconnection

## Client Examples

### Python WebSocket Client

```python
import asyncio
import aiohttp
import json

class BilibiliClient:
    """Example client that consumes Bilibili live messages."""

    def __init__(self, service_url: str = "ws://localhost:8002/ws/stream"):
        self.service_url = service_url

    async def start(self):
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(self.service_url) as ws:
                # Subscribe to channels
                await ws.send_json({
                    "type": "subscribe",
                    "channels": ["danmaku", "superchat", "gift"]
                })

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        await self._handle_message(data)

    async def _handle_message(self, data: dict):
        msg_type = data.get("type")
        message = data.get("data", {})

        if msg_type == "danmaku":
            print(f"[Chat] {message['user']}: {message['content']}")
        elif msg_type == "superchat":
            print(f"[SC ¥{message['amount']}] {message['user']}: {message['content']}")
        elif msg_type == "gift":
            print(f"[Gift] {message['user']} {message['action']} {message['gift_name']}x{message['num']}")
        elif msg_type == "guard":
            print(f"[Guard] {message['user']}: {message['toast_msg']}")
        elif msg_type == "superchat_delete":
            print(f"[SC Delete] IDs: {message['ids']}")

if __name__ == "__main__":
    asyncio.run(BilibiliClient().start())
```

### JavaScript WebSocket Client

```javascript
const ws = new WebSocket('ws://localhost:8002/ws/stream');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'subscribe',
    channels: ['danmaku', 'superchat', 'gift']
  }));
};

ws.onmessage = (event) => {
  const { type, data } = JSON.parse(event.data);

  switch (type) {
    case 'danmaku':
      console.log(`[Chat] ${data.user}: ${data.content}`);
      break;
    case 'superchat':
      console.log(`[SC ¥${data.amount}] ${data.user}: ${data.content}`);
      break;
    case 'gift':
      console.log(`[Gift] ${data.user} ${data.action} ${data.gift_name}x${data.num}`);
      break;
    case 'guard':
      console.log(`[Guard] ${data.toast_msg}`);
      break;
    case 'superchat_delete':
      console.log(`[SC Delete] IDs: ${data.ids}`);
      break;
  }
};
```

## Integration with Orchestrator

The orchestrator connects to the Bilibili service via `BilibiliWebSocketSource` (see `orchestrator/sources/bilibili_websocket_source.py`):

1. **Connection:** Opens WebSocket to `ws://localhost:8002/ws/stream`
2. **Subscription:** Subscribes to `danmaku` and `superchat` channels
3. **Event Publishing:** Publishes received messages to orchestrator EventBus as:
   - `BILIBILI_DANMAKU` events (for danmaku)
   - `BILIBILI_SUPERCHAT` events (for superchat messages)
4. **Auto-Reconnection:** Handles connection loss with exponential backoff

The orchestrator's `QueueManager` subscribes to these events and enqueues messages with appropriate priorities.

## Troubleshooting

### Service Won't Start

**Issue:** `Config not found` error
**Solution:** Ensure `bilibili/config.yaml` exists with valid configuration.

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

**Issue:** Connected but no messages appear

**Solutions:**
1. Check if the Bilibili room is live: `curl http://localhost:8002/state`
2. Check if SESSDATA cookie is valid (may expire after ~6 months)
3. Verify room_id is correct: `curl http://localhost:8002/config`

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
1. Verify service is running and dashboard loads in a regular browser
2. Check OBS browser source settings:
   - URL is correct (with `http://`, not `ws://`)
   - Width/height are set
   - "Shutdown source when not visible" may prevent initial load
3. Check OBS logs for browser errors

## Development

### Running Tests

```bash
uv run pytest tests/bilibili/
```

### Code Structure

```
bilibili/
├── __init__.py           # Package init
├── bilibili_client.py    # BilibiliClient wrapper + BilibiliClientHandler
├── blivedm/              # Bundled Bilibili live chat library
├── config.yaml           # Service configuration
├── settings.py           # Configuration loader (Pydantic models)
├── server.py             # Service entry point
├── manager.py            # BilibiliManager (core business logic)
├── api.py                # FastAPI routes (REST + WebSocket)
├── models.py             # Pydantic request/response models
├── static/
│   ├── dashboard.html    # Full dashboard
│   ├── obs.html          # OBS overlay
│   └── client-example.py # Example WebSocket client
└── README.md             # This file
```

## License

Part of the Voice Agent project. See main project LICENSE for details.
