"""
WebSocket-based Bilibili source that subscribes to the standalone Bilibili service.
Replaces direct BilibiliClient integration with WebSocket streaming.
"""

import asyncio
import json
import aiohttp
from typing import Optional
from core.event_bus import EventBus, Event
from core.logging import get_logger
from core.settings import get_settings
from orchestrator.sources.base import BaseSource
from orchestrator.events import EventType

logger = get_logger(__name__)


class BilibiliWebSocketSource(BaseSource):
    """
    WebSocket-based Bilibili source that subscribes to Bilibili service.
    Replaces direct EventBus integration with WebSocket streaming.
    """

    def __init__(self, event_bus: EventBus):
        super().__init__(event_bus)
        settings = get_settings()

        # Build WebSocket URL from base URL
        base_url = settings.services.bilibili_base_url
        # Convert http:// to ws://
        ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
        self.ws_url = f"{ws_url}/ws/stream"

        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 30.0
        logger.info(f"BilibiliWebSocketSource initialized with URL: {self.ws_url}")

    async def start(self):
        """Start the WebSocket connection."""
        if self.running:
            return

        settings = get_settings()
        if not settings.services.bilibili_enabled:
            logger.info("Bilibili service disabled in config, not starting WebSocket source")
            return

        self.running = True
        self.session = aiohttp.ClientSession()
        self._ws_task = asyncio.create_task(self._ws_loop())
        logger.info("BilibiliWebSocketSource started")

    async def stop(self):
        """Stop the WebSocket connection."""
        if not self.running:
            return

        self.running = False

        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass

        if self.ws:
            await self.ws.close()

        if self.session:
            await self.session.close()

        logger.info("BilibiliWebSocketSource stopped")

    async def _ws_loop(self):
        """Maintain WebSocket connection with auto-reconnection."""
        while self.running:
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if self.running:
                    logger.info(f"Reconnecting in {self._reconnect_delay}s...")
                    await asyncio.sleep(self._reconnect_delay)
                    self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)

    async def _connect_and_listen(self):
        """Connect to WebSocket and listen for messages."""
        try:
            async with self.session.ws_connect(self.ws_url) as ws:
                self.ws = ws
                logger.info("Connected to Bilibili service WebSocket")
                self._reconnect_delay = 1.0  # Reset delay on success

                # Subscribe to all channels
                await ws.send_json({
                    "type": "subscribe",
                    "channels": ["danmaku", "superchat"]
                })
                logger.debug("Subscribed to danmaku and superchat channels")

                # Listen for messages
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        await self._handle_message(data)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"WebSocket error: {ws.exception()}")
                        break
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        logger.warning("WebSocket connection closed by server")
                        break
        except aiohttp.ClientConnectorError as e:
            logger.warning(f"Failed to connect to Bilibili service: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in WebSocket connection: {e}")
            raise

    async def _handle_message(self, data: dict):
        """Handle incoming WebSocket message."""
        msg_type = data.get("type")

        if msg_type == "danmaku":
            # Publish danmaku event to orchestrator event bus
            await self.event_bus.publish(Event(
                EventType.BILIBILI_DANMAKU.value,
                data["data"]
            ))
            logger.debug(f"Published danmaku event: {data['data']['user']}: {data['data']['content']}")

        elif msg_type == "superchat":
            # Publish superchat event to orchestrator event bus
            await self.event_bus.publish(Event(
                EventType.BILIBILI_SUPERCHAT.value,
                data["data"]
            ))
            logger.info(f"Published superchat event: {data['data']['user']} (¥{data['data']['amount']})")

        elif msg_type == "state_changed":
            # Log state changes from service
            state = data.get("data", {})
            logger.info(f"Bilibili service state changed: {state}")

        elif msg_type == "pong":
            # Heartbeat response
            pass

        elif msg_type == "error":
            logger.error(f"Bilibili service error: {data.get('message')}")
