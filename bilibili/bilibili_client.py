import asyncio
import aiohttp
import http.cookies
from typing import Optional, Callable, Awaitable
import blivedm
import blivedm.models.web as web_models
from core.logging import get_logger

logger = get_logger(__name__)


class BilibiliClientHandler(blivedm.BaseHandler):
    """Handler for Bilibili live events."""

    def __init__(
        self,
        on_danmaku: Optional[Callable[[web_models.DanmakuMessage], Awaitable[None]]] = None,
        on_super_chat: Optional[Callable[[web_models.SuperChatMessage], Awaitable[None]]] = None,
        on_gift: Optional[Callable[[web_models.GiftMessage], Awaitable[None]]] = None,
        on_guard: Optional[Callable[[web_models.UserToastV2Message], Awaitable[None]]] = None,
        on_super_chat_delete: Optional[Callable[[web_models.SuperChatDeleteMessage], Awaitable[None]]] = None,
        on_client_stopped: Optional[Callable] = None,
    ):
        self.on_danmaku = on_danmaku
        self.on_super_chat = on_super_chat
        self.on_gift = on_gift
        self.on_guard = on_guard
        self.on_super_chat_delete = on_super_chat_delete
        self._on_client_stopped_cb = on_client_stopped

    def _on_danmaku(self, client: blivedm.BLiveClient, message: web_models.DanmakuMessage):
        if self.on_danmaku:
            asyncio.create_task(self.on_danmaku(message))

    def _on_super_chat(self, client: blivedm.BLiveClient, message: web_models.SuperChatMessage):
        if self.on_super_chat:
            asyncio.create_task(self.on_super_chat(message))

    def _on_gift(self, client: blivedm.BLiveClient, message: web_models.GiftMessage):
        if self.on_gift:
            asyncio.create_task(self.on_gift(message))

    def _on_user_toast_v2(self, client: blivedm.BLiveClient, message: web_models.UserToastV2Message):
        if self.on_guard:
            asyncio.create_task(self.on_guard(message))

    def _on_super_chat_delete(self, client: blivedm.BLiveClient, message: web_models.SuperChatDeleteMessage):
        if self.on_super_chat_delete:
            asyncio.create_task(self.on_super_chat_delete(message))

    def on_client_stopped(self, client, exception):
        """Called when the blivedm client WebSocket stops."""
        if self._on_client_stopped_cb:
            self._on_client_stopped_cb(client, exception)

    def handle(self, client: blivedm.BLiveClient, command: dict):
        """Override handle to silence unknown command warnings."""
        cmd = command.get('cmd', '')
        pos = cmd.find(':')
        if pos != -1:
            cmd = cmd[:pos]

        if cmd in self._CMD_CALLBACK_DICT:
            return super().handle(client, command)
        # Just ignore unknown commands without logging
        return None


class BilibiliClient:
    """Wrapper for blivedm.BLiveClient with easier lifecycle management."""

    def __init__(self, room_id: int, sessdata: Optional[str] = None):
        self.room_id = room_id
        self.sessdata = sessdata
        self.session: Optional[aiohttp.ClientSession] = None
        self.client: Optional[blivedm.BLiveClient] = None
        self._running = False
        self._on_danmaku_cb: Optional[Callable] = None
        self._on_super_chat_cb: Optional[Callable] = None
        self._on_gift_cb: Optional[Callable] = None
        self._on_guard_cb: Optional[Callable] = None
        self._on_super_chat_delete_cb: Optional[Callable] = None
        self._on_client_stopped_cb: Optional[Callable] = None

    def set_handlers(
        self,
        on_danmaku: Optional[Callable] = None,
        on_super_chat: Optional[Callable] = None,
        on_gift: Optional[Callable] = None,
        on_guard: Optional[Callable] = None,
        on_super_chat_delete: Optional[Callable] = None,
        on_client_stopped: Optional[Callable] = None,
    ):
        """Set callbacks for all Bilibili live events."""
        self._on_danmaku_cb = on_danmaku
        self._on_super_chat_cb = on_super_chat
        self._on_gift_cb = on_gift
        self._on_guard_cb = on_guard
        self._on_super_chat_delete_cb = on_super_chat_delete
        self._on_client_stopped_cb = on_client_stopped

    async def start(self):
        """Start the bilibili client."""
        if self._running:
            return

        self._running = True

        # Initialize session with optional SESSDATA
        self.session = aiohttp.ClientSession()
        if self.sessdata:
            cookies = http.cookies.SimpleCookie()
            cookies['SESSDATA'] = self.sessdata
            cookies['SESSDATA']['domain'] = 'bilibili.com'
            self.session.cookie_jar.update_cookies(cookies)

        self.client = blivedm.BLiveClient(self.room_id, session=self.session)
        handler = BilibiliClientHandler(
            on_danmaku=self._on_danmaku_cb,
            on_super_chat=self._on_super_chat_cb,
            on_gift=self._on_gift_cb,
            on_guard=self._on_guard_cb,
            on_super_chat_delete=self._on_super_chat_delete_cb,
            on_client_stopped=self._on_client_stopped_cb,
        )
        self.client.set_handler(handler)

        self.client.start()
        logger.info(f"BilibiliClient started for room {self.room_id}")

    async def stop(self):
        """Stop the bilibili client."""
        if not self._running:
            return

        self._running = False
        if self.client:
            await self.client.stop_and_close()
            self.client = None

        if self.session:
            await self.session.close()
            self.session = None

        logger.info(f"BilibiliClient stopped for room {self.room_id}")

    @property
    def is_running(self) -> bool:
        return self._running
