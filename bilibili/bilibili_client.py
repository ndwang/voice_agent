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
    
    def __init__(self, on_danmaku: Optional[Callable[[web_models.DanmakuMessage], Awaitable[None]]] = None,
                 on_super_chat: Optional[Callable[[web_models.SuperChatMessage], Awaitable[None]]] = None):
        self.on_danmaku = on_danmaku
        self.on_super_chat = on_super_chat

    def _on_danmaku(self, client: blivedm.BLiveClient, message: web_models.DanmakuMessage):
        if self.on_danmaku:
            asyncio.create_task(self.on_danmaku(message))

    def _on_super_chat(self, client: blivedm.BLiveClient, message: web_models.SuperChatMessage):
        if self.on_super_chat:
            asyncio.create_task(self.on_super_chat(message))

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
        self._on_danmaku_cb: Optional[Callable[[web_models.DanmakuMessage], Awaitable[None]]] = None
        self._on_super_chat_cb: Optional[Callable[[web_models.SuperChatMessage], Awaitable[None]]] = None

    def set_handlers(self, on_danmaku: Optional[Callable[[web_models.DanmakuMessage], Awaitable[None]]] = None,
                     on_super_chat: Optional[Callable[[web_models.SuperChatMessage], Awaitable[None]]] = None):
        """Set callbacks for danmaku and superChat events."""
        self._on_danmaku_cb = on_danmaku
        self._on_super_chat_cb = on_super_chat

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
        handler = BilibiliClientHandler(on_danmaku=self._on_danmaku_cb, on_super_chat=self._on_super_chat_cb)
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

