"""
Example Bilibili service WebSocket client for LLM summarization.

This demonstrates how to build a custom client that consumes chat messages
from the Bilibili service and processes them (e.g., for LLM summarization).

Usage:
    python client-example.py
"""

import asyncio
import aiohttp
import json
from typing import Optional
from collections import deque
from datetime import datetime


class BilibiliLLMClient:
    """
    Example client that consumes Bilibili chat messages for LLM processing.

    This client:
    - Connects to the Bilibili service WebSocket
    - Subscribes to danmaku and superchat channels
    - Buffers messages for summarization
    - Can be extended to call an LLM API for summarization
    """

    def __init__(self, service_url: str = "ws://localhost:8002/ws/stream"):
        self.service_url = service_url
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self.running = False

        # Message buffer for summarization
        self.message_buffer = deque(maxlen=100)  # Last 100 messages
        self.superchat_buffer = deque(maxlen=20)  # Last 20 superchats

        # Summarization settings
        self.summarize_interval = 60  # Summarize every 60 seconds
        self.min_messages_for_summary = 10  # Minimum messages to trigger summary

    async def start(self):
        """Connect and start listening to Bilibili messages"""
        self.running = True

        # Start summarization task
        summarize_task = asyncio.create_task(self._summarization_loop())

        # Start WebSocket connection
        try:
            await self._connect_and_listen()
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.running = False
            summarize_task.cancel()
            try:
                await summarize_task
            except asyncio.CancelledError:
                pass

    async def stop(self):
        """Disconnect and stop"""
        self.running = False
        if self.ws:
            await self.ws.close()

    async def _connect_and_listen(self):
        """Connect to WebSocket and process messages"""
        reconnect_delay = 1.0
        max_reconnect_delay = 30.0

        while self.running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(self.service_url) as ws:
                        self.ws = ws
                        print(f"✅ Connected to Bilibili service at {self.service_url}")
                        reconnect_delay = 1.0  # Reset delay on success

                        # Subscribe to both channels
                        await ws.send_json({
                            "type": "subscribe",
                            "channels": ["danmaku", "superchat"]
                        })
                        print("📡 Subscribed to danmaku and superchat channels")

                        # Listen for messages
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                await self._handle_message(data)
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                print(f"❌ WebSocket error: {ws.exception()}")
                                break

            except aiohttp.ClientError as e:
                print(f"❌ Connection error: {e}")
            except Exception as e:
                print(f"❌ Unexpected error: {e}")

            # Reconnect with exponential backoff
            if self.running:
                print(f"🔄 Reconnecting in {reconnect_delay}s...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

    async def _handle_message(self, data: dict):
        """Process incoming WebSocket message"""
        msg_type = data.get("type")

        if msg_type == "danmaku":
            message = data["data"]
            timestamp = datetime.fromtimestamp(message["timestamp"]).strftime("%H:%M:%S")
            print(f"[{timestamp}] 💬 {message['user']}: {message['content']}")

            # Add to buffer for summarization
            self.message_buffer.append(message)

        elif msg_type == "superchat":
            message = data["data"]
            timestamp = datetime.fromtimestamp(message["timestamp"]).strftime("%H:%M:%S")
            print(f"[{timestamp}] 💰 {message['user']} (¥{message['amount']}): {message['content']}")

            # Add to both buffers (superchats are higher priority)
            self.message_buffer.append(message)
            self.superchat_buffer.append(message)

        elif msg_type == "state_changed":
            state = data["data"]
            print(f"🔄 State changed: {state}")

        elif msg_type == "pong":
            # Heartbeat response
            pass

        elif msg_type == "error":
            print(f"❌ Server error: {data.get('message')}")

    async def _summarization_loop(self):
        """Periodically generate summaries of collected messages"""
        while self.running:
            try:
                await asyncio.sleep(self.summarize_interval)

                if len(self.message_buffer) >= self.min_messages_for_summary:
                    await self._generate_summary()

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ Error in summarization loop: {e}")

    async def _generate_summary(self):
        """
        Generate a summary of recent messages using an LLM.

        This is a placeholder implementation. In a real application, you would:
        1. Extract messages from the buffer
        2. Format them for LLM input
        3. Call your LLM API (e.g., OpenAI, Gemini, local LLM)
        4. Process the summary output
        """
        print("\n" + "="*60)
        print("📝 Generating summary...")
        print(f"Messages in buffer: {len(self.message_buffer)}")
        print(f"Superchats: {len(self.superchat_buffer)}")

        # Example: Extract last N messages
        recent_messages = list(self.message_buffer)[-20:]

        # Example: Format for LLM prompt
        messages_text = "\n".join([
            f"{msg['user']}: {msg['content']}"
            for msg in recent_messages
        ])

        print("\nRecent messages:")
        print(messages_text)

        # TODO: Call your LLM API here
        # Example:
        # summary = await call_llm_api(messages_text)
        # print(f"\n💡 Summary: {summary}")

        print("="*60 + "\n")

        # Clear buffer after summarizing
        self.message_buffer.clear()


async def main():
    """Main entry point"""
    print("🚀 Bilibili LLM Client Example")
    print("Press Ctrl+C to stop\n")

    client = BilibiliLLMClient()
    await client.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
