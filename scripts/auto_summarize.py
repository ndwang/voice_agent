#!/usr/bin/env python
"""
Auto Summarizer Script

Automatically triggers chat summarization at regular intervals.
Run this alongside the chat_summarizer service to get periodic analysis.

Usage:
    python scripts/auto_summarize.py --interval 30 --max-messages 50
"""
import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime
import aiohttp

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.logging import get_logger
import logging

logger = get_logger(__name__)

# Also set up file logging for debugging - use absolute path to ensure it's in project root
log_file_path = project_root / 'auto_summarize.log'
file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)


class AutoSummarizer:
    """Automatically triggers chat summarization at regular intervals."""

    def __init__(
        self,
        service_url: str = "http://localhost:8005",
        interval_seconds: int = 30,
        max_messages: int = None,
        time_window_seconds: int = None
    ):
        """
        Initialize auto summarizer.

        Args:
            service_url: Base URL of the chat summarizer service
            interval_seconds: How often to trigger summarization
            max_messages: Number of messages to analyze each time (None = all messages)
            time_window_seconds: Optional time window for message selection
        """
        self.service_url = service_url.rstrip('/')
        self.interval_seconds = interval_seconds
        self.max_messages = max_messages
        self.time_window_seconds = time_window_seconds
        self.running = False
        self._task = None

    async def check_service_health(self) -> bool:
        """Check if the chat summarizer service is running."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.service_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Service health check failed: {e}")
            return False

    async def trigger_summarization(self) -> dict:
        """
        Trigger a single summarization.

        Returns:
            Summary response dict
        """
        try:
            payload = {}
            if self.max_messages is not None:
                payload["max_messages"] = self.max_messages
            if self.time_window_seconds:
                payload["time_window_seconds"] = self.time_window_seconds

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.service_url}/summarize",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"Summarization failed with status {response.status}: {error_text}")
                        return None

        except asyncio.TimeoutError:
            logger.error("Summarization request timed out")
            return None
        except Exception as e:
            logger.error(f"Summarization request failed: {e}")
            return None

    def print_summary(self, result: dict):
        """Print summary results in a nice format."""
        if not result:
            return

        timestamp = result.get("timestamp", "")
        messages_analyzed = result.get("messages_analyzed", 0)
        sentiment = result.get("overall_sentiment", "")
        reasoning = result.get("reasoning", "")
        interesting_msg = result.get("most_interesting_message")

        print("\n" + "=" * 80)
        print(f"📊 CHAT SUMMARY - {timestamp}")
        print("=" * 80)
        print(f"Messages analyzed: {messages_analyzed}")
        print()
        print(f"Overall Sentiment:")
        print(f"  {sentiment}")
        print()

        if interesting_msg:
            user = interesting_msg.get("user", "Unknown")
            content = interesting_msg.get("content", "")
            msg_type = interesting_msg.get("type", "")
            amount = interesting_msg.get("amount")

            type_str = f"[SuperChat ¥{amount}]" if amount else f"[{msg_type}]"
            print(f"Most Interesting Message: {type_str}")
            print(f"  User: {user}")
            print(f"  Content: {content}")
            print()
            print(f"Reasoning:")
            print(f"  {reasoning}")
        else:
            print(f"Most Interesting Message: None")
            print(f"  {reasoning}")

        print("=" * 80 + "\n")

    async def run_loop(self):
        """Main loop that triggers summarization at regular intervals."""
        max_msg_str = "all" if self.max_messages is None else str(self.max_messages)
        logger.info(f"Auto summarizer started (interval: {self.interval_seconds}s, max_messages: {max_msg_str})")

        iteration = 0
        while self.running:
            try:
                iteration += 1
                logger.info(f"[{iteration}] Triggering summarization...")

                result = await self.trigger_summarization()

                if result:
                    self.print_summary(result)
                    logger.info(f"[{iteration}] ✓ Summarization completed")
                else:
                    logger.warning(f"[{iteration}] ✗ Summarization failed")

                # Wait for next interval
                logger.debug(f"Waiting {self.interval_seconds}s until next summarization...")
                await asyncio.sleep(self.interval_seconds)

            except asyncio.CancelledError:
                logger.info("Auto summarizer loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in auto summarizer loop: {e}", exc_info=True)
                await asyncio.sleep(self.interval_seconds)

    async def start(self):
        """Start the auto summarizer."""
        if self.running:
            logger.warning("Auto summarizer already running")
            return

        # Check if service is available
        logger.info(f"Checking if chat summarizer service is running at {self.service_url}...")
        if not await self.check_service_health():
            logger.error("Chat summarizer service is not running!")
            logger.error(f"Please start the service first: python -m chat_summarizer.server")
            return False

        logger.info("✓ Chat summarizer service is running")

        self.running = True
        self._task = asyncio.create_task(self.run_loop())
        return True

    async def stop(self):
        """Stop the auto summarizer."""
        if not self.running:
            return

        logger.info("Stopping auto summarizer...")
        self.running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("✓ Auto summarizer stopped")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Automatically trigger chat summarization at regular intervals"
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=int,
        default=30,
        help="Interval between summarizations in seconds (default: 30)"
    )
    parser.add_argument(
        "--max-messages",
        "-m",
        type=int,
        default=None,
        help="Maximum number of messages to analyze (default: all messages)"
    )
    parser.add_argument(
        "--time-window",
        "-w",
        type=int,
        default=None,
        help="Only analyze messages from last N seconds (optional)"
    )
    parser.add_argument(
        "--url",
        "-u",
        type=str,
        default="http://localhost:8005",
        help="Chat summarizer service URL (default: http://localhost:8005)"
    )

    args = parser.parse_args()

    # Create auto summarizer
    auto_summarizer = AutoSummarizer(
        service_url=args.url,
        interval_seconds=args.interval,
        max_messages=args.max_messages,
        time_window_seconds=args.time_window
    )

    # Start the auto summarizer
    print("=" * 80)
    print("Auto Summarizer Script")
    print("=" * 80)
    print(f"Service URL: {args.url}")
    print(f"Interval: {args.interval} seconds")
    print(f"Max messages: {'all' if args.max_messages is None else args.max_messages}")
    if args.time_window:
        print(f"Time window: {args.time_window} seconds")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 80)
    print()

    started = await auto_summarizer.start()
    if not started:
        logger.error("Failed to start auto summarizer")
        return 1

    try:
        # Keep running until interrupted
        while auto_summarizer.running:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n\nReceived interrupt signal...")

    await auto_summarizer.stop()
    print("\nAuto summarizer stopped. Goodbye!")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
