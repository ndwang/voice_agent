"""
Chat Summarizer Service - Main Entry Point

Standalone service for analyzing Bilibili chat messages with LLM.
"""
import uvicorn
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Ensure project root is in path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.server import create_app
from core.settings import get_settings
from core.logging import get_logger
from chat_summarizer.api import router as summarizer_router
from chat_summarizer.message_buffer import ChatMessageBuffer
from chat_summarizer.summarizer import ChatSummarizer
import chat_summarizer.api
import logging

logger = get_logger(__name__)

# Set up file logging for debugging
file_handler = logging.FileHandler('chat_summarizer.log', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Add to root logger to catch all module logs
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)
root_logger.setLevel(logging.DEBUG)


@asynccontextmanager
async def lifespan(app):
    """
    Application lifespan manager.

    Handles startup and shutdown of the service components.
    """
    settings = get_settings()

    # Check if Bilibili is configured
    if settings.bilibili.room_id == 0:
        logger.error("Bilibili room_id not configured in config.yaml")
        logger.error("Please set bilibili.room_id to a valid Bilibili live room ID")
        raise ValueError("Bilibili room_id is required")

    # Create LLM provider
    logger.info(f"Initializing LLM provider: {settings.llm.provider}")
    from orchestrator.utils.llm_factory import create_provider
    llm_provider = create_provider(settings.llm)

    # Create message buffer
    logger.info(f"Connecting to Bilibili room {settings.bilibili.room_id}...")
    message_buffer = ChatMessageBuffer(
        room_id=settings.bilibili.room_id,
        sessdata=settings.bilibili.sessdata,
        ttl_seconds=settings.bilibili.danmaku_ttl_seconds
    )

    # Create summarizer
    summarizer = ChatSummarizer(llm_provider)

    # Inject into API module
    chat_summarizer.api.message_buffer = message_buffer
    chat_summarizer.api.summarizer = summarizer

    # Start message buffer
    await message_buffer.start()
    logger.info("✓ Chat Summarizer Service started successfully")
    logger.info(f"  - Bilibili room: {settings.bilibili.room_id}")
    logger.info(f"  - LLM provider: {settings.llm.provider}")
    logger.info(f"  - Message TTL: {settings.bilibili.danmaku_ttl_seconds}s")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down Chat Summarizer Service...")
    await message_buffer.stop()
    logger.info("✓ Service stopped")


def main():
    """Main entry point."""
    # Use default port 8005 (can be made configurable later)
    HOST = "0.0.0.0"
    PORT = 8005

    app = create_app(
        title="Chat Summarizer Service",
        description="Bilibili chat message analyzer with LLM-based summarization",
        version="1.0.0",
        lifespan=lifespan
    )

    app.include_router(summarizer_router)

    logger.info("=" * 60)
    logger.info("Chat Summarizer Service")
    logger.info("=" * 60)
    logger.info(f"Starting server on {HOST}:{PORT}...")

    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
