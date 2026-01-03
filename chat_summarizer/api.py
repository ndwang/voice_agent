"""
FastAPI routes for Chat Summarizer service.
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime
from core.logging import get_logger
from chat_summarizer.models import (
    SummarizeRequest,
    SummarizeResponse,
    BufferStatsResponse,
    MessageInfo
)

logger = get_logger(__name__)
router = APIRouter()

# Global instances (injected by server.py)
message_buffer = None
summarizer = None


@router.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Chat Summarizer",
        "description": "Bilibili chat analysis with LLM-based summarization",
        "version": "1.0.0"
    }


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_chat(request: SummarizeRequest):
    """
    Analyze recent chat messages and provide summary.

    This endpoint:
    1. Retrieves recent messages from the buffer
    2. Sends them to LLM for analysis
    3. Returns overall sentiment and most interesting message

    Args:
        request: Summarization parameters (max_messages, time_window_seconds)

    Returns:
        Summary with sentiment analysis and selected message
    """
    if message_buffer is None or summarizer is None:
        raise HTTPException(status_code=500, detail="Service not initialized")

    try:
        # Get recent messages
        messages = await message_buffer.get_all_messages(
            max_messages=request.max_messages if request.max_messages else None,
            time_window_seconds=request.time_window_seconds
        )

        if not messages:
            return SummarizeResponse(
                overall_sentiment="No messages available to analyze.",
                most_interesting_message=None,
                reasoning="Message buffer is empty.",
                messages_analyzed=0,
                timestamp=datetime.now()
            )

        # Perform LLM analysis
        logger.info(f"Analyzing {len(messages)} messages...")
        result = await summarizer.summarize(messages)

        # Find the selected message
        selected_message = None
        if result.get("most_interesting_message_id"):
            selected_message = next(
                (m for m in messages if m["id"] == result["most_interesting_message_id"]),
                None
            )
            if selected_message:
                selected_message = MessageInfo(**selected_message)

        response = SummarizeResponse(
            overall_sentiment=result["overall_sentiment"],
            most_interesting_message=selected_message,
            reasoning=result["reasoning"],
            messages_analyzed=len(messages),
            timestamp=datetime.now()
        )

        # Log results
        logger.info(f"✓ Analyzed {len(messages)} messages")
        logger.info(f"  Sentiment: {response.overall_sentiment}")
        if response.most_interesting_message:
            logger.info(f"  Selected: [{response.most_interesting_message.user}] {response.most_interesting_message.content[:100]}")
            logger.info(f"  Reason: {response.reasoning}")
        else:
            logger.info(f"  No interesting message selected")

        return response

    except Exception as e:
        logger.error(f"Summarization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Summarization error: {str(e)}")


@router.get("/buffer", response_model=BufferStatsResponse)
async def get_buffer_status():
    """
    Get current buffer statistics.

    Useful for debugging and monitoring the message collection.

    Returns:
        Buffer statistics including message counts and samples
    """
    if message_buffer is None:
        raise HTTPException(status_code=500, detail="Service not initialized")

    try:
        stats = await message_buffer.get_buffer_stats()
        messages = await message_buffer.get_all_messages(max_messages=5)

        # Format sample messages for display
        sample_messages = []
        for msg in messages:
            sample = {
                "type": msg["type"],
                "user": msg["user"],
                "content": msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            }
            if msg.get("amount"):
                sample["amount"] = msg["amount"]
            sample_messages.append(sample)

        return BufferStatsResponse(
            danmaku_count=stats["danmaku_count"],
            superchat_count=stats["superchat_count"],
            total_count=stats["total_count"],
            oldest_timestamp=stats["oldest_timestamp"],
            newest_timestamp=stats["newest_timestamp"],
            sample_messages=sample_messages
        )

    except Exception as e:
        logger.error(f"Failed to get buffer status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
