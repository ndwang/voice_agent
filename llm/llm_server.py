"""
LLM Service

FastAPI server for LLM inference with streaming support.
Supports multiple LLM providers.
"""
import os
import json
import asyncio
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from sse_starlette.sse import EventSourceResponse
import logging
import sys

from llm.models import LLMProvider, GeminiProvider

# Configure logging with time info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

# --- Configuration ---
HOST = "0.0.0.0"
PORT = 8002

# LLM Provider Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# --- Initialize LLM Provider ---
llm_provider: Optional[LLMProvider] = None

if LLM_PROVIDER == "gemini":
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is required for Gemini provider")
    # Default model for Gemini if not specified
    llm_provider = GeminiProvider(model=LLM_MODEL)
else:
    raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}. Supported: gemini")

if llm_provider is None:
    raise ValueError("LLM provider not initialized. Please implement provider initialization.")

# --- FastAPI Server ---
app = FastAPI()


class GenerateRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = None


def format_prompt_for_logging(prompt: str) -> None:
    """
    Format and log the prompt in a readable way.
    
    Args:
        prompt: The prompt string to format and log
    """
    prompt_lines = prompt.split("\n")
    if len(prompt_lines) > 10:
        # Show first few lines and last few lines for long prompts
        preview = "\n".join(prompt_lines[:3]) + "\n...\n" + "\n".join(prompt_lines[-3:])
        logger.info(f"LLM Input - Prompt ({len(prompt_lines)} lines, {len(prompt)} chars):\n{preview}")
    else:
        logger.info(f"LLM Input - Prompt:\n{prompt}")


@app.post("/generate")
async def generate(request: GenerateRequest):
    """
    Generate a complete response (non-streaming).
    """
    try:
        # Log input
        format_prompt_for_logging(request.prompt)
        
        kwargs = {}
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        
        response = await llm_provider.generate(
            prompt=request.prompt,
            **kwargs
        )
        
        # Log output
        logger.info(f"{response}")
        
        return {"response": response}
    except Exception as e:
        # Handle provider-specific errors gracefully
        status_code, error_message = llm_provider.parse_error(e)
        logger.error(f"LLM Error: {error_message}", exc_info=True)
        raise HTTPException(status_code=status_code, detail=error_message)


async def generate_stream_events(request: GenerateRequest):
    """Generator for SSE streaming."""
    try:
        # Log input
        format_prompt_for_logging(request.prompt)
        
        kwargs = {}
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        
        full_response = ""
        async for token in llm_provider.generate_stream(
            prompt=request.prompt,
            **kwargs
        ):
            full_response += token
            yield {
                "event": "token",
                "data": json.dumps({"token": token})
            }
        
        # Log output
        if full_response:
            logger.info(f"{full_response}")
        
        yield {
            "event": "done",
            "data": json.dumps({"status": "complete"})
        }
    except Exception as e:
        # Handle provider-specific errors gracefully
        status_code, error_message = llm_provider.parse_error(e)
        logger.error(f"LLM Error: {error_message}", exc_info=True)
        yield {
            "event": "error",
            "data": json.dumps({
                "error": error_message,
                "status_code": status_code
            })
        }


@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    """
    Generate a streaming response using Server-Sent Events (SSE).
    """
    return EventSourceResponse(generate_stream_events(request))


@app.get("/")
async def root():
    return {
        "message": "LLM Service is running",
        "provider": LLM_PROVIDER,
        "model": LLM_MODEL
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "provider": LLM_PROVIDER}


if __name__ == "__main__":
    logger.info(f"Starting LLM server on {HOST}:{PORT}...")
    uvicorn.run(app, host=HOST, port=PORT)

