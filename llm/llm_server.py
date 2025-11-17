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
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from sse_starlette.sse import EventSourceResponse

from llm.models import LLMProvider, GeminiProvider

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


@app.post("/generate")
async def generate(request: GenerateRequest):
    """
    Generate a complete response (non-streaming).
    """
    try:
        kwargs = {}
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        
        response = await llm_provider.generate(
            prompt=request.prompt,
            **kwargs
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def generate_stream_events(request: GenerateRequest):
    """Generator for SSE streaming."""
    try:
        kwargs = {}
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        
        async for token in llm_provider.generate_stream(
            prompt=request.prompt,
            **kwargs
        ):
            yield {
                "event": "token",
                "data": json.dumps({"token": token})
            }
        
        yield {
            "event": "done",
            "data": json.dumps({"status": "complete"})
        }
    except Exception as e:
        yield {
            "event": "error",
            "data": json.dumps({"error": str(e)})
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
    print(f"Starting LLM server on {HOST}:{PORT}...")
    uvicorn.run(app, host=HOST, port=PORT)

