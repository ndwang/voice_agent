"""
LLM Service

FastAPI server for LLM inference with streaming support.
Supports multiple LLM providers.
"""
import os
import json
import asyncio
from typing import Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from sse_starlette.sse import EventSourceResponse
import logging
import sys

# Add project root to path to import config_loader
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config_loader import get_config
from llm import LLMProvider, GeminiProvider, LlamaCppProvider

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
HOST = get_config("llm", "host", default="0.0.0.0")
PORT = get_config("llm", "port", default=8002)

# LLM Provider Configuration
LLM_PROVIDER = get_config("llm", "provider", default="gemini")
LLM_MODEL = get_config("llm", "providers", "gemini", "model", default="gemini-2.5-flash")
GEMINI_API_KEY = get_config("llm", "providers", "gemini", "api_key", default="")

# Llama.cpp Configuration
LLAMACPP_MODEL_PATH = get_config("llm", "providers", "llamacpp", "model_path", default="")
LLAMACPP_N_CTX = int(get_config("llm", "providers", "llamacpp", "n_ctx", default=4096))
LLAMACPP_N_THREADS = int(get_config("llm", "providers", "llamacpp", "n_threads", default=0)) or None
LLAMACPP_N_GPU_LAYERS = int(get_config("llm", "providers", "llamacpp", "n_gpu_layers", default=-1))

# --- Initialize LLM Provider ---
llm_provider: Optional[LLMProvider] = None

if LLM_PROVIDER == "gemini":
    # If API key is provided in config, pass it to provider (overrides environment)
    # Otherwise, provider will use GEMINI_API_KEY from environment automatically
    api_key = GEMINI_API_KEY if GEMINI_API_KEY else None
    llm_provider = GeminiProvider(model=LLM_MODEL, api_key=api_key)
elif LLM_PROVIDER == "llamacpp":
    if not LLAMACPP_MODEL_PATH:
        raise ValueError("LLAMACPP_MODEL_PATH must be set in config.yaml for Llama.cpp provider")
    llm_provider = LlamaCppProvider(
        model_path=LLAMACPP_MODEL_PATH,
        n_ctx=LLAMACPP_N_CTX,
        n_threads=LLAMACPP_N_THREADS,
        n_gpu_layers=LLAMACPP_N_GPU_LAYERS
    )
else:
    raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}. Supported: gemini, llamacpp")

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
    import time
    request_start_time = time.perf_counter()
    first_token_time = None
    
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
            # Track first token latency
            if first_token_time is None:
                first_token_time = time.perf_counter()
                time_to_first_token = first_token_time - request_start_time
                logger.info(f"LLM time-to-first-token: {time_to_first_token*1000:.0f}ms")
            
            full_response += token
            yield {
                "event": "token",
                "data": json.dumps({"token": token})
            }
        
        # Log output and total latency
        if full_response:
            total_time = time.perf_counter() - request_start_time
            logger.info(f"LLM response: {full_response}")
            logger.info(f"LLM total generation time: {total_time*1000:.0f}ms")
        
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
    model_info = LLM_MODEL if LLM_PROVIDER == "gemini" else LLAMACPP_MODEL_PATH
    return {
        "message": "LLM Service is running",
        "provider": LLM_PROVIDER,
        "model": model_info
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "provider": LLM_PROVIDER}


if __name__ == "__main__":
    logger.info(f"Starting LLM server on {HOST}:{PORT}...")
    uvicorn.run(app, host=HOST, port=PORT)

