"""
OCR Service

FastAPI server for continuous OCR monitoring with WebSocket streaming.
Wraps existing OCR functionality for headless operation.
"""
import os
import json
import asyncio
import hashlib
import time
import glob
import numpy as np
import pyautogui
import logging
import sys
from pathlib import Path
from paddleocr import PaddleOCR
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from typing import Optional, Tuple
import uvicorn

from core.config import get_config
from core.logging import setup_logging, get_logger

# Set up logging
setup_logging()
logger = get_logger(__name__)

# --- Configuration ---
HOST = get_config("ocr", "host", default="0.0.0.0")
PORT = get_config("ocr", "port", default=8004)

# OCR Configuration
LANGUAGE = get_config("ocr", "language", default="ch")  # Chinese and English support
INTERVAL_MS = get_config("ocr", "interval_ms", default=1000)  # Default monitoring interval
TEXTS_STORAGE_FILE_PREFIX = get_config("ocr", "texts_storage_file_prefix", default="ocr_detected_texts")  # Prefix for text storage files

# --- Global OCR Reader ---
logger.info("Loading OCR model...")
try:
    ocr_reader = PaddleOCR(
        lang=LANGUAGE,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        ocr_version="PP-OCRv4"
    )
    logger.info("OCR model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading OCR model: {e}", exc_info=True)
    ocr_reader = None

# --- Monitoring State ---
monitoring_task = None
monitoring_region = None
monitoring_interval_ms = INTERVAL_MS
monitoring_active = False
previous_image_hash = None
websocket_clients = set()
detected_texts = []  # Store all detected texts with timestamps
current_storage_file = None  # Current file name for storing texts


class RegionRequest(BaseModel):
    region: Tuple[int, int, int, int]  # (x, y, width, height)


class MonitorRequest(BaseModel):
    region: Tuple[int, int, int, int]
    interval_ms: Optional[int] = INTERVAL_MS
    clear_texts: Optional[bool] = False  # Clear stored texts when starting monitoring


def process_region(region: Tuple[int, int, int, int], previous_hash: Optional[str] = None):
    """
    Process a single region: take screenshot, check hash, perform OCR if changed.
    
    Returns:
        tuple: (result_data, new_hash)
        - result_data: dict with 'text' key, or None if unchanged
        - new_hash: str - New hash value
    """
    if region is None:
        return None, previous_hash
    
    try:
        # Take screenshot
        x, y, width, height = region
        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        
        # Calculate hash
        img_array = np.array(screenshot)
        current_hash = hashlib.md5(img_array.tobytes()).hexdigest()
        
        # If unchanged, return None
        if current_hash == previous_hash:
            return None, previous_hash
        
        # If changed, perform OCR
        if ocr_reader is None:
            return None, current_hash
        
        results = ocr_reader.predict(img_array)
        
        # Extract text from OCR results
        if results and results[0]:
            texts = results[0]['rec_texts']
            # Join texts into a single string
            text = '\n'.join(texts) if texts else ""
        else:
            text = ""
        
        result_data = {"text": text}
        return result_data, current_hash
    
    except Exception as e:
        logger.error(f"Error processing region: {e}", exc_info=True)
        return None, previous_hash


def get_new_storage_filename():
    """Generate a new storage filename with timestamp."""
    timestamp = int(time.time() * 1000)
    return f"{TEXTS_STORAGE_FILE_PREFIX}_{timestamp}.txt"


def get_latest_storage_file():
    """Find the most recent storage file."""
    pattern = f"{TEXTS_STORAGE_FILE_PREFIX}_*.txt"
    files = glob.glob(pattern)
    if not files:
        return None
    # Sort by modification time, most recent first
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def save_texts_to_file():
    """Save detected texts to plain text file."""
    global current_storage_file
    
    if current_storage_file is None:
        current_storage_file = get_new_storage_filename()
    
    try:
        with open(current_storage_file, 'w', encoding='utf-8') as f:
            for entry in detected_texts:
                text = entry.get('text', '')
                # Write as plain text: just the text, separated by double newlines
                f.write(f"{text}\n\n")
    except Exception as e:
        logger.error(f"Error saving texts to file: {e}", exc_info=True)


def load_texts_from_file():
    """Load detected texts from the most recent plain text file."""
    global detected_texts, current_storage_file
    
    latest_file = get_latest_storage_file()
    if latest_file is None:
        return
    
    current_storage_file = latest_file
    
    try:
        detected_texts.clear()
        with open(latest_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Use file modification time as timestamp for all entries
        file_mtime = int(os.path.getmtime(latest_file) * 1000)
        
        # Parse format: text\n\n (entries separated by double newlines)
        entries = content.split('\n\n')
        for entry_block in entries:
            text = entry_block.strip()
            if text:  # Only add non-empty entries
                detected_texts.append({
                    "text": text,
                    "timestamp": file_mtime
                })
        
        logger.info(f"Loaded {len(detected_texts)} text entries from {latest_file}")
    except Exception as e:
        logger.error(f"Error loading texts from file: {e}", exc_info=True)


async def monitoring_loop():
    """Background task for continuous OCR monitoring."""
    global previous_image_hash, detected_texts
    
    while monitoring_active:
        try:
            if monitoring_region:
                result, new_hash = process_region(monitoring_region, previous_image_hash)
                
                if result is not None:
                    # Text changed, store it instead of sending immediately
                    timestamp = int(time.time() * 1000)
                    text_entry = {
                        "text": result["text"],
                        "timestamp": timestamp
                    }
                    detected_texts.append(text_entry)
                    # Save to file
                    save_texts_to_file()
                    
                    previous_image_hash = new_hash
            
            await asyncio.sleep(monitoring_interval_ms / 1000.0)
        
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}", exc_info=True)
            await asyncio.sleep(1)


# --- FastAPI Server ---
app = FastAPI()


@app.post("/monitor/start")
async def start_monitoring(request: MonitorRequest):
    """Start continuous monitoring of a region."""
    global monitoring_task, monitoring_region, monitoring_interval_ms, monitoring_active, previous_image_hash, detected_texts, current_storage_file
    
    if monitoring_active:
        raise HTTPException(status_code=400, detail="Monitoring is already active")
    
    monitoring_region = request.region
    monitoring_interval_ms = request.interval_ms
    monitoring_active = True
    previous_image_hash = None
    
    # Clear stored texts if requested
    if request.clear_texts:
        detected_texts.clear()
        # Create a new file instead of removing the old one
        current_storage_file = get_new_storage_filename()
        # Save empty file
        save_texts_to_file()
    
    # Start monitoring task
    monitoring_task = asyncio.create_task(monitoring_loop())
    
    return {
        "status": "started",
        "region": monitoring_region,
        "interval_ms": monitoring_interval_ms
    }


@app.post("/monitor/stop")
async def stop_monitoring():
    """Stop continuous monitoring."""
    global monitoring_task, monitoring_active
    
    if not monitoring_active:
        raise HTTPException(status_code=400, detail="Monitoring is not active")
    
    monitoring_active = False
    
    if monitoring_task:
        monitoring_task.cancel()
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass
        monitoring_task = None
    
    return {"status": "stopped"}


@app.websocket("/monitor/stream")
async def monitor_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming OCR updates.
    Clients can request stored texts by sending {"type": "get_texts"}.
    """
    await websocket.accept()
    websocket_clients.add(websocket)
    logger.info(f"OCR WebSocket client connected. Total clients: {len(websocket_clients)}")
    
    try:
        # Send initial status
        await websocket.send_json({
            "type": "status",
            "monitoring": monitoring_active,
            "region": monitoring_region
        })
        
        # Keep connection alive and handle client requests
        while True:
            # Wait for messages from client
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "close":
                    break
                elif message.get("type") == "get_texts":
                    # Send all stored texts to the client
                    await websocket.send_json({
                        "type": "texts",
                        "texts": detected_texts.copy(),
                        "count": len(detected_texts)
                    })
                    
            except WebSocketDisconnect:
                break
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Error in OCR WebSocket: {e}", exc_info=True)
    finally:
        websocket_clients.discard(websocket)
        logger.info(f"OCR WebSocket client disconnected. Total clients: {len(websocket_clients)}")


@app.post("/region/set")
async def set_region(request: RegionRequest):
    """Update monitoring region (restarts monitoring if active)."""
    global monitoring_region, previous_image_hash
    
    was_active = monitoring_active
    
    # Stop monitoring if active
    if was_active:
        await stop_monitoring()
    
    # Update region
    monitoring_region = request.region
    previous_image_hash = None
    
    # Restart monitoring if it was active
    if was_active:
        monitor_request = MonitorRequest(
            region=monitoring_region,
            interval_ms=monitoring_interval_ms
        )
        return await start_monitoring(monitor_request)
    
    return {
        "status": "updated",
        "region": monitoring_region
    }


@app.get("/region/get")
async def get_region():
    """Get current monitoring region."""
    return {
        "region": monitoring_region
    }


@app.get("/texts/get")
async def get_texts():
    """Get all stored detected texts."""
    return {
        "texts": detected_texts.copy(),
        "count": len(detected_texts)
    }


@app.post("/texts/clear")
async def clear_texts():
    """Clear all stored detected texts and start a new file."""
    global detected_texts, current_storage_file
    count = len(detected_texts)
    detected_texts.clear()
    # Create a new file instead of removing the old one
    current_storage_file = get_new_storage_filename()
    # Save empty file
    save_texts_to_file()
    return {
        "status": "cleared",
        "count": count,
        "new_file": current_storage_file
    }


@app.get("/")
async def root():
    return {
        "message": "OCR Service is running",
        "monitoring": monitoring_active,
        "region": monitoring_region
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if ocr_reader is not None else "degraded",
        "monitoring": monitoring_active
    }


if __name__ == "__main__":
    # Load texts from file on startup
    logger.info("Loading stored texts from file...")
    load_texts_from_file()
    
    logger.info(f"Starting OCR server on {HOST}:{PORT}...")
    uvicorn.run(app, host=HOST, port=PORT)

