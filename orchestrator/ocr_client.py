"""
OCR Client

Client for fetching OCR texts from OCR service on demand.
"""
import httpx
from typing import List, Dict, Any
from pathlib import Path
import logging
import sys

from core.config import get_config
from core.logging import setup_logging, get_logger

# Set up logging
setup_logging()
logger = get_logger(__name__)


class OCRClient:
    """Client for OCR service."""
    
    def __init__(self):
        """Initialize OCR client."""
        pass
    
    async def fetch_texts(self) -> List[Dict[str, Any]]:
        """
        Fetch all stored OCR texts from the OCR service.
        
        Returns:
            List of text entries, each containing 'text' and 'timestamp' keys
        """
        ocr_base_url = get_config("services", "ocr_base_url", default="http://localhost:8004")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{ocr_base_url}/texts/get",
                    timeout=5.0
                )
                response.raise_for_status()
                data = response.json()
                return data.get("texts", [])
        except (httpx.ConnectError, httpx.ConnectTimeout, ConnectionRefusedError, OSError) as e:
            # Connection errors - no need for full traceback
            logger.error(f"OCR Client: Cannot connect to OCR service at {ocr_base_url}: {e}")
            return []
        except httpx.RequestError as e:
            # Other HTTP errors - log without full traceback
            logger.error(f"OCR Client: Request error fetching texts: {e}")
            return []
        except Exception as e:
            # Unexpected errors - show full traceback only for truly unexpected cases
            logger.error(f"OCR Client: Unexpected error fetching texts: {e}", exc_info=True)
            return []
    
    async def get_all_texts(self) -> str:
        """
        Fetch all OCR texts and return as a single string.
        
        Returns:
            All OCR texts joined together
        """
        texts = await self.fetch_texts()
        text_contents = [entry.get("text", "") for entry in texts if entry.get("text")]
        return "\n".join(text_contents)

