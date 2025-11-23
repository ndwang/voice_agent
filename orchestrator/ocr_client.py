"""
OCR Client

Client for fetching OCR texts from OCR service on demand.
"""
import httpx
from typing import List, Dict, Any
import logging

from orchestrator.config import Config
from orchestrator.logging_config import setup_logging, get_logger

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
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{Config.get_ocr_base_url()}/texts/get",
                    timeout=5.0
                )
                response.raise_for_status()
                data = response.json()
                return data.get("texts", [])
        except httpx.RequestError as e:
            logger.error(f"OCR Client: Error fetching texts: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"OCR Client: Error fetching texts: {e}", exc_info=True)
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

