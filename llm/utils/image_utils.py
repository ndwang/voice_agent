"""
Image Utilities

Utilities for validating, reading, and processing image files for LLM providers.
"""
import logging
import mimetypes
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}

# Maximum image file size (10MB)
MAX_IMAGE_SIZE = 10 * 1024 * 1024


def validate_image_path(path: str) -> bool:
    """
    Validate that a file path exists and is a supported image format.

    Args:
        path: Path to image file

    Returns:
        True if valid, False otherwise
    """
    try:
        # Convert to Path object and resolve to absolute path
        file_path = Path(path).resolve()

        # Check if file exists
        if not file_path.is_file():
            logger.warning(f"Image file not found: {path}")
            return False

        # Check file extension
        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            logger.warning(f"Unsupported image format '{ext}': {path}")
            return False

        # Check file size
        file_size = file_path.stat().st_size
        if file_size > MAX_IMAGE_SIZE:
            logger.warning(f"Image too large ({file_size} bytes, max {MAX_IMAGE_SIZE}): {path}")
            return False

        return True

    except Exception as e:
        logger.error(f"Error validating image path {path}: {e}")
        return False


def validate_image_paths(paths: List[str]) -> List[str]:
    """
    Validate a list of image paths and return only valid ones.

    Args:
        paths: List of image file paths

    Returns:
        List of valid image paths
    """
    if not paths:
        return []

    valid_paths = []
    for path in paths:
        if validate_image_path(path):
            valid_paths.append(path)

    if len(valid_paths) < len(paths):
        logger.info(f"Validated {len(valid_paths)}/{len(paths)} images")

    return valid_paths


def read_image_file(path: str) -> bytes:
    """
    Read image file and return bytes.

    Args:
        path: Path to image file

    Returns:
        Image file contents as bytes

    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file cannot be read
        ValueError: If file is too large
    """
    file_path = Path(path).resolve()

    # Check file size
    file_size = file_path.stat().st_size
    if file_size > MAX_IMAGE_SIZE:
        raise ValueError(f"Image too large: {file_size} bytes (max {MAX_IMAGE_SIZE})")

    # Read file
    with open(file_path, 'rb') as f:
        return f.read()


def get_mime_type(path: str) -> str:
    """
    Determine MIME type from file extension.

    Args:
        path: Path to image file

    Returns:
        MIME type string (e.g., 'image/jpeg')
    """
    # First try using mimetypes module
    mime_type, _ = mimetypes.guess_type(path)

    if mime_type:
        return mime_type

    # Fallback: map extensions to MIME types
    ext = Path(path).suffix.lower()
    mime_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp'
    }

    return mime_map.get(ext, 'application/octet-stream')
