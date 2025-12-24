"""Bilibili integration utilities."""
import sys
from pathlib import Path

# Add local blivedm to path
blivedm_path = Path(__file__).parent / "blivedm"
if blivedm_path.exists() and str(blivedm_path) not in sys.path:
    sys.path.insert(0, str(blivedm_path))

from .bilibili_client import BilibiliClient

__all__ = ["BilibiliClient"]
