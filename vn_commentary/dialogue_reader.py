"""
Dialogue reader for loading visual novel scripts from JSON files.
"""
import json
from typing import List, Iterator
from pathlib import Path

from core.logging import get_logger
from vn_commentary.models import Dialogue

logger = get_logger(__name__)


class DialogueReader:
    """Reads and parses visual novel dialogue from JSON files."""

    def __init__(self, file_path: str | Path):
        """
        Initialize dialogue reader.

        Args:
            file_path: Path to JSON file containing dialogue array
        """
        self.file_path = Path(file_path)
        self._dialogues: List[Dialogue] = []
        self._load_dialogues()

    def _load_dialogues(self):
        """Load dialogues from JSON file."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dialogue file not found: {self.file_path}")

        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Support both array of dialogues and {"dialogues": [...]} format
            if isinstance(data, list):
                dialogue_list = data
            elif isinstance(data, dict) and "dialogues" in data:
                dialogue_list = data["dialogues"]
            else:
                raise ValueError("JSON must be an array or contain 'dialogues' key")

            self._dialogues = [Dialogue(**item) for item in dialogue_list]
            logger.info(f"Loaded {len(self._dialogues)} dialogues from {self.file_path}")

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {self.file_path}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load dialogues: {e}")

    def __iter__(self) -> Iterator[Dialogue]:
        """Iterate over dialogues."""
        return iter(self._dialogues)

    def __len__(self) -> int:
        """Get number of dialogues."""
        return len(self._dialogues)

    def __getitem__(self, index: int) -> Dialogue:
        """Get dialogue by index."""
        return self._dialogues[index]

    @property
    def dialogues(self) -> List[Dialogue]:
        """Get all dialogues as a list."""
        return self._dialogues
