from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class SystemState:
    """
    Struct-like representation of the system's current activity state.
    Provides type-safety and autocompletion for state tracking.
    """
    transcribing: bool = False
    responding: bool = False
    synthesizing: bool = False
    playing: bool = False
    listening: bool = True
    executing_tools: bool = False

    def update(self, changes: Dict[str, Any]):
        """Update fields from a dictionary, ignoring unknown keys."""
        for key, value in changes.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, bool]:
        """Convert state to a dictionary."""
        return {
            "transcribing": self.transcribing,
            "responding": self.responding,
            "synthesizing": self.synthesizing,
            "playing": self.playing,
            "listening": self.listening,
            "executing_tools": self.executing_tools
        }

