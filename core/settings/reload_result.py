"""
Data classes for configuration reload results.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class ReloadResult:
    """Result of a configuration reload operation"""
    handler_name: str
    success: bool
    changes_applied: List[str] = field(default_factory=list)
    restart_required: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
