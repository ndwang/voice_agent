"""FastAPI dependency injection functions."""
from typing import TYPE_CHECKING
from fastapi import Request

if TYPE_CHECKING:
    from orchestrator.server import OrchestratorServer
    from orchestrator.managers.metrics_manager import MetricsManager
    from orchestrator.tools.registry import ToolRegistry


def get_orchestrator(request: Request) -> "OrchestratorServer":
    """Get orchestrator instance from app state."""
    return request.app.state.orchestrator


def get_metrics_manager(request: Request) -> "MetricsManager":
    """Get metrics manager instance from app state."""
    return request.app.state.metrics_manager


def get_tool_registry(request: Request) -> "ToolRegistry":
    """Get tool registry instance from app state."""
    return request.app.state.tool_registry
