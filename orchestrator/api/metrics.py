from fastapi import APIRouter
from orchestrator.managers.metrics_manager import MetricsManager
from typing import Optional

router = APIRouter()

# Will be injected during server startup
metrics_manager: Optional[MetricsManager] = None

@router.get("/metrics")
async def get_metrics():
    return metrics_manager.history.get_analysis()