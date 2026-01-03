from typing import TYPE_CHECKING
from fastapi import APIRouter, Depends
from orchestrator.api.dependencies import get_metrics_manager

if TYPE_CHECKING:
    from orchestrator.managers.metrics_manager import MetricsManager

router = APIRouter()


@router.get("/metrics")
async def get_metrics(metrics_manager: "MetricsManager" = Depends(get_metrics_manager)):
    return metrics_manager.history.get_analysis()