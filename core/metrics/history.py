from collections import deque
from typing import List, Dict, Any
import statistics
from core.logging import get_logger

logger = get_logger(__name__)

class MetricsHistory:
    def __init__(self, max_size: int = 20):
        self.history: deque[Dict] = deque(maxlen=max_size)

    def add(self, turn_metrics: Dict):
        """Add a finalized turn to the history"""
        logger.debug(f"Adding turn metrics: {turn_metrics}")
        self.history.append(turn_metrics)

    def _get_percentile(self, data: List[float], p: int) -> float:
        """Calculate the percentile of the data if we have enough samples"""
        if len(data) <= int(100 / (100 - p)):
            return None

        sorted_data = sorted(data)
        idx = int((p / 100) * len(sorted_data))
        return sorted_data[min(idx, len(sorted_data)-1)]

    def _summarize(self, data: List[float]) -> Dict[str, float]:
        if not data: 
            return {"avg": 0, "median": 0, "p95": 0, "p99": 0, "min": 0, "max": 0}
        return {
            "avg": statistics.mean(data),
            "median": statistics.median(data),
            "p95": self._get_percentile(data, 95),
            "p99": self._get_percentile(data, 99),
            "min": min(data),
            "max": max(data)
        }

    def get_analysis(self) -> Dict[str, Any]:
        """Compute comprehensive statistics for the current window"""
        if not self.history:
            return {"total_requests": 0}

        # Extract columns
        e2e = [m['e2e_latency_ms'] for m in self.history if 'e2e_latency_ms' in m]
        ttft = [m['llm_ttft_ms'] for m in self.history if 'llm_ttft_ms' in m]
        sent = [m['first_sentence_ms'] for m in self.history if 'first_sentence_ms' in m]
        tps = [m['tokens_per_sec'] for m in self.history if 'tokens_per_sec' in m]

        analysis = {
            "total_requests": len(self.history),
            "e2e": self._summarize(e2e),
            "llm_ttft": self._summarize(ttft),
            "first_sentence": self._summarize(sent),
            "throughput": self._summarize(tps)
        }
        
        # Calculate Bottlenecks (Averages)
        avg_e2e = analysis["e2e"]["avg"]
        if avg_e2e > 0:
            avg_ttft = analysis["llm_ttft"]["avg"]
            avg_first_sentence = analysis["first_sentence"]["avg"]
            avg_tts_synthesis = analysis["e2e"]["avg"]

            analysis["latency_breakdown"] = {
                "llm_wait": round((avg_ttft / avg_e2e) * 100),
                "sentence_formation": round((avg_first_sentence / avg_e2e) * 100),
                "tts_synthesis": round((avg_tts_synthesis / avg_e2e) * 100)
            }

        return analysis
