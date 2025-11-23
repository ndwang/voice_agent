"""
Latency Tracker

Utility for measuring and tracking latencies across the voice agent pipeline.
"""
import time
from typing import Dict, Optional, List
from contextlib import contextmanager
from collections import defaultdict


class LatencyTracker:
    """Tracks timing measurements for latency analysis."""
    
    def __init__(self):
        """Initialize latency tracker."""
        self.measurements: Dict[str, List[float]] = defaultdict(list)
        self.active_timers: Dict[str, float] = {}
        self.marks: Dict[str, float] = {}
        self.current_round: Optional[Dict[str, float]] = None
    
    def start(self, name: str) -> None:
        """
        Start a timer for a named measurement.
        
        Args:
            name: Name of the measurement
        """
        self.active_timers[name] = time.perf_counter()
    
    def end(self, name: str) -> Optional[float]:
        """
        End a timer and record the measurement.
        
        Args:
            name: Name of the measurement
            
        Returns:
            Elapsed time in seconds, or None if timer wasn't started
        """
        if name not in self.active_timers:
            return None
        
        elapsed = time.perf_counter() - self.active_timers[name]
        self.measurements[name].append(elapsed)
        del self.active_timers[name]
        
        # Also record in current round if active
        if self.current_round is not None:
            self.current_round[name] = elapsed
        
        return elapsed
    
    def mark(self, name: str) -> float:
        """
        Mark a timestamp with a name.
        
        Args:
            name: Name of the mark
            
        Returns:
            Current timestamp
        """
        timestamp = time.perf_counter()
        self.marks[name] = timestamp
        return timestamp
    
    def measure_between(self, start_mark: str, end_mark: str, measurement_name: Optional[str] = None) -> Optional[float]:
        """
        Measure time between two marks.
        
        Args:
            start_mark: Name of the start mark
            end_mark: Name of the end mark
            measurement_name: Optional name to record this measurement
            
        Returns:
            Elapsed time in seconds, or None if marks don't exist
        """
        if start_mark not in self.marks or end_mark not in self.marks:
            return None
        
        elapsed = self.marks[end_mark] - self.marks[start_mark]
        
        if measurement_name:
            self.measurements[measurement_name].append(elapsed)
            if self.current_round is not None:
                self.current_round[measurement_name] = elapsed
        
        return elapsed
    
    @contextmanager
    def timer(self, name: str):
        """
        Context manager for automatic timing.
        
        Usage:
            with tracker.timer("my_operation"):
                # do something
        """
        self.start(name)
        try:
            yield
        finally:
            self.end(name)
    
    def start_round(self) -> None:
        """Start a new measurement round (e.g., for a conversation turn)."""
        self.current_round = {}
        self.marks.clear()
        self.active_timers.clear()
    
    def end_round(self) -> Optional[Dict[str, float]]:
        """
        End the current measurement round and return results.
        
        Returns:
            Dictionary of measurements for this round, or None if no round was active
        """
        round_data = self.current_round.copy() if self.current_round else None
        self.current_round = None
        self.marks.clear()
        self.active_timers.clear()
        return round_data
    
    def get_measurements(self, name: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Get all measurements.
        
        Args:
            name: Optional name to filter by
            
        Returns:
            Dictionary of measurements (or single list if name provided)
        """
        if name:
            return {name: self.measurements.get(name, [])}
        return dict(self.measurements)
    
    def get_statistics(self, name: str) -> Optional[Dict[str, float]]:
        """
        Get statistics for a measurement.
        
        Args:
            name: Name of the measurement
            
        Returns:
            Dictionary with min, max, avg, count, or None if no measurements
        """
        if name not in self.measurements or not self.measurements[name]:
            return None
        
        values = self.measurements[name]
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return {
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "avg": sum(values) / n,
            "count": n,
            "p50": sorted_values[n // 2] if n > 0 else 0,
            "p95": sorted_values[int(n * 0.95)] if n > 1 else sorted_values[0],
            "p99": sorted_values[int(n * 0.99)] if n > 1 else sorted_values[0],
        }
    
    def reset(self) -> None:
        """Reset all measurements."""
        self.measurements.clear()
        self.active_timers.clear()
        self.marks.clear()
        self.current_round = None
    
    def format_latency(self, seconds: float) -> str:
        """
        Format latency in human-readable format.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted string (e.g., "245ms", "1.2s")
        """
        if seconds < 0.001:
            return f"{seconds * 1000000:.0f}μs"
        elif seconds < 1.0:
            return f"{seconds * 1000:.0f}ms"
        else:
            return f"{seconds:.2f}s"
    
    def format_round(self, round_data: Dict[str, float]) -> str:
        """
        Format a measurement round for display.
        
        Args:
            round_data: Dictionary of measurements for one round
            
        Returns:
            Formatted string
        """
        if not round_data:
            return "No measurements"
        
        lines = []
        max_name_len = max(len(name) for name in round_data.keys()) if round_data else 0
        
        for name, value in sorted(round_data.items()):
            formatted_value = self.format_latency(value)
            lines.append(f"{name:<{max_name_len + 2}}: {formatted_value:>10}")
        
        return "\n".join(lines)

