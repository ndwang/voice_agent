#!/usr/bin/env python3
"""
Latency Measurement Script

Monitors latency measurements from the voice agent system.
User speaks into microphone, and the script displays latency statistics after each conversation round.
"""
import asyncio
import sys
import time
import signal
import requests
from typing import Optional, Dict, List
from pathlib import Path

# Add parent directory to path to import config_loader
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config_loader import get_config


orchestrator_port = int(get_config("orchestrator", "port", default=8000))
ORCHESTRATOR_URL = f"http://localhost:{orchestrator_port}"


def format_latency(seconds: float) -> str:
    """Format latency in human-readable format."""
    if seconds < 0.001:
        return f"{seconds * 1000000:.0f}μs"
    elif seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    else:
        return f"{seconds:.2f}s"


def print_latency_results(results: Dict, round_number: int = None):
    """Print formatted latency results."""
    print("\n" + "=" * 70)
    if round_number:
        print(f"Latency Measurement Results (Round {round_number})")
    else:
        print("Latency Measurement Results")
    print("=" * 70)
    
    # Define display order and labels
    display_order = [
        ("stt_latency", "STT (speech end → transcript)"),
        ("context_formatting", "Context Formatting"),
        ("llm_time_to_first_token", "LLM Time-to-First Token"),
        ("token_processing_overhead", "Token Processing Overhead"),
        ("llm_total", "LLM Total Generation"),
        ("tts_time_to_first_audio", "TTS Time-to-First Audio"),
        ("audio_playback_latency", "Audio Playback Start"),
        ("end_to_end_speech_to_audio", "End-to-End (speech end → first audio)"),
    ]
    
    # Print measurements in order
    for key, label in display_order:
        if key in results:
            value = results[key]
            if isinstance(value, dict) and "formatted" in value:
                formatted = value["formatted"]
            elif isinstance(value, (int, float)):
                formatted = format_latency(value)
            else:
                formatted = str(value)
            print(f"{label:<45}: {formatted:>10}")
    
    # Print any other measurements not in display order
    for key, value in results.items():
        if key not in [k for k, _ in display_order]:
            if isinstance(value, dict) and "formatted" in value:
                formatted = value["formatted"]
            elif isinstance(value, (int, float)):
                formatted = format_latency(value)
            else:
                formatted = str(value)
            print(f"{key:<45}: {formatted:>10}")
    
    print("=" * 70 + "\n")


def check_services():
    """Check if required services are running."""
    services = {
        "Orchestrator": f"{ORCHESTRATOR_URL}/health",
    }
    
    print("Checking services...")
    all_running = True
    for name, url in services.items():
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"  ✓ {name} is running")
            else:
                print(f"  ✗ {name} returned status {response.status_code}")
                all_running = False
        except requests.exceptions.RequestException as e:
            print(f"  ✗ {name} is not accessible: {e}")
            all_running = False
    
    return all_running


def enable_latency_tracking():
    """Enable latency tracking in config."""
    # Note: This is informational only - actual config is in config.yaml
    print("Note: Enable latency tracking by setting 'enable_latency_tracking: true' in config.yaml")
    print("You may need to restart the orchestrator for this to take effect")


def get_latest_latency() -> Optional[Dict]:
    """Get latest latency results from orchestrator."""
    try:
        # Use a short timeout to make the request more interruptible
        response = requests.get(f"{ORCHESTRATOR_URL}/latency/latest", timeout=1.0)
        if response.status_code == 200:
            data = response.json()
            if "error" in data or "message" in data:
                return None
            return data.get("results")
        return None
    except KeyboardInterrupt:
        # Re-raise KeyboardInterrupt immediately
        raise
    except requests.exceptions.RequestException:
        return None


def results_fingerprint(results: Dict) -> str:
    """Create a fingerprint/hash of results for comparison."""
    import json
    # Create a stable representation of the results
    # Extract just the numeric values to create a fingerprint
    fingerprint_data = {}
    for key, value in sorted(results.items()):
        if isinstance(value, dict) and "seconds" in value:
            fingerprint_data[key] = round(value["seconds"], 6)  # Round to microsecond precision
        elif isinstance(value, (int, float)):
            fingerprint_data[key] = round(value, 6)
    return json.dumps(fingerprint_data, sort_keys=True)


def monitor_latency(round_number: int = 1, seen_fingerprints: set = None):
    """
    Monitor for latency results.
    
    Args:
        round_number: Current round number
        seen_fingerprints: Set of result fingerprints already seen (to avoid duplicates)
    
    Returns:
        Tuple of (results dict, fingerprint string) or (None, None)
    """
    if seen_fingerprints is None:
        seen_fingerprints = set()
    
    print(f"\nWaiting for latency measurement (Round {round_number})...")
    print("Speak into the microphone and wait for the response.")
    print("(Press Ctrl+C to exit)\n")
    
    poll_interval = 0.5  # Poll every 500ms for more responsive Ctrl+C
    
    while True:
        try:
            results = get_latest_latency()
            if results:
                fingerprint = results_fingerprint(results)
                if fingerprint not in seen_fingerprints:
                    print_latency_results(results, round_number)
                    seen_fingerprints.add(fingerprint)
                    return results, fingerprint
            
            # Use shorter sleep intervals for more responsive interrupt handling
            time.sleep(poll_interval)
        except (KeyboardInterrupt, SystemExit):
            # Re-raise to be handled by the main loop
            raise


def main():
    """Main function."""
    print("=" * 70)
    print("Voice Agent Latency Measurement Tool")
    print("=" * 70)
    print()
    print("This script monitors latency measurements from the voice agent.")
    print("Make sure all services are running before starting.")
    print()
    
    # Check services
    if not check_services():
        print("\nError: Some services are not running.")
        print("Please start all services before running this script.")
        print("You can use: uv run python scripts/start_services.py")
        sys.exit(1)
    
    # Enable latency tracking
    enable_latency_tracking()
    print()
    print("IMPORTANT: If the orchestrator is already running, you need to")
    print("restart it for latency tracking to be enabled.")
    print()
    
    response = input("Is the orchestrator running with latency tracking enabled? (y/n): ")
    if response.lower() != 'y':
        print("\nPlease:")
        print("1. Set 'enable_latency_tracking: true' in config.yaml")
        print("2. Restart the orchestrator")
        print("3. Run this script again")
        sys.exit(1)
    
    print("\nStarting latency monitoring...")
    print("Speak into the microphone to trigger measurements.")
    print("Results will be displayed after each conversation round.")
    print()
    
    # Initialize seen_fingerprints with any existing measurement to ignore stale results
    seen_fingerprints = set()
    initial_results = get_latest_latency()
    if initial_results:
        initial_fingerprint = results_fingerprint(initial_results)
        seen_fingerprints.add(initial_fingerprint)
        print("(Ignoring any existing measurements from before monitoring started)")
        print()
    
    round_number = 1
    all_results: List[Dict] = []
    
    try:
        while True:
            results, fingerprint = monitor_latency(round_number, seen_fingerprints)
            if results:
                all_results.append(results)
                round_number += 1
                
                # Ask if user wants to continue
                print("Measurement complete. Speak again for another measurement, or press Ctrl+C to exit.")
                print()
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        print(f"Total measurements: {len(all_results)}")
        if all_results:
            print("\nAll measurements completed. Exiting.")
        print("=" * 70)
        sys.exit(0)


if __name__ == "__main__":
    main()

