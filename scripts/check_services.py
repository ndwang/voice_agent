#!/usr/bin/env python3
"""
Check Status of All Voice Agent Services

Checks if all services are running and shows their status.
Usage: python check_services.py
"""
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

try:
    import requests
    import psutil
    
    # Ensure project root is in path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except ImportError as e:
    print("Error: Missing required package.")
    print("Install dependencies with: pip install psutil requests")
    sys.exit(1)

# Service definitions
SERVICES = [
    {
        "name": "Orchestrator",
        "port": 8000,
        "url": "http://localhost:8000/health",
        "endpoint": "/health"
    },
    {
        "name": "STT Service",
        "port": 8001,
        "url": "http://localhost:8001/health",
        "endpoint": "/health"
    },
    {
        "name": "TTS Service",
        "port": 8003,
        "url": "http://localhost:8003/health",
        "endpoint": "/health"
    },
    {
        "name": "OCR Service",
        "port": 8004,
        "url": "http://localhost:8004/health",
        "endpoint": "/health"
    },
]

def check_port(port: int) -> bool:
    """Check if a port is listening."""
    try:
        for conn in psutil.net_connections(kind='inet'):
            if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                return True
    except (psutil.AccessDenied, AttributeError):
        # Fallback: try to connect
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result == 0
    
    return False

def check_http_service(service: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Check HTTP service health."""
    try:
        response = requests.get(service["url"], timeout=2)
        if response.status_code == 200:
            try:
                data = response.json()
                return True, data
            except ValueError:
                return True, None
        return False, None
    except requests.exceptions.RequestException:
        return False, None

def main():
    """Main function to check all services."""
    print("=" * 60)
    print("Voice Agent Services Status")
    print("=" * 60)
    print()
    
    # Check HTTP services
    for service in SERVICES:
        port_ok = check_port(service["port"])
        http_ok, data = check_http_service(service)
        
        status = "Running" if (port_ok or http_ok) else "Not Running"
        status_symbol = "[+]" if (port_ok or http_ok) else "[X]"
        status_color = "\033[92m" if (port_ok or http_ok) else "\033[91m"
        reset_color = "\033[0m"
        
        print(f"{service['name']} (Port {service['port']}): {status_color}{status_symbol} {status}{reset_color}")
        
        # Show service details from health endpoint
        if data:
            print("  Details:")
            print(f"    Service: {data.get('service', 'N/A')}")
            print(f"    Status:  {data.get('status', 'N/A')}")
    
    # Check audio driver
    print()
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
