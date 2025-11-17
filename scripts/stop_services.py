#!/usr/bin/env python3
"""
Stop All Voice Agent Services

Stops all running services by finding processes on their ports.
Usage: python stop_services.py
"""
import sys
import subprocess
import socket
from pathlib import Path

try:
    import psutil
except ImportError:
    print("Error: 'psutil' package is required.")
    print("Install it with: pip install psutil")
    sys.exit(1)

# Service ports
SERVICE_PORTS = {
    8000: "Orchestrator",
    8001: "STT Service",
    8002: "LLM Service",
    8003: "TTS Service",
    8004: "OCR Service",
}

def find_process_by_port(port):
    """Find process ID using a specific port."""
    try:
        for conn in psutil.net_connections(kind='inet'):
            if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                try:
                    proc = psutil.Process(conn.pid)
                    return proc
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
    except (psutil.AccessDenied, AttributeError):
        # Fallback: try using netstat/ss command
        if sys.platform == "win32":
            try:
                result = subprocess.run(
                    ["netstat", "-ano"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                for line in result.stdout.split('\n'):
                    if f":{port} " in line and "LISTENING" in line:
                        parts = line.split()
                        if len(parts) > 0:
                            try:
                                pid = int(parts[-1])
                                return psutil.Process(pid)
                            except (ValueError, psutil.NoSuchProcess):
                                pass
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
        else:
            # Linux/Mac: use lsof or ss
            try:
                result = subprocess.run(
                    ["lsof", "-ti", f":{port}"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                pid = int(result.stdout.strip())
                return psutil.Process(pid)
            except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
                pass
    
    return None

def find_audio_driver_process():
    """Find audio driver process by command line."""
    project_root = str(Path.cwd())
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline', [])
            if cmdline:
                cmdline_str = ' '.join(cmdline)
                if 'audio_driver' in cmdline_str and project_root in cmdline_str:
                    return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    return None

def stop_process(proc, service_name):
    """Stop a process gracefully, then forcefully if needed."""
    try:
        print(f"Stopping {service_name} (PID: {proc.pid})...", end=" ")
        proc.terminate()
        
        # Wait up to 5 seconds for graceful shutdown
        try:
            proc.wait(timeout=5)
            print("✓ Stopped")
            return True
        except psutil.TimeoutExpired:
            # Force kill if still running
            proc.kill()
            proc.wait(timeout=2)
            print("✓ Force stopped")
            return True
    except psutil.NoSuchProcess:
        print("(already stopped)")
        return False
    except psutil.AccessDenied:
        print("✗ Permission denied")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Main function to stop all services."""
    print("=" * 60)
    print("Stopping Voice Agent Services")
    print("=" * 60)
    print()
    
    stopped_count = 0
    
    # Stop services by port
    for port, service_name in SERVICE_PORTS.items():
        proc = find_process_by_port(port)
        if proc:
            if stop_process(proc, service_name):
                stopped_count += 1
        else:
            print(f"{service_name}: Not running on port {port}")
    
    # Stop audio driver
    print()
    audio_driver = find_audio_driver_process()
    if audio_driver:
        if stop_process(audio_driver, "Audio Driver"):
            stopped_count += 1
    else:
        print("Audio Driver: Not running")
    
    print()
    print("=" * 60)
    if stopped_count > 0:
        print(f"Stopped {stopped_count} service(s)")
    else:
        print("No services were running")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

