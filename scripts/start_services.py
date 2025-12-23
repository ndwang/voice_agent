#!/usr/bin/env python3
"""
Start All Voice Agent Services

Starts all services in separate processes/windows.
Usage: python start_services.py
"""
import sys
import subprocess
import time
import os
import logging
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.logging import setup_logging, get_logger

# Set up logging
setup_logging()
logger = get_logger(__name__)

# Service definitions
SERVICES = [
    {
        "name": "STT Service",
        "module": "stt.server",
        "port": 8001,
        "description": "STT Service (Port 8001) - Speech-to-Text"
    },
    {
        "name": "TTS Service",
        "module": "tts.server",
        "port": 8003,
        "description": "TTS Service (Port 8003) - Text-to-Speech"
    },
    {
        "name": "Orchestrator",
        "module": "orchestrator.server",
        "port": 8000,
        "description": "Orchestrator (Port 8000) - Main Coordinator (includes Audio Driver)"
    },
    # Audio Driver is now integrated into the Orchestrator and started automatically
]

def check_venv():
    """Check if virtual environment exists."""
    venv_path = Path(".venv")
    if not venv_path.exists():
        logger.error("Virtual environment not found. Run 'uv sync' first.")
        sys.exit(1)
    
    # Check for activation script
    if sys.platform == "win32":
        activate_script = venv_path / "Scripts" / "activate"
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        activate_script = venv_path / "bin" / "activate"
        python_exe = venv_path / "bin" / "python"
    
    if not python_exe.exists():
        logger.error(f"Python executable not found at {python_exe}")
        sys.exit(1)
    
    return python_exe

def start_service(python_exe, service, project_root):
    """Start a service in a new window/process."""
    logger.info(f"Starting {service['name']}...")
    
    # Build command
    cmd = [str(python_exe), "-m", service["module"]]
    
    # Platform-specific window creation
    if sys.platform == "win32":
        # Windows: Create new console window
        creationflags = subprocess.CREATE_NEW_CONSOLE
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 1  # SW_SHOW = 1 (Windows API constant)
        
        subprocess.Popen(
            cmd,
            cwd=project_root,
            creationflags=creationflags,
            startupinfo=startupinfo
        )
    elif sys.platform == "darwin":
        # macOS: Use osascript to open new terminal
        script = f'''
        tell application "Terminal"
            do script "cd '{project_root}' && {python_exe} -m {service['module']}"
            activate
        end tell
        '''
        subprocess.run(["osascript", "-e", script])
    else:
        # Linux: Try to use xterm or gnome-terminal
        try:
            # Try gnome-terminal first
            subprocess.Popen([
                "gnome-terminal", "--", "bash", "-c",
                f"cd '{project_root}' && {python_exe} -m {service['module']}; exec bash"
            ])
        except FileNotFoundError:
            # Fallback to xterm
            try:
                subprocess.Popen([
                    "xterm", "-e",
                    f"cd '{project_root}' && {python_exe} -m {service['module']}; exec bash"
                ])
            except FileNotFoundError:
                # Last resort: run in background
                logger.warning(f"No terminal emulator found. Running {service['name']} in background.")
                subprocess.Popen(
                    cmd,
                    cwd=project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

def main():
    """Main function to start all services."""
    logger.info("=" * 60)
    logger.info("Starting Voice Agent Services")
    logger.info("=" * 60)
    logger.info("")
    
    # Check virtual environment
    python_exe = check_venv()
    project_root = Path.cwd()
    
    # Start services in order
    for i, service in enumerate(SERVICES):
        start_service(python_exe, service, project_root)
        
        # Wait between services (longer wait before orchestrator)
        if service["name"] == "OCR Service":
            logger.info("  Waiting for services to initialize...")
            time.sleep(3)
        elif service["name"] == "Orchestrator":
            logger.info("  Waiting before starting orchestrator...")
            time.sleep(2)
        else:
            time.sleep(1)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("All services started!")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Check each window for service status.")
    logger.info("")
    logger.info("To check orchestrator health:")
    logger.info("  python check_services.py")
    logger.info("  or")
    logger.info("  curl http://localhost:8000/health")
    logger.info("")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted. Services may still be running.")
        logger.info("Use 'python stop_services.py' to stop all services.")
        sys.exit(0)

