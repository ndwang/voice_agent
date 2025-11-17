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
from pathlib import Path

# Service definitions
SERVICES = [
    {
        "name": "STT Service",
        "module": "stt.stt_server",
        "port": 8001,
        "description": "STT Service (Port 8001) - Speech-to-Text"
    },
    {
        "name": "LLM Service",
        "module": "llm.llm_server",
        "port": 8002,
        "description": "LLM Service (Port 8002) - Language Model"
    },
    {
        "name": "TTS Service",
        "module": "tts.tts_server",
        "port": 8003,
        "description": "TTS Service (Port 8003) - Text-to-Speech"
    },
    {
        "name": "OCR Service",
        "module": "ocr.ocr_server",
        "port": 8004,
        "description": "OCR Service (Port 8004) - Optical Character Recognition"
    },
    {
        "name": "Orchestrator",
        "module": "orchestrator.agent",
        "port": 8000,
        "description": "Orchestrator (Port 8000) - Main Coordinator"
    },
    {
        "name": "Audio Driver",
        "module": "audio.audio_driver",
        "port": None,
        "description": "Audio Driver - Microphone Input"
    },
]

def check_venv():
    """Check if virtual environment exists."""
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("Error: Virtual environment not found. Run 'uv sync' first.")
        sys.exit(1)
    
    # Check for activation script
    if sys.platform == "win32":
        activate_script = venv_path / "Scripts" / "activate"
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        activate_script = venv_path / "bin" / "activate"
        python_exe = venv_path / "bin" / "python"
    
    if not python_exe.exists():
        print(f"Error: Python executable not found at {python_exe}")
        sys.exit(1)
    
    return python_exe

def start_service(python_exe, service, project_root):
    """Start a service in a new window/process."""
    print(f"Starting {service['name']}...")
    
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
                print(f"  Warning: No terminal emulator found. Running {service['name']} in background.")
                subprocess.Popen(
                    cmd,
                    cwd=project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

def main():
    """Main function to start all services."""
    print("=" * 60)
    print("Starting Voice Agent Services")
    print("=" * 60)
    print()
    
    # Check virtual environment
    python_exe = check_venv()
    project_root = Path.cwd()
    
    # Start services in order
    for i, service in enumerate(SERVICES):
        start_service(python_exe, service, project_root)
        
        # Wait between services (longer wait before orchestrator)
        if service["name"] == "OCR Service":
            print("  Waiting for services to initialize...")
            time.sleep(3)
        elif service["name"] == "Orchestrator":
            print("  Waiting before starting orchestrator...")
            time.sleep(2)
        else:
            time.sleep(1)
    
    print()
    print("=" * 60)
    print("All services started!")
    print("=" * 60)
    print()
    print("Check each window for service status.")
    print()
    print("To check orchestrator health:")
    print("  python check_services.py")
    print("  or")
    print("  curl http://localhost:8000/health")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Services may still be running.")
        print("Use 'python stop_services.py' to stop all services.")
        sys.exit(0)

