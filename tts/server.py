import uvicorn
import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.server import create_app
from core.settings import get_settings
from core.logging import get_logger
from tts.api import router as tts_router

logger = get_logger(__name__)

def main():
    settings = get_settings()
    HOST = settings.tts.host
    PORT = settings.tts.port
    
    app = create_app(
        title="TTS Service",
        description="Text-to-Speech Service with Streaming Support",
        version="1.0.0",
        service_name="tts"
    )
    
    app.include_router(tts_router)
    
    # Configure uvicorn to send WebSocket pings
    logger.info(f"Starting TTS server on {HOST}:{PORT}...")
    uvicorn.run(
        app, 
        host=HOST, 
        port=PORT,
        ws_ping_interval=5.0,
        ws_ping_timeout=None
    )

if __name__ == "__main__":
    main()


