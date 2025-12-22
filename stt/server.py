import uvicorn
import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.server import create_app
from core.config import get_config
from core.logging import get_logger
from stt.api import router as stt_router

logger = get_logger(__name__)

def main():
    HOST = get_config("stt", "host", default="0.0.0.0")
    PORT = get_config("stt", "port", default=8001)
    
    app = create_app(
        title="STT Service",
        description="Real-time Speech-to-Text Service",
        version="1.0.0"
    )
    
    # Register router
    app.include_router(stt_router)
    
    logger.info(f"Starting STT server on {HOST}:{PORT}...")
    uvicorn.run(app, host=HOST, port=PORT)

if __name__ == "__main__":
    main()


