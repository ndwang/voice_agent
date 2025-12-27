"""
OBS WebSocket Client

Client for connecting to OBS Studio via WebSocket API.
"""
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from obswebsocket import obsws, requests  # noqa: E402
from core.config import get_config
from core.logging import get_logger
from orchestrator.clients.base import BaseClient

logger = get_logger(__name__)

# Connection timeout in seconds - fail fast if OBS is not available
CONNECTION_TIMEOUT = 2.0
# Operation timeout for OBS calls
OPERATION_TIMEOUT = 1.0
# Time to wait before retrying connection after failure (seconds)
RETRY_DELAY = 30.0


class OBSClient(BaseClient):
    """Client for OBS Studio WebSocket API."""
    
    def __init__(self):
        """Initialize OBS client."""
        super().__init__()
        self.ws = None
        self._connection_failed = False  # Track if connection has failed to avoid repeated attempts
        self._connection_failed_time = None  # Timestamp of last connection failure
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="obs_client")
        
        # Load configuration
        host = get_config("obs", "websocket", "host", default="localhost")
        port = get_config("obs", "websocket", "port", default=4455)
        password = get_config("obs", "websocket", "password", default="")
        
        # Fallback to old config structure for backward compatibility
        if not password:
            password = get_config("obs", "password", default="")
        
        if not password:
            logger.warning("No OBS WebSocket password configured!")
        
        # Store connection info
        self.host = host
        self.port = port
        self.password = password
    
    async def connect(self):
        """Connect to OBS WebSocket with timeout to avoid blocking."""
        if self.ws is not None:
            return
        
        # If we've already failed to connect, check if enough time has passed to retry
        if self._connection_failed:
            if self._connection_failed_time is not None:
                time_since_failure = time.time() - self._connection_failed_time
                if time_since_failure < RETRY_DELAY:
                    return  # Too soon to retry
                # Enough time has passed, reset failure flag and try again
                self._connection_failed = False
                self._connection_failed_time = None
            else:
                return  # Failed but no timestamp (shouldn't happen, but be safe)
        
        self.ws = obsws(self.host, self.port, self.password)
        try:
            # Run blocking connect in thread pool with timeout
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(self._executor, self.ws.connect),
                timeout=CONNECTION_TIMEOUT
            )
            logger.info("Connected to OBS WebSocket")
            self._connection_failed = False
        except asyncio.TimeoutError:
            logger.debug(f"OBS connection timeout after {CONNECTION_TIMEOUT}s - OBS may not be available")
            self.ws = None
            self._connection_failed = True
            self._connection_failed_time = time.time()
        except Exception as e:
            logger.debug(f"Could not connect to OBS: {e}")
            self.ws = None
            self._connection_failed = True
            self._connection_failed_time = time.time()
    
    async def _safe_call(self, request, timeout=OPERATION_TIMEOUT):
        """Safely call OBS WebSocket API with timeout."""
        if not self.ws:
            return None
        
        try:
            loop = asyncio.get_event_loop()
            # Run blocking call in thread pool with timeout
            return await asyncio.wait_for(
                loop.run_in_executor(self._executor, self.ws.call, request),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.debug(f"OBS call timeout after {timeout}s")
            # Mark connection as failed so we don't keep trying
            self._connection_failed = True
            self._connection_failed_time = time.time()
            self.ws = None
            return None
        except Exception as e:
            logger.debug(f"OBS call failed: {e}")
            # Mark connection as failed so we don't keep trying
            self._connection_failed = True
            self._connection_failed_time = time.time()
            self.ws = None
            return None
    
    async def disconnect(self):
        """Disconnect from OBS WebSocket."""
        if self.ws:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self._executor, self.ws.disconnect)
                logger.info("Disconnected from OBS WebSocket")
            except Exception as e:
                logger.debug(f"Error disconnecting from OBS: {e}")
            finally:
                self.ws = None
                self._connection_failed = False
                self._connection_failed_time = None
    
    async def get_current_scene(self) -> str:
        """Get the current program scene name."""
        if not self.ws:
            return ""
        response = await self._safe_call(requests.GetCurrentProgramScene())
        if not response:
            return ""
        return response.datain.get("currentProgramSceneName", "")
    
    async def set_scene(self, new_scene: str):
        """Set the current scene."""
        if not self.ws:
            return
        await self._safe_call(requests.SetCurrentProgramScene(sceneName=new_scene))
    
    async def set_filter_visibility(self, source_name: str, filter_name: str, filter_enabled: bool = True):
        """Set the visibility of any source's filters."""
        if not self.ws:
            return
        await self._safe_call(requests.SetSourceFilterEnabled(
            sourceName=source_name, 
            filterName=filter_name, 
            filterEnabled=filter_enabled
        ))
    
    async def set_source_visibility(self, scene_name: str, source_name: str, source_visible: bool = True):
        """Set the visibility of any source."""
        if not self.ws:
            return
        response = await self._safe_call(requests.GetSceneItemId(sceneName=scene_name, sourceName=source_name))
        if not response:
            return
        myItemID = response.datain['sceneItemId']
        await self._safe_call(requests.SetSceneItemEnabled(
            sceneName=scene_name, 
            sceneItemId=myItemID, 
            sceneItemEnabled=source_visible
        ))
    
    async def get_text(self, source_name: str) -> str:
        """Returns the current text of a text source."""
        if not self.ws:
            return ""
        response = await self._safe_call(requests.GetInputSettings(inputName=source_name))
        if not response:
            return ""
        return response.datain.get("inputSettings", {}).get("text", "")
    
    async def set_text(self, source_name: str, new_text: str):
        """Set the text of a text source."""
        if not self.ws:
            return
        await self._safe_call(requests.SetInputSettings(
            inputName=source_name, 
            inputSettings={'text': new_text}
        ))
    
    async def get_source_transform(self, scene_name: str, source_name: str) -> dict:
        """Get source transform information."""
        if not self.ws:
            return {}
        response = await self._safe_call(requests.GetSceneItemId(sceneName=scene_name, sourceName=source_name))
        if not response:
            return {}
        myItemID = response.datain['sceneItemId']
        response = await self._safe_call(requests.GetSceneItemTransform(sceneName=scene_name, sceneItemId=myItemID))
        if not response:
            return {}
        transform = {}
        transform_data = response.datain.get("sceneItemTransform", {})
        transform["positionX"] = transform_data.get("positionX", 0)
        transform["positionY"] = transform_data.get("positionY", 0)
        transform["scaleX"] = transform_data.get("scaleX", 1.0)
        transform["scaleY"] = transform_data.get("scaleY", 1.0)
        transform["rotation"] = transform_data.get("rotation", 0)
        transform["sourceWidth"] = transform_data.get("sourceWidth", 0)
        transform["sourceHeight"] = transform_data.get("sourceHeight", 0)
        transform["width"] = transform_data.get("width", 0)
        transform["height"] = transform_data.get("height", 0)
        transform["cropLeft"] = transform_data.get("cropLeft", 0)
        transform["cropRight"] = transform_data.get("cropRight", 0)
        transform["cropTop"] = transform_data.get("cropTop", 0)
        transform["cropBottom"] = transform_data.get("cropBottom", 0)
        return transform
    
    async def set_source_transform(self, scene_name: str, source_name: str, new_transform: dict):
        """
        Set source transform.
        
        The transform should be a dictionary containing any of the following keys:
        positionX, positionY, scaleX, scaleY, rotation, width, height, sourceWidth, 
        sourceHeight, cropTop, cropBottom, cropLeft, cropRight
        """
        if not self.ws:
            return
        response = await self._safe_call(requests.GetSceneItemId(sceneName=scene_name, sourceName=source_name))
        if not response:
            return
        myItemID = response.datain['sceneItemId']
        await self._safe_call(requests.SetSceneItemTransform(
            sceneName=scene_name, 
            sceneItemId=myItemID, 
            sceneItemTransform=new_transform
        ))
    
    async def get_input_settings(self, input_name: str):
        """Get input-specific settings (e.g., font, color for text sources)."""
        if not self.ws:
            return None
        return await self._safe_call(requests.GetInputSettings(inputName=input_name))
    
    async def get_input_kind_list(self):
        """Get list of all the input types."""
        if not self.ws:
            return None
        return await self._safe_call(requests.GetInputKindList())
    
    async def get_scene_items(self, scene_name: str):
        """Get list of all items in a certain scene."""
        if not self.ws:
            return None
        return await self._safe_call(requests.GetSceneItemList(sceneName=scene_name))

