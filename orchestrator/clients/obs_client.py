"""
OBS WebSocket Client

Client for connecting to OBS Studio via WebSocket API.
"""
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any
from obswebsocket import obsws, requests  # noqa: E402
from core.settings import get_settings
from core.logging import get_logger
from orchestrator.clients.base import BaseClient

logger = get_logger(__name__)

# Connection timeout in seconds - fail fast if OBS is not available
CONNECTION_TIMEOUT = 2.0
# Operation timeout for OBS calls
OPERATION_TIMEOUT = 1.0
# Constant retry delay - will keep trying at this interval indefinitely
RETRY_DELAY = 5.0


class OBSClient(BaseClient):
    """Client for OBS Studio WebSocket API."""

    def __init__(self):
        """Initialize OBS client."""
        super().__init__()
        self.ws = None
        self._connection_failed = False  # Track if connection has failed to avoid repeated attempts
        self._connection_failed_time = None  # Timestamp of last connection failure
        self._last_successful_connection = None  # Timestamp of last successful connection
        self._failed_operations = []  # Track operations that failed due to connection issues
        self._is_connected = False  # Track connection state
        self._last_reconnect_log_time = None  # Track when we last logged reconnection attempt
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="obs_client")

        # Load configuration
        settings = get_settings()
        host = settings.obs.websocket.host
        port = settings.obs.websocket.port
        password = settings.obs.websocket.password

        if not password:
            logger.warning("No OBS WebSocket password configured!")

        # Store connection info
        self.host = host
        self.port = port
        self.password = password

        logger.info(f"OBS client initialized - target: {host}:{port}")

    async def connect(self):
        """Connect to OBS WebSocket with timeout to avoid blocking."""
        if self.ws is not None and self._is_connected:
            return

        # If we've already failed to connect, check if enough time has passed to retry
        if self._connection_failed:
            if self._connection_failed_time is not None:
                time_since_failure = time.time() - self._connection_failed_time
                if time_since_failure < RETRY_DELAY:
                    return  # Too soon to retry

                # Log reconnection attempt (but not every time to avoid spam)
                # Only log every 30 seconds
                current_time = time.time()
                should_log = (
                    self._last_reconnect_log_time is None or
                    current_time - self._last_reconnect_log_time >= 30.0
                )
                if should_log:
                    logger.info("Attempting to reconnect to OBS WebSocket...")
                    self._last_reconnect_log_time = current_time

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

            # Connection successful
            was_reconnect = self._last_successful_connection is not None and not self._is_connected
            self._is_connected = True
            self._connection_failed = False
            self._last_successful_connection = time.time()

            if was_reconnect:
                logger.info("Successfully reconnected to OBS WebSocket")
                self._failed_operations.clear()
            else:
                logger.info(f"Connected to OBS WebSocket at {self.host}:{self.port}")

        except asyncio.TimeoutError:
            self._is_connected = False
            self.ws = None
            self._connection_failed = True
            self._connection_failed_time = time.time()

            # Only log the first timeout warning, subsequent ones will be logged every 30s
            if self._last_successful_connection is None:
                logger.warning(
                    f"OBS connection timeout after {CONNECTION_TIMEOUT}s - "
                    f"OBS may not be running at {self.host}:{self.port}. Will retry every {RETRY_DELAY}s"
                )

        except Exception as e:
            self._is_connected = False
            self.ws = None
            self._connection_failed = True
            self._connection_failed_time = time.time()

            # Only log the first connection error, subsequent ones will be logged every 30s
            if self._last_successful_connection is None:
                logger.warning(
                    f"Could not connect to OBS at {self.host}:{self.port}: {e}. "
                    f"Will retry every {RETRY_DELAY}s"
                )
    
    def _track_failed_operation(self, operation_name: str):
        """Track a failed operation for logging."""
        # Keep only last 50 failed operations to avoid memory growth
        if len(self._failed_operations) >= 50:
            self._failed_operations = self._failed_operations[-25:]
        self._failed_operations.append(operation_name)

    async def _safe_call(self, request, timeout=OPERATION_TIMEOUT, operation_name: Optional[str] = None):
        """Safely call OBS WebSocket API with timeout and auto-reconnect."""
        # Try to connect if not connected
        if not self.ws or not self._is_connected:
            await self.connect()

        # If still not connected, log and return
        if not self.ws or not self._is_connected:
            if operation_name:
                self._track_failed_operation(operation_name)
            return None

        try:
            loop = asyncio.get_event_loop()
            # Run blocking call in thread pool with timeout
            result = await asyncio.wait_for(
                loop.run_in_executor(self._executor, self.ws.call, request),
                timeout=timeout
            )
            return result

        except asyncio.TimeoutError:
            # Operation timed out - likely connection issue
            self._is_connected = False
            self._connection_failed = True
            self._connection_failed_time = time.time()
            self.ws = None

            if operation_name:
                self._track_failed_operation(operation_name)

            # Trigger immediate reconnection attempt on next call
            return None

        except Exception as e:
            # Operation failed - likely connection issue
            self._is_connected = False
            self._connection_failed = True
            self._connection_failed_time = time.time()
            self.ws = None

            if operation_name:
                self._track_failed_operation(operation_name)

            # Trigger immediate reconnection attempt on next call
            return None
    
    async def disconnect(self):
        """Disconnect from OBS WebSocket."""
        if self.ws:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self._executor, self.ws.disconnect)
                logger.info("Disconnected from OBS WebSocket")
            except Exception as e:
                logger.warning(f"Error disconnecting from OBS: {e}")
            finally:
                self.ws = None
                self._is_connected = False
                self._connection_failed = False
                self._connection_failed_time = None
    
    async def get_current_scene(self) -> str:
        """Get the current program scene name."""
        response = await self._safe_call(
            requests.GetCurrentProgramScene(),
            operation_name="get_current_scene"
        )
        if not response:
            return ""
        return response.datain.get("currentProgramSceneName", "")

    async def set_scene(self, new_scene: str):
        """Set the current scene."""
        await self._safe_call(
            requests.SetCurrentProgramScene(sceneName=new_scene),
            operation_name=f"set_scene({new_scene})"
        )
    
    async def set_filter_visibility(self, source_name: str, filter_name: str, filter_enabled: bool = True):
        """Set the visibility of any source's filters."""
        await self._safe_call(
            requests.SetSourceFilterEnabled(
                sourceName=source_name,
                filterName=filter_name,
                filterEnabled=filter_enabled
            ),
            operation_name=f"set_filter_visibility({source_name}/{filter_name}={filter_enabled})"
        )

    async def set_source_visibility(self, scene_name: str, source_name: str, source_visible: bool = True):
        """Set the visibility of any source."""
        response = await self._safe_call(
            requests.GetSceneItemId(sceneName=scene_name, sourceName=source_name),
            operation_name=f"get_scene_item_id({scene_name}/{source_name})"
        )
        if not response:
            return
        myItemID = response.datain['sceneItemId']
        await self._safe_call(
            requests.SetSceneItemEnabled(
                sceneName=scene_name,
                sceneItemId=myItemID,
                sceneItemEnabled=source_visible
            ),
            operation_name=f"set_source_visibility({scene_name}/{source_name}={source_visible})"
        )
    
    async def get_text(self, source_name: str) -> str:
        """Returns the current text of a text source."""
        response = await self._safe_call(
            requests.GetInputSettings(inputName=source_name),
            operation_name=f"get_text({source_name})"
        )
        if not response:
            return ""
        return response.datain.get("inputSettings", {}).get("text", "")

    async def set_text(self, source_name: str, new_text: str):
        """Set the text of a text source."""
        await self._safe_call(
            requests.SetInputSettings(
                inputName=source_name,
                inputSettings={'text': new_text}
            ),
            operation_name=f"set_text({source_name})"
        )
    
    async def get_source_transform(self, scene_name: str, source_name: str) -> dict:
        """Get source transform information."""
        response = await self._safe_call(
            requests.GetSceneItemId(sceneName=scene_name, sourceName=source_name),
            operation_name=f"get_scene_item_id({scene_name}/{source_name})"
        )
        if not response:
            return {}
        myItemID = response.datain['sceneItemId']
        response = await self._safe_call(
            requests.GetSceneItemTransform(sceneName=scene_name, sceneItemId=myItemID),
            operation_name=f"get_source_transform({scene_name}/{source_name})"
        )
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
        response = await self._safe_call(
            requests.GetSceneItemId(sceneName=scene_name, sourceName=source_name),
            operation_name=f"get_scene_item_id({scene_name}/{source_name})"
        )
        if not response:
            return
        myItemID = response.datain['sceneItemId']
        await self._safe_call(
            requests.SetSceneItemTransform(
                sceneName=scene_name,
                sceneItemId=myItemID,
                sceneItemTransform=new_transform
            ),
            operation_name=f"set_source_transform({scene_name}/{source_name})"
        )
    
    async def get_input_settings(self, input_name: str):
        """Get input-specific settings (e.g., font, color for text sources)."""
        return await self._safe_call(
            requests.GetInputSettings(inputName=input_name),
            operation_name=f"get_input_settings({input_name})"
        )

    async def get_input_kind_list(self):
        """Get list of all the input types."""
        return await self._safe_call(
            requests.GetInputKindList(),
            operation_name="get_input_kind_list"
        )

    async def get_scene_items(self, scene_name: str):
        """Get list of all items in a certain scene."""
        return await self._safe_call(
            requests.GetSceneItemList(sceneName=scene_name),
            operation_name=f"get_scene_items({scene_name})"
        )

    def is_connected(self) -> bool:
        """Check if currently connected to OBS."""
        return self._is_connected and self.ws is not None

    def get_connection_status(self) -> Dict[str, Any]:
        """Get detailed connection status information."""
        status = {
            "connected": self.is_connected(),
            "host": self.host,
            "port": self.port,
            "failed_operations_count": len(self._failed_operations),
        }

        if self._last_successful_connection:
            status["last_connected"] = time.time() - self._last_successful_connection
        else:
            status["last_connected"] = None

        if self._connection_failed and self._connection_failed_time:
            status["time_since_failure"] = time.time() - self._connection_failed_time
            status["next_retry_in"] = max(0, RETRY_DELAY - status["time_since_failure"])
        else:
            status["time_since_failure"] = None
            status["next_retry_in"] = None

        if self._failed_operations:
            status["recent_failed_operations"] = self._failed_operations[-5:]

        return status

