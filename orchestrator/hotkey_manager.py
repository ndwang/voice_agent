"""
Hotkey Manager

Systematic hotkey registration system using pynput for system-wide hotkey detection.
Allows registering multiple hotkeys with callback functions.
"""
import asyncio
import threading
import logging
from typing import Dict, Callable, Optional
from pynput import keyboard
from pynput.keyboard import GlobalHotKeys

logger = logging.getLogger(__name__)


class HotkeyManager:
    """Manages system-wide hotkey registration and callbacks."""
    
    def __init__(self):
        """Initialize hotkey manager."""
        self._hotkeys: Dict[str, Dict[str, any]] = {}  # hotkey_id -> {hotkey_str, callback, parsed_str}
        self._lock = threading.Lock()
        self._listener: Optional[keyboard.Listener] = None
        self._running = False
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
    
    def _parse_hotkey(self, hotkey_str: str) -> str:
        """
        Parse hotkey string into pynput GlobalHotKeys format.
        
        Supports formats:
        - "ctrl+shift+l" (lowercase) -> "<ctrl>+<shift>+l"
        - "Control+Shift+KeyL" (mixed case) -> "<ctrl>+<shift>+l"
        - "Ctrl+Shift+L" (mixed case) -> "<ctrl>+<shift>+l"
        - "f1", "F1" (function keys) -> "<f1>"
        - "space", "enter" (special keys) -> "<space>", "<enter>"
        
        Returns:
            Hotkey string in pynput GlobalHotKeys format
        """
        parts = [p.strip().lower() for p in hotkey_str.split('+')]
        parsed_parts = []
        
        # Map of modifier names
        modifier_map = {
            'ctrl': '<ctrl>',
            'control': '<ctrl>',
            'shift': '<shift>',
            'alt': '<alt>',
            'cmd': '<cmd>',
            'meta': '<cmd>',
            'super': '<cmd>',
        }
        
        # Map of special key names
        special_key_map = {
            'space': '<space>',
            'enter': '<enter>',
            'return': '<enter>',
            'tab': '<tab>',
            'esc': '<esc>',
            'escape': '<esc>',
            'backspace': '<backspace>',
            'delete': '<delete>',
            'up': '<up>',
            'down': '<down>',
            'left': '<left>',
            'right': '<right>',
            'home': '<home>',
            'end': '<end>',
            'page_up': '<page_up>',
            'page_down': '<page_down>',
            'insert': '<insert>',
        }
        
        # Function keys
        function_keys = {
            'f1': '<f1>', 'f2': '<f2>', 'f3': '<f3>', 'f4': '<f4>',
            'f5': '<f5>', 'f6': '<f6>', 'f7': '<f7>', 'f8': '<f8>',
            'f9': '<f9>', 'f10': '<f10>', 'f11': '<f11>', 'f12': '<f12>',
        }
        
        for part in parts:
            if part in modifier_map:
                parsed_parts.append(modifier_map[part])
            elif part in special_key_map:
                parsed_parts.append(special_key_map[part])
            elif part in function_keys:
                parsed_parts.append(function_keys[part])
            elif part.startswith('key') and len(part) > 3:
                # Handle "KeyL" format - extract the letter
                key_char = part[3].lower()
                parsed_parts.append(key_char)
            elif len(part) == 1:
                # Single character key
                parsed_parts.append(part)
            else:
                raise ValueError(f"Unknown key in hotkey: {part} (from {hotkey_str})")
        
        if not parsed_parts:
            raise ValueError(f"No keys specified in hotkey: {hotkey_str}")
        
        return '+'.join(parsed_parts)
    
    def register_hotkey(self, hotkey_id: str, hotkey_str: str, callback: Callable) -> bool:
        """
        Register a hotkey with ID and callback.
        
        Args:
            hotkey_id: Unique identifier for the hotkey
            hotkey_str: Hotkey string (e.g., "ctrl+shift+l")
            callback: Callback function to execute when hotkey is pressed (can be async)
            
        Returns:
            True if registration successful, False otherwise
        """
        with self._lock:
            if hotkey_id in self._hotkeys:
                logger.warning(f"Hotkey {hotkey_id} already registered, updating...")
                # Unregister old hotkey first
                self._unregister_hotkey_internal(hotkey_id)
            
            try:
                parsed = self._parse_hotkey(hotkey_str)
                
                # Create wrapper callback that handles async
                # Use a closure that references self._event_loop so it gets updated value
                def callback_wrapper():
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            # Async callback - schedule in event loop
                            # Get current event loop reference
                            loop = self._event_loop
                            if loop is None:
                                try:
                                    loop = asyncio.get_event_loop()
                                except RuntimeError:
                                    logger.error(f"No event loop available for hotkey {hotkey_id}")
                                    return
                            if loop and loop.is_running():
                                asyncio.run_coroutine_threadsafe(callback(), loop)
                        else:
                            # Sync callback - call directly
                            callback()
                    except Exception as e:
                        logger.error(f"Error executing hotkey callback for {hotkey_id}: {e}", exc_info=True)
                
                self._hotkeys[hotkey_id] = {
                    'hotkey_str': hotkey_str,
                    'parsed': parsed,
                    'callback': callback_wrapper,
                }
                logger.info(f"Registered hotkey {hotkey_id}: {hotkey_str} (parsed: {parsed})")
                
                # Restart listener to pick up new hotkey
                if self._running:
                    self._restart_listener()
                
                return True
            except Exception as e:
                logger.error(f"Failed to register hotkey {hotkey_id} ({hotkey_str}): {e}", exc_info=True)
                return False
    
    def unregister_hotkey(self, hotkey_id: str) -> bool:
        """
        Unregister a hotkey by ID.
        
        Args:
            hotkey_id: Unique identifier for the hotkey
            
        Returns:
            True if unregistration successful, False otherwise
        """
        with self._lock:
            return self._unregister_hotkey_internal(hotkey_id)
    
    def _unregister_hotkey_internal(self, hotkey_id: str) -> bool:
        """Internal unregister method (assumes lock is held)."""
        if hotkey_id not in self._hotkeys:
            logger.warning(f"Hotkey {hotkey_id} not found for unregistration")
            return False
        
        del self._hotkeys[hotkey_id]
        logger.info(f"Unregistered hotkey {hotkey_id}")
        
        # Restart listener
        if self._running:
            self._restart_listener()
        
        return True
    
    def update_hotkey(self, hotkey_id: str, hotkey_str: str) -> bool:
        """
        Update existing hotkey string.
        
        Args:
            hotkey_id: Unique identifier for the hotkey
            hotkey_str: New hotkey string
            
        Returns:
            True if update successful, False otherwise
        """
        with self._lock:
            if hotkey_id not in self._hotkeys:
                logger.warning(f"Hotkey {hotkey_id} not found for update")
                return False
            
            callback = self._hotkeys[hotkey_id]['callback']
            return self.register_hotkey(hotkey_id, hotkey_str, callback)
    
    def get_registered_hotkeys(self) -> Dict[str, str]:
        """
        Get all registered hotkeys.
        
        Returns:
            Dictionary mapping hotkey_id -> hotkey_str
        """
        with self._lock:
            return {hotkey_id: data['hotkey_str'] for hotkey_id, data in self._hotkeys.items()}
    
    def get_hotkey(self, hotkey_id: str) -> Optional[str]:
        """
        Get hotkey string for a specific ID.
        
        Args:
            hotkey_id: Unique identifier for the hotkey
            
        Returns:
            Hotkey string if found, None otherwise
        """
        with self._lock:
            if hotkey_id in self._hotkeys:
                return self._hotkeys[hotkey_id]['hotkey_str']
            return None
    
    def _restart_listener(self):
        """Restart the keyboard listener with all registered hotkeys."""
        if self._listener:
            try:
                self._listener.stop()
            except Exception:
                pass
        
        # Build hotkey dict for GlobalHotKeys
        hotkey_dict = {}
        for hotkey_id, hotkey_data in self._hotkeys.items():
            hotkey_dict[hotkey_data['parsed']] = hotkey_data['callback']
        
        if hotkey_dict:
            # Start new listener with all hotkeys
            self._listener = GlobalHotKeys(hotkey_dict)
            self._listener.start()
        else:
            self._listener = None
    
    def start(self, event_loop: Optional[asyncio.AbstractEventLoop] = None):
        """
        Start the hotkey listener.
        
        Args:
            event_loop: Optional asyncio event loop for async callbacks
        """
        with self._lock:
            if self._running:
                logger.warning("Hotkey manager already running")
                return
            
            self._event_loop = event_loop or asyncio.get_event_loop()
            self._running = True
            
            # Start listener with all registered hotkeys
            self._restart_listener()
            
            logger.info("Hotkey manager started")
    
    def stop(self):
        """Stop the hotkey listener and cleanup."""
        with self._lock:
            if not self._running:
                return
            
            self._running = False
            
            if self._listener:
                try:
                    self._listener.stop()
                except Exception as e:
                    logger.warning(f"Error stopping hotkey listener: {e}")
                self._listener = None
            
            logger.info("Hotkey manager stopped")

