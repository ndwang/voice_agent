"""
Hotkey Manager

Systematic hotkey registration system using pynput for system-wide hotkey detection.
Allows registering multiple hotkeys with callback functions.

Uses manual key detection instead of GlobalHotKeys to support synthetic key sequences.
"""
import asyncio
import threading
import logging
from typing import Dict, Callable, Optional, Set, Union
from pynput import keyboard
from pynput.keyboard import Key, KeyCode

logger = logging.getLogger(__name__)


class HotkeyManager:
    """Manages system-wide hotkey registration and callbacks."""
    
    def __init__(self):
        """Initialize hotkey manager."""
        self._hotkeys: Dict[str, Dict[str, any]] = {}  # hotkey_id -> {hotkey_str, callback, key_set}
        self._lock = threading.Lock()
        self._listener: Optional[keyboard.Listener] = None
        self._running = False
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._pressed_keys: Set[Union[Key, KeyCode]] = set()  # Currently pressed keys
        self._triggered_hotkeys: Set[str] = set()  # Hotkeys that have been triggered (to avoid repeats)
    
    def _normalize_key_name(self, key_name: str) -> str:
        """
        Normalize key name from hotkey string to a standard format.
        
        Args:
            key_name: Key name from hotkey string (e.g., "ctrl", "KeyL", "f1")
            
        Returns:
            Normalized key name
        """
        key_name = key_name.strip().lower()
        
        # Map of modifier names
        modifier_map = {
            'ctrl': 'ctrl',
            'control': 'ctrl',
            'shift': 'shift',
            'alt': 'alt',
            'cmd': 'cmd',
            'meta': 'cmd',
            'super': 'cmd',
        }
        
        # Map of special key names
        special_key_map = {
            'space': 'space',
            'enter': 'enter',
            'return': 'enter',
            'tab': 'tab',
            'esc': 'esc',
            'escape': 'esc',
            'backspace': 'backspace',
            'delete': 'delete',
            'up': 'up',
            'down': 'down',
            'left': 'left',
            'right': 'right',
            'home': 'home',
            'end': 'end',
            'page_up': 'page_up',
            'page_down': 'page_down',
            'insert': 'insert',
        }
        
        # Function keys
        function_keys = {
            'f1': 'f1', 'f2': 'f2', 'f3': 'f3', 'f4': 'f4',
            'f5': 'f5', 'f6': 'f6', 'f7': 'f7', 'f8': 'f8',
            'f9': 'f9', 'f10': 'f10', 'f11': 'f11', 'f12': 'f12',
        }
        
        if key_name in modifier_map:
            return modifier_map[key_name]
        elif key_name in special_key_map:
            return special_key_map[key_name]
        elif key_name in function_keys:
            return function_keys[key_name]
        elif key_name.startswith('key') and len(key_name) > 3:
            # Handle "KeyL" format - extract the letter
            return key_name[3].lower()
        elif len(key_name) == 1:
            # Single character key
            return key_name
        else:
            raise ValueError(f"Unknown key in hotkey: {key_name}")
    
    def _key_to_normalized_name(self, key: Union[Key, KeyCode]) -> Optional[str]:
        """
        Convert pynput Key or KeyCode object to normalized key name.
        
        Args:
            key: pynput Key or KeyCode object
            
        Returns:
            Normalized key name, or None if key cannot be normalized
        """
        # Handle Key enum values first
        if isinstance(key, Key):
            # Handle special keys
            if key == Key.space:
                return 'space'
            elif key == Key.enter:
                return 'enter'
            elif key == Key.tab:
                return 'tab'
            elif key == Key.esc:
                return 'esc'
            elif key == Key.backspace:
                return 'backspace'
            elif key == Key.delete:
                return 'delete'
            elif key == Key.up:
                return 'up'
            elif key == Key.down:
                return 'down'
            elif key == Key.left:
                return 'left'
            elif key == Key.right:
                return 'right'
            elif key == Key.home:
                return 'home'
            elif key == Key.end:
                return 'end'
            elif key == Key.page_up:
                return 'page_up'
            elif key == Key.page_down:
                return 'page_down'
            elif key == Key.insert:
                return 'insert'
            elif key == Key.ctrl_l or key == Key.ctrl_r:
                return 'ctrl'
            elif key == Key.shift_l or key == Key.shift_r:
                return 'shift'
            elif key == Key.alt_l or key == Key.alt_r:
                return 'alt'
            elif key == Key.cmd_l or key == Key.cmd_r:
                return 'cmd'
            elif key == Key.f1:
                return 'f1'
            elif key == Key.f2:
                return 'f2'
            elif key == Key.f3:
                return 'f3'
            elif key == Key.f4:
                return 'f4'
            elif key == Key.f5:
                return 'f5'
            elif key == Key.f6:
                return 'f6'
            elif key == Key.f7:
                return 'f7'
            elif key == Key.f8:
                return 'f8'
            elif key == Key.f9:
                return 'f9'
            elif key == Key.f10:
                return 'f10'
            elif key == Key.f11:
                return 'f11'
            elif key == Key.f12:
                return 'f12'
            else:
                return None
        else:
            # Handle KeyCode objects (character keys)
            try:
                if hasattr(key, 'char') and key.char:
                    char = key.char
                    # Handle control characters (0x01-0x1A = Ctrl+A through Ctrl+Z)
                    # Control characters are in the range 0x01-0x1F
                    if isinstance(char, str) and len(char) == 1:
                        ord_char = ord(char)
                        # Control characters: 0x01 (Ctrl+A) through 0x1A (Ctrl+Z)
                        if 0x01 <= ord_char <= 0x1A:
                            # Convert to letter: '\x01' -> 'a', '\x02' -> 'b', etc.
                            return chr(ord_char + ord('a') - 1)
                        # Regular printable characters
                        elif ord_char >= 0x20:  # Printable ASCII starts at 0x20 (space)
                            return char.lower()
            except (AttributeError, TypeError):
                pass
            return None
    
    def _parse_hotkey(self, hotkey_str: str) -> Set[str]:
        """
        Parse hotkey string into a set of normalized key names.
        
        Supports formats:
        - "ctrl+shift+l" (lowercase)
        - "Control+Shift+KeyL" (mixed case)
        - "Ctrl+Shift+L" (mixed case)
        - "f1", "F1" (function keys)
        - "space", "enter" (special keys)
        
        Returns:
            Set of normalized key names
        """
        parts = [p.strip() for p in hotkey_str.split('+')]
        key_set = set()
        
        for part in parts:
            normalized = self._normalize_key_name(part)
            key_set.add(normalized)
        
        if not key_set:
            raise ValueError(f"No keys specified in hotkey: {hotkey_str}")
        
        return key_set
    
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
                key_set = self._parse_hotkey(hotkey_str)
                
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
                    'key_set': key_set,
                    'callback': callback_wrapper,
                    'original_callback': callback,  # Store original for updates
                }
                logger.info(f"Registered hotkey {hotkey_id}: {hotkey_str} (keys: {key_set})")
                
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
            
            callback = self._hotkeys[hotkey_id]['original_callback']
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
    
    def _on_press(self, key: Union[Key, KeyCode]):
        """Handle key press events."""
        try:
            normalized = self._key_to_normalized_name(key)
            if normalized:
                with self._lock:
                    # Add to pressed keys set
                    # We need to track the actual Key object, not just the normalized name
                    # because multiple keys can map to the same normalized name (e.g., ctrl_l and ctrl_r)
                    self._pressed_keys.add(key)
                    
                    # Get current set of normalized pressed keys
                    pressed_normalized = set()
                    for pressed_key in self._pressed_keys:
                        norm = self._key_to_normalized_name(pressed_key)
                        if norm:
                            pressed_normalized.add(norm)
                    
                    # Check if any hotkey matches
                    for hotkey_id, hotkey_data in self._hotkeys.items():
                        key_set = hotkey_data['key_set']
                        # Check if all required keys are pressed
                        if key_set.issubset(pressed_normalized):
                            # Only trigger if we haven't already triggered this hotkey
                            if hotkey_id not in self._triggered_hotkeys:
                                logger.debug(f"Hotkey matched: {hotkey_id} (keys: {sorted(key_set)}, pressed: {sorted(pressed_normalized)})")
                                self._triggered_hotkeys.add(hotkey_id)
                                callback = hotkey_data['callback']
                                # Execute callback in a separate thread to avoid blocking
                                threading.Thread(target=callback, daemon=True).start()
        except Exception as e:
            logger.error(f"Error in key press handler: {e}", exc_info=True)
    
    def _on_release(self, key: Union[Key, KeyCode]):
        """Handle key release events."""
        try:
            with self._lock:
                # Remove from pressed keys
                self._pressed_keys.discard(key)
                
                # Get current set of normalized pressed keys
                pressed_normalized = set()
                for pressed_key in self._pressed_keys:
                    norm = self._key_to_normalized_name(pressed_key)
                    if norm:
                        pressed_normalized.add(norm)
                
                # Remove hotkeys from triggered set if they're no longer fully pressed
                to_remove = []
                for hotkey_id, hotkey_data in self._hotkeys.items():
                    key_set = hotkey_data['key_set']
                    if not key_set.issubset(pressed_normalized):
                        to_remove.append(hotkey_id)
                
                for hotkey_id in to_remove:
                    self._triggered_hotkeys.discard(hotkey_id)
        except Exception as e:
            logger.error(f"Error in key release handler: {e}", exc_info=True)
    
    def _restart_listener(self):
        """
        Restart the keyboard listener.
        
        Note: This method assumes the lock is already held when called from within
        locked methods. The listener.stop() call is non-blocking.
        """
        # Stop old listener first (non-blocking)
        old_listener = self._listener
        if old_listener:
            try:
                old_listener.stop()
            except Exception:
                pass
        
        # Clear pressed keys and triggered hotkeys (lock should already be held)
        self._pressed_keys.clear()
        self._triggered_hotkeys.clear()
        
        # Start new listener
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self._listener.start()
    
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
            
            # Start listener with all registered hotkeys (lock is held)
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

