"""
Debug script for hotkey manager.

Prints all key press/release events and when hotkeys are matched.
"""
import sys
import time
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from orchestrator.hotkey_manager import HotkeyManager
from core.logging import setup_logging

# Setup logging
setup_logging()
import logging
logger = logging.getLogger(__name__)

# Create a custom hotkey manager with debug output
class DebugHotkeyManager(HotkeyManager):
    """Hotkey manager with debug output."""
    
    def _on_press(self, key):
        """Handle key press events with debug output."""
        normalized = self._key_to_normalized_name(key)
        key_repr = f"{key}" if hasattr(key, 'char') and key.char else str(key)
        
        print(f"[KEY PRESS] {key_repr} -> normalized: {normalized}")
        
        # Call parent implementation
        super()._on_press(key)
        
        # Print current state after processing
        with self._lock:
            pressed_normalized = set()
            for pressed_key in self._pressed_keys:
                norm = self._key_to_normalized_name(pressed_key)
                if norm:
                    pressed_normalized.add(norm)
            print(f"  -> Currently pressed keys: {sorted(pressed_normalized)}")
            print(f"  -> Triggered hotkeys: {sorted(self._triggered_hotkeys)}")
    
    def _on_release(self, key):
        """Handle key release events with debug output."""
        normalized = self._key_to_normalized_name(key)
        key_repr = f"{key}" if hasattr(key, 'char') and key.char else str(key)
        
        print(f"[KEY RELEASE] {key_repr} -> normalized: {normalized}")
        
        # Call parent implementation
        super()._on_release(key)
        
        # Print current state after processing
        with self._lock:
            pressed_normalized = set()
            for pressed_key in self._pressed_keys:
                norm = self._key_to_normalized_name(pressed_key)
                if norm:
                    pressed_normalized.add(norm)
            print(f"  -> Currently pressed keys: {sorted(pressed_normalized)}")
            print(f"  -> Triggered hotkeys: {sorted(self._triggered_hotkeys)}")
    
    def register_hotkey(self, hotkey_id, hotkey_str, callback):
        """Register hotkey with debug output."""
        print(f"[REGISTER] Hotkey '{hotkey_id}': {hotkey_str}")
        result = super().register_hotkey(hotkey_id, hotkey_str, callback)
        if result:
            with self._lock:
                key_set = self._hotkeys[hotkey_id]['key_set']
                print(f"  -> Key set: {sorted(key_set)}")
        return result


def test_callback(hotkey_id):
    """Test callback that prints when hotkey is triggered."""
    print(f"\n{'='*60}")
    print(f"[HOTKEY TRIGGERED] {hotkey_id}")
    print(f"{'='*60}\n")


async def async_test_callback(hotkey_id):
    """Async test callback that prints when hotkey is triggered."""
    print(f"\n{'='*60}")
    print(f"[HOTKEY TRIGGERED (ASYNC)] {hotkey_id}")
    print(f"{'='*60}\n")


def main():
    """Main debug function."""
    print("="*60)
    print("Hotkey Manager Debug Script")
    print("="*60)
    print("\nPress keys to see debug output.")
    print("Registered hotkeys:")
    print("  - ctrl+shift+l: Toggle listening")
    print("  - ctrl+shift+c: Cancel speech")
    print("  - ctrl+alt+t: Test hotkey")
    print("\nPress Ctrl+C to exit.\n")
    print("="*60)
    print()
    
    # Create debug hotkey manager
    manager = DebugHotkeyManager()
    
    # Get event loop
    loop = asyncio.get_event_loop()
    
    # Register test hotkeys
    manager.register_hotkey(
        "toggle_listening",
        "ctrl+shift+l",
        lambda: test_callback("toggle_listening")
    )
    
    manager.register_hotkey(
        "cancel_speech",
        "ctrl+shift+c",
        lambda: test_callback("cancel_speech")
    )
    
    manager.register_hotkey(
        "test_hotkey",
        "ctrl+alt+t",
        lambda: test_callback("test_hotkey")
    )
    
    # Register an async hotkey
    async def async_callback():
        await async_test_callback("async_test")
    
    manager.register_hotkey(
        "async_test",
        "ctrl+alt+a",
        async_callback
    )
    
    # Start the manager
    print("\n[STARTING] Hotkey manager...")
    manager.start(loop)
    print("[STARTED] Hotkey manager is running. Press keys to test.\n")
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n[STOPPING] Hotkey manager...")
        manager.stop()
        print("[STOPPED] Exiting.")


if __name__ == "__main__":
    main()

