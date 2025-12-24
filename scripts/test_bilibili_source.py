import asyncio
import time
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.event_bus import EventBus
from orchestrator.sources.bilibili_source import BilibiliSource

async def test_bilibili_source_logic():
    print("=" * 60)
    print("BilibiliSource Internal Logic Test")
    print("=" * 60)
    
    event_bus = EventBus()
    source = BilibiliSource(event_bus)
    
    # Configure a temporary short TTL for testing
    source.ttl = 2 
    
    # Mock message objects (simulating blivedm models)
    # Danmaku message mock
    mock_dm = MagicMock()
    mock_dm.uid = 12345
    mock_dm.uname = "DanmakuUser"
    mock_dm.msg = "Hello Bilibili!"
    
    # SuperChat message mock
    mock_sc = MagicMock()
    mock_sc.uid = 67890
    mock_sc.uname = "SuperUser"
    mock_sc.message = "Support you!"
    mock_sc.price = 100
    
    print("\n1. Testing Danmaku Handling...")
    await source._on_danmaku(mock_dm)
    print(f"[OK] Danmaku added. Count: {source.get_danmaku_count()}")
    snapshot = source.get_danmaku_snapshot()
    if snapshot:
        print(f"[OK] Snapshot content: {snapshot[0]['user']}: {snapshot[0]['content']}")
    else:
        print("[FAIL] Snapshot is empty")
    
    print("\n2. Testing SuperChat Handling...")
    await source._on_super_chat(mock_sc)
    print(f"[OK] SuperChat added. Queue count: {source.get_superchat_count()}")
    sc = await source.get_superchat()
    print(f"[OK] Retrieved SC: {sc['user']} sent {sc['amount']} - {sc['content']}")
    print(f"[OK] Queue count after retrieval: {source.get_superchat_count()}")
    
    print(f"\n3. Testing TTL Expiry (TTL={source.ttl}s)...")
    await source._on_danmaku(mock_dm)
    print(f"Count immediately: {source.get_danmaku_count()}")
    print("Waiting 2.1s...")
    await asyncio.sleep(2.1)
    print(f"Count after wait: {source.get_danmaku_count()}")
    if source.get_danmaku_count() == 0:
        print("[OK] TTL Expiry works correctly")
    else:
        print("[FAIL] TTL Expiry FAILED (Make sure _cleanup_loop isn't interfering or wait longer)")

async def test_real_connection(room_id: int):
    print("\n" + "=" * 60)
    print(f"BilibiliSource Real Connection Test (Room: {room_id})")
    print("=" * 60)
    
    event_bus = EventBus()
    source = BilibiliSource(event_bus)
    
    # Manually override settings for test
    source.room_id = room_id
    source.client.room_id = room_id
    source.ttl = 30 # 30s TTL for real test
    
    print(f"Starting source for room {room_id}...")
    await source.start()
    
    print("Listening for 30 seconds. Send some danmaku in the room!")
    print("Press Ctrl+C to stop early.")
    
    try:
        start_time = time.time()
        seen_dm_ids = set()
        
        while time.time() - start_time < 30:
            await asyncio.sleep(0.5)
            
            # Check Danmaku
            snapshot = source.get_danmaku_snapshot()
            for dm in snapshot:
                if dm['id'] not in seen_dm_ids:
                    try:
                        print(f"[DM] {dm['user']}: {dm['content']}")
                    except UnicodeEncodeError:
                        # Fallback for terminals that don't support Chinese characters
                        print(f"[DM] {dm['user'].encode('ascii', 'replace').decode()}: {dm['content'].encode('ascii', 'replace').decode()}")
                    seen_dm_ids.add(dm['id'])
            
            # Check SuperChat (non-blocking)
            while source.get_superchat_count() > 0:
                sc = await source.get_superchat()
                try:
                    print(f"[SC] {sc['user']} (¥{sc['amount']}): {sc['content']}")
                except UnicodeEncodeError:
                    print(f"[SC] {sc['user'].encode('ascii', 'replace').decode()} (Y{sc['amount']}): {sc['content'].encode('ascii', 'replace').decode()}")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nError during real test: {e}")
    finally:
        await source.stop()
        print("\nSource stopped.")

async def main():
    print("Bilibili Source Test Script")
    
    # 1. Logic Test
    await test_bilibili_source_logic()
    
    # 2. Real Connection Test (if room ID provided as argument)
    room_id = None
    if len(sys.argv) > 1:
        try:
            room_id = int(sys.argv[1])
        except ValueError:
            pass
            
    if room_id:
        await test_real_connection(room_id)
    else:
        print("\n(Tip: Provide a room ID as an argument to run a real connection test, e.g., 'python scripts/test_bilibili_source.py 31232063')")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

