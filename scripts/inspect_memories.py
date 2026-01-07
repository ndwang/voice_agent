"""
Memory Inspection CLI

Command-line tool to inspect and manage stored memories.

Usage:
    uv run python scripts/inspect_memories.py list
    uv run python scripts/inspect_memories.py search "query text"
    uv run python scripts/inspect_memories.py tags
    uv run python scripts/inspect_memories.py delete <memory_id>
    uv run python scripts/inspect_memories.py stats
"""
import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from orchestrator.memory.storage import get_memory_storage


def format_timestamp(timestamp: float) -> str:
    """Format Unix timestamp as readable string."""
    try:
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return "Unknown"


def print_memory(mem: dict, include_id: bool = True):
    """Print a single memory with formatting."""
    print("\n" + "=" * 80)
    if include_id:
        print(f"ID: {mem['id']}")
    print(f"Time: {format_timestamp(mem['timestamp'])}")
    print(f"Tags: {', '.join(mem['tags']) if mem['tags'] else '(none)'}")
    print(f"Source: {mem['source']}")
    if 'distance' in mem:
        print(f"Relevance: {1 - mem['distance']:.2%}")
    print("-" * 80)
    print(mem['content'])
    print("=" * 80)


async def cmd_list():
    """List all stored memories."""
    storage = get_memory_storage()
    memories = await storage.list_all_memories()

    if not memories:
        print("No memories stored.")
        return

    print(f"\nFound {len(memories)} total memories:\n")
    for i, mem in enumerate(memories, 1):
        print(f"\n[{i}] ID: {mem['id'][:8]}...")
        print(f"    Time: {format_timestamp(mem['timestamp'])}")
        print(f"    Tags: {', '.join(mem['tags']) if mem['tags'] else '(none)'}")
        print(f"    Content: {mem['content'][:100]}{'...' if len(mem['content']) > 100 else ''}")


async def cmd_search(query: str):
    """Search memories by query."""
    if not query:
        print("Error: Search query required")
        print("Usage: uv run python scripts/inspect_memories.py search \"your query\"")
        return

    storage = get_memory_storage()
    memories = await storage.search_memories(query=query, limit=10)

    if not memories:
        print(f"No memories found matching: {query}")
        return

    print(f"\nSearch results for: '{query}'")
    print(f"Found {len(memories)} relevant memories:\n")

    for i, mem in enumerate(memories, 1):
        print(f"\n[{i}] Relevance: {(1 - mem['distance']):.2%}")
        print_memory(mem, include_id=True)


async def cmd_tags():
    """List all tags with counts."""
    storage = get_memory_storage()
    tag_counts = await storage.list_all_tags()

    if not tag_counts:
        print("No tags found.")
        return

    print("\nAll tags:\n")
    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)

    max_tag_len = max(len(tag) for tag, _ in sorted_tags)
    for tag, count in sorted_tags:
        print(f"  {tag:<{max_tag_len}}  {count:>4} memor{'ies' if count > 1 else 'y'}")

    print(f"\nTotal: {len(sorted_tags)} unique tags")


async def cmd_delete(memory_id: str):
    """Delete a memory by ID."""
    if not memory_id:
        print("Error: Memory ID required")
        print("Usage: uv run python scripts/inspect_memories.py delete <memory_id>")
        return

    # Confirm deletion
    print(f"\nWARNING: About to delete memory: {memory_id}")
    response = input("Are you sure? (yes/no): ").strip().lower()

    if response != "yes":
        print("Cancelled.")
        return

    storage = get_memory_storage()
    success = await storage.delete_memory(memory_id)

    if success:
        print(f"Memory {memory_id} deleted successfully.")
    else:
        print(f"Failed to delete memory {memory_id}. It may not exist.")


async def cmd_stats():
    """Show memory storage statistics."""
    storage = get_memory_storage()
    memories = await storage.list_all_memories()
    tag_counts = await storage.list_all_tags()

    print("\nMemory Storage Statistics:")
    print("=" * 50)
    print(f"Total memories: {len(memories)}")
    print(f"Total tags: {len(tag_counts)}")

    if memories:
        oldest = min(mem['timestamp'] for mem in memories)
        newest = max(mem['timestamp'] for mem in memories)
        print(f"Oldest memory: {format_timestamp(oldest)}")
        print(f"Newest memory: {format_timestamp(newest)}")

        # Content statistics
        total_chars = sum(len(mem['content']) for mem in memories)
        avg_chars = total_chars / len(memories)
        print(f"Average memory length: {avg_chars:.0f} characters")

        # Tag statistics
        tagged_count = sum(1 for mem in memories if mem['tags'])
        print(f"Tagged memories: {tagged_count} ({tagged_count/len(memories)*100:.1f}%)")

    # Storage size
    try:
        storage_size = sum(f.stat().st_size for f in storage.storage_path.rglob('*') if f.is_file())
        print(f"Storage size: {storage_size / (1024*1024):.2f} MB")
    except:
        pass

    print("=" * 50)


def print_usage():
    """Print usage instructions."""
    print(__doc__)


async def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print_usage()
        return

    command = sys.argv[1].lower()

    try:
        if command == "list":
            await cmd_list()
        elif command == "search":
            query = sys.argv[2] if len(sys.argv) > 2 else ""
            await cmd_search(query)
        elif command == "tags":
            await cmd_tags()
        elif command == "delete":
            memory_id = sys.argv[2] if len(sys.argv) > 2 else ""
            await cmd_delete(memory_id)
        elif command == "stats":
            await cmd_stats()
        else:
            print(f"Unknown command: {command}\n")
            print_usage()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
