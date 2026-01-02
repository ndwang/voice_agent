"""
Test script for image input support.

This script demonstrates how to send a message with images to the LLM provider.
"""
import asyncio
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.event_bus import EventBus, Event
from orchestrator.events import EventType


async def test_image_chat():
    """Send a test message with an image."""
    print("=" * 60)
    print("Testing Image Input Support")
    print("=" * 60)

    # Initialize event bus
    event_bus = EventBus()

    # Example 1: Single image
    print("\n[Test 1] Sending message with single image...")
    test_image_path = input("Enter path to test image (or press Enter to skip): ").strip()

    if test_image_path:
        await event_bus.publish(Event(
            EventType.INPUT_RECEIVED.value,
            {
                "text": "What's in this image?",
                "images": [test_image_path],
                "source": "test",
                "priority": 10
            }
        ))
        print(f"Published event with image: {test_image_path}")
    else:
        print("Skipped (no image path provided)")

    # Example 2: Multiple images
    print("\n[Test 2] Sending message with multiple images...")
    multi_image = input("Enter paths to multiple images (comma-separated) or press Enter to skip: ").strip()

    if multi_image:
        image_paths = [p.strip() for p in multi_image.split(",")]
        await event_bus.publish(Event(
            EventType.INPUT_RECEIVED.value,
            {
                "text": "Compare these images.",
                "images": image_paths,
                "source": "test",
                "priority": 10
            }
        ))
        print(f"Published event with {len(image_paths)} images")
    else:
        print("Skipped (no image paths provided)")

    # Example 3: Text-only (backward compatibility)
    print("\n[Test 3] Sending text-only message (backward compatibility)...")
    await event_bus.publish(Event(
        EventType.INPUT_RECEIVED.value,
        {
            "text": "Hello! This is a text-only message.",
            "source": "test",
            "priority": 10
        }
    ))
    print("Published text-only event (no images field)")

    print("\n" + "=" * 60)
    print("Test events published successfully!")
    print("=" * 60)
    print("\nNote: These events were published to the event bus.")
    print("To see actual LLM responses, run the full orchestrator service.")
    print("\nTo run the full service:")
    print("  uv run python scripts/start_services.py")


async def test_direct_provider():
    """Test providers directly with image input."""
    from core.settings import get_settings
    from orchestrator.utils.llm_factory import create_provider

    print("\n" + "=" * 60)
    print("Testing Providers Directly")
    print("=" * 60)

    settings = get_settings()
    provider = create_provider(settings.llm)

    print(f"\nProvider: {settings.llm.provider}")
    print(f"Model: {provider.model if hasattr(provider, 'model') else 'N/A'}")

    # Get test image path
    test_image_path = input("\nEnter path to test image: ").strip()
    if not test_image_path:
        print("No image path provided. Exiting.")
        return

    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": "Please describe this image in detail.",
            "images": [test_image_path]
        }
    ]

    print(f"\nSending request with image: {test_image_path}")
    print("Generating response...\n")

    try:
        # Generate response
        response = await provider.generate(messages)
        print("-" * 60)
        print("LLM Response:")
        print("-" * 60)
        print(response)
        print("-" * 60)
        print(f"\nTokens used: {provider.last_token_count}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main test function."""
    print("\nImage Input Support Test Script")
    print("=" * 60)
    print("\nChoose test mode:")
    print("1. Test event bus (publish events)")
    print("2. Test provider directly (get LLM response)")
    print("3. Both")

    choice = input("\nEnter choice (1/2/3): ").strip()

    if choice == "1":
        await test_image_chat()
    elif choice == "2":
        await test_direct_provider()
    elif choice == "3":
        await test_image_chat()
        await test_direct_provider()
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
