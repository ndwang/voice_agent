#!/usr/bin/env python3
"""
Test Audio Player

Tests the audio player component to verify it can receive and play audio chunks correctly.
"""
import sys
import asyncio
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from audio.audio_player import AudioPlayer


async def test_basic_playback():
    """Test 1: Basic playback functionality."""
    print("=" * 60)
    print("Test 1: Basic Playback")
    print("=" * 60)
    
    player = AudioPlayer()
    
    # Generate test audio (1 second of sine wave at 440 Hz)
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Convert to int16 bytes
    audio_int16 = (audio * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    
    print(f"Generated {len(audio_bytes)} bytes of audio")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {duration} seconds")
    print(f"Frequency: {frequency} Hz")
    print()
    print("Playing test tone (440 Hz, 1 second)...")
    
    await player.play_audio_chunk(audio_bytes)
    
    # Wait for playback to complete
    await asyncio.sleep(2)
    
    print("✓ Basic playback test complete")
    print()
    
    await player.stop()


async def test_streaming_chunks():
    """Test 2: Streaming multiple chunks."""
    print("=" * 60)
    print("Test 2: Streaming Chunks")
    print("=" * 60)
    
    player = AudioPlayer()
    
    sample_rate = 16000
    chunk_duration = 0.2  # 200ms chunks
    frequency = 440.0
    num_chunks = 5
    
    print(f"Generating {num_chunks} chunks of {chunk_duration}s each...")
    print("Playing continuous tone...")
    
    for i in range(num_chunks):
        t = np.linspace(
            i * chunk_duration,
            (i + 1) * chunk_duration,
            int(sample_rate * chunk_duration)
        )
        audio = np.sin(2 * np.pi * frequency * t)
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        await player.play_audio_chunk(audio_bytes)
        print(f"  Chunk {i+1}/{num_chunks} sent")
        await asyncio.sleep(0.05)  # Small delay between chunks
    
    # Wait for all chunks to play
    await asyncio.sleep(2)
    
    print("✓ Streaming chunks test complete")
    print()
    
    await player.stop()


async def test_stop_functionality():
    """Test 3: Stop functionality."""
    print("=" * 60)
    print("Test 3: Stop Functionality")
    print("=" * 60)
    
    player = AudioPlayer()
    
    sample_rate = 16000
    duration = 3.0  # 3 second audio
    frequency = 440.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t)
    audio_int16 = (audio * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    
    print("Starting 3-second playback...")
    await player.play_audio_chunk(audio_bytes)
    
    # Wait a bit, then stop
    await asyncio.sleep(0.5)
    print("Stopping playback...")
    await player.stop()
    
    await asyncio.sleep(0.5)
    print("✓ Stop functionality test complete")
    print()


async def test_invalid_data():
    """Test 4: Invalid data handling."""
    print("=" * 60)
    print("Test 4: Invalid Data Handling")
    print("=" * 60)
    
    player = AudioPlayer()
    
    # Test empty bytes
    print("Testing empty bytes...")
    try:
        await player.play_audio_chunk(b"")
        await asyncio.sleep(0.5)
        print("  ✓ Empty bytes handled gracefully")
    except Exception as e:
        print(f"  ✗ Error with empty bytes: {e}")
    
    # Test very small chunk
    print("Testing very small chunk...")
    try:
        small_audio = np.array([1000, -1000], dtype=np.int16)
        await player.play_audio_chunk(small_audio.tobytes())
        await asyncio.sleep(0.5)
        print("  ✓ Small chunk handled gracefully")
    except Exception as e:
        print(f"  ✗ Error with small chunk: {e}")
    
    await player.stop()
    print("✓ Invalid data test complete")
    print()


async def test_multiple_tones():
    """Test 5: Multiple different tones."""
    print("=" * 60)
    print("Test 5: Multiple Tones")
    print("=" * 60)
    
    player = AudioPlayer()
    
    sample_rate = 16000
    duration = 0.5
    frequencies = [440, 554, 659]  # A, C#, E notes
    
    print("Playing sequence of tones...")
    
    for i, freq in enumerate(frequencies):
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * freq * t)
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        await player.play_audio_chunk(audio_bytes)
        print(f"  Tone {i+1}: {freq} Hz")
        await asyncio.sleep(0.6)  # Wait for playback + small gap
    
    await asyncio.sleep(0.5)
    print("✓ Multiple tones test complete")
    print()
    
    await player.stop()


async def main():
    """Run all audio player tests."""
    print()
    print("=" * 60)
    print("Audio Player Test Suite")
    print("=" * 60)
    print()
    print("Make sure your speakers/headphones are connected and volume is up!")
    print("You should hear test tones during these tests.")
    print()
    
    try:
        # Run tests
        await test_basic_playback()
        await asyncio.sleep(0.5)
        
        await test_streaming_chunks()
        await asyncio.sleep(0.5)
        
        await test_stop_functionality()
        await asyncio.sleep(0.5)
        
        await test_invalid_data()
        await asyncio.sleep(0.5)
        
        await test_multiple_tones()
        
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

