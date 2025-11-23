#!/usr/bin/env python3
"""
Test TTS Service Endpoints

Tests all TTS service endpoints (Test 1.3c, 1.3d from TESTING_PLAN.md):
- Non-streaming REST endpoint (POST /synthesize)
- Streaming WebSocket endpoint (WS /synthesize/stream)
- Voice listing (GET /voices)

Automatically converts PCM audio to WAV format for easy playback.
"""
import asyncio
import json
import sys
import websockets
import requests
from pathlib import Path

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("Warning: pydub not available. WAV conversion will be skipped.")

# Configuration
TTS_BASE_URL = "http://localhost:8003"
TTS_WS_URL = "ws://localhost:8003/synthesize/stream"
OUTPUT_FILE = "streamed_audio"
SAMPLE_RATE = 16000  # TTS output sample rate
CHANNELS = 1  # Mono
SAMPLE_WIDTH = 2  # 16-bit (2 bytes per sample)


def convert_pcm_to_wav(pcm_data: bytes, output_wav_path: str) -> bool:
    """
    Convert PCM audio data to WAV format.
    
    Args:
        pcm_data: Raw PCM audio bytes (int16, 16kHz, mono)
        output_wav_path: Output WAV file path
        
    Returns:
        True if conversion successful, False otherwise
    """
    if not PYDUB_AVAILABLE:
        return False
    
    try:
        # Load PCM data using pydub
        # Parameters: data, sample_width, channels, frame_rate
        audio = AudioSegment(
            pcm_data,
            sample_width=SAMPLE_WIDTH,
            channels=CHANNELS,
            frame_rate=SAMPLE_RATE
        )
        
        # Export as WAV
        audio.export(output_wav_path, format="wav")
        return True
    except Exception as e:
        print(f"  Warning: Failed to convert to WAV: {e}")
        return False


async def test_tts_websocket(text: str, voice: str = "zh-CN-XiaoxiaoNeural", output_file: str = None):
    """
    Test TTS WebSocket streaming endpoint.
    
    Args:
        text: Text to synthesize
        voice: Voice to use (for Edge TTS)
        output_file: Output file path for audio (default: streamed_audio.pcm)
    """
    if output_file is None:
        output_file = OUTPUT_FILE
    
    # Ensure output_file doesn't have extension (we'll add .pcm and .wav)
    output_file = str(Path(output_file).with_suffix(''))
    
    print("=" * 60)
    print("TTS WebSocket Streaming Test")
    print("=" * 60)
    print(f"WebSocket URL: {TTS_WS_URL}")
    print(f"Text: {text}")
    print(f"Voice: {voice}")
    print(f"Output file: {output_file}.wav")
    print()
    
    try:
        print("Connecting to WebSocket...")
        async with websockets.connect(TTS_WS_URL) as ws:
            print("✓ Connected successfully")
            print()
            
            # Send text chunk with finalize flag
            message = {
                "type": "text",
                "text": text,
                "finalize": True,
                "voice": voice  # Optional for Edge TTS
            }
            
            print(f"Sending message: {json.dumps(message, ensure_ascii=False)}")
            await ws.send(json.dumps(message, ensure_ascii=False))
            print("✓ Message sent")
            print()
            print("Receiving audio chunks...")
            print("-" * 60)
            
            # Receive audio chunks
            audio_chunks = []
            chunk_count = 0
            done_received = False
            
            async for message in ws:
                if isinstance(message, bytes):
                    # Audio chunk received
                    chunk_count += 1
                    chunk_size = len(message)
                    audio_chunks.append(message)
                    print(f"Chunk {chunk_count}: {chunk_size} bytes")
                else:
                    # JSON message received
                    try:
                        data = json.loads(message)
                        print(f"Message: {json.dumps(data, ensure_ascii=False)}")
                        
                        if data.get("type") == "done":
                            print("✓ Synthesis complete")
                            done_received = True
                            break
                        elif data.get("type") == "error":
                            print(f"✗ Error: {data.get('message', 'Unknown error')}")
                            return False
                    except json.JSONDecodeError:
                        print(f"Warning: Invalid JSON message: {message}")
            
            print("-" * 60)
            print()
            
            # Save complete audio
            if audio_chunks:
                total_size = sum(len(chunk) for chunk in audio_chunks)
                audio_data = b"".join(audio_chunks)
                
                print(f"✓ Received {chunk_count} audio chunks")
                print(f"  Total size: {total_size} bytes")
                print(f"  Average chunk size: {total_size // chunk_count if chunk_count > 0 else 0} bytes")
                
                # Convert and save as WAV
                wav_path = f"{output_file}.wav"
                if convert_pcm_to_wav(audio_data, wav_path):
                    print(f"✓ WAV audio saved to: {wav_path}")
                    print(f"  You can play this file directly: {wav_path}")
                    
                    # Verify WAV file was created
                    wav_file_path = Path(wav_path)
                    if wav_file_path.exists():
                        actual_size = wav_file_path.stat().st_size
                        print(f"  WAV file size: {actual_size} bytes")
                else:
                    print(f"✗ Error: Failed to convert to WAV (pydub not available or conversion failed)")
                    print(f"  Note: Install pydub for WAV conversion support")
                
                if done_received:
                    print()
                    print("✓ Test PASSED: Audio streamed successfully via WebSocket")
                    return True
                else:
                    print()
                    print("⚠ Warning: 'done' message not received, but audio chunks were saved")
                    return True
            else:
                print("✗ No audio chunks received")
                return False
                
    except websockets.exceptions.ConnectionRefused:
        print(f"✗ Error: Could not connect to {TTS_WS_URL}")
        print("Make sure the TTS service is running on port 8003.")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_multiple_chunks():
    """Test sending multiple text chunks before finalizing."""
    print("\n" + "=" * 60)
    print("Testing Multiple Text Chunks")
    print("=" * 60)
    
    try:
        async with websockets.connect(TTS_WS_URL) as ws:
            print("✓ Connected")
            
            # Send multiple text chunks
            chunks = [
                "这是第一段",
                "这是第二段",
                "这是第三段"
            ]
            
            for i, chunk in enumerate(chunks):
                message = {
                    "type": "text",
                    "text": chunk,
                    "finalize": False,
                    "voice": "zh-CN-XiaoxiaoNeural"
                }
                print(f"Sending chunk {i+1}: {chunk}")
                await ws.send(json.dumps(message, ensure_ascii=False))
                await asyncio.sleep(0.1)  # Small delay between chunks
            
            # Finalize
            finalize_message = {
                "type": "text",
                "text": "",
                "finalize": True
            }
            print("Sending finalize message...")
            await ws.send(json.dumps(finalize_message))
            
            # Receive audio
            audio_chunks = []
            async for message in ws:
                if isinstance(message, bytes):
                    audio_chunks.append(message)
                else:
                    data = json.loads(message)
                    if data.get("type") == "done":
                        break
            
            if audio_chunks:
                audio_data = b"".join(audio_chunks)
                output_base = "streamed_audio_multiple"
                
                # Convert and save as WAV
                wav_path = f"{output_base}.wav"
                if convert_pcm_to_wav(audio_data, wav_path):
                    print(f"✓ Saved WAV audio to {wav_path}")
                    print(f"  You can play this file directly: {wav_path}")
                else:
                    print(f"✗ Error: Failed to convert to WAV")
                
                return True
            else:
                print("✗ No audio received")
                return False
                
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_voice_listing():
    """Test voice listing endpoint (GET /voices)."""
    print("=" * 60)
    print("Test: Voice Listing (GET /voices)")
    print("=" * 60)
    
    try:
        url = f"{TTS_BASE_URL}/voices"
        print(f"Requesting: {url}")
        response = requests.get(url, timeout=10)
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            voices = data.get("voices", [])
            print(f"✓ Successfully retrieved {len(voices)} voices")
            print()
            
            if voices:
                print("Sample voices (first 5):")
                for i, voice in enumerate(voices[:5], 1):
                    if isinstance(voice, dict):
                        name = voice.get("name", voice.get("short_name", "Unknown"))
                        locale = voice.get("locale", "N/A")
                        gender = voice.get("gender", "N/A")
                        print(f"  {i}. {name} ({locale}, {gender})")
                    else:
                        print(f"  {i}. {voice}")
                
                if len(voices) > 5:
                    print(f"  ... and {len(voices) - 5} more")
            else:
                print("  No voices returned")
            
            print()
            return True
        else:
            print(f"✗ Error: Status {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"✗ Error: Could not connect to {TTS_BASE_URL}")
        print("Make sure the TTS service is running on port 8003.")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rest_synthesize(text: str, voice: str = "zh-CN-XiaoxiaoNeural", output_file: str = None):
    """Test non-streaming REST endpoint (POST /synthesize)."""
    if output_file is None:
        output_file = "rest_audio"
    
    output_file = str(Path(output_file).with_suffix(''))
    
    print("=" * 60)
    print("Test: Non-Streaming REST Endpoint (POST /synthesize)")
    print("=" * 60)
    print(f"URL: {TTS_BASE_URL}/synthesize")
    print(f"Text: {text}")
    print(f"Voice: {voice}")
    print(f"Output file: {output_file}.wav")
    print()
    
    try:
        url = f"{TTS_BASE_URL}/synthesize"
        headers = {
            "Content-Type": "application/json; charset=utf-8"
        }
        data = {
            "text": text,
            "voice": voice
        }
        
        print(f"Sending POST request...")
        response = requests.post(
            url,
            headers=headers,
            json=data,
            timeout=30
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            audio_data = response.content
            print(f"✓ Audio data received: {len(audio_data)} bytes")
            
            # Convert and save as WAV
            wav_path = f"{output_file}.wav"
            if convert_pcm_to_wav(audio_data, wav_path):
                print(f"✓ WAV audio saved to: {wav_path}")
                print(f"  You can play this file directly: {wav_path}")
                
                # Verify WAV file was created
                wav_file_path = Path(wav_path)
                if wav_file_path.exists():
                    actual_size = wav_file_path.stat().st_size
                    print(f"  WAV file size: {actual_size} bytes")
                
                print()
                print("✓ Test PASSED: REST endpoint works correctly")
                return True
            else:
                print(f"✗ Error: Failed to convert to WAV")
                return False
        else:
            print(f"✗ Error: Status {response.status_code}")
            try:
                error_data = response.json()
                print(f"  Error: {json.dumps(error_data, ensure_ascii=False, indent=2)}")
            except:
                print(f"  Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"✗ Error: Could not connect to {TTS_BASE_URL}")
        print("Make sure the TTS service is running on port 8003.")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    # Parse command line arguments
    test_voices = "--test-voices" in sys.argv or "--all" in sys.argv
    test_rest = "--test-rest" in sys.argv or "--all" in sys.argv
    test_websocket = "--test-websocket" in sys.argv or "--all" in sys.argv or not any([
        "--test-voices", "--test-rest", "--test-websocket", "--all"
    ])
    
    # Default test text
    if len(sys.argv) > 1 and not sys.argv[-1].startswith("--"):
        # Get text from arguments (skip flags)
        text_args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
        text = " ".join(text_args) if text_args else "这是语音合成测试"
    else:
        text = "这是语音合成测试"
    
    # Optional voice argument
    voice = "zh-CN-XiaoxiaoNeural"
    if "--voice" in sys.argv:
        idx = sys.argv.index("--voice")
        if idx + 1 < len(sys.argv):
            voice = sys.argv[idx + 1]
    
    # Optional output file argument
    output_file = None
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_file = sys.argv[idx + 1]
    
    # Run tests
    results = []
    
    # Test 1: Voice listing
    if test_voices:
        print("\n")
        results.append(("Voice Listing", test_voice_listing()))
    
    # Test 2: REST endpoint
    if test_rest:
        print("\n")
        rest_output = f"{output_file}_rest" if output_file else "rest_audio"
        results.append(("REST Endpoint", test_rest_synthesize(text, voice, rest_output)))
    
    # Test 3: WebSocket endpoint
    if test_websocket:
        print("\n")
        ws_output = f"{output_file}_ws" if output_file else "streamed_audio"
        results.append(("WebSocket Endpoint", asyncio.run(test_tts_websocket(text, voice, ws_output))))
        
        # Optionally test multiple chunks
        if "--test-multiple" in sys.argv:
            print("\n")
            results.append(("WebSocket Multiple Chunks", asyncio.run(test_multiple_chunks())))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    all_passed = True
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name}: {status}")
        if not result:
            all_passed = False
    print("=" * 60)
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

