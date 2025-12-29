#!/usr/bin/env python3
"""
Test script for GPT-SoVITS TTS generation

This script demonstrates how to use GPT-SoVITS for text-to-speech synthesis.
It can work with either a local API server or by making direct HTTP requests.
"""

import json
import os
import subprocess
import sys
import time
import requests
import tempfile
import wave
from pathlib import Path

# Configuration
GPT_SOVITS_DIR = Path("tts/GPT-SoVITS").resolve()
API_HOST = "127.0.0.1"
API_PORT = 9880
API_URL = f"http://{API_HOST}:{API_PORT}"
CONFIG_PATH = GPT_SOVITS_DIR / "GPT_SoVITS/configs/tts_infer.yaml"

def check_api_server_running():
    """Check if GPT-SoVITS API server is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def start_api_server():
    """Start the GPT-SoVITS API server using api_v2.py"""
    print(f"Starting GPT-SoVITS API server at {API_URL}...")
    
    # Change to GPT-SoVITS directory
    os.chdir(GPT_SOVITS_DIR)
    
    # Start the API server in background
    cmd = [
        sys.executable, "api_v2.py",
        "-a", API_HOST,
        "-p", str(API_PORT),
        "-c", str(CONFIG_PATH)
    ]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to start
    print("Waiting for server to start...")
    for i in range(30):  # Wait up to 30 seconds
        if check_api_server_running():
            print("✓ API server is running!")
            return process
        time.sleep(1)
        print(f"Waiting... ({i+1}/30)")
    
    print("✗ Failed to start API server")
    process.terminate()
    return None

def generate_tts(text, output_file="output.wav", text_lang="zh", ref_audio_path=None, prompt_text=None, prompt_lang="zh"):
    """
    Generate TTS using GPT-SoVITS API
    
    Args:
        text: Text to synthesize
        output_file: Output audio file path
        text_lang: Language of the text (zh/en/ja/ko)
        ref_audio_path: Reference audio file path (optional)
        prompt_text: Text corresponding to reference audio (optional)
        prompt_lang: Language of the prompt text
    """
    
    # Prepare request data
    data = {
        "text": text,
        "text_lang": text_lang,
        "prompt_lang": prompt_lang,
        "batch_size": 1,
        "streaming_mode": False,
        "media_type": "wav",
        "top_k": 15,
        "top_p": 1.0,
        "temperature": 1.0,
        "speed_factor": 1.0,
        "seed": -1
    }
    
    # Add reference audio if provided
    if ref_audio_path:
        data["ref_audio_path"] = ref_audio_path
    if prompt_text:
        data["prompt_text"] = prompt_text
    
    print(f"Generating TTS for: '{text}'")
    print(f"Language: {text_lang}")
    if ref_audio_path:
        print(f"Reference audio: {ref_audio_path}")
    
    try:
        # Make API request
        response = requests.post(
            f"{API_URL}/tts",
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            # Save audio file
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"✓ Audio saved to: {output_file}")
            
            # Print audio file info
            try:
                with wave.open(output_file, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    duration = frames / float(rate)
                    print(f"  Duration: {duration:.2f} seconds")
                    print(f"  Sample rate: {rate} Hz")
                    print(f"  Channels: {wav_file.getnchannels()}")
            except Exception as e:
                print(f"  Could not read audio info: {e}")
            
        else:
            print(f"✗ API request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Request failed: {e}")

def main():
    """Main function to run TTS test"""
    print("GPT-SoVITS TTS Test Script")
    print("=" * 40)
    
    # Check if GPT-SoVITS directory exists
    if not GPT_SOVITS_DIR.exists():
        print(f"✗ GPT-SoVITS directory not found at: {GPT_SOVITS_DIR}")
        print("Please ensure GPT-SoVITS is installed in the correct location.")
        return
    
    print(f"✓ GPT-SoVITS found at: {GPT_SOVITS_DIR}")
    
    # Check if API server is already running
    if not check_api_server_running():
        print("API server not running, attempting to start...")
        process = start_api_server()
        if not process:
            print("✗ Could not start API server")
            return
    else:
        print("✓ API server is already running")
        process = None
    
    try:
        # Test cases
        test_cases = [
            {
                "text": "Hello, this is a test of GPT-SoVITS text to speech synthesis.",
                "output_file": "test_english.wav",
                "text_lang": "en"
            },
            {
                "text": "你好，这是GPT-SoVITS语音合成的测试。",
                "output_file": "test_chinese.wav", 
                "text_lang": "zh"
            },
            {
                "text": "こんにちは、これはGPT-SoVITSのテストです。",
                "output_file": "test_japanese.wav",
                "text_lang": "ja"
            }
        ]
        
        print("\nRunning test cases...")
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i} ---")
            generate_tts(**test_case)
            
        print("\n✓ All test cases completed!")
        print("\nGenerated files:")
        for test_case in test_cases:
            output_file = test_case["output_file"]
            if os.path.exists(output_file):
                size = os.path.getsize(output_file)
                print(f"  - {output_file} ({size:,} bytes)")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    finally:
        # Cleanup: stop the API server if we started it
        if process:
            print("\nStopping API server...")
            process.terminate()
            process.wait()

if __name__ == "__main__":
    main()