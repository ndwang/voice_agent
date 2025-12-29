#!/usr/bin/env python3
"""
Simple GPT-SoVITS test script using the library directly
"""

import os
import sys
from pathlib import Path

# Add GPT-SoVITS to Python path
GPT_SOVITS_DIR = Path("tts/GPT-SoVITS").resolve()
sys.path.insert(0, str(GPT_SOVITS_DIR))
sys.path.insert(0, str(GPT_SOVITS_DIR / "GPT_SoVITS"))

def main():
    print("GPT-SoVITS Simple Test")
    print("=" * 30)
    
    # Change to GPT-SoVITS directory
    original_cwd = os.getcwd()
    os.chdir(GPT_SOVITS_DIR)
    
    # Import TTS class
    from TTS_infer_pack.TTS import TTS, TTS_Config
    
    # Use v2 configuration (most common)
    config = TTS_Config("v2")
    config.device = "cpu"  # Use CPU for compatibility
    config.is_half = False
    
    print(f"Device: {config.device}")
    print("Loading models...")
    
    # Initialize TTS
    tts = TTS(config)
    
    # Test input
    inputs = {
        "text": "你好，这是一个测试。",
        "text_lang": "zh",
        "ref_audio_path": "",  # Empty for default
        "prompt_text": "",
        "prompt_lang": "zh"
    }
    
    print(f"Generating TTS for: {inputs['text']}")
    
    # Generate TTS
    sr, audio_data = tts.run(inputs)
    
    # Save output
    output_file = os.path.join(original_cwd, "test_output.wav")
    import soundfile as sf
    sf.write(output_file, audio_data, sr)
    
    print(f"✓ Audio saved: {output_file}")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {len(audio_data) / sr:.2f}s")
    
    # Restore directory
    os.chdir(original_cwd)

if __name__ == "__main__":
    main()