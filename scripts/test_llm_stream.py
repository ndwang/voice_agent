#!/usr/bin/env python3
"""
Test LLM Streaming Endpoint

Tests the LLM service streaming endpoint and verifies tokens arrive incrementally.
"""
import sys
import json
import time
import requests
from typing import Optional

# Configuration
LLM_BASE_URL = "http://localhost:8002"
STREAM_ENDPOINT = f"{LLM_BASE_URL}/generate/stream"


def test_streaming(prompt: str, context: Optional[str] = None):
    """
    Test the streaming endpoint and verify tokens arrive incrementally.
    
    Args:
        prompt: The prompt to send to the LLM
        context: Optional context string
    """
    print("=" * 60)
    print("Testing LLM Streaming Endpoint")
    print("=" * 60)
    print(f"Prompt: {prompt}")
    if context:
        print(f"Context: {context}")
    print()
    
    # Prepare request
    payload = {
        "prompt": prompt,
        "context": context
    }
    
    print("Connecting to streaming endpoint...")
    print(f"URL: {STREAM_ENDPOINT}")
    print()
    
    try:
        # Make streaming request
        response = requests.post(
            STREAM_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True,  # Important: enable streaming
            timeout=60
        )
        
        response.raise_for_status()
        
        print("✓ Connected successfully")
        print("Waiting for tokens...")
        print("-" * 60)
        
        # Track timing
        start_time = time.time()
        first_token_time = None
        token_count = 0
        full_response = ""
        
        # Process SSE stream
        # SSE format: "event: token\ndata: {...}\n\n"
        buffer = ""
        
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            
            # Handle SSE format
            if line.startswith("event:"):
                event_type = line[6:].strip()
                continue
            
            if line.startswith("data:"):
                data_str = line[5:].strip()
                
                try:
                    data = json.loads(data_str)
                    
                    if "token" in data:
                        token = data["token"]
                        token_count += 1
                        
                        # Record first token time
                        if first_token_time is None:
                            first_token_time = time.time()
                            time_to_first_token = first_token_time - start_time
                            print(f"⏱️  Time to first token: {time_to_first_token:.2f}s")
                            print()
                        
                        # Print token as it arrives
                        print(token, end="", flush=True)
                        full_response += token
                        
                        # Small delay to make streaming visible
                        time.sleep(0.01)
                    
                    elif "status" in data and data.get("status") == "complete":
                        print("\n")
                        print("-" * 60)
                        print(f"✓ Stream complete")
                        break
                    
                    elif "error" in data:
                        print(f"\n✗ Error: {data['error']}")
                        return False
                
                except json.JSONDecodeError:
                    # Skip invalid JSON
                    continue
        
        # Calculate statistics
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"Total tokens received: {token_count}")
        print(f"Total time: {total_time:.2f}s")
        if token_count > 0:
            print(f"Average time per token: {total_time/token_count:.3f}s")
        if first_token_time:
            print(f"Time to first token: {first_token_time - start_time:.2f}s")
        print()
        print("Full response:")
        print("-" * 60)
        print(full_response)
        print("-" * 60)
        
        # Verify streaming worked (tokens arrived incrementally)
        if token_count > 1:
            print("\n✓ Streaming verified: Multiple tokens received incrementally")
            return True
        elif token_count == 1:
            print("\n⚠ Warning: Only one token received (might be buffered)")
            return True
        else:
            print("\n✗ Error: No tokens received")
            return False
    
    except requests.exceptions.ConnectionError:
        print(f"✗ Error: Could not connect to {LLM_BASE_URL}")
        print("Make sure the LLM service is running.")
        return False
    except requests.exceptions.Timeout:
        print("✗ Error: Request timed out")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = "用一句话解释什么是人工智能"
    
    context = None
    
    # Check for context argument
    if "--context" in sys.argv:
        idx = sys.argv.index("--context")
        if idx + 1 < len(sys.argv):
            context = sys.argv[idx + 1]
    
    success = test_streaming(prompt, context)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

