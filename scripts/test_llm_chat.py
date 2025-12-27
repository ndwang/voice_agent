#!/usr/bin/env python3
"""
Test script for chatting with LLM provider in terminal.

This script loads the LLM provider from config.yaml and provides an interactive
terminal chat interface to test the LLM responses.
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import load_config, get_config
from llm.providers import OllamaProvider, GeminiProvider


async def chat_with_llm(streaming: bool = True):
    """
    Interactive chat interface with LLM provider.
    
    Args:
        streaming: If True, use streaming responses. If False, wait for complete response.
    """
    # Load configuration
    config = load_config()
    
    # Get LLM provider configuration
    provider_name = get_config("llm", "provider", default="ollama")
    provider_config = get_config("llm", "providers", provider_name, default={})
    
    print(f"Loading LLM provider: {provider_name}")
    print(f"Provider config: {provider_config}")
    print("-" * 60)
    
    # Initialize provider
    try:
        if provider_name == "ollama":
            provider = OllamaProvider(
                model=provider_config.get("model", "Qwen3-8B-Q4-8kcontext"),
                base_url=provider_config.get("base_url", "http://localhost:11434"),
                timeout=provider_config.get("timeout", 300.0),
                disable_thinking=provider_config.get("disable_thinking", False),
                generation_config=provider_config.get("generation_config", {})
            )
        elif provider_name == "gemini":
            provider = GeminiProvider(
                model=provider_config.get("model", "gemini-2.5-flash"),
                api_key=provider_config.get("api_key") or None,
                generation_config=provider_config.get("generation_config", {})
            )
        else:
            print(f"Error: Unknown provider '{provider_name}'")
            print("Supported providers: ollama, gemini")
            return
    except Exception as e:
        print(f"Error initializing provider: {e}")
        return
    
    print(f"✓ Initialized {provider_name} provider")
    print(f"Model: {provider_config.get('model', 'N/A')}")
    print("-" * 60)
    print("Chat started! Type your messages (or 'exit'/'quit' to end)")
    print("=" * 60)
    
    # Conversation history
    messages = []
    
    # Optional: Load system prompt if available
    system_prompt = None
    system_prompt_file = get_config("orchestrator", "system_prompt_file")
    if system_prompt_file:
        system_prompt_path = project_root / system_prompt_file
        if system_prompt_path.exists():
            try:
                with open(system_prompt_path, 'r', encoding='utf-8') as f:
                    system_prompt = f.read().strip()
                print(f"Loaded system prompt from: {system_prompt_file}")
            except Exception as e:
                print(f"Warning: Could not load system prompt: {e}")
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
        print("\nSystem Prompt:")
        print("-" * 60)
        print(system_prompt)
        print("-" * 60)
    
    # Main chat loop
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            # Check for exit commands
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye!")
                break
            
            # Add user message to history
            messages.append({"role": "user", "content": user_input})
            
            # Generate response
            print("\nAssistant: ", end="", flush=True)
            
            try:
                if streaming:
                    # Streaming response
                    full_response = ""
                    async for token in provider.generate_stream(
                        messages=messages,
                        system_prompt=system_prompt
                    ):
                        print(token, end="", flush=True)
                        full_response += token
                    print()  # New line after streaming
                    
                    # Add assistant response to history
                    messages.append({"role": "assistant", "content": full_response})
                else:
                    # Non-streaming response
                    response = await provider.generate(
                        messages=messages,
                        system_prompt=system_prompt
                    )
                    print(response)
                    
                    # Add assistant response to history
                    messages.append({"role": "assistant", "content": response})
                    
            except Exception as e:
                status_code, error_msg = provider.parse_error(e)
                print(f"\nError [{status_code}]: {error_msg}")
                # Remove the user message from history on error
                messages.pop()
                
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'exit' to quit or continue chatting.")
        except EOFError:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test script for chatting with LLM provider"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming responses (wait for complete response)"
    )
    
    args = parser.parse_args()
    
    # Run async chat
    try:
        asyncio.run(chat_with_llm(streaming=not args.no_stream))
    except KeyboardInterrupt:
        print("\n\nExiting...")


if __name__ == "__main__":
    main()

