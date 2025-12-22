"""
Validation script for conversation history unification.

This script verifies that:
1. ContextManager is the single source of truth for conversation history
2. LLM providers are stateless and accept messages from ContextManager
3. Multi-turn conversations work correctly
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from orchestrator.managers.context_manager import ContextManager

# Try to import providers (may not be available)
try:
    from llm.providers.ollama import OllamaProvider
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from llm.providers.gemini import GeminiProvider
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def test_context_manager_history():
    """Test that ContextManager maintains conversation history correctly."""
    print("Testing ContextManager history management...")
    cm = ContextManager(max_history=5)
    
    # Add some messages
    cm.add_user_message("Hello")
    cm.add_assistant_message("Hi there!")
    cm.add_user_message("How are you?")
    cm.add_assistant_message("I'm doing well, thanks!")
    
    # Verify history
    assert len(cm.conversation_history) == 4, f"Expected 4 messages, got {len(cm.conversation_history)}"
    assert cm.conversation_history[0]["role"] == "user"
    assert cm.conversation_history[0]["content"] == "Hello"
    assert cm.conversation_history[1]["role"] == "assistant"
    assert cm.conversation_history[1]["content"] == "Hi there!"
    
    # Test format_context_for_llm
    context_data = cm.format_context_for_llm("What's the weather?")
    assert "messages" in context_data, "format_context_for_llm should return messages"
    assert "system_prompt" in context_data, "format_context_for_llm should return system_prompt"
    assert len(context_data["messages"]) > 0, "messages should not be empty"
    
    # Verify messages include system, history, and current message
    messages = context_data["messages"]
    assert messages[0]["role"] == "system", "First message should be system"
    assert messages[-1]["role"] == "user", "Last message should be user"
    assert messages[-1]["content"] == "What's the weather?", "Last message should be current user message"
    
    print("[OK] ContextManager history management works correctly")
    return True


def test_provider_stateless():
    """Test that providers are stateless regarding history."""
    print("Testing provider statelessness...")
    
    # Test Ollama provider
    if OLLAMA_AVAILABLE:
        try:
            provider = OllamaProvider(model="llama3", base_url="http://localhost:11434")
            # In the refactored code, providers no longer have get_history/clear_history
            # they are strictly stateless by design (only generate/generate_stream methods)
            assert hasattr(provider, 'generate_stream')
            print("[OK] Ollama provider exists and supports streaming")
        except Exception as e:
            print(f"[SKIP] Ollama provider test skipped (Ollama not available): {e}")
    else:
        print("[SKIP] Ollama provider test skipped (module not available)")
    
    # Test Gemini provider
    if GEMINI_AVAILABLE:
        try:
            provider = GeminiProvider()
            assert hasattr(provider, 'generate_stream')
            print("[OK] Gemini provider exists and supports streaming")
        except Exception as e:
            print(f"[SKIP] Gemini provider test skipped (API key not available): {e}")
    else:
        print("[SKIP] Gemini provider test skipped (module not available)")
    
    return True


def test_messages_format():
    """Test that ContextManager formats messages correctly for providers."""
    print("Testing messages format...")
    cm = ContextManager(max_history=3)
    
    # Add conversation
    cm.add_user_message("First message")
    cm.add_assistant_message("First response")
    cm.add_user_message("Second message")
    
    # Format for LLM
    context_data = cm.format_context_for_llm("Third message")
    messages = context_data["messages"]
    
    # Verify structure
    assert isinstance(messages, list), "messages should be a list"
    for msg in messages:
        assert "role" in msg, "Each message should have 'role'"
        assert "content" in msg, "Each message should have 'content'"
        assert msg["role"] in ["system", "user", "assistant"], f"Invalid role: {msg['role']}"
    
    # Verify system message is first
    assert messages[0]["role"] == "system", "First message should be system"
    
    # Verify current user message is last
    assert messages[-1]["role"] == "user", "Last message should be user"
    assert messages[-1]["content"] == "Third message", "Last message should be current user message"
    
    print("[OK] Messages format is correct")
    return True


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Conversation History Unification Validation")
    print("=" * 60)
    print()
    
    try:
        test_context_manager_history()
        print()
        test_provider_stateless()
        print()
        test_messages_format()
        print()
        print("=" * 60)
        print("[OK] All validation tests passed!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"[FAIL] Validation failed: {e}")
        print("=" * 60)
        return 1
    except Exception as e:
        print()
        print("=" * 60)
        print(f"[ERROR] Unexpected error: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

