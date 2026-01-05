"""
LLM Factory Utility

Encapsulates LLM provider instantiation logic.
"""
import os
from core.settings import LLMSettings
from llm.providers import GeminiProvider, OllamaProvider, OpenAIProvider


def create_provider(llm_settings: LLMSettings):
    """
    Create and return the appropriate LLM provider based on settings.

    Args:
        llm_settings: LLM configuration settings

    Returns:
        Initialized LLM provider (GeminiProvider or OllamaProvider)
    """
    provider_config = llm_settings.get_provider_config()

    if llm_settings.provider == "gemini":
        # Use environment variable GEMINI_API_KEY if not set in config
        api_key = provider_config.api_key or os.getenv("GEMINI_API_KEY")
        return GeminiProvider(
            model=provider_config.model,
            api_key=api_key,
            generation_config=provider_config.generation_config
        )
    elif llm_settings.provider == "openai":
        # Use environment variable OPENAI_API_KEY if not set in config
        api_key = provider_config.api_key or os.getenv("OPENAI_API_KEY")
        return OpenAIProvider(
            model=provider_config.model,
            api_key=api_key,
            base_url=provider_config.base_url,
            organization=provider_config.organization,
            timeout=provider_config.timeout,
            max_retries=provider_config.max_retries,
            generation_config=provider_config.generation_config
        )
    else:  # ollama
        return OllamaProvider(
            model=provider_config.model,
            base_url=provider_config.base_url,
            timeout=provider_config.timeout,
            disable_thinking=provider_config.disable_thinking,
            generation_config=provider_config.generation_config
        )
