"""
LLM Factory Utility

Encapsulates LLM provider instantiation logic.
"""
from core.settings import LLMSettings
from llm.providers import GeminiProvider, OllamaProvider


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
        return GeminiProvider(
            model=provider_config.model,
            api_key=provider_config.api_key,
            generation_config=provider_config.generation_config
        )
    else:
        return OllamaProvider(
            model=provider_config.model,
            base_url=provider_config.base_url,
            timeout=provider_config.timeout,
            disable_thinking=provider_config.disable_thinking,
            generation_config=provider_config.generation_config
        )
