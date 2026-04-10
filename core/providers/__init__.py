from __future__ import annotations

from core.providers.anthropic import AnthropicTranslationProvider
from core.providers.base import TranslationProvider
from core.providers.gemini import GeminiTranslationProvider
from core.providers.openai import OpenAITranslationProvider

ANTHROPIC_PROVIDER = AnthropicTranslationProvider()
GEMINI_PROVIDER = GeminiTranslationProvider()
OPENAI_PROVIDER = OpenAITranslationProvider()
DEFAULT_PROVIDER = GEMINI_PROVIDER
DEFAULT_PROVIDER_NAME = DEFAULT_PROVIDER.name
SUPPORTED_TRANSLATION_PROVIDERS = {
    ANTHROPIC_PROVIDER.name: ANTHROPIC_PROVIDER,
    GEMINI_PROVIDER.name: GEMINI_PROVIDER,
    OPENAI_PROVIDER.name: OPENAI_PROVIDER,
}


def get_translation_provider(name: str | None = None) -> TranslationProvider:
    provider_name = (name or DEFAULT_PROVIDER_NAME).strip().lower()
    provider = SUPPORTED_TRANSLATION_PROVIDERS.get(provider_name)
    if provider is None:
        supported = ", ".join(sorted(SUPPORTED_TRANSLATION_PROVIDERS))
        raise ValueError(f"Unsupported provider: {provider_name!r}. Supported providers: {supported}")
    return provider


__all__ = [
    "DEFAULT_PROVIDER",
    "DEFAULT_PROVIDER_NAME",
    "ANTHROPIC_PROVIDER",
    "GEMINI_PROVIDER",
    "OPENAI_PROVIDER",
    "SUPPORTED_TRANSLATION_PROVIDERS",
    "TranslationProvider",
    "get_translation_provider",
]
