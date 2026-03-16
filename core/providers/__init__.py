from __future__ import annotations

from core.providers.base import TranslationProvider
from core.providers.gemini import GeminiTranslationProvider

GEMINI_PROVIDER = GeminiTranslationProvider()
DEFAULT_PROVIDER = GEMINI_PROVIDER
DEFAULT_PROVIDER_NAME = DEFAULT_PROVIDER.name
SUPPORTED_TRANSLATION_PROVIDERS = {
    DEFAULT_PROVIDER_NAME: DEFAULT_PROVIDER,
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
    "GEMINI_PROVIDER",
    "SUPPORTED_TRANSLATION_PROVIDERS",
    "TranslationProvider",
    "get_translation_provider",
]
