from __future__ import annotations

from core.providers.base import ProviderSpec, TranslationProvider
from core.providers.gemini import GeminiTranslationProvider

GEMINI_PROVIDER = GeminiTranslationProvider()
SUPPORTED_TRANSLATION_PROVIDERS = {
    GEMINI_PROVIDER.name: GEMINI_PROVIDER,
}


def get_translation_provider(name: str | None = None) -> TranslationProvider:
    provider_name = (name or GEMINI_PROVIDER.name).strip().lower()
    provider = SUPPORTED_TRANSLATION_PROVIDERS.get(provider_name)
    if provider is None:
        supported = ", ".join(sorted(SUPPORTED_TRANSLATION_PROVIDERS))
        raise ValueError(f"Unsupported provider: {provider_name!r}. Supported providers: {supported}")
    return provider


__all__ = [
    "GEMINI_PROVIDER",
    "ProviderSpec",
    "SUPPORTED_TRANSLATION_PROVIDERS",
    "TranslationProvider",
    "get_translation_provider",
]
