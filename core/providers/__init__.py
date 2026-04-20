from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from typing import Any

from core.cli_errors import CliError
from core.providers.base import TranslationProvider


@dataclass(frozen=True)
class _ProviderSpec:
    name: str
    module_name: str
    class_name: str
    default_model: str
    api_key_env: str | None
    supports_structured_json: bool
    supports_structured_input: bool
    supports_thinking: bool
    supports_flex_mode: bool
    supports_seed: bool


_PROVIDER_SPECS = {
    "anthropic": _ProviderSpec(
        name="anthropic",
        module_name="core.providers.anthropic",
        class_name="AnthropicTranslationProvider",
        default_model="claude-sonnet-4-20250514",
        api_key_env="ANTHROPIC_API_KEY",
        supports_structured_json=True,
        supports_structured_input=False,
        supports_thinking=True,
        supports_flex_mode=False,
        supports_seed=False,
    ),
    "gemini": _ProviderSpec(
        name="gemini",
        module_name="core.providers.gemini",
        class_name="GeminiTranslationProvider",
        default_model="gemini-3-flash-preview",
        api_key_env="GOOGLE_API_KEY",
        supports_structured_json=True,
        supports_structured_input=True,
        supports_thinking=True,
        supports_flex_mode=True,
        supports_seed=True,
    ),
    "openai": _ProviderSpec(
        name="openai",
        module_name="core.providers.openai",
        class_name="OpenAITranslationProvider",
        default_model="gpt-5-mini",
        api_key_env="OPENAI_API_KEY",
        supports_structured_json=True,
        supports_structured_input=False,
        supports_thinking=True,
        supports_flex_mode=True,
        supports_seed=False,
    ),
}

DEFAULT_PROVIDER_NAME = "gemini"


@lru_cache(maxsize=None)
def _load_translation_provider(provider_name: str) -> TranslationProvider:
    spec = _PROVIDER_SPECS[provider_name]
    try:
        module = import_module(spec.module_name)
    except ImportError as exc:
        raise CliError(
            f"Provider '{spec.name}' requires an optional dependency that is not installed."
        ) from exc
    provider_class = getattr(module, spec.class_name)
    return provider_class()


class LazyTranslationProvider:
    def __init__(self, name: str):
        if name not in _PROVIDER_SPECS:
            raise ValueError(f"Unknown provider spec: {name!r}")
        self._name = name

    @property
    def _spec(self) -> _ProviderSpec:
        return _PROVIDER_SPECS[self._name]

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def default_model(self) -> str:
        return self._spec.default_model

    @property
    def api_key_env(self) -> str | None:
        return self._spec.api_key_env

    @property
    def supports_structured_json(self) -> bool:
        return self._spec.supports_structured_json

    @property
    def supports_structured_input(self) -> bool:
        return self._spec.supports_structured_input

    @property
    def supports_thinking(self) -> bool:
        return self._spec.supports_thinking

    @property
    def supports_flex_mode(self) -> bool:
        return self._spec.supports_flex_mode

    @property
    def supports_seed(self) -> bool:
        return self._spec.supports_seed

    def _provider(self) -> TranslationProvider:
        return _load_translation_provider(self._name)

    def __getattr__(self, attr_name: str) -> Any:
        return getattr(self._provider(), attr_name)

    def __repr__(self) -> str:
        return f"<LazyTranslationProvider name={self.name!r}>"


ANTHROPIC_PROVIDER = LazyTranslationProvider("anthropic")
GEMINI_PROVIDER = LazyTranslationProvider("gemini")
OPENAI_PROVIDER = LazyTranslationProvider("openai")
DEFAULT_PROVIDER = GEMINI_PROVIDER
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
        raise CliError(f"Unsupported provider: {provider_name!r}. Supported providers: {supported}")
    return provider


def validate_provider_seed(provider: TranslationProvider, seed: int | None) -> None:
    if seed is not None and not getattr(provider, "supports_seed", False):
        raise CliError(f"Provider '{provider.name}' does not support --seed.")


__all__ = [
    "DEFAULT_PROVIDER",
    "DEFAULT_PROVIDER_NAME",
    "ANTHROPIC_PROVIDER",
    "GEMINI_PROVIDER",
    "OPENAI_PROVIDER",
    "SUPPORTED_TRANSLATION_PROVIDERS",
    "LazyTranslationProvider",
    "TranslationProvider",
    "get_translation_provider",
    "validate_provider_seed",
]
