from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class TranslationProvider(Protocol):
    name: str

    def create_client_from_env(self) -> Any:
        ...

    def build_translation_config(
        self,
        *,
        thinking_level: str | None,
        response_schema: Any,
    ) -> Any:
        ...

    async def generate_with_retry(
        self,
        *,
        client: Any,
        model: str,
        prompt: str,
        batch_label: str,
        max_attempts: int,
        config: Any,
    ) -> Any:
        ...


@dataclass(frozen=True)
class ProviderSpec:
    provider: TranslationProvider
    model: str
