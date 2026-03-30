from __future__ import annotations

from typing import Any, Protocol


class TranslationProvider(Protocol):
    name: str
    default_model: str
    api_key_env: str | None
    supports_structured_json: bool
    supports_structured_input: bool
    supports_thinking: bool

    def create_client_from_env(self) -> Any:
        ...

    def build_request_contents(
        self,
        *,
        task_instruction: str,
        function_name: str,
        payload: dict[str, Any],
        fallback_prompt: str,
    ) -> Any:
        ...

    def build_generation_config(
        self,
        *,
        thinking_level: str | None,
        json_schema: dict[str, Any] | None,
        system_instruction: str | None,
    ) -> Any:
        ...

    async def generate_with_retry(
        self,
        *,
        client: Any,
        model: str,
        contents: Any,
        batch_label: str,
        max_attempts: int,
        config: Any,
    ) -> Any:
        ...
