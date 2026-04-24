from __future__ import annotations

import asyncio
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from core.cli_errors import CliError


@dataclass(frozen=True)
class OpenAICompatibleResponse:
    parsed: Any | None = None
    text: str = ""


class OpenAIBaseProvider:
    """Base class for providers using the OpenAI Python SDK."""
    api_key_env: str
    timeout_env: str | None = None
    default_timeout_seconds: float = 600.0

    def _read_timeout_seconds(self) -> float:
        if not self.timeout_env:
            return self.default_timeout_seconds
        raw_timeout = str(os.getenv(self.timeout_env, "")).strip()
        if not raw_timeout:
            return self.default_timeout_seconds
        try:
            timeout_seconds = float(raw_timeout)
        except ValueError:
            raise CliError(f"{self.timeout_env} must be a positive number")
        if timeout_seconds <= 0:
            raise CliError(f"{self.timeout_env} must be a positive number")
        return timeout_seconds

    def _read_api_key(self) -> str:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise CliError(f"{self.api_key_env} environment variable is not set")

        cleaned_api_key = str(api_key).strip()
        if not cleaned_api_key:
            raise CliError(f"{self.api_key_env} environment variable is empty")

        if "API Error [" in cleaned_api_key or "Retrying [" in cleaned_api_key:
            raise CliError(
                f"{self.api_key_env} appears to contain log output instead of an API key. "
                "Ensure the environment variable contains only the raw API key."
            )

        if any(ch.isspace() for ch in cleaned_api_key):
            raise CliError(
                f"{self.api_key_env} contains whitespace or line breaks. "
                "Paste only the raw API key."
            )

        return cleaned_api_key

    def _describe_api_error(self, exc: Exception) -> str:
        detail = str(exc).strip() or exc.__class__.__name__
        cause = exc.__cause__ or exc.__context__

        if isinstance(exc, APIStatusError):
            status_code = getattr(exc, "status_code", None)
            request_id = getattr(exc, "request_id", None)
            suffix_parts = []
            if status_code is not None:
                suffix_parts.append(f"status={status_code}")
            if request_id:
                suffix_parts.append(f"request_id={request_id}")
            if suffix_parts:
                detail = f"{detail} ({', '.join(suffix_parts)})"

        if isinstance(exc, APITimeoutError):
            detail = f"{detail} (timeout after {self._read_timeout_seconds():g}s)"

        if isinstance(exc, APIConnectionError) and cause is not None:
            cause_text = str(cause).strip() or cause.__class__.__name__
            detail = f"{detail} ({cause.__class__.__name__}: {cause_text})"

        return detail

    def _normalize_response_schema(self, json_schema: dict[str, Any]) -> dict[str, Any]:
        schema = deepcopy(json_schema)
        return self._normalize_schema_node(schema)

    def _normalize_schema_node(self, node: Any) -> Any:
        if not isinstance(node, dict):
            return node

        normalized = {key: self._normalize_schema_node(value) for key, value in node.items()}

        properties = normalized.get("properties")
        if isinstance(properties, dict):
            normalized["properties"] = {
                str(name): self._normalize_schema_node(value)
                for name, value in properties.items()
            }

        items = normalized.get("items")
        if isinstance(items, dict):
            normalized["items"] = self._normalize_schema_node(items)

        # Handle specific OpenAI strict mode requirement: additionalProperties must be false
        schema_type = normalized.get("type")
        if isinstance(normalized.get("properties"), dict) and schema_type in (None, "object"):
            normalized["type"] = "object"
            normalized["additionalProperties"] = False

        return normalized

    async def _perform_retry_loop(
        self,
        *,
        async_api_call: Any,
        batch_label: str,
        max_attempts: int,
        parse_output_fn: Any,
    ) -> OpenAICompatibleResponse:
        for attempt in range(1, max_attempts + 1):
            try:
                response = await async_api_call()
                return parse_output_fn(response)
            except Exception as exc:
                print(
                    f"\nAPI Error [{batch_label}] (Attempt {attempt}/{max_attempts}): "
                    f"{self._describe_api_error(exc)}"
                )
                if attempt == max_attempts:
                    raise RuntimeError(f"Aborting [{batch_label}] due to repeated API errors.") from exc
                wait_time = 2 ** attempt
                print(f"Retrying [{batch_label}] in {wait_time}s...")
                await asyncio.sleep(wait_time)
