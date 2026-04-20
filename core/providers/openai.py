from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import os
from dataclasses import dataclass
from typing import Any, Dict

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from core.cli_errors import CliError
from core.entries import json_load_maybe


@dataclass(frozen=True)
class OpenAIProviderResponse:
    parsed: Any | None = None
    text: str = ""


@dataclass(frozen=True)
class OpenAIRequestContents:
    """Separates the per-task instruction from the per-batch payload.

    The task instruction is merged into ``instructions`` at request time
    so OpenAI can cache it across batches, while the payload varies per batch.
    """
    task_instruction: str
    payload_text: str


class OpenAITranslationProvider:
    name = "openai"
    default_model = "gpt-5-mini"
    api_key_env = "OPENAI_API_KEY"
    base_url_env = "OPENAI_BASE_URL"
    timeout_env = "OPENAI_TIMEOUT_SECONDS"
    default_timeout_seconds = 600.0
    supports_structured_json = True
    supports_structured_input = False
    supports_thinking = True
    supports_flex_mode = True
    supports_seed = False

    def _read_timeout_seconds(self) -> float:
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
                "Clear the GUI field or environment variable and paste the real OpenAI API key."
            )

        if any(ch.isspace() for ch in cleaned_api_key):
            raise CliError(
                f"{self.api_key_env} contains whitespace or line breaks. "
                "Paste only the raw OpenAI API key."
            )

        return cleaned_api_key

    def create_client_from_env(self, *, flex_mode: bool = False) -> OpenAI:
        _ = flex_mode
        api_key = self._read_api_key()
        base_url = str(os.getenv(self.base_url_env, "")).strip() or None
        timeout = self._read_timeout_seconds()
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )

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

        defs = normalized.get("$defs")
        if isinstance(defs, dict):
            normalized["$defs"] = {
                str(name): self._normalize_schema_node(value)
                for name, value in defs.items()
            }

        any_of = normalized.get("anyOf")
        if isinstance(any_of, list):
            normalized["anyOf"] = [self._normalize_schema_node(value) for value in any_of]

        one_of = normalized.get("oneOf")
        if isinstance(one_of, list):
            normalized["oneOf"] = [self._normalize_schema_node(value) for value in one_of]

        all_of = normalized.get("allOf")
        if isinstance(all_of, list):
            normalized["allOf"] = [self._normalize_schema_node(value) for value in all_of]

        schema_type = normalized.get("type")
        if isinstance(normalized.get("properties"), dict) and schema_type in (None, "object"):
            normalized["type"] = "object"
            normalized["additionalProperties"] = False

        return normalized

    def build_request_contents(
        self,
        *,
        task_instruction: str,
        function_name: str,
        payload: dict[str, Any],
        fallback_prompt: str,
    ) -> OpenAIRequestContents:
        _ = fallback_prompt
        _ = function_name
        payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
        return OpenAIRequestContents(
            task_instruction=task_instruction,
            payload_text=f"Batch payload (JSON):\n{payload_json}",
        )

    def build_generation_config(
        self,
        *,
        thinking_level: str | None,
        json_schema: dict[str, Any] | None,
        system_instruction: str | None,
        flex_mode: bool = False,
        seed: int | None = None,
    ) -> dict[str, Any]:
        _ = seed
        config: Dict[str, Any] = {}
        if json_schema is not None:
            normalized_schema = self._normalize_response_schema(json_schema)
            config["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "response_payload",
                    "schema": normalized_schema,
                    "strict": True,
                }
            }
        if system_instruction and system_instruction.strip():
            config["instructions"] = system_instruction.strip()
        if flex_mode:
            config["service_tier"] = "flex"
        if thinking_level is not None:
            normalized = str(thinking_level).strip().lower()
            if normalized:
                config["reasoning"] = {"effort": normalized}
        return config

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

    async def generate_with_retry(
        self,
        *,
        client: OpenAI,
        model: str,
        contents: Any,
        batch_label: str,
        max_attempts: int,
        config: dict[str, Any] | None,
    ) -> OpenAIProviderResponse:
        effective_config = dict(config) if config else {}
        if isinstance(contents, OpenAIRequestContents):
            input_text = contents.payload_text
            if "instructions" in effective_config and contents.task_instruction:
                effective_config["instructions"] = (
                    f"{effective_config['instructions']}\n\n{contents.task_instruction}"
                )
            elif contents.task_instruction:
                effective_config["instructions"] = contents.task_instruction
        else:
            input_text = contents
        request_kwargs: Dict[str, Any] = {"model": model, "input": input_text}
        if effective_config:
            request_kwargs.update(effective_config)

        for attempt in range(1, max_attempts + 1):
            try:
                response = await asyncio.to_thread(client.responses.create, **request_kwargs)
                text = getattr(response, "output_text", "") or ""
                return OpenAIProviderResponse(
                    parsed=json_load_maybe(text),
                    text=text,
                )
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
