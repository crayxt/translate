from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import os
from dataclasses import dataclass
from typing import Any, Dict

from core.cli_errors import CliError
from core.entries import json_load_maybe


def _import_anthropic_sdk() -> Any:
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError(
            "Anthropic provider requires the `anthropic` package. "
            "Install project dependencies from requirements.txt."
        ) from exc
    return Anthropic


@dataclass(frozen=True)
class AnthropicProviderResponse:
    parsed: Any | None = None
    text: str = ""


class AnthropicTranslationProvider:
    name = "anthropic"
    default_model = "claude-sonnet-4-20250514"
    api_key_env = "ANTHROPIC_API_KEY"
    base_url_env = "ANTHROPIC_BASE_URL"
    timeout_env = "ANTHROPIC_TIMEOUT_SECONDS"
    default_timeout_seconds = 600.0
    default_max_tokens = 8192
    response_tool_name = "response_payload"
    supports_structured_json = True
    supports_structured_input = False
    supports_thinking = True
    supports_flex_mode = False
    supports_seed = False

    _THINKING_BUDGETS = {
        "minimal": 1024,
        "low": 1536,
        "medium": 2048,
        "high": 4096,
    }

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
                "Clear the GUI field or environment variable and paste the real Anthropic API key."
            )

        if any(ch.isspace() for ch in cleaned_api_key):
            raise CliError(
                f"{self.api_key_env} contains whitespace or line breaks. "
                "Paste only the raw Anthropic API key."
            )

        return cleaned_api_key

    def create_client_from_env(self, *, flex_mode: bool = False) -> Any:
        _ = flex_mode
        Anthropic = _import_anthropic_sdk()
        return Anthropic(
            api_key=self._read_api_key(),
            base_url=str(os.getenv(self.base_url_env, "")).strip() or None,
            timeout=self._read_timeout_seconds(),
            max_retries=0,
        )

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
            schema_type = normalized.get("type")
            if schema_type in (None, "object"):
                normalized["type"] = "object"
                normalized.setdefault("additionalProperties", False)

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

        return normalized

    def _normalize_response_schema(self, json_schema: dict[str, Any]) -> dict[str, Any]:
        return self._normalize_schema_node(deepcopy(json_schema))

    def _build_tool_instruction(self, task_instruction: str) -> str:
        return (
            f"{task_instruction.strip()}\n\n"
            f"Instead of emitting raw JSON text, call the `{self.response_tool_name}` tool "
            "with the final structured result. Do not add commentary outside the tool call."
        )

    def build_request_contents(
        self,
        *,
        task_instruction: str,
        function_name: str,
        payload: dict[str, Any],
        fallback_prompt: str,
    ) -> Any:
        _ = function_name
        _ = fallback_prompt
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._build_tool_instruction(task_instruction)},
                    {
                        "type": "text",
                        "text": "Batch payload (JSON):\n"
                        + json.dumps(payload, ensure_ascii=False, indent=2),
                    },
                ],
            }
        ]

    def _resolve_thinking_budget(self, thinking_level: str | None) -> int | None:
        normalized = str(thinking_level or "").strip().lower()
        if not normalized:
            return None
        return self._THINKING_BUDGETS.get(normalized)

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
        _ = flex_mode
        config: Dict[str, Any] = {"max_tokens": self.default_max_tokens}
        if system_instruction and system_instruction.strip():
            config["system"] = system_instruction.strip()

        thinking_budget = self._resolve_thinking_budget(thinking_level)
        if thinking_budget is not None:
            config["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }

        if json_schema is not None:
            config["tools"] = [
                {
                    "name": self.response_tool_name,
                    "description": "Return the final structured task result.",
                    "input_schema": self._normalize_response_schema(json_schema),
                    "strict": True,
                }
            ]
            if thinking_budget is None:
                config["tool_choice"] = {
                    "type": "tool",
                    "name": self.response_tool_name,
                }
            else:
                config["tool_choice"] = {"type": "auto"}
        return config

    def _describe_api_error(self, exc: Exception) -> str:
        detail = str(exc).strip() or exc.__class__.__name__
        cause = exc.__cause__ or exc.__context__

        status_code = getattr(exc, "status_code", None)
        request_id = getattr(exc, "request_id", None)
        suffix_parts = []
        if status_code is not None:
            suffix_parts.append(f"status={status_code}")
        if request_id:
            suffix_parts.append(f"request_id={request_id}")
        if suffix_parts:
            detail = f"{detail} ({', '.join(suffix_parts)})"

        if exc.__class__.__name__ == "APITimeoutError":
            detail = f"{detail} (timeout after {self._read_timeout_seconds():g}s)"

        if exc.__class__.__name__ == "APIConnectionError" and cause is not None:
            cause_text = str(cause).strip() or cause.__class__.__name__
            detail = f"{detail} ({cause.__class__.__name__}: {cause_text})"

        return detail

    @staticmethod
    def _block_attr(block: Any, name: str) -> Any:
        if isinstance(block, dict):
            return block.get(name)
        return getattr(block, name, None)

    def _parse_response_content(self, response: Any) -> AnthropicProviderResponse:
        parsed: Any | None = None
        text_chunks: list[str] = []

        for block in getattr(response, "content", []) or []:
            block_type = self._block_attr(block, "type")
            if block_type == "tool_use":
                name = self._block_attr(block, "name")
                if name == self.response_tool_name:
                    parsed = self._block_attr(block, "input")
            elif block_type == "text":
                text = self._block_attr(block, "text")
                if text:
                    text_chunks.append(str(text))

        text = "\n".join(text_chunks).strip()
        if parsed is None:
            parsed = json_load_maybe(text)
        return AnthropicProviderResponse(parsed=parsed, text=text)

    async def generate_with_retry(
        self,
        *,
        client: Any,
        model: str,
        contents: Any,
        batch_label: str,
        max_attempts: int,
        config: dict[str, Any] | None,
    ) -> AnthropicProviderResponse:
        request_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": contents,
        }
        if config:
            request_kwargs.update(config)

        for attempt in range(1, max_attempts + 1):
            try:
                response = await asyncio.to_thread(client.messages.create, **request_kwargs)
                return self._parse_response_content(response)
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
