from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

from openai import OpenAI

from core.entries import json_load_maybe
from core.providers.openai_base import OpenAIBaseProvider, OpenAICompatibleResponse


class DeepSeekTranslationProvider(OpenAIBaseProvider):
    name = "deepseek"
    default_model = "deepseek-v4-flash"
    api_key_env = "DEEPSEEK_API_KEY"
    default_base_url = "https://api.deepseek.com"
    timeout_env = "DEEPSEEK_TIMEOUT_SECONDS"
    supports_structured_json = True
    supports_structured_input = False
    supports_thinking = True
    supports_flex_mode = False
    supports_seed = False

    def create_client_from_env(self, *, flex_mode: bool = False) -> OpenAI:
        _ = flex_mode
        api_key = self._read_api_key()
        base_url = self.default_base_url
        timeout = self._read_timeout_seconds()
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )

    def build_request_contents(
        self,
        *,
        task_instruction: str,
        function_name: str,
        payload: dict[str, Any],
        fallback_prompt: str,
    ) -> str:
        _ = function_name
        _ = fallback_prompt
        payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
        return f"{task_instruction.strip()}\n\nBatch payload (JSON):\n{payload_json}"

    def build_generation_config(
        self,
        *,
        thinking_level: str | None,
        json_schema: dict[str, Any] | None,
        system_instruction: str | None,
        flex_mode: bool = False,
        seed: int | None = None,
    ) -> dict[str, Any]:
        _ = flex_mode
        _ = thinking_level
        _ = seed
        config: Dict[str, Any] = {}
        if json_schema is not None:
            config["response_format"] = {"type": "json_object"}
        if system_instruction and system_instruction.strip():
            config["system_instruction_text"] = system_instruction.strip()
        return config

    async def generate_with_retry(
        self,
        *,
        client: OpenAI,
        model: str,
        contents: Any,
        batch_label: str,
        max_attempts: int,
        config: dict[str, Any] | None,
    ) -> OpenAICompatibleResponse:
        messages = []
        effective_config = dict(config) if config else {}
        
        system_instr = effective_config.pop("system_instruction_text", None)
        if system_instr:
            messages.append({"role": "system", "content": system_instr})
            
        messages.append({"role": "user", "content": str(contents)})

        request_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if effective_config:
            request_kwargs.update(effective_config)

        def parse_output(response: Any) -> OpenAICompatibleResponse:
            text = response.choices[0].message.content or ""
            return OpenAICompatibleResponse(
                parsed=json_load_maybe(text),
                text=text,
            )

        return await self._perform_retry_loop(
            async_api_call=lambda: asyncio.to_thread(client.chat.completions.create, **request_kwargs),
            batch_label=batch_label,
            max_attempts=max_attempts,
            parse_output_fn=parse_output,
        )
