from __future__ import annotations

import os
import sys
from typing import Any, Dict

from google import genai
from google.genai import types as genai_types

from core.runtime import build_thinking_config, generate_with_retry


class GeminiTranslationProvider:
    name = "gemini"
    default_model = "gemini-3-flash-preview"
    api_key_env = "GOOGLE_API_KEY"

    def create_client_from_env(self) -> genai.Client:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            sys.exit(f"ERROR: {self.api_key_env} environment variable is not set")
        return genai.Client(api_key=api_key)

    def build_translation_config(
        self,
        *,
        thinking_level: str | None,
        response_schema: Any,
    ) -> genai_types.GenerateContentConfig:
        config_kwargs: Dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_schema": response_schema,
        }
        thinking_config = build_thinking_config(thinking_level)
        if thinking_config is not None:
            config_kwargs["thinking_config"] = thinking_config
        return genai_types.GenerateContentConfig(**config_kwargs)

    async def generate_with_retry(
        self,
        *,
        client: genai.Client,
        model: str,
        prompt: str,
        batch_label: str,
        max_attempts: int,
        config: genai_types.GenerateContentConfig | None,
    ) -> Any:
        return await generate_with_retry(
            client=client,
            model=model,
            prompt=prompt,
            batch_label=batch_label,
            max_attempts=max_attempts,
            config=config,
        )
