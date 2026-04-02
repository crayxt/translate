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
    supports_structured_json = True
    supports_structured_input = True
    supports_thinking = True
    supports_flex_mode = True
    flex_timeout_seconds = 600

    _SCHEMA_TYPE_MAP = {
        "string": genai_types.Type.STRING,
        "number": genai_types.Type.NUMBER,
        "integer": genai_types.Type.INTEGER,
        "boolean": genai_types.Type.BOOLEAN,
        "array": genai_types.Type.ARRAY,
        "object": genai_types.Type.OBJECT,
    }

    def _build_response_schema(
        self,
        json_schema: dict[str, Any] | None,
    ) -> genai_types.Schema | None:
        if json_schema is None:
            return None

        schema_kwargs: Dict[str, Any] = {}
        schema_type = json_schema.get("type")
        if schema_type is not None:
            resolved_type = self._SCHEMA_TYPE_MAP.get(str(schema_type).lower())
            if resolved_type is None:
                raise ValueError(f"Unsupported JSON schema type for Gemini provider: {schema_type!r}")
            schema_kwargs["type"] = resolved_type

        properties = json_schema.get("properties")
        if isinstance(properties, dict):
            schema_kwargs["properties"] = {
                str(name): self._build_response_schema(value)
                for name, value in properties.items()
                if isinstance(value, dict)
            }

        items = json_schema.get("items")
        if isinstance(items, dict):
            schema_kwargs["items"] = self._build_response_schema(items)

        required = json_schema.get("required")
        if isinstance(required, list):
            schema_kwargs["required"] = [str(item) for item in required]

        description = json_schema.get("description")
        if description is not None:
            schema_kwargs["description"] = str(description)

        return genai_types.Schema(**schema_kwargs)

    def create_client_from_env(self, *, flex_mode: bool = False) -> genai.Client:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            sys.exit(f"ERROR: {self.api_key_env} environment variable is not set")
        http_options = None
        if flex_mode:
            http_options = genai_types.HttpOptions(timeout=self.flex_timeout_seconds)
        return genai.Client(api_key=api_key, http_options=http_options)

    def build_request_contents(
        self,
        *,
        task_instruction: str,
        function_name: str,
        payload: dict[str, Any],
        fallback_prompt: str,
    ) -> Any:
        _ = fallback_prompt
        return [
            genai_types.Content(
                role="user",
                parts=[
                    genai_types.Part.from_text(text=task_instruction.strip()),
                    genai_types.Part.from_function_response(
                        name=function_name,
                        response=payload,
                    ),
                ],
            )
        ]

    def build_generation_config(
        self,
        *,
        thinking_level: str | None,
        json_schema: dict[str, Any] | None,
        system_instruction: str | None,
        flex_mode: bool = False,
    ) -> genai_types.GenerateContentConfig:
        config_kwargs: Dict[str, Any] = {}
        response_schema = self._build_response_schema(json_schema)
        if response_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema
        if system_instruction and system_instruction.strip():
            config_kwargs["system_instruction"] = system_instruction.strip()
        if flex_mode:
            config_kwargs["model_selection_config"] = genai_types.ModelSelectionConfig(
                feature_selection_preference=genai_types.FeatureSelectionPreference.PRIORITIZE_COST,
            )
        thinking_config = build_thinking_config(thinking_level)
        if thinking_config is not None:
            config_kwargs["thinking_config"] = thinking_config
        return genai_types.GenerateContentConfig(**config_kwargs)

    async def generate_with_retry(
        self,
        *,
        client: genai.Client,
        model: str,
        contents: Any,
        batch_label: str,
        max_attempts: int,
        config: genai_types.GenerateContentConfig | None,
    ) -> Any:
        return await generate_with_retry(
            client=client,
            model=model,
            contents=contents,
            batch_label=batch_label,
            max_attempts=max_attempts,
            config=config,
        )
