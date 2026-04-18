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
    vertex_flag_env = "GOOGLE_GENAI_USE_VERTEXAI"
    vertex_location_env = "GOOGLE_CLOUD_LOCATION"
    default_vertex_location = "global"
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

    def _use_vertex_from_env(self) -> bool:
        raw_value = str(os.getenv(self.vertex_flag_env, "")).strip().lower()
        return raw_value in {"1", "true", "yes", "on"}

    def _build_http_options(self, *, use_vertex: bool, flex_mode: bool) -> genai_types.HttpOptions | None:
        options_kwargs: Dict[str, Any] = {}
        if use_vertex:
            options_kwargs["api_version"] = "v1"
        if flex_mode:
            options_kwargs["timeout"] = self.flex_timeout_seconds
        if not options_kwargs:
            return None
        return genai_types.HttpOptions(**options_kwargs)

    def create_client_from_env(self, *, flex_mode: bool = False) -> genai.Client:
        use_vertex = self._use_vertex_from_env()
        http_options = self._build_http_options(use_vertex=use_vertex, flex_mode=flex_mode)

        if use_vertex:
            api_key = str(os.getenv(self.api_key_env, "")).strip()
            location = (
                str(os.getenv(self.vertex_location_env, "")).strip()
                or self.default_vertex_location
            )
            if not api_key:
                sys.exit(f"ERROR: {self.api_key_env} environment variable is not set")
            if location.lower() != self.default_vertex_location:
                sys.exit(
                    "ERROR: Gemini Vertex API-key mode currently supports only the global endpoint"
                )
            client_kwargs: Dict[str, Any] = {
                "vertexai": True,
                "api_key": api_key,
            }
            if http_options is not None:
                client_kwargs["http_options"] = http_options
            return genai.Client(**client_kwargs)

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            sys.exit(f"ERROR: {self.api_key_env} environment variable is not set")
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
        seed: int | None = None,
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
        if seed is not None:
            config_kwargs["seed"] = int(seed)
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
