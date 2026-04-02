import asyncio
import os
import unittest
from unittest.mock import patch

import httpx
from openai import APIConnectionError

from core.providers.openai import OpenAITranslationProvider


class _DummyResponses:
    def __init__(self, output_text: str):
        self.output_text = output_text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("Response", (), {"output_text": self.output_text})()


class _DummyClient:
    def __init__(self, output_text: str):
        self.responses = _DummyResponses(output_text)


class OpenAIProviderSmokeTests(unittest.TestCase):
    def test_read_api_key_rejects_log_output(self):
        provider = OpenAITranslationProvider()

        with patch.dict(
            os.environ,
            {
                provider.api_key_env: "API Error [batch 1/1]: Connection error.\nRetrying...",
            },
            clear=True,
        ):
            with self.assertRaises(SystemExit) as ctx:
                provider._read_api_key()

        self.assertIn("appears to contain log output", str(ctx.exception))

    def test_read_api_key_rejects_whitespace(self):
        provider = OpenAITranslationProvider()

        with patch.dict(
            os.environ,
            {
                provider.api_key_env: "sk-test key",
            },
            clear=True,
        ):
            with self.assertRaises(SystemExit) as ctx:
                provider._read_api_key()

        self.assertIn("contains whitespace or line breaks", str(ctx.exception))

    def test_create_client_from_env_uses_timeout_and_optional_base_url(self):
        provider = OpenAITranslationProvider()

        with patch.dict(
            os.environ,
            {
                provider.api_key_env: "test-key",
                provider.base_url_env: "https://example.invalid/v1",
                provider.timeout_env: "123",
            },
            clear=True,
        ):
            with patch("core.providers.openai.OpenAI", return_value="client") as openai_cls:
                client = provider.create_client_from_env()

        self.assertEqual(client, "client")
        self.assertEqual(openai_cls.call_args.kwargs["api_key"], "test-key")
        self.assertEqual(openai_cls.call_args.kwargs["base_url"], "https://example.invalid/v1")
        self.assertEqual(openai_cls.call_args.kwargs["timeout"], 123.0)
        self.assertEqual(openai_cls.call_args.kwargs["max_retries"], 0)

    def test_build_generation_config_includes_schema_instruction_and_reasoning(self):
        provider = OpenAITranslationProvider()

        config = provider.build_generation_config(
            thinking_level="minimal",
            json_schema={
                "properties": {
                    "ok": {"type": "string"},
                    "nested": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string"},
                        },
                    },
                },
            },
            system_instruction="You are a translator.",
        )

        self.assertEqual(config["instructions"], "You are a translator.")
        self.assertEqual(config["reasoning"], {"effort": "minimal"})
        self.assertEqual(config["text"]["format"]["type"], "json_schema")
        self.assertTrue(config["text"]["format"]["strict"])
        schema = config["text"]["format"]["schema"]
        self.assertEqual(schema["type"], "object")
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["required"], ["ok", "nested"])
        self.assertFalse(schema["properties"]["nested"]["additionalProperties"])
        self.assertEqual(schema["properties"]["nested"]["required"], ["value"])

    def test_describe_api_error_includes_connection_cause(self):
        provider = OpenAITranslationProvider()
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        try:
            raise APIConnectionError(request=request) from OSError("dns lookup failed")
        except APIConnectionError as exc:
            description = provider._describe_api_error(exc)

        self.assertIn("Connection error.", description)
        self.assertIn("OSError: dns lookup failed", description)

    def test_generate_with_retry_returns_parsed_json_text(self):
        provider = OpenAITranslationProvider()
        client = _DummyClient('{"translations":[{"id":"0","text":"Ashu"}]}')

        response = asyncio.run(
            provider.generate_with_retry(
                client=client,
                model="gpt-5-mini",
                contents="prompt text",
                batch_label="batch 1/1",
                max_attempts=1,
                config={
                    "instructions": "Translate",
                    "reasoning": {"effort": "low"},
                },
            )
        )

        self.assertEqual(response.text, '{"translations":[{"id":"0","text":"Ashu"}]}')
        self.assertEqual(response.parsed["translations"][0]["text"], "Ashu")
        self.assertEqual(client.responses.calls[0]["model"], "gpt-5-mini")
        self.assertEqual(client.responses.calls[0]["input"], "prompt text")
        self.assertEqual(client.responses.calls[0]["instructions"], "Translate")
        self.assertEqual(client.responses.calls[0]["reasoning"], {"effort": "low"})


if __name__ == "__main__":
    unittest.main()
