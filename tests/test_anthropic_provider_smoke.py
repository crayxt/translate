import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.providers.anthropic import AnthropicTranslationProvider


class _DummyAnthropicClient:
    def __init__(self, response):
        self.messages = SimpleNamespace(create=self._create)
        self._response = response
        self.calls: list[dict] = []

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class AnthropicProviderSmokeTests(unittest.TestCase):
    def test_create_client_from_env_uses_timeout_and_optional_base_url(self):
        provider = AnthropicTranslationProvider()

        captured: dict[str, object] = {}

        class _FakeAnthropic:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with patch.dict(
            os.environ,
            {
                provider.api_key_env: "test-key",
                provider.base_url_env: "https://example.invalid",
                provider.timeout_env: "123",
            },
            clear=True,
        ):
            with patch("core.providers.anthropic._import_anthropic_sdk", return_value=_FakeAnthropic):
                provider.create_client_from_env()

        self.assertEqual(captured["api_key"], "test-key")
        self.assertEqual(captured["base_url"], "https://example.invalid")
        self.assertEqual(captured["timeout"], 123.0)
        self.assertEqual(captured["max_retries"], 0)

    def test_build_request_contents_uses_native_messages_shape(self):
        provider = AnthropicTranslationProvider()

        contents = provider.build_request_contents(
            task_instruction="Translate items",
            function_name="sample_batch",
            payload={"items": [{"id": "0"}]},
            fallback_prompt="plain prompt",
        )

        self.assertEqual(contents[0]["role"], "user")
        self.assertEqual(contents[0]["content"][0]["type"], "text")
        self.assertIn("response_payload", contents[0]["content"][0]["text"])
        self.assertIn('"id": "0"', contents[0]["content"][1]["text"])

    def test_build_generation_config_uses_tool_choice_when_no_thinking(self):
        provider = AnthropicTranslationProvider()

        config = provider.build_generation_config(
            thinking_level=None,
            json_schema={
                "type": "object",
                "properties": {
                    "translations": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["translations"],
            },
            system_instruction="You are a translator.",
        )

        self.assertEqual(config["system"], "You are a translator.")
        self.assertEqual(config["tool_choice"], {"type": "tool", "name": "response_payload"})
        self.assertEqual(config["tools"][0]["name"], "response_payload")
        self.assertTrue(config["tools"][0]["strict"])

    def test_build_generation_config_enables_thinking_with_auto_tool_choice(self):
        provider = AnthropicTranslationProvider()

        config = provider.build_generation_config(
            thinking_level="minimal",
            json_schema={"type": "object", "properties": {"ok": {"type": "string"}}},
            system_instruction="You are a translator.",
        )

        self.assertEqual(
            config["thinking"],
            {"type": "enabled", "budget_tokens": 1024},
        )
        self.assertEqual(config["tool_choice"], {"type": "auto"})

    def test_parse_response_content_prefers_tool_use_payload(self):
        provider = AnthropicTranslationProvider()
        response = SimpleNamespace(
            content=[
                {"type": "text", "text": "Working..."},
                {
                    "type": "tool_use",
                    "name": "response_payload",
                    "input": {"translations": [{"id": "0", "text": "Ashu"}]},
                },
            ]
        )

        parsed = provider._parse_response_content(response)

        self.assertEqual(parsed.parsed["translations"][0]["text"], "Ashu")
        self.assertEqual(parsed.text, "Working...")

    def test_generate_with_retry_falls_back_to_json_text(self):
        provider = AnthropicTranslationProvider()
        response = SimpleNamespace(
            content=[
                {
                    "type": "text",
                    "text": '{"translations":[{"id":"0","text":"Ashu"}]}',
                }
            ]
        )
        client = _DummyAnthropicClient(response)

        parsed = asyncio.run(
            provider.generate_with_retry(
                client=client,
                model="claude-sonnet-4-20250514",
                contents=[{"role": "user", "content": "Translate"}],
                batch_label="batch 1/1",
                max_attempts=1,
                config={"system": "Translate"},
            )
        )

        self.assertEqual(parsed.parsed["translations"][0]["text"], "Ashu")
        self.assertEqual(client.calls[0]["model"], "claude-sonnet-4-20250514")
        self.assertEqual(client.calls[0]["system"], "Translate")


if __name__ == "__main__":
    unittest.main()
