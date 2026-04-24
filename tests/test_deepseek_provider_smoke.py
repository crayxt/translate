import asyncio
import os
import unittest
from unittest.mock import patch

from core.providers.deepseek import DeepSeekTranslationProvider
from core.providers.openai_base import OpenAICompatibleResponse


class _DummyMessage:
    def __init__(self, content: str):
        self.content = content


class _DummyChoice:
    def __init__(self, content: str):
        self.message = _DummyMessage(content)


class _DummyCompletions:
    def __init__(self, output_text: str):
        self.output_text = output_text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("Response", (), {"choices": [_DummyChoice(self.output_text)]})()


class _DummyChat:
    def __init__(self, output_text: str):
        self.completions = _DummyCompletions(output_text)


class _DummyClient:
    def __init__(self, output_text: str):
        self.chat = _DummyChat(output_text)


class DeepSeekProviderSmokeTests(unittest.TestCase):
    def test_create_client_from_env_uses_timeout_and_default_base_url(self):
        provider = DeepSeekTranslationProvider()

        with patch.dict(
            os.environ,
            {
                provider.api_key_env: "test-key",
                provider.timeout_env: "123",
            },
            clear=True,
        ):
            with patch("core.providers.deepseek.OpenAI", return_value="client") as openai_cls:
                client = provider.create_client_from_env()

        self.assertEqual(client, "client")
        self.assertEqual(openai_cls.call_args.kwargs["api_key"], "test-key")
        self.assertEqual(openai_cls.call_args.kwargs["base_url"], "https://api.deepseek.com")
        self.assertEqual(openai_cls.call_args.kwargs["timeout"], 123.0)
        self.assertEqual(openai_cls.call_args.kwargs["max_retries"], 0)

    def test_build_generation_config_includes_json_mode_and_system_instruction(self):
        provider = DeepSeekTranslationProvider()

        config = provider.build_generation_config(
            thinking_level=None,
            json_schema={"type": "object"},
            system_instruction="You are a translator.",
        )

        self.assertEqual(config["response_format"], {"type": "json_object"})
        self.assertEqual(config["system_instruction_text"], "You are a translator.")

    def test_generate_with_retry_returns_parsed_json_text(self):
        provider = DeepSeekTranslationProvider()
        client = _DummyClient('{"translations":[{"id":"0","text":"Ashu"}]}')

        response = asyncio.run(
            provider.generate_with_retry(
                client=client,
                model="deepseek-v4-flash",
                contents="prompt text",
                batch_label="batch 1/1",
                max_attempts=1,
                config={
                    "system_instruction_text": "Translate",
                    "response_format": {"type": "json_object"},
                },
            )
        )

        self.assertEqual(response.text, '{"translations":[{"id":"0","text":"Ashu"}]}')
        self.assertEqual(response.parsed["translations"][0]["text"], "Ashu")
        self.assertEqual(client.chat.completions.calls[0]["model"], "deepseek-v4-flash")
        self.assertEqual(len(client.chat.completions.calls[0]["messages"]), 2)
        self.assertEqual(client.chat.completions.calls[0]["messages"][0]["role"], "system")
        self.assertEqual(client.chat.completions.calls[0]["messages"][0]["content"], "Translate")
        self.assertEqual(client.chat.completions.calls[0]["messages"][1]["role"], "user")
        self.assertEqual(client.chat.completions.calls[0]["messages"][1]["content"], "prompt text")


if __name__ == "__main__":
    unittest.main()
