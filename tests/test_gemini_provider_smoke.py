import os
import unittest
from unittest.mock import patch

from core.providers.gemini import GeminiTranslationProvider


class GeminiProviderSmokeTests(unittest.TestCase):
    def test_create_client_from_env_uses_ai_studio_api_key(self):
        provider = GeminiTranslationProvider()

        with patch.dict(
            os.environ,
            {
                provider.api_key_env: "studio-key",
                provider.vertex_flag_env: "false",
            },
            clear=True,
        ):
            with patch("core.providers.gemini.genai.Client", return_value="client") as client_cls:
                client = provider.create_client_from_env()

        self.assertEqual(client, "client")
        self.assertEqual(client_cls.call_args.kwargs["api_key"], "studio-key")
        self.assertNotIn("vertexai", client_cls.call_args.kwargs)

    def test_create_client_from_env_uses_vertex_api_key_mode(self):
        provider = GeminiTranslationProvider()

        with patch.dict(
            os.environ,
            {
                provider.api_key_env: "vertex-key",
                provider.vertex_flag_env: "true",
                provider.vertex_location_env: "global",
            },
            clear=True,
        ):
            with patch("core.providers.gemini.genai.Client", return_value="client") as client_cls:
                client = provider.create_client_from_env(flex_mode=True)

        self.assertEqual(client, "client")
        self.assertTrue(client_cls.call_args.kwargs["vertexai"])
        self.assertEqual(client_cls.call_args.kwargs["api_key"], "vertex-key")
        self.assertNotIn("location", client_cls.call_args.kwargs)
        self.assertEqual(client_cls.call_args.kwargs["http_options"].api_version, "v1")
        self.assertEqual(client_cls.call_args.kwargs["http_options"].timeout, provider.flex_timeout_seconds)

    def test_create_client_from_env_requires_api_key_in_vertex_mode(self):
        provider = GeminiTranslationProvider()

        with patch.dict(
            os.environ,
            {
                provider.vertex_flag_env: "true",
                provider.vertex_location_env: "global",
            },
            clear=True,
        ):
            with self.assertRaises(SystemExit) as ctx:
                provider.create_client_from_env()

        self.assertIn(provider.api_key_env, str(ctx.exception))

    def test_create_client_from_env_rejects_non_global_vertex_location(self):
        provider = GeminiTranslationProvider()

        with patch.dict(
            os.environ,
            {
                provider.api_key_env: "vertex-key",
                provider.vertex_flag_env: "true",
                provider.vertex_location_env: "us-central1",
            },
            clear=True,
        ):
            with self.assertRaises(SystemExit) as ctx:
                provider.create_client_from_env()

        self.assertIn("global endpoint", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
