import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import core.providers as providers


class ProviderRegistrySmokeTests(unittest.TestCase):
    def test_provider_registry_imports_no_sdk_backed_provider_modules(self):
        code = r'''
import sys
import core.providers as providers

provider_modules = (
    "core.providers.anthropic",
    "core.providers.gemini",
    "core.providers.openai",
)
loaded = [name for name in provider_modules if name in sys.modules]
if loaded:
    raise SystemExit(f"loaded after registry import: {loaded}")

_ = providers.DEFAULT_PROVIDER.default_model
_ = providers.get_translation_provider("openai").api_key_env
loaded = [name for name in provider_modules if name in sys.modules]
if loaded:
    raise SystemExit(f"loaded after metadata access: {loaded}")
'''
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    def test_lazy_provider_loads_selected_module_on_behavior_access(self):
        providers._load_translation_provider.cache_clear()
        provider_impl = SimpleNamespace(create_client_from_env=lambda flex_mode=False: "client")
        provider_cls = Mock(return_value=provider_impl)
        module = SimpleNamespace(OpenAITranslationProvider=provider_cls)

        with patch("core.providers.import_module", return_value=module) as import_module:
            provider = providers.get_translation_provider("openai")

            self.assertEqual(provider.default_model, "gpt-5-mini")
            import_module.assert_not_called()

            client = provider.create_client_from_env()

        self.assertEqual(client, "client")
        import_module.assert_called_once_with("core.providers.openai")
        provider_cls.assert_called_once_with()
        providers._load_translation_provider.cache_clear()


if __name__ == "__main__":
    unittest.main()
