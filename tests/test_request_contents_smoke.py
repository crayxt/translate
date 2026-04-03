import unittest

from core.providers.anthropic import AnthropicTranslationProvider
from core.providers.gemini import GeminiTranslationProvider
from core.providers.openai import OpenAITranslationProvider
from core.request_contents import TaskRequestSpec, build_task_request_contents
from tasks import check_translations
from tasks import extract_terms
from tasks import revise_translations
from tasks import translate


class _FallbackProvider:
    pass


class RequestContentsSmokeTests(unittest.TestCase):
    def test_build_task_request_contents_falls_back_to_prompt_without_builder(self):
        fallback = build_task_request_contents(
            provider=_FallbackProvider(),
            task_spec=TaskRequestSpec(
                task_intro="Do work",
                output_lines=("Return valid JSON.",),
            ),
            function_name="sample_batch",
            payload={"items": []},
        )

        self.assertIn("Do work", fallback)
        self.assertIn('"items": []', fallback)

    def test_gemini_provider_build_request_contents_returns_function_response(self):
        provider = GeminiTranslationProvider()

        contents = provider.build_request_contents(
            task_instruction="Translate items",
            function_name="sample_batch",
            payload={"items": [{"id": "0"}]},
            fallback_prompt="plain prompt",
        )

        self.assertEqual(len(contents), 1)
        self.assertEqual(contents[0].role, "user")
        self.assertEqual(contents[0].parts[0].text, "Translate items")
        self.assertEqual(contents[0].parts[1].function_response.name, "sample_batch")
        self.assertEqual(contents[0].parts[1].function_response.response["items"][0]["id"], "0")

    def test_openai_provider_build_request_contents_uses_text_fallback(self):
        provider = OpenAITranslationProvider()

        contents = provider.build_request_contents(
            task_instruction="Translate items",
            function_name="sample_batch",
            payload={"items": [{"id": "0"}]},
            fallback_prompt="plain prompt",
        )

        self.assertEqual(contents, "plain prompt")

    def test_anthropic_provider_build_request_contents_uses_message_blocks(self):
        provider = AnthropicTranslationProvider()

        contents = provider.build_request_contents(
            task_instruction="Translate items",
            function_name="sample_batch",
            payload={"items": [{"id": "0"}]},
            fallback_prompt="plain prompt",
        )

        self.assertEqual(contents[0]["role"], "user")
        self.assertIn("response_payload", contents[0]["content"][0]["text"])
        self.assertIn('"id": "0"', contents[0]["content"][1]["text"])

    def test_translate_request_contents_use_structured_batch_payload(self):
        contents = translate.build_translation_request_contents(
            messages={"0": {"source": "Open file"}},
            source_lang="en",
            target_lang="kk",
            vocabulary="open - ashu",
            translation_rules="Use imperative tone.",
            provider=translate.DEFAULT_PROVIDER,
        )

        payload = contents[0].parts[1].function_response.response
        self.assertEqual(contents[0].parts[1].function_response.name, "translation_batch")
        self.assertEqual(payload["source_lang"], "en")
        self.assertEqual(payload["messages"]["0"]["source"], "Open file")

    def test_translate_request_contents_can_include_message_scoped_vocabulary(self):
        contents = translate.build_translation_request_contents(
            messages={
                "0": {
                    "source": "Start playback",
                    "relevant_vocabulary": [
                        {
                            "source_term": "start",
                            "target_term": "бастау",
                            "part_of_speech": "verb",
                        }
                    ],
                }
            },
            source_lang="en",
            target_lang="kk",
            vocabulary="start|бастау|verb|Start playback",
            translation_rules="Use imperative tone.",
            provider=translate.DEFAULT_PROVIDER,
        )

        payload = contents[0].parts[1].function_response.response
        self.assertEqual(
            payload["messages"]["0"]["relevant_vocabulary"][0]["source_term"],
            "start",
        )

    def test_translate_request_spec_explicitly_mentions_context_and_variant_selection(self):
        spec = translate.build_translation_request_spec()

        self.assertIn(
            "Each message includes source text and may also include `context`, `note`, and `relevant_vocabulary`.",
            spec.payload_lines,
        )
        self.assertIn(
            "Use `message.context` and `message.note` to disambiguate meaning and select the correct approved terminology for that message.",
            spec.output_lines,
        )
        self.assertIn(
            "If multiple `message.relevant_vocabulary` entries share the same `source_term`, choose the variant whose `part_of_speech` and `context_note` best match `message.context` and `message.note`.",
            spec.output_lines,
        )

    def test_check_request_contents_use_structured_batch_payload(self):
        contents = check_translations.build_check_request_contents(
            messages={"0": {"source": "Open", "translation": "Ashu"}},
            source_lang="en",
            target_lang="kk",
            vocabulary="open - ashu",
            translation_rules="Use imperative tone.",
            provider=check_translations.DEFAULT_PROVIDER,
        )

        payload = contents[0].parts[1].function_response.response
        self.assertEqual(contents[0].parts[1].function_response.name, "translation_check_batch")
        self.assertEqual(payload["messages"]["0"]["translation"], "Ashu")

    def test_extract_request_contents_use_structured_batch_payload(self):
        contents = extract_terms.build_term_request_contents(
            messages={"0": {"source": "Open file", "context": "Toolbar"}},
            source_lang="en",
            target_lang="kk",
            mode="missing",
            vocabulary="open - ashu",
            max_terms_per_batch=25,
            provider=extract_terms.DEFAULT_PROVIDER,
        )

        payload = contents[0].parts[1].function_response.response
        self.assertEqual(contents[0].parts[1].function_response.name, "term_extraction_batch")
        self.assertEqual(payload["mode"], "missing")
        self.assertEqual(payload["messages"]["0"]["source"], "Open file")
        self.assertEqual(payload["messages"]["0"]["context"], "Toolbar")

    def test_revision_request_contents_use_structured_batch_payload(self):
        contents = revise_translations.build_revision_request_contents(
            messages={"0": {"source": "Save", "current_translation": "Saqtau"}},
            source_lang="en",
            target_lang="kk",
            instruction="Use shorter wording",
            vocabulary="save - saqtau",
            translation_rules="Keep labels short.",
            provider=revise_translations.DEFAULT_PROVIDER,
        )

        payload = contents[0].parts[1].function_response.response
        self.assertEqual(contents[0].parts[1].function_response.name, "translation_revision_batch")
        self.assertEqual(payload["instruction"], "Use shorter wording")
        self.assertEqual(payload["items"]["0"]["current_translation"], "Saqtau")


if __name__ == "__main__":
    unittest.main()
