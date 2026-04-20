import os
import unittest
from datetime import datetime
from unittest.mock import patch

import process_gui


class _FakeVar:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _FakeWidget:
    def __init__(self):
        self.config = {}

    def configure(self, **kwargs):
        self.config.update(kwargs)


class ProcessGuiSmokeTests(unittest.TestCase):
    def test_check_filetypes_include_ts(self):
        self.assertIn(("Qt TS files", "*.ts"), process_gui.CHECK_FILETYPES)

    def test_filetype_filters_include_xliff(self):
        self.assertIn(("XLIFF files", "*.xlf *.xliff"), process_gui.TRANSLATABLE_FILETYPES)
        self.assertIn(("XLIFF files", "*.xlf *.xliff"), process_gui.CHECK_FILETYPES)
        self.assertIn(("XLIFF files", "*.xlf *.xliff"), process_gui.LOCAL_EXTRACT_FILETYPES)

    def test_summarize_input_files_formats_multi_file_display(self):
        summary = process_gui.summarize_input_files(
            [r"C:\tmp\one.po", r"C:\tmp\two.po", r"C:\tmp\three.po"]
        )

        self.assertEqual(summary, r"C:\tmp\one.po (+2 more)")

    def test_summarize_recursive_input_folder_includes_count(self):
        summary = process_gui.summarize_recursive_input_folder(r"C:\tmp\tree", 3)

        self.assertEqual(summary, r"C:\tmp\tree (3 recursive files)")

    def test_get_local_extract_file_dialog_config_uses_source_files_in_extract_mode(self):
        title, filetypes = process_gui.get_local_extract_file_dialog_config(False)

        self.assertEqual(title, "Select source file for local extraction")
        self.assertEqual(filetypes, process_gui.LOCAL_EXTRACT_SOURCE_FILETYPES)

    def test_get_local_extract_file_dialog_config_uses_json_files_in_to_po_mode(self):
        title, filetypes = process_gui.get_local_extract_file_dialog_config(True)

        self.assertEqual(title, "Select local extraction JSON file")
        self.assertEqual(filetypes, process_gui.JSON_FILETYPES)

    def test_build_system_prompt_preview_for_translate_uses_translation_system_prompt(self):
        preview = process_gui.build_system_prompt_preview("process", "kk")
        self.assertIn("professional software localization translator", preview)
        self.assertIn("MANDATORY LOCALIZATION INVARIANTS", preview)
        self.assertIn("Placeholders must be preserved exactly", preview)

    def test_build_system_prompt_preview_for_check_uses_target_script_guidance(self):
        preview = process_gui.build_system_prompt_preview("check", "kk")
        self.assertIn("software localization QA reviewer", preview)
        self.assertIn("real Kazakh Cyrillic alphabet", preview)

    def test_build_system_prompt_preview_for_local_extract_explains_local_flow(self):
        preview = process_gui.build_system_prompt_preview("extract_local", "kk")
        self.assertIn("No model system prompt is used", preview)
        self.assertIn("core/term_extraction.py", preview)
        self.assertIn("core/term_handoff.py", preview)

    def test_detect_default_resource_paths_prefers_data_dir(self):
        data_dir = os.path.join(os.getcwd(), "_tmp_gui_data", "data", "locales", "fr")
        base_dir = os.path.join(os.getcwd(), "_tmp_gui_data")
        os.makedirs(data_dir, exist_ok=True)
        glossary_path = os.path.join(data_dir, "glossary.po")
        rules_path = os.path.join(data_dir, "rules.md")
        try:
            with open(glossary_path, "w", encoding="utf-8") as handle:
                handle.write('msgid "save"\nmsgstr "enregistrer"\n')
            with open(rules_path, "w", encoding="utf-8") as handle:
                handle.write("Use imperative tone.\n")

            detected_glossary, detected_rules = process_gui.detect_default_resource_paths(
                "fr",
                base_dir=base_dir,
            )

            self.assertEqual(detected_glossary, glossary_path)
            self.assertEqual(detected_rules, rules_path)
        finally:
            for path in (glossary_path, rules_path):
                if os.path.exists(path):
                    os.remove(path)
            if os.path.isdir(data_dir):
                os.removedirs(data_dir)

    def test_detect_default_resource_paths_ignores_legacy_root_glossary(self):
        base_dir = os.path.join(os.getcwd(), "_tmp_gui_legacy_root")
        glossary_path = os.path.join(base_dir, "glossary-fr.po")
        try:
            os.makedirs(base_dir, exist_ok=True)
            with open(glossary_path, "w", encoding="utf-8") as handle:
                handle.write('msgid "legacy"\nmsgstr "legacy"\n')

            detected_glossary, detected_rules = process_gui.detect_default_resource_paths(
                "fr",
                base_dir=base_dir,
            )

            self.assertEqual(detected_glossary, "")
            self.assertEqual(detected_rules, "")
        finally:
            if os.path.exists(glossary_path):
                os.remove(glossary_path)
            if os.path.isdir(base_dir):
                os.rmdir(base_dir)

    def test_detect_default_resource_paths_supports_glossary_directory(self):
        data_dir = os.path.join(os.getcwd(), "_tmp_gui_data_dir", "data", "locales", "fr")
        legacy_dir = os.path.join(os.getcwd(), "_tmp_gui_data_dir")
        glossary_dir = os.path.join(data_dir, "glossary")
        glossary_file = os.path.join(glossary_dir, "colors.po")
        rules_path = os.path.join(data_dir, "rules.md")
        try:
            os.makedirs(glossary_dir, exist_ok=True)
            with open(glossary_file, "w", encoding="utf-8") as handle:
                handle.write('msgid "blue"\nmsgstr "bleu"\n')
            with open(rules_path, "w", encoding="utf-8") as handle:
                handle.write("Use imperative tone.\n")

            detected_glossary, detected_rules = process_gui.detect_default_resource_paths(
                "fr",
                base_dir=legacy_dir,
            )

            self.assertEqual(detected_glossary, glossary_dir)
            self.assertEqual(detected_rules, rules_path)
        finally:
            for path in (glossary_file, rules_path):
                if os.path.exists(path):
                    os.remove(path)
            if os.path.isdir(glossary_dir):
                os.rmdir(glossary_dir)
            if os.path.isdir(data_dir):
                os.removedirs(data_dir)

    def test_detect_default_resource_paths_prefers_glossary_po_over_directory_bundle(self):
        data_dir = os.path.join(os.getcwd(), "_tmp_gui_glossary_pref", "data", "locales", "fr")
        legacy_dir = os.path.join(os.getcwd(), "_tmp_gui_glossary_pref")
        glossary_path = os.path.join(data_dir, "glossary.po")
        glossary_bundle_path = os.path.join(data_dir, "glossary", "common.po")
        rules_path = os.path.join(data_dir, "rules.md")
        try:
            os.makedirs(os.path.dirname(glossary_bundle_path), exist_ok=True)
            with open(glossary_path, "w", encoding="utf-8") as handle:
                handle.write('msgid "save"\nmsgstr "enregistrer"\n')
            with open(glossary_bundle_path, "w", encoding="utf-8") as handle:
                handle.write('msgid "save"\nmsgstr "enregistrer"\n')
            with open(rules_path, "w", encoding="utf-8") as handle:
                handle.write("Keep labels short.\n")

            detected_glossary, detected_rules = process_gui.detect_default_resource_paths(
                "fr",
                base_dir=legacy_dir,
            )

            self.assertEqual(detected_glossary, glossary_path)
            self.assertEqual(detected_rules, rules_path)
        finally:
            for path in (glossary_path, glossary_bundle_path, rules_path):
                if os.path.exists(path):
                    os.remove(path)
            glossary_parent = os.path.dirname(glossary_bundle_path)
            if os.path.isdir(glossary_parent):
                os.rmdir(glossary_parent)
            if os.path.isdir(data_dir):
                os.removedirs(data_dir)

    def test_detect_default_glossary_source_path_prefers_canonical_jsonl(self):
        glossary_dir = os.path.join(os.getcwd(), "_tmp_gui_glossary", "data", "glossary")
        base_dir = os.path.join(os.getcwd(), "_tmp_gui_glossary")
        glossary_path = os.path.join(glossary_dir, "glossary.jsonl")
        try:
            os.makedirs(glossary_dir, exist_ok=True)
            with open(glossary_path, "w", encoding="utf-8") as handle:
                handle.write("{}\n")

            detected = process_gui.detect_default_glossary_source_path(base_dir)

            self.assertEqual(detected, glossary_path)
        finally:
            if os.path.exists(glossary_path):
                os.remove(glossary_path)
            if os.path.isdir(glossary_dir):
                os.removedirs(glossary_dir)

    def test_choose_resource_field_value_preserves_manual_path(self):
        value, auto_value = process_gui.choose_resource_field_value(
            current_value="C:\\manual\\glossary.po",
            previous_auto_value="C:\\auto\\old.txt",
            new_auto_value="C:\\auto\\new.txt",
        )

        self.assertEqual(value, "C:\\manual\\glossary.po")
        self.assertEqual(auto_value, "C:\\auto\\old.txt")

    def test_choose_resource_field_value_updates_auto_managed_path(self):
        value, auto_value = process_gui.choose_resource_field_value(
            current_value="C:\\auto\\old.txt",
            previous_auto_value="C:\\auto\\old.txt",
            new_auto_value="C:\\auto\\new.txt",
        )

        self.assertEqual(value, "C:\\auto\\new.txt")
        self.assertEqual(auto_value, "C:\\auto\\new.txt")

    def test_validate_config_requires_input_file_and_api_key(self):
        config = process_gui.ProcessGuiConfig(input_file="")

        errors = process_gui.validate_process_gui_config(config, environ={})

        self.assertIn("Input file is required.", errors)
        self.assertTrue(any("GOOGLE_API_KEY is not set" in item for item in errors))

    def test_validate_config_requires_api_key_for_gemini_vertex(self):
        input_path = os.path.join(os.getcwd(), "_tmp_gui_vertex.po")
        try:
            with open(input_path, "w", encoding="utf-8") as handle:
                handle.write('msgid "Open"\nmsgstr ""\n')

            config = process_gui.ProcessGuiConfig(
                input_file=input_path,
                provider="gemini",
                gemini_backend="vertex",
            )

            errors = process_gui.validate_process_gui_config(config, environ={})

            self.assertTrue(any("GOOGLE_API_KEY is not set" in item for item in errors))
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)

    def test_validate_config_rejects_non_global_location_for_gemini_vertex(self):
        input_path = os.path.join(os.getcwd(), "_tmp_gui_vertex_location.po")
        try:
            with open(input_path, "w", encoding="utf-8") as handle:
                handle.write('msgid "Open"\nmsgstr ""\n')

            config = process_gui.ProcessGuiConfig(
                input_file=input_path,
                provider="gemini",
                gemini_backend="vertex",
                google_cloud_location="us-central1",
                api_key="vertex-key",
            )

            errors = process_gui.validate_process_gui_config(config, environ={})

            self.assertIn(
                "Gemini Vertex API-key mode currently supports only the global endpoint.",
                errors,
            )
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)

    def test_validate_config_rejects_bad_optional_values(self):
        input_path = os.path.join(os.getcwd(), "_tmp_gui_input.po")
        try:
            with open(input_path, "w", encoding="utf-8") as handle:
                handle.write('msgid "Open"\nmsgstr ""\n')

            config = process_gui.ProcessGuiConfig(
                input_file=input_path,
                batch_size="abc",
                parallel_requests="0",
                seed="-1",
                glossary_path="missing-glossary.po",
                rules_path="missing-rules.md",
                api_key="test-key",
            )

            errors = process_gui.validate_process_gui_config(config, environ={})

            self.assertIn("Batch size must be a whole number.", errors)
            self.assertIn("Parallel requests must be greater than 0.", errors)
            self.assertIn("Seed must be 0 or greater.", errors)
            self.assertIn("Glossary file or directory does not exist: missing-glossary.po", errors)
            self.assertIn("Rules file does not exist: missing-rules.md", errors)
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)

    def test_validate_config_accepts_vocabulary_directory(self):
        input_path = os.path.join(os.getcwd(), "_tmp_gui_glossary_dir.po")
        glossary_dir = os.path.join(os.getcwd(), "_tmp_gui_glossary_dir")
        glossary_file = os.path.join(glossary_dir, "colors.txt")
        try:
            with open(input_path, "w", encoding="utf-8") as handle:
                handle.write('msgid "Open"\nmsgstr ""\n')
            os.makedirs(glossary_dir, exist_ok=True)
            with open(glossary_file, "w", encoding="utf-8") as handle:
                handle.write("blue|bleu|adjective|\n")

            config = process_gui.ProcessGuiConfig(
                input_file=input_path,
                glossary_path=glossary_dir,
                api_key="test-key",
            )

            errors = process_gui.validate_process_gui_config(config, environ={})

            self.assertFalse(
                any("Glossary file or directory does not exist" in item for item in errors)
            )
        finally:
            for path in (input_path, glossary_file):
                if os.path.exists(path):
                    os.remove(path)
            if os.path.isdir(glossary_dir):
                os.rmdir(glossary_dir)

    def test_validate_process_config_rejects_mixed_file_types(self):
        input_po = os.path.join(os.getcwd(), "_tmp_gui_input.po")
        input_ts = os.path.join(os.getcwd(), "_tmp_gui_input.ts")
        try:
            with open(input_po, "w", encoding="utf-8") as handle:
                handle.write('msgid "Open"\nmsgstr ""\n')
            with open(input_ts, "w", encoding="utf-8") as handle:
                handle.write("<TS></TS>\n")

            config = process_gui.ProcessGuiConfig(
                input_file=input_po,
                input_files=(input_po, input_ts),
                api_key="test-key",
            )

            errors = process_gui.validate_process_gui_config(config, environ={})

            self.assertIn(
                "Multi-file translation requires all input files to use the same format.",
                errors,
            )
        finally:
            for path in (input_po, input_ts):
                if os.path.exists(path):
                    os.remove(path)

    def test_validate_process_config_accepts_input_directory(self):
        input_dir = os.path.join(os.getcwd(), "_tmp_gui_input_dir")
        input_path = os.path.join(input_dir, "first.po")
        try:
            os.makedirs(input_dir, exist_ok=True)
            with open(input_path, "w", encoding="utf-8") as handle:
                handle.write('msgid "Open"\nmsgstr ""\n')

            config = process_gui.ProcessGuiConfig(
                input_file=input_dir,
                target_lang="fr",
                api_key="test-key",
            )

            errors = process_gui.validate_process_gui_config(config, environ={})

            self.assertEqual(errors, [])
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.isdir(input_dir):
                os.rmdir(input_dir)

    def test_build_process_command_includes_required_and_optional_flags(self):
        input_path = os.path.join(os.getcwd(), "_tmp_gui_input.po")
        script_path = os.path.join(os.getcwd(), "_tmp_process_script.py")
        try:
            with open(input_path, "w", encoding="utf-8") as handle:
                handle.write('msgid "Open"\nmsgstr ""\n')
            with open(script_path, "w", encoding="utf-8") as handle:
                handle.write("print('stub')\n")

            config = process_gui.ProcessGuiConfig(
                input_file=input_path,
                source_lang="en",
                target_lang="fr",
                model="gemini-test",
                thinking_level="low",
                seed="42",
                batch_size="50",
                parallel_requests="3",
                glossary_path=input_path,
                rules_path=script_path,
                rules_str="Use imperative tone.",
                api_key="test-key",
                retranslate_all=True,
            )

            command = process_gui.build_process_command(
                config,
                python_executable="python",
                script_path=script_path,
            )

            self.assertEqual(
                command,
                [
                    "python",
                    "-u",
                    os.path.abspath(script_path),
                    "translate",
                    input_path,
                    "--source-lang",
                    "en",
                    "--target-lang",
                    "fr",
                    "--provider",
                    "gemini",
                    "--model",
                    "gemini-test",
                    "--thinking-level",
                    "low",
                    "--seed",
                    "42",
                    "--gemini-backend",
                    "vertex",
                    "--google-cloud-location",
                    "global",
                    "--batch-size",
                    "50",
                    "--parallel-requests",
                    "3",
                    "--glossary",
                    input_path,
                    "--rules",
                    script_path,
                    "--rules-str",
                    "Use imperative tone.",
                    "--retranslate-all",
                ],
            )
        finally:
            for path in (input_path, script_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_build_process_command_accepts_input_directory(self):
        input_dir = os.path.join(os.getcwd(), "_tmp_gui_process_dir")
        input_path = os.path.join(input_dir, "first.po")
        script_path = os.path.join(os.getcwd(), "_tmp_process_script.py")
        try:
            os.makedirs(input_dir, exist_ok=True)
            with open(input_path, "w", encoding="utf-8") as handle:
                handle.write('msgid "Open"\nmsgstr ""\n')
            with open(script_path, "w", encoding="utf-8") as handle:
                handle.write("print('stub')\n")

            config = process_gui.ProcessGuiConfig(
                input_file=input_dir,
                target_lang="fr",
                api_key="test-key",
            )

            command = process_gui.build_process_command(
                config,
                python_executable="python",
                script_path=script_path,
            )

            self.assertEqual(
                command[:5],
                [
                    "python",
                    "-u",
                    os.path.abspath(script_path),
                    "translate",
                    input_dir,
                ],
            )
        finally:
            for path in (input_path, script_path):
                if os.path.exists(path):
                    os.remove(path)
            if os.path.isdir(input_dir):
                os.rmdir(input_dir)

    def test_build_process_command_uses_provider_default_model_when_blank(self):
        input_path = os.path.join(os.getcwd(), "_tmp_gui_default_model.po")
        script_path = os.path.join(os.getcwd(), "_tmp_process_script.py")
        try:
            with open(input_path, "w", encoding="utf-8") as handle:
                handle.write('msgid "Open"\nmsgstr ""\n')
            with open(script_path, "w", encoding="utf-8") as handle:
                handle.write("print('stub')\n")

            config = process_gui.ProcessGuiConfig(
                input_file=input_path,
                target_lang="fr",
                provider="openai",
                model="",
                api_key="openai-key",
            )

            command = process_gui.build_process_command(
                config,
                python_executable="python",
                script_path=script_path,
            )

            self.assertIn("--model", command)
            self.assertEqual(command[command.index("--model") + 1], "gpt-5-mini")
        finally:
            for path in (input_path, script_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_build_process_command_supports_multiple_input_files(self):
        input_one = os.path.join(os.getcwd(), "_tmp_gui_one.po")
        input_two = os.path.join(os.getcwd(), "_tmp_gui_two.po")
        script_path = os.path.join(os.getcwd(), "_tmp_process_script.py")
        try:
            for path in (input_one, input_two):
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write('msgid "Open"\nmsgstr ""\n')
            with open(script_path, "w", encoding="utf-8") as handle:
                handle.write("print('stub')\n")

            config = process_gui.ProcessGuiConfig(
                input_file=input_one,
                input_files=(input_one, input_two),
                source_lang="en",
                target_lang="fr",
                model="gemini-test",
                api_key="test-key",
            )

            command = process_gui.build_process_command(
                config,
                python_executable="python",
                script_path=script_path,
            )

            self.assertEqual(
                command,
                [
                    "python",
                    "-u",
                    os.path.abspath(script_path),
                    "translate",
                    input_one,
                    input_two,
                    "--source-lang",
                    "en",
                    "--target-lang",
                    "fr",
                    "--provider",
                    "gemini",
                    "--model",
                    "gemini-test",
                    "--gemini-backend",
                    "vertex",
                    "--google-cloud-location",
                    "global",
                    "--batch-size",
                    "50",
                    "--parallel-requests",
                    "1",
                ],
            )
        finally:
            for path in (input_one, input_two, script_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_build_process_command_includes_flex_for_supported_provider(self):
        input_path = os.path.join(os.getcwd(), "_tmp_gui_flex.po")
        script_path = os.path.join(os.getcwd(), "_tmp_process_script.py")
        try:
            with open(input_path, "w", encoding="utf-8") as handle:
                handle.write('msgid "Open"\nmsgstr ""\n')
            with open(script_path, "w", encoding="utf-8") as handle:
                handle.write("print('stub')\n")

            config = process_gui.ProcessGuiConfig(
                input_file=input_path,
                target_lang="fr",
                provider="openai",
                model="gpt-4.1-mini",
                api_key="test-key",
                flex_mode=True,
            )

            command = process_gui.build_process_command(
                config,
                python_executable="python",
                script_path=script_path,
            )

            self.assertIn("--flex", command)
        finally:
            for path in (input_path, script_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_build_process_command_includes_warnings_report_flag(self):
        input_path = os.path.join(os.getcwd(), "_tmp_gui_warn.po")
        script_path = os.path.join(os.getcwd(), "_tmp_process_script.py")
        try:
            with open(input_path, "w", encoding="utf-8") as handle:
                handle.write('msgid "Open"\nmsgstr ""\n')
            with open(script_path, "w", encoding="utf-8") as handle:
                handle.write("print('stub')\n")

            config = process_gui.ProcessGuiConfig(
                input_file=input_path,
                target_lang="fr",
                provider="gemini",
                model="gemini-test",
                api_key="test-key",
                warnings_report=True,
            )

            command = process_gui.build_process_command(
                config,
                python_executable="python",
                script_path=script_path,
            )

            self.assertIn("--warnings-report", command)
        finally:
            for path in (input_path, script_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_build_process_command_includes_gemini_vertex_args(self):
        input_path = os.path.join(os.getcwd(), "_tmp_gui_vertex_args.po")
        script_path = os.path.join(os.getcwd(), "_tmp_process_script.py")
        try:
            with open(input_path, "w", encoding="utf-8") as handle:
                handle.write('msgid "Open"\nmsgstr ""\n')
            with open(script_path, "w", encoding="utf-8") as handle:
                handle.write("print('stub')\n")

            config = process_gui.ProcessGuiConfig(
                input_file=input_path,
                target_lang="fr",
                provider="gemini",
                gemini_backend="vertex",
                google_cloud_location="global",
                model="gemini-3-flash-preview",
                api_key="vertex-key",
            )

            command = process_gui.build_process_command(
                config,
                python_executable="python",
                script_path=script_path,
            )

            self.assertIn("--gemini-backend", command)
            self.assertIn("vertex", command)
            self.assertIn("--google-cloud-location", command)
            self.assertIn("global", command)
        finally:
            for path in (input_path, script_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_build_process_env_prefers_gui_api_key(self):
        config = process_gui.ProcessGuiConfig(
            input_file="dummy.po",
            api_key="gui-key",
        )

        env = process_gui.build_process_env(
            config,
            base_env={"GOOGLE_API_KEY": "env-key", "PATH": "x"},
        )

        self.assertEqual(env["GOOGLE_API_KEY"], "gui-key")
        self.assertEqual(env["PATH"], "x")

    def test_build_process_env_sets_gemini_vertex_env(self):
        config = process_gui.ProcessGuiConfig(
            input_file="dummy.po",
            provider="gemini",
            gemini_backend="vertex",
            google_cloud_location="global",
            api_key="vertex-key",
        )

        env = process_gui.build_process_env(
            config,
            base_env={"PATH": "x"},
        )

        self.assertEqual(env["GOOGLE_API_KEY"], "vertex-key")
        self.assertEqual(env["GOOGLE_GENAI_USE_VERTEXAI"], "true")
        self.assertEqual(env["GOOGLE_CLOUD_LOCATION"], "global")

    def test_build_process_env_uses_openai_api_key_env_for_openai_provider(self):
        config = process_gui.ProcessGuiConfig(
            input_file="dummy.po",
            provider="openai",
            api_key="openai-key",
        )

        env = process_gui.build_process_env(
            config,
            base_env={"PATH": "x"},
        )

        self.assertEqual(env["OPENAI_API_KEY"], "openai-key")
        self.assertEqual(env["PATH"], "x")

    def test_build_process_env_uses_anthropic_api_key_env_for_anthropic_provider(self):
        config = process_gui.ProcessGuiConfig(
            input_file="dummy.po",
            provider="anthropic",
            api_key="anthropic-key",
        )

        env = process_gui.build_process_env(
            config,
            base_env={"PATH": "x"},
        )

        self.assertEqual(env["ANTHROPIC_API_KEY"], "anthropic-key")
        self.assertEqual(env["PATH"], "x")

    def test_build_extract_command_includes_extract_specific_flags(self):
        input_path = os.path.join(os.getcwd(), "_tmp_gui_extract.po")
        script_path = os.path.join(os.getcwd(), "_tmp_extract_script.py")
        try:
            with open(input_path, "w", encoding="utf-8") as handle:
                handle.write('msgid "Open"\nmsgstr ""\n')
            with open(script_path, "w", encoding="utf-8") as handle:
                handle.write("print('stub')\n")

            config = process_gui.ExtractGuiConfig(
                input_file=input_path,
                source_lang="en",
                target_lang="fr",
                model="gemini-test",
                thinking_level="minimal",
                batch_size="80",
                parallel_requests="4",
                glossary_path=input_path,
                api_key="test-key",
                mode="all",
                out_format="json",
                out_path="terms.json",
                max_terms_per_batch="25",
                max_attempts="7",
            )

            command = process_gui.build_extract_command(
                config,
                python_executable="python",
                script_path=script_path,
            )

            self.assertEqual(
                command,
                [
                    "python",
                    "-u",
                    os.path.abspath(script_path),
                    "extract-terms",
                    input_path,
                    "--source-lang",
                    "en",
                    "--target-lang",
                    "fr",
                    "--provider",
                    "gemini",
                    "--model",
                    "gemini-test",
                    "--thinking-level",
                    "minimal",
                    "--gemini-backend",
                    "vertex",
                    "--google-cloud-location",
                    "global",
                    "--batch-size",
                    "80",
                    "--parallel-requests",
                    "4",
                    "--glossary",
                    input_path,
                    "--mode",
                    "all",
                    "--out-format",
                    "json",
                    "--out",
                    "terms.json",
                    "--max-terms-per-batch",
                    "25",
                    "--max-attempts",
                    "7",
                ],
            )
        finally:
            for path in (input_path, script_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_validate_local_extract_config_does_not_require_api_key(self):
        input_path = os.path.join(os.getcwd(), "_tmp_gui_local_extract.po")
        try:
            with open(input_path, "w", encoding="utf-8") as handle:
                handle.write('msgid "Open"\nmsgstr ""\n')

            config = process_gui.LocalExtractGuiConfig(
                input_file=input_path,
                source_lang="en",
            )

            errors = process_gui.validate_local_extract_gui_config(config)

            self.assertEqual(errors, [])
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)

    def test_validate_local_extract_config_accepts_directory_input(self):
        input_dir = os.path.join(os.getcwd(), "_tmp_gui_local_extract_dir")
        try:
            os.makedirs(input_dir, exist_ok=True)

            config = process_gui.LocalExtractGuiConfig(
                input_file=input_dir,
                source_lang="en",
            )

            errors = process_gui.validate_local_extract_gui_config(config)

            self.assertEqual(errors, [])
        finally:
            if os.path.isdir(input_dir):
                os.rmdir(input_dir)

    def test_local_extract_browse_file_uses_json_dialog_in_to_po_mode(self):
        tab = object.__new__(process_gui.LocalExtractToolTab)
        tab.to_po_var = _FakeVar(True)
        tab.input_file_var = _FakeVar("")

        with patch(
            "process_gui.filedialog.askopenfilename",
            return_value=r"C:\tmp\terms.json",
        ) as askopenfilename:
            tab._browse_local_extract_file()

        askopenfilename.assert_called_once_with(
            title="Select local extraction JSON file",
            filetypes=process_gui.JSON_FILETYPES,
        )
        self.assertEqual(tab.input_file_var.get(), r"C:\tmp\terms.json")

    def test_local_extract_browse_folder_does_not_fallback_to_file_dialog(self):
        tab = object.__new__(process_gui.LocalExtractToolTab)
        tab.to_po_var = _FakeVar(False)
        tab.input_file_var = _FakeVar("")

        with (
            patch("process_gui.filedialog.askdirectory", return_value="") as askdirectory,
            patch("process_gui.filedialog.askopenfilename") as askopenfilename,
        ):
            tab._browse_local_extract_folder()

        askdirectory.assert_called_once_with(
            title="Select source folder for local extraction",
            mustexist=True,
        )
        askopenfilename.assert_not_called()
        self.assertEqual(tab.input_file_var.get(), "")

    def test_refresh_local_mode_controls_updates_input_buttons_for_json_mode(self):
        tab = object.__new__(process_gui.LocalExtractToolTab)
        tab.to_po_var = _FakeVar(True)
        tab.mode_combo = _FakeWidget()
        tab.max_length_combo = _FakeWidget()
        tab.include_rejected_button = _FakeWidget()
        tab.include_borderline_button = _FakeWidget()
        tab.input_file_button = _FakeWidget()
        tab.input_folder_button = _FakeWidget()
        tab.include_rejected_var = _FakeVar(True)

        tab._refresh_local_mode_controls()

        self.assertEqual(tab.input_file_button.config["text"], "JSON file")
        self.assertEqual(tab.input_folder_button.config["state"], "disabled")
        self.assertEqual(tab.include_rejected_var.get(), False)

    def test_validate_local_extract_config_rejects_directory_in_json_to_po_mode(self):
        input_dir = os.path.join(os.getcwd(), "_tmp_gui_local_extract_json_dir")
        try:
            os.makedirs(input_dir, exist_ok=True)

            config = process_gui.LocalExtractGuiConfig(
                input_file=input_dir,
                to_po=True,
            )

            errors = process_gui.validate_local_extract_gui_config(config)

            self.assertIn(f"Input file does not exist: {input_dir}", errors)
        finally:
            if os.path.isdir(input_dir):
                os.rmdir(input_dir)

    def test_validate_local_extract_config_rejects_missing_glossary_source(self):
        input_path = os.path.join(os.getcwd(), "_tmp_gui_local_extract.po")
        try:
            with open(input_path, "w", encoding="utf-8") as handle:
                handle.write('msgid "Open"\nmsgstr ""\n')

            config = process_gui.LocalExtractGuiConfig(
                input_file=input_path,
                source_lang="en",
                glossary_source_path="missing-glossary.jsonl",
            )

            errors = process_gui.validate_local_extract_gui_config(config)

            self.assertIn(
                "Glossary source file does not exist: missing-glossary.jsonl",
                errors,
            )
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)

    def test_build_local_extract_command_includes_local_flags(self):
        input_path = os.path.join(os.getcwd(), "_tmp_gui_local_extract.po")
        script_path = os.path.join(os.getcwd(), "_tmp_local_extract_script.py")
        try:
            with open(input_path, "w", encoding="utf-8") as handle:
                handle.write('msgid "Open"\nmsgstr ""\n')
            with open(script_path, "w", encoding="utf-8") as handle:
                handle.write("print('stub')\n")

            config = process_gui.LocalExtractGuiConfig(
                input_file=input_path,
                source_lang="en",
                glossary_source_path=input_path,
                mode="all",
                max_length="3",
                out_path="terms.json",
                include_rejected=True,
            )

            command = process_gui.build_local_extract_command(
                config,
                python_executable="python",
                script_path=script_path,
            )

            self.assertEqual(
                command,
                [
                    "python",
                    "-u",
                    os.path.abspath(script_path),
                    "extract-terms-local",
                    input_path,
                    "--source-lang",
                    "en",
                    "--mode",
                    "all",
                    "--max-length",
                    "3",
                    "--glossary-source",
                    input_path,
                    "--include-rejected",
                    "--out",
                    "terms.json",
                ],
            )
        finally:
            for path in (input_path, script_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_build_local_extract_command_supports_json_to_po_mode(self):
        input_path = os.path.join(os.getcwd(), "_tmp_gui_local_extract.prototype-missing-terms.json")
        script_path = os.path.join(os.getcwd(), "_tmp_local_extract_script.py")
        try:
            with open(input_path, "w", encoding="utf-8") as handle:
                handle.write("{}\n")
            with open(script_path, "w", encoding="utf-8") as handle:
                handle.write("print('stub')\n")

            config = process_gui.LocalExtractGuiConfig(
                input_file=input_path,
                to_po=True,
                include_borderline=True,
                out_path="terms.po",
            )

            command = process_gui.build_local_extract_command(
                config,
                python_executable="python",
                script_path=script_path,
            )

            self.assertEqual(
                command,
                [
                    "python",
                    "-u",
                    os.path.abspath(script_path),
                    "extract-terms-local",
                    input_path,
                    "--to-po",
                    "--include-borderline",
                    "--out",
                    "terms.po",
                ],
            )
        finally:
            for path in (input_path, script_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_build_local_extract_command_supports_one_shot_json_and_po_output(self):
        input_path = os.path.join(os.getcwd(), "_tmp_gui_local_extract.po")
        script_path = os.path.join(os.getcwd(), "_tmp_local_extract_script.py")
        try:
            with open(input_path, "w", encoding="utf-8") as handle:
                handle.write('msgid "Access token"\nmsgstr ""\n')
            with open(script_path, "w", encoding="utf-8") as handle:
                handle.write("print('stub')\n")

            config = process_gui.LocalExtractGuiConfig(
                input_file=input_path,
                source_lang="en",
                also_po=True,
                include_borderline=True,
                out_path="terms.json",
            )

            command = process_gui.build_local_extract_command(
                config,
                python_executable="python",
                script_path=script_path,
            )

            self.assertEqual(
                command,
                [
                    "python",
                    "-u",
                    os.path.abspath(script_path),
                    "extract-terms-local",
                    input_path,
                    "--source-lang",
                    "en",
                    "--mode",
                    "missing",
                    "--max-length",
                    "1",
                    "--also-po",
                    "--include-borderline",
                    "--out",
                    "terms.json",
                ],
            )
        finally:
            for path in (input_path, script_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_build_check_command_includes_check_specific_flags(self):
        input_path = os.path.join(os.getcwd(), "_tmp_gui_check.po")
        script_path = os.path.join(os.getcwd(), "_tmp_check_script.py")
        try:
            with open(input_path, "w", encoding="utf-8") as handle:
                handle.write('msgid "Open"\nmsgstr "Ouvrir"\n')
            with open(script_path, "w", encoding="utf-8") as handle:
                handle.write("print('stub')\n")

            config = process_gui.CheckGuiConfig(
                input_file=input_path,
                source_lang="en",
                target_lang="fr",
                model="gemini-test",
                thinking_level="high",
                batch_size="60",
                parallel_requests="2",
                glossary_path=input_path,
                rules_path=script_path,
                rules_str="Review carefully.",
                api_key="test-key",
                num_messages="10",
                out_path="report.json",
                include_ok=True,
                max_attempts="9",
            )

            command = process_gui.build_check_command(
                config,
                python_executable="python",
                script_path=script_path,
            )

            self.assertEqual(
                command,
                [
                    "python",
                    "-u",
                    os.path.abspath(script_path),
                    "check",
                    input_path,
                    "--source-lang",
                    "en",
                    "--target-lang",
                    "fr",
                    "--provider",
                    "gemini",
                    "--model",
                    "gemini-test",
                    "--thinking-level",
                    "high",
                    "--gemini-backend",
                    "vertex",
                    "--google-cloud-location",
                    "global",
                    "--batch-size",
                    "60",
                    "--parallel-requests",
                    "2",
                    "--glossary",
                    input_path,
                    "--rules",
                    script_path,
                    "--rules-str",
                    "Review carefully.",
                    "--probe",
                    "10",
                    "--out",
                    "report.json",
                    "--include-ok",
                    "--max-attempts",
                    "9",
                ],
            )
        finally:
            for path in (input_path, script_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_validate_revise_config_requires_source_file_for_strings(self):
        input_path = os.path.join(os.getcwd(), "_tmp_gui_revise.strings")
        try:
            with open(input_path, "w", encoding="utf-8") as handle:
                handle.write('/* "app|Name" = "Viewer"; */\n')

            config = process_gui.ReviseGuiConfig(
                input_file=input_path,
                instruction="Change viewer to browser",
                api_key="test-key",
            )

            errors = process_gui.validate_revise_gui_config(config, environ={})

            self.assertIn(
                "Source file is required for Android .xml, .strings, .resx, and .txt revision runs.",
                errors,
            )
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)

    def test_validate_revise_config_accepts_xliff_without_source_file(self):
        input_path = os.path.join(os.getcwd(), "_tmp_gui_revise.xliff")
        try:
            with open(input_path, "w", encoding="utf-8", newline="") as handle:
                handle.write(
                    """<?xml version="1.0" encoding="utf-8"?>
<xliff version="1.2" xmlns="urn:oasis:names:tc:xliff:document:1.2">
  <file original="messages.json">
    <body>
      <trans-unit id="openAction" resname="openAction">
        <source>Open</source>
        <target state="translated">Ouvrir</target>
      </trans-unit>
    </body>
  </file>
</xliff>
"""
                )

            config = process_gui.ReviseGuiConfig(
                input_file=input_path,
                target_lang="fr",
                instruction="Change to imperative mood",
                api_key="test-key",
            )

            errors = process_gui.validate_revise_gui_config(config, environ={})

            self.assertNotIn(
                "Source file is required for Android .xml, .strings, .resx, and .txt revision runs.",
                errors,
            )
            self.assertEqual(errors, [])
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)

    def test_build_revise_command_includes_revision_specific_flags(self):
        input_path = os.path.join(os.getcwd(), "_tmp_gui_revise.ai-translated.strings")
        source_path = os.path.join(os.getcwd(), "_tmp_gui_revise.strings")
        script_path = os.path.join(os.getcwd(), "_tmp_revise_script.py")
        rules_path = os.path.join(os.getcwd(), "_tmp_revise_rules.md")
        try:
            with open(input_path, "w", encoding="utf-8") as handle:
                handle.write('"app|Name" = "Viewer";\n')
            with open(source_path, "w", encoding="utf-8") as handle:
                handle.write('/* "app|Name" = "Document Viewer"; */\n')
            with open(script_path, "w", encoding="utf-8") as handle:
                handle.write("print('stub')\n")
            with open(rules_path, "w", encoding="utf-8") as handle:
                handle.write("Keep labels short.\n")

            config = process_gui.ReviseGuiConfig(
                input_file=input_path,
                source_file=source_path,
                source_lang="en",
                target_lang="fr",
                model="gemini-test",
                thinking_level="medium",
                batch_size="40",
                parallel_requests="2",
                glossary_path=source_path,
                rules_path=rules_path,
                rules_str="Use approved wording.",
                api_key="test-key",
                instruction="Change viewer to browser",
                num_messages="10",
                out_path="revised.strings",
                max_attempts="7",
                dry_run=True,
            )

            command = process_gui.build_revise_command(
                config,
                python_executable="python",
                script_path=script_path,
            )

            self.assertEqual(
                command,
                [
                    "python",
                    "-u",
                    os.path.abspath(script_path),
                    "revise",
                    input_path,
                    "--instruction",
                    "Change viewer to browser",
                    "--source-lang",
                    "en",
                    "--target-lang",
                    "fr",
                    "--provider",
                    "gemini",
                    "--model",
                    "gemini-test",
                    "--thinking-level",
                    "medium",
                    "--gemini-backend",
                    "vertex",
                    "--google-cloud-location",
                    "global",
                    "--batch-size",
                    "40",
                    "--parallel-requests",
                    "2",
                    "--glossary",
                    source_path,
                    "--source-file",
                    source_path,
                    "--rules",
                    rules_path,
                    "--rules-str",
                    "Use approved wording.",
                    "--probe",
                    "10",
                    "--out",
                    "revised.strings",
                    "--dry-run",
                    "--max-attempts",
                    "7",
                ],
            )
        finally:
            for path in (input_path, source_path, script_path, rules_path):
                if os.path.exists(path):
                    os.remove(path)

    def test_build_run_log_path_uses_logs_dir_and_timestamp(self):
        path = process_gui.build_run_log_path(
            "revise",
            r"C:\tmp\demo file.po",
            base_dir=os.getcwd(),
            now=datetime(2026, 3, 15, 12, 34, 56),
        )

        self.assertEqual(
            path,
            os.path.join(os.getcwd(), "logs", "revise-demo-file-20260315-123456.log"),
        )

    def test_widget_supports_clipboard_for_entry_and_text_classes(self):
        self.assertTrue(process_gui.widget_supports_clipboard("TEntry"))
        self.assertTrue(process_gui.widget_supports_clipboard("Text"))
        self.assertFalse(process_gui.widget_supports_clipboard("Button"))

    def test_widget_is_editable_respects_disabled_and_readonly_states(self):
        self.assertTrue(process_gui.widget_is_editable("TEntry", "normal"))
        self.assertFalse(process_gui.widget_is_editable("TEntry", "readonly"))
        self.assertFalse(process_gui.widget_is_editable("Text", "disabled"))

    def test_parse_progress_percent_supports_percent_and_batches(self):
        self.assertEqual(
            process_gui.parse_progress_percent("Progress: 42.5% (85/200), completed batches: 2/5"),
            42.5,
        )
        self.assertAlmostEqual(
            process_gui.parse_progress_percent("Progress: completed batches 1/4 (latest: 1/4), raw terms collected: 10"),
            25.0,
        )
        self.assertEqual(
            process_gui.parse_progress_percent("Translation QA complete."),
            100.0,
        )
        self.assertIsNone(process_gui.parse_progress_percent("Startup configuration:"))


if __name__ == "__main__":
    unittest.main()
