import os
import unittest
from datetime import datetime

import process_gui


class ProcessGuiSmokeTests(unittest.TestCase):
    def test_summarize_input_files_formats_multi_file_display(self):
        summary = process_gui.summarize_input_files(
            [r"C:\tmp\one.po", r"C:\tmp\two.po", r"C:\tmp\three.po"]
        )

        self.assertEqual(summary, r"C:\tmp\one.po (+2 more)")

    def test_build_system_prompt_preview_for_translate_uses_translation_system_prompt(self):
        preview = process_gui.build_system_prompt_preview("process", "kk")
        self.assertIn("professional software localization translator", preview)
        self.assertIn("Preserve all placeholders EXACTLY", preview)

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
        legacy_dir = os.path.join(os.getcwd(), "_tmp_gui_data")
        os.makedirs(data_dir, exist_ok=True)
        vocab_path = os.path.join(data_dir, "vocab.txt")
        rules_path = os.path.join(data_dir, "rules.md")
        legacy_vocab_path = os.path.join(legacy_dir, "vocab-fr.txt")
        try:
            with open(vocab_path, "w", encoding="utf-8") as handle:
                handle.write("save|enregistrer|verb|\n")
            with open(rules_path, "w", encoding="utf-8") as handle:
                handle.write("Use imperative tone.\n")
            with open(legacy_vocab_path, "w", encoding="utf-8") as handle:
                handle.write("legacy\n")

            detected_vocab, detected_rules = process_gui.detect_default_resource_paths(
                "fr",
                base_dir=legacy_dir,
            )

            self.assertEqual(detected_vocab, vocab_path)
            self.assertEqual(detected_rules, rules_path)
        finally:
            for path in (vocab_path, rules_path, legacy_vocab_path):
                if os.path.exists(path):
                    os.remove(path)
            if os.path.isdir(data_dir):
                os.removedirs(data_dir)

    def test_choose_resource_field_value_preserves_manual_path(self):
        value, auto_value = process_gui.choose_resource_field_value(
            current_value="C:\\manual\\vocab.txt",
            previous_auto_value="C:\\auto\\old.txt",
            new_auto_value="C:\\auto\\new.txt",
        )

        self.assertEqual(value, "C:\\manual\\vocab.txt")
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
                vocab_path="missing-vocab.txt",
                rules_path="missing-rules.md",
                api_key="test-key",
            )

            errors = process_gui.validate_process_gui_config(config, environ={})

            self.assertIn("Batch size must be a whole number.", errors)
            self.assertIn("Parallel requests must be greater than 0.", errors)
            self.assertIn("Vocabulary file does not exist: missing-vocab.txt", errors)
            self.assertIn("Rules file does not exist: missing-rules.md", errors)
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)

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
                batch_size="50",
                parallel_requests="3",
                vocab_path=input_path,
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
                    "--batch-size",
                    "50",
                    "--parallel-requests",
                    "3",
                    "--vocab",
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
                vocab_path=input_path,
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
                    "--batch-size",
                    "80",
                    "--parallel-requests",
                    "4",
                    "--vocab",
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
                target_lang="kk",
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
                target_lang="kk",
            )

            errors = process_gui.validate_local_extract_gui_config(config)

            self.assertEqual(errors, [])
        finally:
            if os.path.isdir(input_dir):
                os.rmdir(input_dir)

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
                target_lang="fr",
                vocab_path=input_path,
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
                    "--target-lang",
                    "fr",
                    "--mode",
                    "all",
                    "--max-length",
                    "3",
                    "--vocab",
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
                target_lang="kk",
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
                    "--target-lang",
                    "kk",
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
                vocab_path=input_path,
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
                    "--batch-size",
                    "60",
                    "--parallel-requests",
                    "2",
                    "--vocab",
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
                vocab_path=source_path,
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
                    "--batch-size",
                    "40",
                    "--parallel-requests",
                    "2",
                    "--vocab",
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
