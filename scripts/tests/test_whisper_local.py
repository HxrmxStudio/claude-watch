"""Tests for the local whisper.cpp backend (no model is run here)."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import whisper  # noqa: E402

# Shape of `whisper-cli -oj` output (offsets in milliseconds).
WHISPER_CPP_JSON = {
    "result": {"language": "en"},
    "transcription": [
        {"offsets": {"from": 0, "to": 5320}, "text": " Hello, friends."},
        {"offsets": {"from": 5320, "to": 6000}, "text": "   "},
        {"offsets": {"from": 6000, "to": 11520}, "text": " This repo has 162,000 stars."},
    ],
}


class TestWhisperCppParsing(unittest.TestCase):

    def test_segments_convert_milliseconds_and_skip_blank_text(self):
        segments = whisper._segments_from_whisper_cpp(WHISPER_CPP_JSON)
        self.assertEqual(segments, [
            {"start": 0.0, "end": 5.32, "text": "Hello, friends."},
            {"start": 6.0, "end": 11.52, "text": "This repo has 162,000 stars."},
        ])

    def test_words_use_the_same_shape_as_the_cloud_backends(self):
        words = whisper._words_from_whisper_cpp(WHISPER_CPP_JSON)
        self.assertEqual(words[0], {"word": "Hello, friends.", "start": 0.0, "end": 5.32})
        self.assertEqual(len(words), 2)


class TestCloudResponseTimes(unittest.TestCase):

    def test_bad_cloud_times_become_system_exit(self):
        for value in ("nan", "inf", "abc", [1]):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                whisper._segments_from_response({"segments": [{"start": value, "end": 1, "text": "hi"}]})

    def test_valid_cloud_times_parse(self):
        segments = whisper._segments_from_response(
            {"segments": [{"start": 1.234, "end": "2.5", "text": " hi "}]},
        )
        self.assertEqual(segments, [{"start": 1.23, "end": 2.5, "text": "hi"}])


class TestTranscribeLocal(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="watch-local-run-"))
        self.audio = self.tmp / "audio.mp3"
        self.audio.write_bytes(b"fake")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_reads_the_json_whisper_cli_writes_next_to_output_base(self):
        def fake_run(cmd, **_kwargs):
            if cmd[0] == whisper.LOCAL_BINARY:
                # whisper-cli appends ".json" to the -of value as given.
                output_base = cmd[cmd.index("-of") + 1]
                Path(output_base + ".json").write_text(
                    json.dumps(WHISPER_CPP_JSON), encoding="utf-8",
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with mock.patch.object(whisper.subprocess, "run", side_effect=fake_run):
            segments, words = whisper._transcribe_local(self.audio, "model.bin")
        self.assertEqual(len(segments), 2)
        self.assertEqual(words, [])


    def test_unreadable_json_becomes_system_exit(self):
        # Callers only catch SystemExit; any other error would abort /watch
        # instead of falling back to frames-only.
        def fake_run(cmd, **_kwargs):
            if cmd[0] == whisper.LOCAL_BINARY:
                output_base = cmd[cmd.index("-of") + 1]
                Path(output_base + ".json").write_bytes(b'{"transcription": [\xff\xfe')
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with mock.patch.object(whisper.subprocess, "run", side_effect=fake_run), \
                self.assertRaises(SystemExit):
            whisper._transcribe_local(self.audio, "model.bin")


    def test_unexpected_json_shapes_become_system_exit(self):
        shapes = ("[]", "null", '{"transcription": ["x"]}',
                  '{"transcription": [{"offsets": {"from": "abc"}, "text": "hi"}]}',
                  '{"transcription": [{"offsets": {"from": ' + "9" * 400 + '}, "text": "hi"}]}',
                  '{"transcription": [{"offsets": {"from": NaN}, "text": "hi"}]}',
                  '{"transcription": [{"offsets": {"to": Infinity}, "text": "hi"}]}')
        for payload in shapes:
            def fake_run(cmd, payload=payload, **_kwargs):
                if cmd[0] == whisper.LOCAL_BINARY:
                    output_base = cmd[cmd.index("-of") + 1]
                    Path(output_base + ".json").write_text(payload, encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            with self.subTest(payload=payload), \
                    mock.patch.object(whisper.subprocess, "run", side_effect=fake_run), \
                    self.assertRaises(SystemExit):
                whisper._transcribe_local(self.audio, "model.bin")


class TestLocalBackendDetection(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="watch-local-whisper-"))
        self.model = self.tmp / "ggml-test.bin"
        self.model.write_bytes(b"model")
        self.env = mock.patch.dict(os.environ, {"WHISPER_CPP_MODEL": str(self.model)}, clear=False)
        self.env.start()
        for name in ("GROQ_API_KEY", "OPENAI_API_KEY"):
            os.environ.pop(name, None)
        self.no_dotenv = mock.patch.object(whisper, "_dotenv_paths", return_value=[])
        self.no_dotenv.start()

    def tearDown(self):
        self.no_dotenv.stop()
        self.env.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_local_is_used_when_no_cloud_key_and_model_present(self):
        with mock.patch.object(whisper.shutil, "which", return_value="/usr/bin/whisper-cli"):
            self.assertEqual(whisper.load_api_key(), ("local", str(self.model)))

    def test_cloud_key_still_wins_over_local(self):
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test"}), \
                mock.patch.object(whisper.shutil, "which", return_value="/usr/bin/whisper-cli"):
            self.assertEqual(whisper.load_api_key(), ("groq", "gsk_test"))

    def test_local_needs_the_binary(self):
        with mock.patch.object(whisper.shutil, "which", return_value=None):
            self.assertEqual(whisper.load_api_key(), (None, None))

    def test_local_needs_the_model_file(self):
        self.model.unlink()
        with mock.patch.object(whisper.shutil, "which", return_value="/usr/bin/whisper-cli"):
            self.assertEqual(whisper.load_api_key("local"), (None, None))


if __name__ == "__main__":
    unittest.main()
