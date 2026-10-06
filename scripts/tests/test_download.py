"""Tests for the stale yt-dlp hint shown when a download fails."""
import datetime as dt
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from download import stale_ytdlp_hint, ytdlp_age_days  # noqa: E402

TODAY = dt.date(2026, 10, 6)


class TestYtdlpAge(unittest.TestCase):

    def test_parses_release_versions(self):
        self.assertEqual(ytdlp_age_days("2026.07.04", TODAY), 94)

    def test_parses_nightly_versions(self):
        self.assertEqual(ytdlp_age_days("2026.08.19.231500", TODAY), 48)

    def test_unknown_format_returns_none(self):
        self.assertIsNone(ytdlp_age_days("unknown", TODAY))


class TestStaleHint(unittest.TestCase):

    def test_hints_when_older_than_threshold(self):
        hint = stale_ytdlp_hint("2026.07.04", TODAY)
        self.assertIn("94 days old", hint)
        self.assertIn("yt-dlp -U", hint)

    def test_silent_when_recent(self):
        self.assertEqual(stale_ytdlp_hint("2026.09.30", TODAY), "")

    def test_silent_when_version_unknown(self):
        self.assertEqual(stale_ytdlp_hint(None, TODAY), "")


if __name__ == "__main__":
    unittest.main()
