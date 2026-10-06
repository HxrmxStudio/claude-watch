"""Tests for caption parsing and subtitle selection."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from download import _pick_subtitle  # noqa: E402
from transcribe import parse_vtt  # noqa: E402

# Shape of YouTube auto-captions: every cue repeats the previous cue's last
# line as its first line, with 10 ms "settle" cues in between. <SP> stands for
# the whitespace-only placeholder lines YouTube emits (kept explicit here so
# editors that trim trailing whitespace cannot change the fixture).
ROLLING_AUTO_VTT = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:01.630 align:start position:0%
<SP>
Hello<00:00:00.120><c> friends,</c><00:00:00.480><c> it</c><00:00:00.600><c> occurs</c>

00:00:01.630 --> 00:00:01.640 align:start position:0%
Hello friends, it occurs
<SP>

00:00:01.640 --> 00:00:04.350 align:start position:0%
Hello friends, it occurs
to<00:00:02.200><c> me</c><00:00:02.880><c> that</c>

00:00:04.350 --> 00:00:04.360 align:start position:0%
to me that
<SP>

00:00:04.360 --> 00:00:06.350 align:start position:0%
to me that
I've<00:00:04.920><c> never</c><00:00:05.080><c> made</c><00:00:05.240><c> a</c><00:00:05.680><c> tutorial.</c>
""".replace("<SP>", " ")

# Manual captions never overlap, and real speech can repeat a word across a
# cue boundary ("think that / that takes"); that repetition must survive.
MANUAL_VTT = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:03.030
it's easy to think that

00:00:03.030 --> 00:00:05.070
that takes a ton of skills.
"""


class TestParseVtt(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="watch-vtt-test-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _parse(self, body: str) -> str:
        path = self.tmp / "captions.vtt"
        path.write_text(body, encoding="utf-8")
        return " ".join(segment["text"] for segment in parse_vtt(str(path)))

    def test_rolling_auto_captions_are_not_duplicated(self):
        text = self._parse(ROLLING_AUTO_VTT)
        self.assertEqual(
            text, "Hello friends, it occurs to me that I've never made a tutorial."
        )

    def test_manual_captions_keep_real_repeated_words(self):
        text = self._parse(MANUAL_VTT)
        self.assertEqual(text, "it's easy to think that that takes a ton of skills.")

    def test_empty_cue_with_spaced_separator_does_not_swallow_next_cue(self):
        body = (
            "WEBVTT\n\n1\n00:00:01.000 --> 00:00:02.000\n \n"
            "2\n00:00:02.000 --> 00:00:03.000\nSecond cue\n"
        )
        path = self.tmp / "captions.vtt"
        path.write_text(body, encoding="utf-8")
        self.assertEqual(
            parse_vtt(str(path)), [{"start": 2.0, "end": 3.0, "text": "Second cue"}],
        )

    def test_spaced_separators_without_cue_ids_keep_every_cue(self):
        # YouTube cues carry no ids; whitespace-only separators (space, tab)
        # and CR-only line endings must not swallow the following cue.
        for separator, newline in ((" ", "\n"), ("\t", "\n"), (" ", "\r")):
            body = newline.join([
                "WEBVTT", "",
                "00:00:01.000 --> 00:00:02.000", separator,
                "00:00:02.000 --> 00:00:03.000", "Second cue", "",
                "00:00:03.000 --> 00:00:04.000", "Third", "",
            ])
            path = self.tmp / "captions.vtt"
            path.write_text(body, encoding="utf-8", newline="")
            with self.subTest(separator=repr(separator), newline=repr(newline)):
                self.assertEqual(
                    [(segment["start"], segment["text"]) for segment in parse_vtt(str(path))],
                    [(2.0, "Second cue"), (3.0, "Third")],
                )

    def test_timestamps_without_hours(self):
        # WebVTT allows MM:SS.mmm; hand-made and third-party files use it.
        text = self._parse("WEBVTT\n\n01:02.500 --> 01:04.000\nShort form\n")
        self.assertEqual(text, "Short form")
        path = self.tmp / "captions.vtt"
        self.assertEqual(parse_vtt(str(path))[0]["start"], 62.5)

    def test_skips_several_placeholder_lines(self):
        text = self._parse("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n \n \nhello\n")
        self.assertEqual(text, "hello")

    def test_decodes_html_entities(self):
        text = self._parse("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nR&amp;D &gt; hype\n")
        self.assertEqual(text, "R&D > hype")

    def test_escaped_line_breaks_become_spaces(self):
        # Manual YouTube tracks can carry "&lt;br&gt;"; decoding must not leave a
        # literal tag gluing words, while real escaped text like "&lt;b&gt;" stays.
        text = self._parse(
            "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n"
            "the&lt;br&gt;image and&lt;BR /&gt;more, keep &lt;b&gt;literal\n"
        )
        self.assertEqual(text, "the image and more, keep <b>literal")

    def test_raw_line_breaks_become_spaces(self):
        text = self._parse("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\na<br>b<BR/>c <i>d</i>\n")
        self.assertEqual(text, "a b c d")

    def test_non_breaking_spaces_collapse(self):
        text = self._parse("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nsee&nbsp; their&nbsp;&nbsp;\n")
        self.assertEqual(text, "see their")

    def test_rolling_segments_keep_start_times(self):
        path = self.tmp / "captions.vtt"
        path.write_text(ROLLING_AUTO_VTT, encoding="utf-8")
        starts = [segment["start"] for segment in parse_vtt(str(path))]
        self.assertEqual(starts, sorted(starts))
        self.assertEqual(starts[0], 0.0)


class TestPickSubtitle(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="watch-subs-test-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, *names: str, manual: tuple[str, ...] = ()) -> None:
        for name in names:
            (self.tmp / name).write_text("WEBVTT\n", encoding="utf-8")
        info = {"subtitles": {lang: [] for lang in manual}, "automatic_captions": {}}
        (self.tmp / "video.info.json").write_text(json.dumps(info), encoding="utf-8")

    def test_prefers_manual_over_auto_original(self):
        # Sorted by name, video.en-orig.vtt comes first; it is the auto track.
        self._write("video.en-orig.vtt", "video.en.vtt", manual=("en",))
        self.assertEqual(_pick_subtitle(self.tmp).name, "video.en.vtt")

    def test_prefers_manual_regional_english(self):
        self._write("video.en-GB.vtt", "video.en-orig.vtt", manual=("en-GB",))
        self.assertEqual(_pick_subtitle(self.tmp).name, "video.en-GB.vtt")

    def test_without_manual_prefers_original_language_auto_track(self):
        # With no manual track, "en" is a machine translation of "en-orig".
        self._write("video.en-orig.vtt", "video.en.vtt")
        self.assertEqual(_pick_subtitle(self.tmp).name, "video.en-orig.vtt")

    def test_without_info_json_still_picks_a_track(self):
        (self.tmp / "video.en.vtt").write_text("WEBVTT\n", encoding="utf-8")
        self.assertEqual(_pick_subtitle(self.tmp).name, "video.en.vtt")

    def test_returns_none_without_tracks(self):
        self.assertIsNone(_pick_subtitle(self.tmp))


if __name__ == "__main__":
    unittest.main()
