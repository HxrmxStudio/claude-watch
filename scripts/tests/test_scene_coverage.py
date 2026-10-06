"""Tests for repeated-shot removal, gap filling and contact sheets."""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from frames import (  # noqa: E402
    build_contact_sheets, drop_repeated_shots, extract_scene_change, fill_scene_gaps,
    plan_scene_frames,
)

PIXELS = 100


def _screen(value: int) -> bytes:
    return bytes([value] * PIXELS)


def _probes(*stretches: tuple[int, int]) -> list[bytes]:
    """Probe samples built from (grey value, sample count) stretches."""
    return [_screen(value) for value, count in stretches for _ in range(count)]


class TestDropRepeatedShots(unittest.TestCase):

    def test_drops_a_shot_that_returns_to_an_earlier_framing(self):
        # Talking head (A), slide (B), back to the talking head (A).
        probes = _probes((50, 8), (200, 8), (50, 8))
        self.assertEqual(drop_repeated_shots([0.0, 4.0, 8.0], probes, every_seconds=0.5), [0.0, 4.0])

    def test_keeps_forced_times_even_when_repeated(self):
        probes = _probes((50, 8), (200, 8), (50, 8))
        self.assertEqual(
            drop_repeated_shots([0.0, 4.0, 8.0], probes, every_seconds=0.5, keep=(8.0,)),
            [0.0, 4.0, 8.0],
        )

    def test_judges_the_shot_after_its_fade_settles(self):
        # A, B, then a fade back to A: mid-fade the cut looks new, settled it is A.
        probes = _probes((50, 10), (200, 10), (170, 1), (140, 1), (110, 1), (80, 1), (50, 16))
        self.assertEqual(
            drop_repeated_shots([5.0, 10.0], probes, every_seconds=0.5, keep=(0.0,)), [0.0, 5.0],
        )

    def test_keeps_shots_that_differ(self):
        probes = _probes((50, 8), (200, 8), (120, 8))
        self.assertEqual(
            drop_repeated_shots([0.0, 4.0, 8.0], probes, every_seconds=0.5), [0.0, 4.0, 8.0],
        )


class TestFillSceneGaps(unittest.TestCase):

    def test_finds_a_screen_change_inside_a_long_stretch_without_cuts(self):
        # Cuts at 0 and 2 s, then 18 s with no cut but a new screen at 12 s.
        probes = _probes((50, 4), (120, 20), (200, 16))
        fillers = fill_scene_gaps([0.0, 2.0], probes, every_seconds=0.5, max_gap_seconds=10.0)
        self.assertEqual(fillers, [12.0])

    def test_an_animation_gets_only_its_fair_share(self):
        # 18 s with no cut where the screen changes on every sample: the
        # stretch gets its fair share (one per half max gap), not 36 frames.
        probes = _probes((50, 4)) + [_screen(60), _screen(200)] * 18
        fillers = fill_scene_gaps([0.0, 2.0], probes, every_seconds=0.5, max_gap_seconds=8.0)
        self.assertEqual(len(fillers), 4)

    def test_small_changes_inside_a_long_shot_get_no_fillers(self):
        # A gesture in a talking head moves well under GAP_CHANGE of the screen.
        probes = _probes((50, 4), (120, 20)) + [bytes([200] * 10 + [120] * 90)] * 16
        self.assertEqual(
            fill_scene_gaps([0.0, 2.0], probes, every_seconds=0.5, max_gap_seconds=10.0), [],
        )

    def test_filler_lands_after_a_crossfade(self):
        # Slide A, a 2 s fade (every sample moves), then slide B from 12 s.
        probes = _probes((50, 20), (80, 1), (110, 1), (140, 1), (170, 1), (200, 20))
        self.assertEqual(
            fill_scene_gaps([0.0], probes, every_seconds=0.5, max_gap_seconds=4.0), [12.0],
        )

    def test_short_stretches_get_no_fillers(self):
        probes = _probes((50, 4), (120, 20), (200, 16))
        self.assertEqual(
            fill_scene_gaps([0.0, 2.0, 12.0], probes, every_seconds=0.5, max_gap_seconds=10.0), [],
        )


class TestPlanSceneFrames(unittest.TestCase):

    def _ui_edits(self) -> list[bytes]:
        """A 40 s uncut screen recording: a small UI edit (10% of it) every 4 s."""
        probes = []
        for step in range(10):
            screen = [200] * PIXELS
            screen[step * 10:step * 10 + 10] = [40] * 10
            probes += [bytes(screen)] * 8
        return probes

    def test_spare_budget_goes_to_smaller_screen_changes(self):
        times = plan_scene_frames([], self._ui_edits(), duration=40.0, max_frames=20)
        self.assertGreaterEqual(len(times), 8, times)

    def test_tight_budget_is_still_respected(self):
        times = plan_scene_frames([], self._ui_edits(), duration=40.0, max_frames=4)
        self.assertLessEqual(len(times), 4)

    def test_without_probes_cuts_are_planned_as_before(self):
        self.assertEqual(
            plan_scene_frames([5.0, 10.0], [], duration=20.0, max_frames=10, anchors=(7.456,)),
            [0.0, 5.0, 7.46, 10.0],
        )

    def test_a_short_shot_is_not_settled_into_the_next_one(self):
        # Talking head, a 1 s slide at 10 s, then the talking head again. The
        # return cut is dropped as a repeat; the slide must still be the frame.
        probes = _probes((50, 20), (200, 2), (50, 38))
        times = plan_scene_frames([10.0, 11.0], probes, duration=30.0, max_frames=10)
        self.assertEqual(times, [0.0, 10.5])

    def test_cuts_move_past_their_transition(self):
        # A cut at 2 s whose new shot fades in until 4 s.
        probes = _probes((50, 4), (80, 1), (110, 1), (140, 1), (170, 1), (200, 12))
        self.assertEqual(plan_scene_frames([2.0], probes, duration=10.0, max_frames=10), [0.0, 4.0])


def _edited_video(out: Path) -> Path:
    """Cuts between two framings that keep returning, then one long shot whose
    content changes halfway without a cut (a panel moves, like a new slide)."""
    segments = [
        "color=c=red:size=192x108:duration=2",
        "color=c=blue:size=192x108:duration=2",
        "color=c=red:size=192x108:duration=2",
        "color=c=blue:size=192x108:duration=2",
        "color=c=red:size=192x108:duration=2",
        "color=c=gray:size=192x108:duration=10,drawbox=x=10:y=24:w=60:h=60:color=white:t=fill",
        "color=c=gray:size=192x108:duration=10,drawbox=x=120:y=24:w=60:h=60:color=white:t=fill",
    ]
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for segment in segments:
        cmd += ["-f", "lavfi", "-i", segment]
    inputs = "".join(f"[{idx}:v]" for idx in range(len(segments)))
    cmd += [
        "-filter_complex", f"{inputs}concat=n={len(segments)}:v=1:a=0[v]",
        "-map", "[v]", "-pix_fmt", "yuv420p", str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


class TestSceneCoverage(unittest.TestCase):

    def setUp(self):
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not available")
        self.tmp = Path(tempfile.mkdtemp(prefix="watch-coverage-test-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_repeated_shots_dropped_and_uncut_change_found(self):
        video = _edited_video(self.tmp / "edited.mp4")
        frames = extract_scene_change(
            str(video), self.tmp / "frames", resolution=64, max_frames=10, uniform_fallback_min=3,
        )
        times = [frame["timestamp_seconds"] for frame in frames]
        # Red and blue once each, the grey shot, and the moved panel at ~20 s.
        self.assertEqual(len(times), 4, times)
        self.assertTrue(any(19.5 <= time <= 22.0 for time in times), times)

    def test_contact_sheets_cover_every_frame(self):
        video = _edited_video(self.tmp / "edited.mp4")
        frames = extract_scene_change(
            str(video), self.tmp / "frames", resolution=64, max_frames=10, uniform_fallback_min=3,
        )
        sheets = build_contact_sheets(frames, self.tmp / "sheets", columns=2, rows=1)
        self.assertEqual(len(sheets), 2)
        self.assertTrue(all(Path(sheet["path"]).exists() for sheet in sheets))
        self.assertEqual(
            [time for sheet in sheets for time in sheet["timestamps"]],
            [frame["timestamp_seconds"] for frame in frames],
        )

    def test_contact_sheets_survive_an_apostrophe_in_the_path(self):
        video = _edited_video(self.tmp / "edited.mp4")
        frames = extract_scene_change(
            str(video), self.tmp / "it's frames", resolution=64, max_frames=10, uniform_fallback_min=3,
        )
        sheets = build_contact_sheets(frames, self.tmp / "it's sheets", columns=2, rows=1)
        self.assertEqual(len(sheets), 2)

    def test_contact_sheets_of_no_frames(self):
        self.assertEqual(build_contact_sheets([], self.tmp / "sheets"), [])


if __name__ == "__main__":
    unittest.main()
