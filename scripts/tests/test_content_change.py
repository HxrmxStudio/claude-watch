"""Tests for content-change sampling on screencasts (few or no hard cuts)."""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from frames import extract_scene_change, pick_content_changes, plan_content_changes  # noqa: E402

PIXELS = 100


def _frame(changed: int, value: int = 200) -> bytes:
    """A probe frame where the first `changed` pixels differ from a dark base."""
    return bytes([value] * changed + [10] * (PIXELS - changed))


class TestPickContentChanges(unittest.TestCase):

    def test_ignores_small_changes(self):
        frames = [_frame(0), _frame(2), _frame(3), _frame(1)]
        self.assertEqual(pick_content_changes(frames, every_seconds=1.0, changed_ratio=0.1), [0.0])

    def test_accumulated_small_changes_trigger_a_new_screen(self):
        # A terminal filling line by line: each step adds 4% of new pixels,
        # never enough on its own, but compared with the last *kept* screen
        # the change adds up and crosses 10% on the third step.
        frames = [_frame(0), _frame(4), _frame(8), _frame(12), _frame(16)]
        self.assertEqual(
            pick_content_changes(frames, every_seconds=2.0, changed_ratio=0.1),
            [0.0, 6.0],
        )

    def test_anchors_and_max_gap_force_a_screen(self):
        frames = [_frame(0)] * 10
        times = pick_content_changes(
            frames, every_seconds=1.0, changed_ratio=0.5,
            anchors=(3.0,), max_gap_seconds=4.0,
        )
        self.assertEqual(times, [0.0, 3.0, 7.0])


class TestPlanContentChanges(unittest.TestCase):

    def test_dense_chapters_do_not_truncate_the_end(self):
        # 300 samples of a static screen (600 s), 30 chapter anchors and a
        # budget of 20: forced frames exceed the budget, and the plan must
        # still reach the end of the video instead of keeping the first 20.
        frames = [_frame(0)] * 300
        anchors = tuple(float(idx * 20) for idx in range(30))
        times = plan_content_changes(frames, every_seconds=2.0, max_frames=20, anchors=anchors)
        self.assertLessEqual(len(times), 20)
        self.assertGreaterEqual(times[-1], 560.0)


def _screencast(out: Path, positions: int = 8, seconds_each: int = 2) -> Path:
    """A static screen where a small panel moves every `seconds_each` seconds.

    Each change touches well under a third of the frame, so the scene-change
    detector stays quiet while the content clearly changes.
    """
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for idx in range(positions):
        x = 10 + idx * 16
        cmd += [
            "-f", "lavfi", "-i",
            (
                f"color=c=gray:size=192x108:duration={seconds_each},"
                f"drawbox=x={x}:y=30:w=40:h=40:color=white:t=fill"
            ),
        ]
    inputs = "".join(f"[{idx}:v]" for idx in range(positions))
    cmd += [
        "-filter_complex", f"{inputs}concat=n={positions}:v=1:a=0[v]",
        "-map", "[v]", "-pix_fmt", "yuv420p", str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


class TestScreencastFallback(unittest.TestCase):

    def setUp(self):
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not available")
        self.tmp = Path(tempfile.mkdtemp(prefix="watch-content-test-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_screencast_uses_content_change_within_budget(self):
        video = _screencast(self.tmp / "cast.mp4")
        frames = extract_scene_change(
            str(video), self.tmp / "frames",
            resolution=64, max_frames=5, uniform_fallback_min=4,
        )
        self.assertTrue(frames)
        self.assertTrue(all(frame["source"] == "content-change" for frame in frames))
        self.assertLessEqual(len(frames), 5)
        self.assertGreaterEqual(len(frames), 3)
        self.assertGreaterEqual(frames[-1]["timestamp_seconds"], 10.0)


if __name__ == "__main__":
    unittest.main()
