"""Tests for spreading the scene-change frame budget over the whole video."""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from frames import extract_scene_change, merge_bursts, plan_frame_times  # noqa: E402


class TestMergeBursts(unittest.TestCase):

    def test_keeps_first_cut_of_each_burst(self):
        # An animation can trip the detector on consecutive frames.
        self.assertEqual(
            merge_bursts([1.0, 1.02, 1.05, 5.0, 5.5, 9.0], min_gap=1.0),
            [1.0, 5.0, 9.0],
        )

    def test_sorts_input(self):
        self.assertEqual(merge_bursts([9.0, 1.0], min_gap=1.0), [1.0, 9.0])


class TestPlanFrameTimes(unittest.TestCase):

    def test_keeps_every_scene_under_budget(self):
        self.assertEqual(
            plan_frame_times([0.0, 10.0, 20.0], duration=30.0, max_frames=10),
            [0.0, 10.0, 20.0],
        )

    def test_always_includes_the_opening_frame(self):
        self.assertEqual(plan_frame_times([5.0], duration=30.0, max_frames=10)[0], 0.0)

    def test_spreads_budget_to_the_end_of_dense_videos(self):
        # 400 cuts packed in the first minute, then a few late ones: the
        # budget must still reach the end instead of filling up early.
        early = [idx * 0.15 for idx in range(400)]
        late = [300.0, 450.0, 590.0]
        times = plan_frame_times(early + late, duration=600.0, max_frames=20)
        self.assertLessEqual(len(times), 20)
        self.assertIn(590.0, times)
        self.assertIn(300.0, times)

    def test_includes_anchors_even_without_nearby_cuts(self):
        times = plan_frame_times(
            [idx * 0.5 for idx in range(200)], duration=600.0,
            max_frames=12, anchors=(240.0, 480.0),
        )
        self.assertIn(240.0, times)
        self.assertIn(480.0, times)
        self.assertLessEqual(len(times), 12)

    def test_ignores_cuts_outside_the_range(self):
        early = [0.1 * idx for idx in range(1, 300)]
        in_range = [100.0 + idx for idx in range(50)]
        times = plan_frame_times(early + in_range, duration=50.0, max_frames=10, range_start=100.0)
        self.assertLessEqual(len(times), 10)
        self.assertTrue(all(100.0 <= time < 150.0 for time in times))

    def test_more_anchors_than_budget_still_reach_the_end(self):
        anchors = tuple(float(idx * 20) for idx in range(1, 31))  # 30 chapters
        times = plan_frame_times([], duration=620.0, max_frames=10, anchors=anchors)
        self.assertEqual(len(times), 10)
        self.assertEqual(times[0], 0.0)
        self.assertEqual(times[-1], 600.0)

    def test_result_is_sorted_and_unique(self):
        times = plan_frame_times(
            [0.0, 1.0, 2.0, 3.0], duration=10.0, max_frames=10, anchors=(2.0,),
        )
        self.assertEqual(times, sorted(set(times)))


def _color_strip(out: Path, colors: list[str], seconds_each: int = 1) -> Path:
    """One solid-color segment per entry: a hard cut every `seconds_each`."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for color in colors:
        cmd += ["-f", "lavfi", "-i", f"color=c={color}:size=160x120:duration={seconds_each}"]
    inputs = "".join(f"[{idx}:v]" for idx in range(len(colors)))
    cmd += [
        "-filter_complex", f"{inputs}concat=n={len(colors)}:v=1:a=0[v]",
        "-map", "[v]", "-pix_fmt", "yuv420p", str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


class TestSceneChangeCoverage(unittest.TestCase):

    def setUp(self):
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not available")
        self.tmp = Path(tempfile.mkdtemp(prefix="watch-budget-test-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_frame_cap_does_not_truncate_the_tail(self):
        colors = ["red", "green", "blue", "white", "black", "yellow",
                  "cyan", "magenta", "gray", "orange", "purple", "pink"]
        video = _color_strip(self.tmp / "strip.mp4", colors)
        frames = extract_scene_change(
            str(video), self.tmp / "frames",
            resolution=64, max_frames=4, uniform_fallback_min=2,
        )
        self.assertLessEqual(len(frames), 4)
        self.assertGreaterEqual(frames[-1]["timestamp_seconds"], 8.0)
        for frame in frames:
            self.assertTrue(Path(frame["path"]).exists())


if __name__ == "__main__":
    unittest.main()
