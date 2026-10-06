#!/usr/bin/env python3
"""Probe video metadata and extract frames at an auto-scaled fps.

Auto-fps targets a frame budget, not a fixed rate. Token cost scales with frame
count, so budget-by-duration keeps short videos dense and long videos capped.
When a user-specified range is passed, focused-mode budgets denser (they are
zooming in for detail).
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path


MAX_FPS = 2.0


def _clamp_fps(fps: float, duration_seconds: float, max_frames: int) -> tuple[float, int]:
    fps = min(fps, MAX_FPS)
    target = min(max_frames, max(1, int(round(fps * duration_seconds))))
    return fps, target


def parse_time(value: str | float | int | None) -> float | None:
    """Parse SS, MM:SS, or HH:MM:SS (with optional .ms) into seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    parts = s.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError:
        pass
    raise SystemExit(f"Cannot parse time value: {value!r} (expected SS, MM:SS, or HH:MM:SS)")


def format_time(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def get_metadata(video_path: str) -> dict:
    if shutil.which("ffprobe") is None:
        raise SystemExit("ffprobe is not installed. Install with: brew install ffmpeg")

    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(Path(video_path).resolve()),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"ffprobe failed: {result.stderr.strip()}")

    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])
    fmt = data.get("format", {})
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = float(fmt.get("duration") or video_stream.get("duration") or 0)
    return {
        "duration_seconds": duration,
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "codec": video_stream.get("codec_name"),
        "size_bytes": int(fmt.get("size") or 0),
        "has_audio": audio_stream is not None,
    }


def auto_fps(duration_seconds: float, max_frames: int = 100) -> tuple[float, int]:
    """Pick fps that targets a sensible frame budget for full-video scans."""
    if duration_seconds <= 0:
        return 1.0, 1

    if duration_seconds <= 30:
        target = min(max_frames, max(12, int(round(duration_seconds))))
    elif duration_seconds <= 60:
        target = min(max_frames, 40)
    elif duration_seconds <= 180:  # 3 min
        target = min(max_frames, 60)
    elif duration_seconds <= 600:  # 10 min
        target = min(max_frames, 80)
    else:
        target = max_frames

    return _clamp_fps(target / duration_seconds, duration_seconds, max_frames)


def auto_fps_focus(duration_seconds: float, max_frames: int = 100) -> tuple[float, int]:
    """Denser budget for user-specified ranges — they are zooming in for detail."""
    if duration_seconds <= 0:
        return min(MAX_FPS, 2.0), 2

    if duration_seconds <= 5:
        target = min(max_frames, max(10, int(round(duration_seconds * 6))))
    elif duration_seconds <= 15:
        target = min(max_frames, max(30, int(round(duration_seconds * 4))))
    elif duration_seconds <= 30:
        target = min(max_frames, 60)
    elif duration_seconds <= 60:
        target = min(max_frames, 80)
    elif duration_seconds <= 180:
        target = max_frames
    else:
        target = max_frames

    return _clamp_fps(target / duration_seconds, duration_seconds, max_frames)


def extract(
    video_path: str,
    out_dir: Path,
    fps: float,
    resolution: int = 512,
    max_frames: int = 100,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> list[dict]:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")

    out_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_dir.glob("frame_*.jpg"):
        existing.unlink()

    output_pattern = str(out_dir / "frame_%04d.jpg")
    cmd: list[str] = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
    ]

    # -ss before -i = fast seek (keyframe-snap, good enough for preview frames).
    if start_seconds is not None:
        cmd += ["-ss", f"{start_seconds:.3f}"]
    if end_seconds is not None:
        cmd += ["-to", f"{end_seconds:.3f}"]

    cmd += [
        "-i", str(Path(video_path).resolve()),
        "-vf", f"fps={fps},scale={resolution}:-2",
        "-frames:v", str(max_frames),
        "-q:v", "4",
        output_pattern,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg frame extraction failed: {result.stderr.strip()}")

    offset = start_seconds or 0.0
    frames = sorted(out_dir.glob("frame_*.jpg"))
    return [
        {
            "index": i,
            "timestamp_seconds": round(offset + (i / fps if fps > 0 else 0.0), 2),
            "path": str(p),
        }
        for i, p in enumerate(frames)
    ]


BURST_GAP_SECONDS = 1.0
SCENE_THRESHOLD = 0.3  # ffmpeg scene score; catches hard cuts and most dissolves


def detect_scene_times(
    video_path: str,
    scene_threshold: float = SCENE_THRESHOLD,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> list[float]:
    """Return every scene-change timestamp (absolute seconds) in the range.

    Detection runs as a decode-only pass with no frame cap, so the frame
    budget can later be spread across the whole video instead of being spent
    on the first N cuts.
    """
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")

    cmd: list[str] = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if start_seconds is not None:
        cmd += ["-ss", f"{start_seconds:.3f}"]
    if end_seconds is not None:
        cmd += ["-to", f"{end_seconds:.3f}"]
    cmd += [
        "-i", str(Path(video_path).resolve()),
        "-vf", f"select='gt(scene\\,{scene_threshold})',metadata=mode=print:file=-",
        "-an", "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg scene detection failed: {result.stderr.strip()}")

    offset = start_seconds or 0.0
    times: list[float] = []
    for stream in (result.stdout, result.stderr):
        for token in stream.split():
            for prefix in ("pts_time:", "pts_time="):
                if token.startswith(prefix):
                    try:
                        times.append(round(offset + float(token[len(prefix):]), 2))
                    except ValueError:
                        pass
    return sorted(set(times))


def merge_bursts(times: list[float], min_gap: float = BURST_GAP_SECONDS) -> list[float]:
    """Keep the first cut of each burst of cuts closer than `min_gap` seconds.

    Animations and flashes trip the detector on consecutive frames; without
    this, one transition can eat most of the frame budget.
    """
    merged: list[float] = []
    for time in sorted(times):
        if not merged or time - merged[-1] >= min_gap:
            merged.append(time)
    return merged


def plan_frame_times(
    scene_times: list[float],
    duration: float,
    max_frames: int,
    anchors: tuple[float, ...] = (),
    range_start: float = 0.0,
) -> list[float]:
    """Choose at most `max_frames` timestamps that cover the whole range.

    The range opening and the `anchors` (e.g. chapter starts) are always kept.
    The remaining budget goes to the first cut of each equal-length time
    window, so a cut-dense opening cannot starve the rest of the video; any
    budget left after that goes to the remaining cuts, evenly by index.
    """
    range_end = range_start + duration
    fixed = spread_evenly(
        sorted({round(range_start, 2), *(round(anchor, 2) for anchor in anchors)}), max_frames,
    )
    candidates = [
        time for time in sorted(set(scene_times))
        if range_start <= time < range_end and time not in fixed
    ]
    budget = max_frames - len(fixed)
    if len(candidates) <= budget:
        return sorted(fixed + candidates)

    chosen: list[float] = []
    if budget > 0:
        window = max(duration, 1e-6) / budget
        taken_windows: set[int] = set()
        for time in candidates:
            slot = min(budget - 1, int((time - range_start) / window))
            if slot not in taken_windows:
                taken_windows.add(slot)
                chosen.append(time)

    leftover = [time for time in candidates if time not in chosen]
    chosen += spread_evenly(leftover, budget - len(chosen))
    return sorted(set(fixed + chosen))


def spread_evenly(items: list[float], count: int) -> list[float]:
    """Pick `count` items evenly by index, always keeping the first and last.

    Used whenever a list must be cut to a budget: keeping the first N would
    silently drop the end of the video.
    """
    if count <= 0:
        return []
    if len(items) <= count:
        return list(items)
    if count == 1:
        return [items[0]]
    step = (len(items) - 1) / (count - 1)
    return [items[round(idx * step)] for idx in range(count)]


def extract_at(
    video_path: str,
    out_dir: Path,
    times: list[float],
    resolution: int = 512,
    source: str = "scene-change",
) -> list[dict]:
    """Extract one JPEG per timestamp, named in chronological order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_dir.glob("frame_*.jpg"):
        existing.unlink()

    frames: list[dict] = []
    for index, time in enumerate(sorted(times)):
        target = out_dir / f"frame_{index + 1:04d}.jpg"
        result = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{time:.3f}", "-i", str(Path(video_path).resolve()),
                "-frames:v", "1", "-vf", f"scale={resolution}:-2", "-q:v", "4",
                str(target),
            ],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0 or not target.exists():
            print(f"[watch] frame at {time:.2f}s failed: {result.stderr.strip()}", file=sys.stderr)
            continue
        frames.append({
            "index": len(frames),
            "timestamp_seconds": round(time, 2),
            "path": str(target),
            "source": source,
        })
    return frames


PROBE_WIDTH, PROBE_HEIGHT = 64, 36
PIXEL_DELTA = 24  # grey levels; smaller differences are compression noise
CONTENT_SAMPLE_SECONDS = 2.0
# Share of probe bytes that moved: pixels in grey probes, colour channels in
# rgb24 ones (a hue-only change moves fewer channels than a brightness change).
MIN_CONTENT_CHANGE = 0.05


def probe_gray_frames(
    video_path: str,
    every_seconds: float,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    pixel_format: str = "gray",
) -> list[bytes]:
    """Tiny samples (one every `every_seconds`) for cheap comparison.

    Greyscale by default; "rgb24" keeps hue, so two screens of equal
    brightness but different colour do not look alike.
    """
    cmd: list[str] = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if start_seconds is not None:
        cmd += ["-ss", f"{start_seconds:.3f}"]
    if end_seconds is not None:
        cmd += ["-to", f"{end_seconds:.3f}"]
    cmd += [
        "-i", str(Path(video_path).resolve()),
        "-vf", f"fps=1/{every_seconds},scale={PROBE_WIDTH}:{PROBE_HEIGHT},format={pixel_format}",
        "-f", "rawvideo", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg probe failed: {result.stderr.decode(errors='replace').strip()}")
    size = PROBE_WIDTH * PROBE_HEIGHT * (3 if pixel_format == "rgb24" else 1)
    raw = result.stdout
    return [raw[offset:offset + size] for offset in range(0, len(raw) - size + 1, size)]


def _changed_ratio(first: bytes, second: bytes) -> float:
    changed = sum(1 for left, right in zip(first, second) if abs(left - right) > PIXEL_DELTA)
    return changed / max(1, len(first))


def pick_content_changes(
    frames: list[bytes],
    every_seconds: float,
    changed_ratio: float,
    anchors: tuple[float, ...] = (),
    max_gap_seconds: float = float("inf"),
    range_start: float = 0.0,
) -> list[float]:
    """Keep a sample when it differs enough from the last *kept* sample.

    Comparing against the last kept screen (not the previous sample) lets
    slow changes, like a terminal filling line by line, add up until they
    count as a new screen. Anchors and `max_gap_seconds` force a screen so
    no stretch of the video goes unseen.
    """
    kept: list[float] = []
    reference: bytes | None = None
    pending = sorted(anchors)
    for index, frame in enumerate(frames):
        time = round(range_start + index * every_seconds, 2)
        anchor_reached = bool(pending) and time >= pending[0]
        if anchor_reached:
            pending = [anchor for anchor in pending if anchor > time]
        gap_exceeded = bool(kept) and time - kept[-1] >= max_gap_seconds
        if (reference is None or anchor_reached or gap_exceeded
                or _changed_ratio(frame, reference) > changed_ratio):
            kept.append(time)
            reference = frame
    return kept


def plan_content_changes(
    frames: list[bytes],
    every_seconds: float,
    max_frames: int,
    anchors: tuple[float, ...] = (),
    range_start: float = 0.0,
) -> list[float]:
    """Binary-search the change ratio so the kept screens fit `max_frames`.

    Screencasts vary a lot in how much changes per minute, so a fixed ratio
    either floods or starves the budget. The forced gap scales with the
    budget, so it cannot overflow the budget on its own.
    """
    duration = len(frames) * every_seconds
    max_gap = max(every_seconds, 2 * duration / max(1, max_frames))
    low, high = 0.0, 1.0
    best = pick_content_changes(frames, every_seconds, high, anchors, max_gap, range_start)
    for _ in range(12):
        middle = (low + high) / 2
        kept = pick_content_changes(frames, every_seconds, middle, anchors, max_gap, range_start)
        if len(kept) <= max_frames:
            best, high = kept, middle
        else:
            low = middle
    # Anchors plus forced gaps can exceed the budget even at the loosest
    # ratio; spread the excess rather than dropping the end of the video.
    return spread_evenly(best, max_frames)


SHOT_SAMPLE_SECONDS = 0.5  # fine enough that two nearby cuts get their own sample
SETTLE_SAMPLES = 1  # look one sample past a cut, once any dissolve has settled
SETTLE_LIMIT = 4  # samples to wait for a transition to end; longer motion is content
REPEAT_LOOKBACK = 30  # kept shots compared per candidate; bounds long videos
GAP_CHANGE = 0.2  # with no spare budget, only a new screen (not a gesture) earns a frame


def _sample_index(time: float, sample_count: int, every_seconds: float, range_start: float) -> int:
    first_after = math.ceil((time - range_start) / every_seconds - 1e-9)
    return max(0, min(sample_count - 1, first_after + SETTLE_SAMPLES))


def _settle(index: int, probes: list[bytes], end: int | None = None) -> int:
    """Move past a fade or slide-in until two samples in a row look alike.

    Never reaches `end` (exclusive), the first sample of the next shot.
    """
    end = len(probes) if end is None else min(end, len(probes))
    for _ in range(SETTLE_LIMIT):
        if index + 1 >= end or _changed_ratio(probes[index], probes[index + 1]) <= MIN_CONTENT_CHANGE:
            break
        index += 1
    return index


def _shot_sample(time: float, next_cut: float, probes: list[bytes],
                 every_seconds: float, range_start: float) -> int:
    """Settled sample of the shot opening at `time`, kept before `next_cut`."""
    end = len(probes)
    if next_cut != float("inf"):
        end = math.ceil((next_cut - range_start) / every_seconds - 1e-9)
    start = min(_sample_index(time, len(probes), every_seconds, range_start), max(0, end - 1))
    return _settle(start, probes, end)


def drop_repeated_shots(
    times: list[float],
    probes: list[bytes],
    every_seconds: float,
    keep: tuple[float, ...] = (),
    range_start: float = 0.0,
) -> list[float]:
    """Drop shots that look like a shot already kept.

    Edited videos cut back to the same framing over and over (a talking head
    between slides); each return costs a frame and shows nothing new. `keep`
    times (opening, chapter starts) always stay and count as seen.
    """
    if not probes:
        return sorted(set(times) | set(keep))
    forced = set(keep)
    kept: list[float] = []
    seen: list[bytes] = []
    ordered = sorted(set(times) | forced)
    for position, time in enumerate(ordered):
        # The settled sample is the one extracted later, so judge that one.
        next_cut = ordered[position + 1] if position + 1 < len(ordered) else float("inf")
        sample = probes[_shot_sample(time, next_cut, probes, every_seconds, range_start)]
        repeated = any(
            _changed_ratio(sample, earlier) <= MIN_CONTENT_CHANGE
            for earlier in seen[-REPEAT_LOOKBACK:]
        )
        if time in forced or not repeated:
            kept.append(time)
            seen.append(sample)
    return kept


def fill_scene_gaps(
    scene_times: list[float],
    probes: list[bytes],
    every_seconds: float,
    max_gap_seconds: float,
    range_start: float = 0.0,
    changed_ratio: float = GAP_CHANGE,
) -> list[float]:
    """Content-change times inside stretches longer than `max_gap_seconds` with no cut.

    Slides that cross-fade or a screen share inside one shot never trip the
    scene detector, so a whole section can go unseen. Within those stretches
    a new frame is taken whenever `changed_ratio` of the screen changed, once
    the change has settled; nothing is forced by time alone, so a long
    unchanging shot stays one frame. An animation changes the screen on every
    sample, so a stretch gets at most its fair share of the budget (one filler
    per half `max_gap_seconds`), spread over the stretch.
    """
    range_end = range_start + len(probes) * every_seconds
    bounds = sorted({range_start, range_end, *(
        time for time in scene_times if range_start <= time < range_end
    )})
    fillers: list[float] = []
    for left, right in zip(bounds, bounds[1:]):
        if right - left <= max_gap_seconds:
            continue
        # Start once the opening shot has settled, so its own fade is not a change.
        first = _shot_sample(left, right, probes, every_seconds, range_start)
        last = int((right - range_start) / every_seconds)  # samples before the next cut
        changes = pick_content_changes(
            probes[first:last], every_seconds, changed_ratio,
            range_start=range_start + first * every_seconds,
        )
        # The first change is the shot that opened the stretch. Pick before
        # settling: settling compares samples and most changes are dropped.
        picked = spread_evenly(changes[1:], int(2 * (right - left) // max_gap_seconds))
        fillers += sorted({
            round(range_start + _settle(round((time - range_start) / every_seconds), probes, last)
                  * every_seconds, 2)
            for time in picked
        })
    return fillers


def _settled_cut_times(times: list[float], probes: list[bytes], every_seconds: float,
                       cuts: set[float], all_cuts: list[float], range_start: float) -> list[float]:
    """Move each cut past its transition, never onto or beyond the next frame.

    Only cuts move: fillers are settled already, and anchors keep their time.
    Settling stops before the next detected cut even when that cut was not
    kept, or a short shot would be replaced by the one after it.
    """
    settled: list[float] = []
    for position, time in enumerate(times):
        next_time = times[position + 1] if position + 1 < len(times) else float("inf")
        if time in cuts:
            next_cut = next((cut for cut in all_cuts if cut > time), float("inf"))
            index = _shot_sample(time, next_cut, probes, every_seconds, range_start)
            later = round(range_start + index * every_seconds, 2)
            if time < later < next_time:
                time = later
        settled.append(time)
    return settled


def plan_scene_frames(
    scene_times: list[float],
    probes: list[bytes],
    duration: float,
    max_frames: int,
    anchors: tuple[float, ...] = (),
    range_start: float = 0.0,
) -> list[float]:
    """Pick frame times for an edited video: cuts, minus repeated framings,
    plus screen changes inside long uncut stretches.

    The change needed for a filler starts at GAP_CHANGE and is lowered while
    the result still fits the budget, so spare budget goes to smaller screen
    changes (a UI edit) instead of going unused. Frames land after transitions.
    """
    anchors = tuple(round(anchor, 2) for anchor in anchors)
    if not probes:
        return plan_frame_times(scene_times, duration, max_frames, anchors, range_start)
    keep = (round(range_start, 2), *anchors)
    max_gap = 2 * duration / max(1, max_frames)
    # Only cuts can return to a framing already seen. A filler is a change from
    # the screen before it by construction; deduplicating fillers would drop the
    # small successive edits a UI walkthrough is about.
    unique_cuts = drop_repeated_shots(
        scene_times, probes, SHOT_SAMPLE_SECONDS, keep=keep, range_start=range_start,
    )

    def candidates_at(ratio: float) -> list[float]:
        fillers = fill_scene_gaps(scene_times, probes, SHOT_SAMPLE_SECONDS, max_gap, range_start, ratio)
        return sorted({*unique_cuts, *fillers})

    best = candidates_at(GAP_CHANGE)
    low, high = MIN_CONTENT_CHANGE, GAP_CHANGE
    while len(best) < max_frames and high - low > 0.01:
        middle = (low + high) / 2
        candidates = candidates_at(middle)
        if len(candidates) <= max_frames:
            best, high = candidates, middle
        else:
            low = middle
    times = plan_frame_times(best, duration, max_frames, anchors, range_start)
    cuts = set(scene_times) - set(keep)
    return _settled_cut_times(
        times, probes, SHOT_SAMPLE_SECONDS, cuts, sorted({*scene_times, *keep}), range_start,
    )


def extract_scene_change(
    video_path: str,
    out_dir: Path,
    scene_threshold: float = SCENE_THRESHOLD,
    resolution: int = 512,
    max_frames: int = 100,
    uniform_fallback_min: int = 10,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    scene_times: list[float] | None = None,
    anchors: tuple[float, ...] = (),
) -> list[dict]:
    """One frame per detected shot, spread over the whole range within budget.

    Uses ffmpeg's `select='gt(scene,T)'` filter — scene change scores in [0,1],
    higher = more visual difference between frames. 0.3 is a permissive cut
    detector that catches hard cuts and most dissolves without firing on motion.

    All cuts are detected first (pass `scene_times` to reuse a detection you
    already ran), bursts are merged, and `plan_frame_times` picks at most
    `max_frames` of them across the whole range plus the opening frame and
    any `anchors`.

    On `uniform_fallback_min`: screen recordings and long talking heads yield
    very few scene changes. They fall back to content-change sampling (a new
    frame whenever enough of the screen changed); a truly static video falls
    back to uniform sampling — sparse frames > almost no frames.
    """
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")

    if scene_times is None:
        scene_times = merge_bursts(detect_scene_times(
            video_path, scene_threshold, start_seconds, end_seconds,
        ))

    meta = get_metadata(video_path)
    eff_start = start_seconds if start_seconds is not None else 0.0
    eff_end = end_seconds if end_seconds is not None else meta["duration_seconds"]
    eff_duration = max(0.1, eff_end - eff_start)

    in_range = tuple(anchor for anchor in anchors if eff_start <= anchor < eff_end)

    # +1: the opening frame always counts as a shot.
    if len(scene_times) + 1 < uniform_fallback_min:
        # Screencasts barely cut: follow what changes on screen instead.
        every = min(CONTENT_SAMPLE_SECONDS, max(0.5, eff_duration / (max_frames * 4)))
        try:
            probes = probe_gray_frames(video_path, every, start_seconds, end_seconds)
        except SystemExit as exc:
            print(f"[watch] content probe failed, sampling uniformly: {exc}", file=sys.stderr)
            probes = []
        if len(pick_content_changes(probes, every, MIN_CONTENT_CHANGE)) > 1:
            times = plan_content_changes(
                probes, every, max_frames, anchors=in_range, range_start=eff_start,
            )
            return extract_at(video_path, out_dir, times, resolution, source="content-change")
        out_dir.mkdir(parents=True, exist_ok=True)
        fps, _ = auto_fps(eff_duration, max_frames=max_frames)
        return extract(
            video_path, out_dir,
            fps=fps, resolution=resolution, max_frames=max_frames,
            start_seconds=start_seconds, end_seconds=end_seconds,
        )

    try:
        probes = probe_gray_frames(
            video_path, SHOT_SAMPLE_SECONDS, start_seconds, end_seconds, pixel_format="rgb24",
        )
    except SystemExit as exc:
        # Cuts alone still cover the video; lose the refinements, not the run.
        print(f"[watch] shot probe failed, keeping plain cuts: {exc}", file=sys.stderr)
        probes = []
    times = plan_scene_frames(scene_times, probes, eff_duration, max_frames, in_range, eff_start)
    return extract_at(video_path, out_dir, times, resolution=resolution)


def _concat_quote(path: str) -> str:
    """Escape a path for a single-quoted line of ffmpeg's concat list."""
    return str(Path(path).resolve()).replace("'", "'\\''")


def build_contact_sheets(
    frames: list[dict],
    out_dir: Path,
    columns: int = 4,
    rows: int = 5,
    cell_width: int = 480,
) -> list[dict]:
    """Tile the frames into grids so a whole video reads in a few images.

    Cells run left to right, top to bottom; each sheet lists the timestamps
    of its cells (ffmpeg here has no drawtext, so they are not burned in).
    """
    if not frames:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_dir.glob("sheet_*.jpg"):
        existing.unlink()

    listing = out_dir / "frames.txt"
    listing.write_text(
        "".join(f"file '{_concat_quote(frame['path'])}'\n" for frame in frames),
        encoding="utf-8",
    )
    cell_height = cell_width * 9 // 16
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(listing),
            "-vf", (
                f"scale={cell_width}:{cell_height}:force_original_aspect_ratio=decrease,"
                f"pad={cell_width}:{cell_height}:(ow-iw)/2:(oh-ih)/2,"
                f"tile={columns}x{rows}:padding=4:margin=4"
            ),
            "-fps_mode", "passthrough", "-q:v", "4", str(out_dir / "sheet_%d.jpg"),
        ],
        capture_output=True, text=True, check=False,
    )
    listing.unlink(missing_ok=True)
    if result.returncode != 0:
        print(f"[watch] contact sheets failed: {result.stderr.strip()}", file=sys.stderr)
        return []

    per_sheet = columns * rows
    stamps = [frame["timestamp_seconds"] for frame in frames]
    sheets = sorted(out_dir.glob("sheet_*.jpg"), key=lambda path: int(path.stem.split("_")[1]))
    if len(sheets) != math.ceil(len(frames) / per_sheet):
        # A skipped frame would shift every later cell's timestamp.
        print("[watch] contact sheets skipped frames; read single frames instead", file=sys.stderr)
        return []
    return [
        {"path": str(path), "timestamps": stamps[idx * per_sheet:(idx + 1) * per_sheet]}
        for idx, path in enumerate(sheets)
    ]


def select_hero_frames(
    frames: list[dict],
    pacing: dict | None = None,
    hook_end_seconds: float = 10.0,
    max_hero: int = 5,
    min_hero: int = 3,
) -> list[dict]:
    """Pick 3-5 'hero' frames for embedding in the wiki source page.

    Heuristic, deterministic:
      1. First frame after a scene-change at or before `hook_end_seconds`.
      2. First frame of the highest-motion shot from pacing.shots.
      3. First frame of the longest sustained shot.
      4. Plus 1-2 evenly-spaced extras to round out to max_hero.

    Falls back to uniform picks from `frames` if pacing data is unavailable
    or any heuristic yields no candidate.
    """
    if not frames:
        return []

    chosen_indices: list[int] = []

    def _add(idx: int) -> None:
        if 0 <= idx < len(frames) and idx not in chosen_indices:
            chosen_indices.append(idx)

    for i, f in enumerate(frames):
        if f["timestamp_seconds"] <= hook_end_seconds:
            _add(i)
            break

    if pacing and pacing.get("shots"):
        shots = pacing["shots"]
        if shots:
            top_motion = max(shots, key=lambda s: s.get("motion_score", 0) or 0)
            longest = max(shots, key=lambda s: s.get("duration_seconds", 0))
            for shot in (top_motion, longest):
                start_t = shot.get("start_seconds", 0)
                for i, f in enumerate(frames):
                    if f["timestamp_seconds"] >= start_t:
                        _add(i)
                        break

    if len(chosen_indices) < min_hero:
        gap = max(1, len(frames) // max_hero)
        for i in range(0, len(frames), gap):
            _add(i)
            if len(chosen_indices) >= max_hero:
                break

    chosen_indices = sorted(set(chosen_indices))[:max_hero]
    return [frames[i] for i in chosen_indices]


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "usage: frames.py <video-path> <out-dir> [--fps F] [--resolution W] "
            "[--max-frames N] [--start T] [--end T]",
            file=sys.stderr,
        )
        raise SystemExit(2)

    video = sys.argv[1]
    out = Path(sys.argv[2])
    args = sys.argv[3:]

    fps_override = None
    resolution = 512
    max_frames = 100
    start_arg = None
    end_arg = None
    i = 0
    while i < len(args):
        if args[i] == "--fps":
            fps_override = float(args[i + 1]); i += 2
        elif args[i] == "--resolution":
            resolution = int(args[i + 1]); i += 2
        elif args[i] == "--max-frames":
            max_frames = int(args[i + 1]); i += 2
        elif args[i] == "--start":
            start_arg = args[i + 1]; i += 2
        elif args[i] == "--end":
            end_arg = args[i + 1]; i += 2
        else:
            i += 1

    meta = get_metadata(video)
    start_sec = parse_time(start_arg)
    end_sec = parse_time(end_arg)
    full_duration = meta["duration_seconds"]

    effective_start = start_sec if start_sec is not None else 0.0
    effective_end = end_sec if end_sec is not None else full_duration
    effective_duration = max(0.0, effective_end - effective_start)

    focused = start_sec is not None or end_sec is not None
    if focused:
        fps, target = auto_fps_focus(effective_duration, max_frames=max_frames)
    else:
        fps, target = auto_fps(effective_duration, max_frames=max_frames)
    if fps_override is not None:
        fps = fps_override
        target = max(1, int(round(fps * effective_duration)))

    frames = extract(
        video, out,
        fps=fps,
        resolution=resolution,
        max_frames=max_frames,
        start_seconds=start_sec,
        end_seconds=end_sec,
    )
    print(json.dumps(
        {"meta": meta, "fps": fps, "target": target, "focused": focused, "frames": frames},
        indent=2,
    ))
