#!/usr/bin/env python3
"""Parse a WebVTT subtitle file into a clean, timestamped transcript.

YouTube auto-subs emit rolling-duplicate cues (each line appears 2-3 times as it
scrolls). We dedupe consecutive identical cues and merge their time ranges.
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path


# Hours are optional in WebVTT (MM:SS.mmm is valid).
TS_RE = re.compile(
    r"(?:(\d{2,}):)?(\d{2}):(\d{2})[.,](\d{3})\s+-->\s+(?:(\d{2,}):)?(\d{2}):(\d{2})[.,](\d{3})"
)
TAG_RE = re.compile(r"<[^>]+>")


def _to_seconds(h: str | None, m: str, s: str, ms: str) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_vtt(path: str) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    segments: list[dict] = []
    previous_lines: list[str] = []
    i = 0
    while i < len(lines):
        match = TS_RE.match(lines[i])
        if not match:
            i += 1
            continue

        start = _to_seconds(*match.groups()[:4])
        end = _to_seconds(*match.groups()[4:])
        i += 1

        # YouTube auto-captions open some cues with a whitespace-only
        # placeholder line followed by the cue text. A whitespace-only line
        # with no text after it is just a separator and ends the cue.
        placeholder_end = i
        while placeholder_end < len(lines) and lines[placeholder_end] and not lines[placeholder_end].strip():
            placeholder_end += 1
        if (placeholder_end > i and placeholder_end < len(lines)
                and lines[placeholder_end].strip() and not _starts_next_cue(lines, placeholder_end)):
            i = placeholder_end

        cue_lines: list[str] = []
        while i < len(lines) and lines[i].strip() and not _starts_next_cue(lines, i):
            # Decode entities (manual captions carry &nbsp;) and collapse the
            # resulting non-breaking and repeated spaces.
            cleaned = " ".join(html.unescape(TAG_RE.sub("", lines[i])).split())
            if cleaned:
                cue_lines.append(cleaned)
            i += 1

        new_lines = _drop_rolled_line(previous_lines, cue_lines)
        if cue_lines:
            previous_lines = cue_lines
        cue_text = " ".join(new_lines).strip()
        if cue_text:
            segments.append({"start": round(start, 2), "end": round(end, 2), "text": cue_text})
        # Step over the separator, but never over the next cue's timing line.
        if i < len(lines) and not TS_RE.match(lines[i]):
            i += 1

    return _dedupe(segments)


def _starts_next_cue(lines: list[str], index: int) -> bool:
    """True when lines[index] is the next cue's timing line or its identifier.

    Guards cue text against running into the following cue when a separator
    line holds only whitespace.
    """
    if TS_RE.match(lines[index]):
        return True
    return index + 1 < len(lines) and bool(TS_RE.match(lines[index + 1]))


def _drop_rolled_line(previous_lines: list[str], cue_lines: list[str]) -> list[str]:
    """Drop the line YouTube auto-captions roll over from the previous cue.

    Rolling captions show two lines: the first repeats the previous cue's last
    line. Comparing whole lines (not word overlap) keeps genuine repetitions
    across cue boundaries in manual captions ("think that / that takes").
    """
    if previous_lines and cue_lines and cue_lines[0] == previous_lines[-1]:
        return cue_lines[1:]
    return cue_lines


def _dedupe(segments: list[dict]) -> list[dict]:
    """Collapse rolling duplicates common in YouTube auto-subs."""
    out: list[dict] = []
    for seg in segments:
        if out and seg["text"] == out[-1]["text"]:
            out[-1]["end"] = seg["end"]
            continue
        if out and seg["text"].startswith(out[-1]["text"] + " "):
            out[-1]["text"] = seg["text"]
            out[-1]["end"] = seg["end"]
            continue
        out.append(seg)
    return out


def filter_range(
    segments: list[dict],
    start_seconds: float | None,
    end_seconds: float | None,
) -> list[dict]:
    """Return segments whose time range overlaps [start, end]."""
    if start_seconds is None and end_seconds is None:
        return segments
    lo = start_seconds if start_seconds is not None else float("-inf")
    hi = end_seconds if end_seconds is not None else float("inf")
    return [seg for seg in segments if seg["end"] >= lo and seg["start"] <= hi]


def format_transcript(segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        start = int(seg["start"])
        stamp = f"[{start // 60:02d}:{start % 60:02d}]"
        lines.append(f"{stamp} {seg['text']}")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: transcribe.py <vtt-path>", file=sys.stderr)
        raise SystemExit(2)
    print(format_transcript(parse_vtt(sys.argv[1])))
