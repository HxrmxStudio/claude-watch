#!/usr/bin/env python3
"""Transcribe a video via Groq or OpenAI Whisper API.

Strategy: extract audio (mono 16kHz mp3, tiny payload), upload to whichever
API has a key. Returns segments in the same shape as transcribe.parse_vtt so
the rest of the pipeline (filter_range, format_transcript) doesn't care where
the transcript came from.

Pure stdlib — no `pip install groq` or `pip install openai` needed.
"""
from __future__ import annotations

import io
import json
import mimetypes
import os
import shutil
import ssl
import subprocess
import sys
import time
import urllib.error
import uuid
from pathlib import Path
from urllib.request import Request, urlopen


GROQ_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = "whisper-large-v3"

OPENAI_ENDPOINT = "https://api.openai.com/v1/audio/transcriptions"
OPENAI_MODEL = "whisper-1"


LOCAL_BINARY = "whisper-cli"
DEFAULT_LOCAL_MODEL = (
    Path.home() / ".local" / "share" / "whisper.cpp" / "models" / "ggml-large-v3-turbo-q5_0.bin"
)


def _dotenv_paths() -> list[Path]:
    return [Path.home() / ".config" / "watch" / ".env", Path.cwd() / ".env"]


def _from_dotenv(path: Path, name: str) -> str | None:
    if not path.exists():
        return None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() != name:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
                value = value[1:-1]
            return value or None
    except OSError:
        return None
    return None


def _read_setting(name: str) -> str | None:
    """Environment first, then ~/.config/watch/.env, then ./.env."""
    value = os.environ.get(name)
    if value and value.strip():
        return value.strip()
    for candidate in _dotenv_paths():
        value = _from_dotenv(candidate, name)
        if value:
            return value
    return None


def local_model_path() -> str | None:
    """whisper.cpp model to use, when both the binary and the model exist.

    WHISPER_CPP_MODEL overrides the default model location.
    """
    if shutil.which(LOCAL_BINARY) is None:
        return None
    model = Path(_read_setting("WHISPER_CPP_MODEL") or DEFAULT_LOCAL_MODEL).expanduser()
    return str(model) if model.is_file() else None


def load_api_key(preferred: str | None = None) -> tuple[str, str] | tuple[None, None]:
    """Return (backend, credential): Groq, then OpenAI, then local whisper.cpp.

    For the cloud backends the credential is the API key; for "local" it is
    the whisper.cpp model path, so audio never leaves the machine. If
    `preferred` names a backend, only that one is considered.
    """
    candidates = (("GROQ_API_KEY", "groq"), ("OPENAI_API_KEY", "openai"), (None, "local"))
    if preferred is not None:
        candidates = tuple(c for c in candidates if c[1] == preferred)

    for key_name, backend in candidates:
        value = local_model_path() if key_name is None else _read_setting(key_name)
        if value:
            return backend, value

    return None, None


def extract_audio(video_path: str, out_path: Path) -> Path:
    """Extract mono 16kHz 64kbps mp3 — ~480 kB/min, fits any Whisper limit."""
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(Path(video_path).resolve()),
        "-vn",
        "-acodec", "libmp3lame",
        "-ar", "16000",
        "-ac", "1",
        "-b:a", "64k",
        str(out_path.resolve()),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg audio extraction failed: {result.stderr.strip()}")
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise SystemExit("ffmpeg produced no audio — video may have no audio track")
    return out_path


def _build_multipart(fields: dict[str, str], file_path: Path) -> tuple[bytes, str]:
    """Assemble a multipart/form-data body the Whisper APIs accept.

    Whisper's multipart upload is small and predictable — doing it by hand
    keeps us on pure stdlib instead of pulling requests/groq/openai SDKs.
    """
    boundary = f"----WatchBoundary{uuid.uuid4().hex}"
    eol = b"\r\n"
    buf = io.BytesIO()

    for name, value in fields.items():
        values = value if isinstance(value, (list, tuple)) else [value]
        for v in values:
            buf.write(f"--{boundary}".encode()); buf.write(eol)
            buf.write(f'Content-Disposition: form-data; name="{name}"'.encode()); buf.write(eol)
            buf.write(eol)
            buf.write(str(v).encode()); buf.write(eol)

    mimetype = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    buf.write(f"--{boundary}".encode()); buf.write(eol)
    buf.write(
        f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"'.encode()
    )
    buf.write(eol)
    buf.write(f"Content-Type: {mimetype}".encode()); buf.write(eol)
    buf.write(eol)
    buf.write(file_path.read_bytes())
    buf.write(eol)
    buf.write(f"--{boundary}--".encode()); buf.write(eol)

    return buf.getvalue(), boundary


MAX_ATTEMPTS = 4       # initial + 3 retries
MAX_429_RETRIES = 2
RETRY_BASE_DELAY = 2.0


def _post_whisper(
    endpoint: str,
    api_key: str,
    model: str,
    audio_path: Path,
    word_timestamps: bool = False,
) -> dict:
    fields: dict = {
        "model": model,
        "response_format": "verbose_json",
        "temperature": "0",
    }
    if word_timestamps:
        # OpenAI/Groq accept timestamp_granularities[] as a repeated form field.
        fields["timestamp_granularities[]"] = "word"
    body, boundary = _build_multipart(fields, audio_path)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        # Groq sits behind Cloudflare — the default `Python-urllib/3.x` UA
        # trips WAF rule 1010 (403) before auth even runs. Any non-default
        # UA clears it; we identify honestly.
        "User-Agent": "watch-skill/1.0 (+claude-code; python-urllib)",
    }

    context = ssl.create_default_context()
    rate_limit_hits = 0
    last_exc: Exception | None = None
    last_detail = ""

    for attempt in range(MAX_ATTEMPTS):
        request = Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=300, context=context) as response:
                payload = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = _read_error_body(exc)
            last_exc, last_detail = exc, detail

            # 4xx other than 429 are client errors — no retry will fix them.
            if 400 <= exc.code < 500 and exc.code != 429:
                raise SystemExit(f"Whisper request failed: {exc}{detail}")

            if exc.code == 429:
                rate_limit_hits += 1
                if rate_limit_hits >= MAX_429_RETRIES:
                    raise SystemExit(f"Whisper request failed: {exc}{detail}")
                delay = _retry_after(exc) or RETRY_BASE_DELAY * (2 ** attempt) + 1
            else:
                delay = RETRY_BASE_DELAY * (2 ** attempt)

            if attempt < MAX_ATTEMPTS - 1:
                print(
                    f"[watch] whisper HTTP {exc.code} — retrying in {delay:.1f}s "
                    f"(attempt {attempt + 2}/{MAX_ATTEMPTS})",
                    file=sys.stderr,
                )
                time.sleep(delay)
            continue
        except (urllib.error.URLError, TimeoutError, ConnectionResetError, OSError) as exc:
            last_exc, last_detail = exc, ""
            if attempt < MAX_ATTEMPTS - 1:
                delay = RETRY_BASE_DELAY * (attempt + 1)
                print(
                    f"[watch] whisper network error ({type(exc).__name__}: {exc}) — "
                    f"retrying in {delay:.1f}s (attempt {attempt + 2}/{MAX_ATTEMPTS})",
                    file=sys.stderr,
                )
                time.sleep(delay)
            continue

        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Whisper returned non-JSON response: {exc}: {payload[:200]}")

    raise SystemExit(
        f"Whisper request failed after {MAX_ATTEMPTS} attempts: {last_exc}{last_detail}"
    )


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read()
    except Exception:
        return ""
    if not body:
        return ""
    try:
        return f" — {body.decode('utf-8', errors='replace')[:400]}"
    except Exception:
        return ""


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    header = exc.headers.get("Retry-After") if getattr(exc, "headers", None) else None
    if not header:
        return None
    try:
        return float(header)
    except ValueError:
        return None


def _segments_from_response(data: dict) -> list[dict]:
    """Convert Whisper verbose_json into our {start, end, text} segment format."""
    out: list[dict] = []
    for seg in data.get("segments") or []:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        out.append({
            "start": round(float(seg.get("start") or 0.0), 2),
            "end": round(float(seg.get("end") or 0.0), 2),
            "text": text,
        })

    if not out:
        full = (data.get("text") or "").strip()
        if full:
            out.append({"start": 0.0, "end": 0.0, "text": full})

    return out


def _segments_from_whisper_cpp(data: dict) -> list[dict]:
    """Convert `whisper-cli -oj` output (millisecond offsets) to our segment shape."""
    out: list[dict] = []
    for item in data.get("transcription") or []:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        offsets = item.get("offsets") or {}
        out.append({
            "start": round(float(offsets.get("from") or 0) / 1000, 2),
            "end": round(float(offsets.get("to") or 0) / 1000, 2),
            "text": text,
        })
    return out


def _words_from_whisper_cpp(data: dict) -> list[dict]:
    """Word entries from a `--max-len 1 --split-on-word` run (one word per item)."""
    return [
        {"word": segment["text"], "start": segment["start"], "end": segment["end"]}
        for segment in _segments_from_whisper_cpp(data)
    ]


def _transcribe_local(
    audio_path: Path, model: str, word_timestamps: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Run whisper.cpp on-device. Returns (segments, words); audio stays local."""
    wav_path = audio_path.with_name(f"{audio_path.stem}-16k.wav")
    convert = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(audio_path.resolve()), "-ac", "1", "-ar", "16000", str(wav_path.resolve())],
        capture_output=True, text=True, check=False,
    )
    if convert.returncode != 0:
        raise SystemExit(f"ffmpeg wav conversion failed: {convert.stderr.strip()}")

    # whisper-cli appends ".json" to the -of value verbatim.
    output_base = str(wav_path.with_suffix("").resolve())
    cmd = [
        LOCAL_BINARY, "-m", model, "-f", str(wav_path.resolve()),
        "-l", "auto", "-oj", "-of", output_base, "-np",
    ]
    if word_timestamps:
        cmd += ["-ml", "1", "-sow"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    json_path = Path(output_base + ".json")
    if result.returncode != 0 or not json_path.exists():
        raise SystemExit(f"whisper.cpp failed (exit {result.returncode}): {result.stderr.strip()[-400:]}")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    segments = _segments_from_whisper_cpp(data)
    words = _words_from_whisper_cpp(data) if word_timestamps else []
    return segments, words


def transcribe_video(
    video_path: str,
    audio_out: Path,
    backend: str | None = None,
    api_key: str | None = None,
) -> tuple[list[dict], str]:
    """Run the full flow: extract audio → upload → parse segments.

    Returns (segments, backend_used). Raises SystemExit on any failure.
    """
    if backend is None or api_key is None:
        detected_backend, detected_key = load_api_key()
        backend = backend or detected_backend
        api_key = api_key or detected_key

    if not backend or not api_key:
        setup_py = Path(__file__).resolve().parent / "setup.py"
        raise SystemExit(
            "No Whisper backend available. Set GROQ_API_KEY (preferred) or OPENAI_API_KEY "
            "in the environment or in ~/.config/watch/.env, or install whisper.cpp "
            f"(`whisper-cli`) with a model at {DEFAULT_LOCAL_MODEL} (or WHISPER_CPP_MODEL). "
            f"Run `python3 {setup_py}` to configure."
        )

    print(f"[watch] extracting audio for Whisper ({backend})…", file=sys.stderr)
    audio_path = extract_audio(video_path, audio_out)
    size_kb = audio_path.stat().st_size / 1024
    if backend == "local":
        print(f"[watch] audio: {size_kb:.0f} kB — transcribing on-device (whisper.cpp)…", file=sys.stderr)
        segments, _ = _transcribe_local(audio_path, api_key)
        if not segments:
            raise SystemExit("Whisper returned no transcript segments")
        print(f"[watch] transcribed {len(segments)} segments via local", file=sys.stderr)
        return segments, backend

    print(f"[watch] audio: {size_kb:.0f} kB — uploading to {backend} Whisper…", file=sys.stderr)

    if backend == "groq":
        response = _post_whisper(GROQ_ENDPOINT, api_key, GROQ_MODEL, audio_path)
    elif backend == "openai":
        response = _post_whisper(OPENAI_ENDPOINT, api_key, OPENAI_MODEL, audio_path)
    else:
        raise SystemExit(f"Unknown whisper backend: {backend}")

    segments = _segments_from_response(response)
    if not segments:
        raise SystemExit("Whisper returned no transcript segments")

    print(f"[watch] transcribed {len(segments)} segments via {backend}", file=sys.stderr)
    return segments, backend


def transcribe_audio(
    audio_path: Path,
    backend: str | None = None,
    api_key: str | None = None,
    word_timestamps: bool = False,
) -> tuple[list[dict], str, list[dict]]:
    """Upload an existing audio file. Returns (segments, backend, words).

    Words is empty unless word_timestamps=True. Each word is
    {"word": str, "start": float, "end": float}.
    """
    if backend is None or api_key is None:
        detected_backend, detected_key = load_api_key()
        backend = backend or detected_backend
        api_key = api_key or detected_key
    if not backend or not api_key:
        raise SystemExit("No Whisper API key available for transcribe_audio()")

    if backend == "local":
        segments, words = _transcribe_local(audio_path, api_key, word_timestamps=word_timestamps)
        return segments, backend, words

    if backend == "groq":
        endpoint, model = GROQ_ENDPOINT, GROQ_MODEL
    elif backend == "openai":
        endpoint, model = OPENAI_ENDPOINT, OPENAI_MODEL
    else:
        raise SystemExit(f"Unknown whisper backend: {backend}")

    response = _post_whisper(
        endpoint, api_key, model, audio_path,
        word_timestamps=word_timestamps,
    )
    segments = _segments_from_response(response)
    words: list[dict] = []
    if word_timestamps:
        for w in response.get("words") or []:
            text = (w.get("word") or "").strip()
            if not text:
                continue
            words.append({
                "word": text,
                "start": round(float(w.get("start") or 0.0), 3),
                "end": round(float(w.get("end") or 0.0), 3),
            })
    return segments, backend, words


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: whisper.py <video-path> [<audio-out.mp3>] [--backend groq|openai|local]", file=sys.stderr)
        raise SystemExit(2)

    video = sys.argv[1]
    audio_out = Path(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else Path("audio.mp3")
    backend_override = None
    if "--backend" in sys.argv:
        backend_override = sys.argv[sys.argv.index("--backend") + 1]

    segments, backend = transcribe_video(video, audio_out, backend=backend_override)
    print(json.dumps({"backend": backend, "segments": segments}, indent=2))
