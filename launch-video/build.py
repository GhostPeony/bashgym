from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
PYTHON = Path(sys.executable)
FFMPEG = Path(imageio_ffmpeg.get_ffmpeg_exe())
PICTURE = OUT / "bashgym-launch-picture.mp4"
VOICE = OUT / "voiceover.mp3"
VOICE_WAV = OUT / "voiceover-48k.wav"
SUBS = OUT / "voiceover.srt"
MUSIC = OUT / "music-original.wav"
FINAL = OUT / "bashgym-launch-final.mp4"
VO_ONLY = OUT / "bashgym-launch-vo-only.mp4"


def run(cmd: list[str | Path]) -> None:
    print(" ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=ROOT, check=True)


def ensure_voiceover() -> None:
    if VOICE.exists() and SUBS.exists():
        return
    edge_tts = ROOT / ".venv" / "Scripts" / "edge-tts.exe"
    run([
        edge_tts, "-f", ROOT / "voiceover.txt", "-v", "en-US-AndrewNeural",
        "--rate=+7%", "--pitch=-2Hz", "--write-media", VOICE,
        "--write-subtitles", SUBS,
    ])


def mix() -> None:
    final_filter = (
        "[1:a]aresample=48000,volume=1.08,asplit=2[vo_sc][vo_mix];"
        "[2:a]aresample=48000,volume=0.30[music];"
        "[music][vo_sc]sidechaincompress=threshold=0.012:ratio=8:attack=18:release=360[ducked];"
        "[ducked][vo_mix]amix=inputs=2:duration=longest:normalize=0,"
        "loudnorm=I=-14:LRA=8:TP=-1.5,apad,atrim=0:72[a]"
    )
    run([
        FFMPEG, "-y", "-hide_banner", "-loglevel", "warning",
        "-i", PICTURE, "-i", VOICE, "-i", MUSIC,
        "-filter_complex", final_filter,
        "-map", "0:v:0", "-map", "[a]", "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-t", "72", "-movflags", "+faststart", FINAL,
    ])
    vo_filter = "aresample=48000,volume=1.08,loudnorm=I=-16:LRA=7:TP=-1.5,apad,atrim=0:72"
    run([
        FFMPEG, "-y", "-hide_banner", "-loglevel", "warning",
        "-i", PICTURE, "-i", VOICE,
        "-filter_complex", f"[1:a]{vo_filter}[a]",
        "-map", "0:v:0", "-map", "[a]", "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-t", "72", "-movflags", "+faststart", VO_ONLY,
    ])


def probe(path: Path) -> dict[str, object]:
    proc = subprocess.run(
        [str(FFMPEG), "-hide_banner", "-i", str(path), "-f", "null", "NUL"],
        capture_output=True, text=True, cwd=ROOT,
    )
    text = proc.stderr
    duration = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", text)
    seconds = None
    if duration:
        seconds = int(duration.group(1)) * 3600 + int(duration.group(2)) * 60 + float(duration.group(3))
    streams = [line.strip() for line in text.splitlines() if "Stream #" in line and ("Video:" in line or "Audio:" in line)]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "duration_seconds": seconds,
        "streams": streams,
        "sha256": digest,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_voiceover()
    if not VOICE_WAV.exists():
        run([FFMPEG, "-y", "-hide_banner", "-loglevel", "warning", "-i", VOICE,
             "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", VOICE_WAV])
    if not MUSIC.exists():
        run([PYTHON, ROOT / "generate_music.py"])
    if not PICTURE.exists() or PICTURE.stat().st_size == 0:
        run([PYTHON, ROOT / "render_video.py"])
    mix()
    metadata = {
        "format": {"width": 1920, "height": 1080, "fps": 30, "duration_seconds": 72},
        "exports": [probe(path) for path in (FINAL, VO_ONLY, PICTURE, VOICE, VOICE_WAV, MUSIC)],
        "replaceable_audio": {
            "voiceover": VOICE.name,
            "voiceover_editing_wav": VOICE_WAV.name,
            "music": MUSIC.name,
            "voiceover_only_video": VO_ONLY.name,
            "subtitles": SUBS.name,
        },
    }
    (OUT / "verification.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
