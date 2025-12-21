# app/editor_api.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os, uuid, subprocess, textwrap

EDIT_ROOT = os.getenv("EDIT_ROOT", "/workspace")        # Basis
EXPORT_DIR = os.path.join(EDIT_ROOT, "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)


class Clip(BaseModel):
    path: str                  # absoluter Pfad zur MP4
    text: Optional[str] = None # Untertitel für diesen Clip
    duration: Optional[float] = 5.0  # Sekunden (falls bekannt/konstant)


class EditRequest(BaseModel):
    clips: List[Clip]
    output_name: Optional[str] = None  # z.B. "test_vid.mp4"


def _run(cmd: list) -> None:
    print("FFMPEG CMD:", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{proc.stdout}")


def render_edit(req: EditRequest) -> Dict[str, Any]:
    if not req.clips:
        return {"ok": False, "error": "no_clips"}

    job_id = uuid.uuid4().hex[:8]
    out_name = req.output_name or f"edit_{job_id}.mp4"

    # 1) Concat-Liste bauen
    list_path = os.path.join(EXPORT_DIR, f"{job_id}_list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for c in req.clips:
            clip_path = c.path
            if not os.path.isfile(clip_path):
                raise FileNotFoundError(clip_path)
            # -safe 0 erlaubt absolute Pfade
            f.write(f"file '{clip_path}'\n")

    tmp_concat = os.path.join(EXPORT_DIR, f"{job_id}_concat.mp4")
    out_path = os.path.join(EXPORT_DIR, out_name)

    # 2) Clips hart zusammenkleben (ohne Re-Encode)
    _run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        tmp_concat,
    ])

    # 3) SRT-Datei mit Untertiteln bauen (ein Block pro Clip)
    #    Annahme: duration pro Clip ≈ c.duration (Standard 5s)
    srt_path = os.path.join(EXPORT_DIR, f"{job_id}.srt")
    current_start = 0.0
    idx = 1

    def _fmt_time(t: float) -> str:
        # t in Sekunden -> "HH:MM:SS,mmm"
        total_ms = int(round(t * 1000))
        s, ms = divmod(total_ms, 1000)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    with open(srt_path, "w", encoding="utf-8") as f:
        for c in req.clips:
            if not c.text:
                # keinen Untertitel für diesen Clip
                current_start += c.duration or 5.0
                continue
            start = current_start
            end = start + (c.duration or 5.0)
            f.write(f"{idx}\n")
            f.write(f"{_fmt_time(start)} --> {_fmt_time(end)}\n")
            f.write(c.text.strip() + "\n\n")
            idx += 1
            current_start = end

    # 4) Untertitel auf das Video brennen
    # Achtung bei Pfad mit Leerzeichen → srt_path am besten ohne Spaces halten
    _run([
        "ffmpeg", "-y",
        "-i", tmp_concat,
        "-vf", f"subtitles={srt_path}",
        "-c:a", "copy",
        out_path,
    ])

    # Optional: Temp-Dateien könnten gelöscht werden
    # os.remove(tmp_concat)
    # os.remove(list_path)
    # os.remove(srt_path)

    # Pfad relativ zu /workspace zurückgeben
    rel_path = os.path.relpath(out_path, EDIT_ROOT)
    return {"ok": True, "output_path": rel_path}
