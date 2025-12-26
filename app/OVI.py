# /workspace/app/OVI.py
import asyncio
import csv
import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


# Public constants (für /health)
OVI_ROOT = os.getenv("OVI_ROOT", "/workspace/Ovi")
OVI_CKPT_DIR = os.getenv("OVI_CKPT_DIR", f"{OVI_ROOT}/ckpts")
OVI_RUN_BASE = os.getenv("OVI_RUN_BASE", f"{OVI_ROOT}/run.json")
OVI_JOBS_DIR = os.getenv("OVI_JOBS_DIR", "/workspace/jobs")
OVI_PYTHON = os.getenv("OVI_PYTHON", "python3")


class OVIJobRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    overrides: Dict[str, Any] = Field(default_factory=dict)
    job_id: Optional[str] = None


@dataclass
class Job:
    id: str
    status: str  # queued | running | succeeded | failed
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None

    job_dir: str = ""
    run_json: str = ""
    prompt_csv: str = ""
    output_dir: str = ""
    log_file: str = ""


class _OVIService:
    def __init__(self):
        self.ovi_root = Path(OVI_ROOT).resolve()
        self.jobs_root = Path(OVI_JOBS_DIR).resolve()
        self.run_base = Path(OVI_RUN_BASE).resolve()
        self.inference_py = (self.ovi_root / "inference.py").resolve()
        self.ckpts_dir = Path(OVI_CKPT_DIR).resolve()

        self.jobs: Dict[str, Job] = {}

        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._lock = asyncio.Lock()               # ✅ nur 1 Job gleichzeitig
        self._worker_task: Optional[asyncio.Task] = None

        self.jobs_root.mkdir(parents=True, exist_ok=True)

    def _sanity(self):
        if not self.ovi_root.exists():
            raise RuntimeError(f"OVI root missing: {self.ovi_root}")
        if not self.inference_py.exists():
            raise RuntimeError(f"inference.py missing: {self.inference_py}")
        if not self.run_base.exists():
            raise RuntimeError(f"run_base.json missing: {self.run_base}")

    async def ensure_worker(self):
        self._sanity()
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker_loop())

    def _make_job_id(self, job_id: Optional[str]) -> str:
        if job_id:
            clean = "".join(ch for ch in job_id if ch.isalnum() or ch in ("-", "_"))
            if not clean:
                raise ValueError("invalid job_id")
            return clean[:64]
        return uuid.uuid4().hex[:12]

    def _load_json(self, p: Path) -> Dict[str, Any]:
        return json.loads(p.read_text(encoding="utf-8"))

    def _save_json(self, p: Path, data: Dict[str, Any]):
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _persist(self, job: Job):
        try:
            jd = Path(job.job_dir)
            (jd / "job_status.json").write_text(
                json.dumps(asdict(job), indent=2),
                encoding="utf-8"
            )
        except Exception:
            pass

    def _write_prompt_csv(self, p: Path, prompt: str):
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["text_prompt"])
            w.writerow([prompt])

    async def create_job(self, prompt: str, overrides: Dict[str, Any], job_id: Optional[str]) -> str:
        await self.ensure_worker()

        jid = self._make_job_id(job_id)
        job_dir = (self.jobs_root / jid).resolve()
        if job_dir.exists():
            raise FileExistsError("job_id already exists")

        job_dir.mkdir(parents=True, exist_ok=True)

        prompt_csv = job_dir / "prompt.csv"
        run_json = job_dir / "run.json"
        output_dir = job_dir / "output"
        log_file = job_dir / "job.log"
        output_dir.mkdir(parents=True, exist_ok=True)

        # prompt.csv schreiben
        self._write_prompt_csv(prompt_csv, prompt)

        # run.json aus run_base bauen und NUR nötig überschreiben (ABSOLUTE Pfade!)
        cfg = self._load_json(self.run_base)

        # OVI soll aus /workspace/Ovi laufen, aber prompt/output pro Job nutzen:
        cfg["text_prompt"] = str(prompt_csv)   # z.B. /workspace/jobs/<id>/prompt.csv
        cfg["output_dir"] = str(output_dir)    # z.B. /workspace/jobs/<id>/output

        for k, v in (overrides or {}).items():
            cfg[k] = v

        self._save_json(run_json, cfg)

        job = Job(
            id=jid,
            status="queued",
            created_at=time.time(),
            job_dir=str(job_dir),
            run_json=str(run_json),
            prompt_csv=str(prompt_csv),
            output_dir=str(output_dir),
            log_file=str(log_file),
        )
        self.jobs[jid] = job
        self._persist(job)

        await self._queue.put(jid)
        return jid

    def get_status(self, job_id: str) -> Dict[str, Any]:
        job = self.jobs.get(job_id)
        if job:
            return asdict(job)

        # fallback: nach restart von disk lesen
        jf = (self.jobs_root / job_id / "job_status.json")
        if jf.exists():
            return self._load_json(jf)

        raise KeyError("job not found")

    def get_file(self, job_id: str, path: Optional[str]) -> Path:
        jd = (self.jobs_root / job_id).resolve()
        if not jd.exists():
            raise FileNotFoundError("job not found")

        if path:
            target = (jd / path).resolve()
            # path traversal block
            if not str(target).startswith(str(jd) + os.sep):
                raise ValueError("invalid path")
            if not target.exists() or not target.is_file():
                raise FileNotFoundError("file not found")
            return target

        # default: newest mp4 from output
        out_dir = jd / "output"
        mp4s = list(out_dir.rglob("*.mp4"))
        if not mp4s:
            raise FileNotFoundError("no mp4 yet")
        mp4s.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return mp4s[0]

    async def _worker_loop(self):
        while True:
            jid = await self._queue.get()
            try:
                await self._run_one(jid)
            finally:
                self._queue.task_done()

    async def _run_one(self, job_id: str):
        async with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                return

            job.status = "running"
            job.started_at = time.time()
            self._persist(job)

            run_json = Path(job.run_json)
            log_file = Path(job.log_file)

            cmd = [OVI_PYTHON, str(self.inference_py), "--config-file", str(run_json)]
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.ovi_root) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

            def _run_blocking() -> int:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                with log_file.open("wb") as lf:
                    lf.write(f"[OVI-API] cmd={' '.join(cmd)}\n".encode())
                    lf.write(f"[OVI-API] cwd={OVI_ROOT}\n".encode())
                    lf.flush()
                    p = subprocess.Popen(
                        cmd,
                        cwd=str(self.ovi_root),  # ✅ OVI läuft aus Repo-Root (keine relativen Pfad-Probleme)
                        env=env,
                        stdout=lf,
                        stderr=subprocess.STDOUT,
                    )
                    return p.wait()

            try:
                rc = await asyncio.to_thread(_run_blocking)
                job.exit_code = rc
                job.finished_at = time.time()
                if rc == 0:
                    job.status = "succeeded"
                    job.error = None
                else:
                    job.status = "failed"
                    job.error = f"inference.py exited with code {rc}"
            except Exception as e:
                job.status = "failed"
                job.error = repr(e)
                job.finished_at = time.time()

            self._persist(job)


_service = _OVIService()


# ---- Public functions used by main.py ----

async def submit_job(req: OVIJobRequest) -> str:
    return await _service.create_job(prompt=req.prompt, overrides=req.overrides, job_id=req.job_id)


def get_status(job_id: str):
    return _service.get_status(job_id)


def get_file(job_id: str, path: Optional[str] = None):
    return _service.get_file(job_id, path=path)
