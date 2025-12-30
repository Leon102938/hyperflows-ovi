import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

router = APIRouter()

# ---- Config ----
ZIMAGE_PY = os.environ.get("ZIMAGE_PY", "/opt/venvs/zimage/bin/python")
HF_HOME = os.environ.get("HF_HOME", "/workspace/.cache/hf")

STATUS_DIR = Path(os.environ.get("STATUS_DIR", "/workspace/status"))
ZIMAGE_READY_FLAG = Path(os.environ.get("ZIMAGE_READY_FLAG", str(STATUS_DIR / "zimage_ready")))

JOBS_ROOT = Path(os.environ.get("ZIMAGE_JOBS_ROOT", "/workspace/jobs/zimage"))


class ZImageJobRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    width: int = Field(768, ge=256, le=2048)
    height: int = Field(768, ge=256, le=2048)
    steps: int = Field(9, ge=1, le=30)
    guidance_scale: float = Field(0.0, ge=0.0, le=20.0)
    seed: Optional[int] = Field(42, ge=0)
    job_id: Optional[str] = None


def _job_dir(job_id: str) -> Path:
    return JOBS_ROOT / job_id


def _write_status(job_id: str, state: str, extra: Optional[Dict[str, Any]] = None) -> None:
    d = _job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    payload = {"job_id": job_id, "state": state, "ts": time.time()}
    if extra:
        payload.update(extra)
    (d / "status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_status(job_id: str) -> Dict[str, Any]:
    p = _job_dir(job_id) / "status.json"
    if not p.exists():
        raise FileNotFoundError
    return json.loads(p.read_text(encoding="utf-8"))


async def _run_job(job_id: str) -> None:
    d = _job_dir(job_id)
    req_path = d / "request.json"
    out_path = d / "out.png"
    stdout_path = d / "stdout.log"
    stderr_path = d / "stderr.log"

    _write_status(job_id, "running")

    worker = r"""
import os, json, torch
from diffusers import ZImagePipeline

with open(os.environ["JOB_JSON"], "r", encoding="utf-8") as f:
    req = json.load(f)

prompt = req["prompt"]
width = int(req["width"])
height = int(req["height"])
steps = int(req["steps"])
guidance = float(req["guidance_scale"])
seed = req.get("seed", None)
out_path = req["out_path"]

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=dtype,
    low_cpu_mem_usage=False,
)

if torch.cuda.is_available():
    pipe = pipe.to("cuda")

gen = None
if seed is not None:
    gen = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(seed))

img = pipe(
    prompt=prompt,
    height=height,
    width=width,
    num_inference_steps=steps,
    guidance_scale=guidance,
    generator=gen,
).images[0]

os.makedirs(os.path.dirname(out_path), exist_ok=True)
img.save(out_path)
print(out_path)
"""

    env = os.environ.copy()
    env["HF_HOME"] = HF_HOME
    env["JOB_JSON"] = str(req_path)

    proc = await asyncio.create_subprocess_exec(
        ZIMAGE_PY, "-c", worker,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    out_b, err_b = await proc.communicate()
    stdout_path.write_bytes(out_b or b"")
    stderr_path.write_bytes(err_b or b"")

    if proc.returncode != 0:
        _write_status(job_id, "failed", extra={
            "error": f"zimage subprocess exit code {proc.returncode}",
            "stderr_log": str(stderr_path),
        })
        return

    if not out_path.exists():
        _write_status(job_id, "failed", extra={
            "error": "out.png missing after generation",
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        })
        return

    _write_status(job_id, "succeeded", extra={
        "output_path": str(out_path),
        "file_endpoint": f"/zimage/jobs/{job_id}/file",
    })


@router.get("/ready")
def zimage_ready():
    ok = ZIMAGE_READY_FLAG.exists()
    return {"ready": ok, "flag": str(ZIMAGE_READY_FLAG), "message": "Z-Image bereit." if ok else "Z-Image nicht bereit."}


@router.post("/jobs")
async def zimage_submit(req: ZImageJobRequest):
    if not ZIMAGE_READY_FLAG.exists():
        raise HTTPException(status_code=503, detail="Z-Image not ready (flag missing).")

    job_id = req.job_id or f"zimg_{uuid.uuid4().hex[:12]}"
    d = _job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)

    request_payload = req.model_dump()
    request_payload["out_path"] = str(d / "out.png")
    (d / "request.json").write_text(json.dumps(request_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_status(job_id, "queued")
    asyncio.create_task(_run_job(job_id))

    return {
        "job_id": job_id,
        "status_url": f"/zimage/jobs/{job_id}",
        "file_url": f"/zimage/jobs/{job_id}/file",
        "state": "queued",
    }


@router.get("/jobs/{job_id}")
def zimage_status(job_id: str):
    try:
        return _read_status(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="job_id not found")


@router.get("/jobs/{job_id}/file")
def zimage_file(job_id: str):
    d = _job_dir(job_id)
    out_path = d / "out.png"

    if not out_path.exists():
        try:
            st = _read_status(job_id)
            if st.get("state") in ("queued", "running"):
                raise HTTPException(status_code=409, detail=f"not ready yet: {st.get('state')}")
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="job_id not found")
        raise HTTPException(status_code=404, detail="file not found")

    return FileResponse(str(out_path), media_type="image/png", filename=f"{job_id}.png")
