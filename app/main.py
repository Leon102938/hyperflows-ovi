import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .editor_api import EditRequest, render_edit
from .OVI import OVIJobRequest, submit_job, get_status, get_file, OVI_ROOT, OVI_CKPT_DIR
from .zimage import router as zimage_router

app = FastAPI(title="OVI API", version="1.0")

# ---- static exports (falls du das nutzt) ----
BASE_DIR = Path(__file__).resolve().parent.parent  # /workspace
EXPORT_DIR = BASE_DIR / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/exports", StaticFiles(directory=str(EXPORT_DIR), html=False), name="exports")

# ---- Routers ----
app.include_router(zimage_router, prefix="/zimage", tags=["zimage"])

# ---- Ready Flags ----
OVI_FLAG_FILE = "/workspace/status/ovi_ready"
ZIMAGE_FLAG_FILE = "/workspace/status/zimage_ready"


@app.get("/health")
def health():
    return {"status": "ok", "OVI_ROOT": OVI_ROOT, "OVI_CKPT_DIR": OVI_CKPT_DIR}


@app.get("/DW/ready")
def dw_ready():
    ready = os.path.exists(OVI_FLAG_FILE)
    return {"ready": ready, "message": "OVI bereit." if ready else "OVI wird noch vorbereitet."}


@app.get("/DW/zimage_ready")
def dw_zimage_ready():
    ready = os.path.exists(ZIMAGE_FLAG_FILE)
    return {"ready": ready, "message": "Z-Image bereit." if ready else "Z-Image wird noch vorbereitet."}


# ---- Editor ----
@app.post("/editor/render")
def editor_render(request: EditRequest):
    return render_edit(request)


# ---- OVI Jobs (Polling wie gehabt) ----
@app.post("/jobs")
async def create_job(body: OVIJobRequest):
    try:
        jid = await submit_job(body)
        return {"id": jid, "status": "queued"}
    except FileExistsError:
        raise HTTPException(status_code=409, detail="job_id already exists")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    try:
        return get_status(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found")


@app.get("/jobs/{job_id}/file")
def job_file(job_id: str, path: str | None = None):
    try:
        p = get_file(job_id, path=path)
        return FileResponse(str(p))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

