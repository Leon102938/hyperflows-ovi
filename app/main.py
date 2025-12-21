# /workspace/app/main.py
from fastapi import FastAPI, BackgroundTasks, Request, HTTPException
from .editor_api import EditRequest, render_edit
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi.responses import FileResponse
import os
from .wan_api import (
    TI2VRequest,
    run_wan_ti2v,
    submit_job,
    get_status,
    get_result,
    WAN_ROOT, WAN_CKPT
)

from .thinksound_api import (
    TSRequest, run_thinksound, submit_ts_job, get_ts_status, get_ts_result, THINK_ROOT
)



# 🔗 Proxy-Basis-URL aus start.sh (BASE_URL)
BASE_URL = os.getenv("BASE_URL", "").rstrip("/")


app = FastAPI(title="WAN 2.2 API", version="1.1")


# Basis-Verzeichnis der App
BASE_DIR = Path(__file__).resolve().parent.parent  # -> /workspace
# Ordner, in dem FFmpeg die Videos speichert
EXPORT_DIR = BASE_DIR / "exports"                 # -> /workspace/exports

# Sicherstellen, dass der Ordner existiert
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# /exports/... als statisches Verzeichnis bereitstellen
app.mount(
    "/exports",
    StaticFiles(directory=str(EXPORT_DIR), html=False),
    name="exports",
)




WAN_FLAG_FILE = "/workspace/status/wan2.2_ready"



@app.get("/health")
def health():
    return {"status":"ok","WAN_ROOT":WAN_ROOT,"WAN_CKPT_DIR":WAN_CKPT}

# ---- Sync (blockierend, wie gehabt) ----
@app.post("/wan/generate")
def wan_generate(request: TI2VRequest):
    return run_wan_ti2v(request)

# ---- Async Pattern: Submit → Status → Result ----
@app.post("/wan/submit")
def wan_submit(request: TI2VRequest, bg: BackgroundTasks):
    job_id = submit_job(request)
    return {"ok": True, "job_id": job_id}

@app.get("/wan/status/{job_id}")
def wan_status(job_id: str):
    return get_status(job_id)

@app.get("/wan/result/{job_id}")
def wan_result(job_id: str):
    return get_result(job_id)

@app.post("/thinksound/generate")
def ts_generate(request: TSRequest):
    return run_thinksound(request)

@app.get("/thinksound/file/{file_path:path}", name="ts_file")
def ts_file(file_path: str):
    # Pfad sicher machen (keine ..-Kletterei)
    safe_path = os.path.normpath(file_path).lstrip(os.sep)
    full_path = os.path.join(THINK_ROOT, safe_path)

    if not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail="File not found")

    # MP4 ausliefern (Browser zeigt Video / Download)
    return FileResponse(full_path, media_type="video/mp4")


@app.post("/thinksound/submit")
def ts_submit(request: TSRequest):
    job_id = submit_ts_job(request)
    return {"ok": True, "job_id": job_id}

@app.get("/thinksound/status/{job_id}")
def ts_status(job_id: str):
    return get_ts_status(job_id)

@app.get("/thinksound/result/{job_id}")
def ts_result(job_id: str, request: Request):
    res = get_ts_result(job_id)
    audio_path = res.get("audio_path")

    if audio_path:
        if BASE_URL:
            # Öffentlicher Proxy-Link wie bei WAN / fal.ai
            res["audio_url"] = f"{BASE_URL}/thinksound/file/{audio_path}"
        else:
            # Fallback: interne URL, falls BASE_URL ausnahmsweise nicht gesetzt ist
            res["audio_url"] = str(
                request.url_for("ts_file", file_path=audio_path)
            )

    return res


@app.post("/editor/render")
def editor_render(request: EditRequest):
    return render_edit(request)



@app.get("/DW/ready")
def dw_ready():
    ready = os.path.exists(WAN_FLAG_FILE)
    return {
        "ready": ready,
        "message": "Wan2.2 Download fertig." if ready else "Wan2.2 wird noch vorbereitet."
    }