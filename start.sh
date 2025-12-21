#!/bin/bash
set -euo pipefail

# ============ 🔧 Anti-Fragmentation für PyTorch ============
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
# (optional; falls root nicht erlaubt, diesen Block weglassen)
if [ -w /etc/profile.d ]; then
  echo 'export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256' > /etc/profile.d/pytorch_alloc.sh || true
fi

# -------- tools.config nur laden, wenn vorhanden --------
if [ -f ./tools.config ]; then
  # shellcheck disable=SC1091
  source ./tools.config
fi

# 🌍 BASE_URL automatisch setzen (RUNPOD_POD_ID sicher expandieren)
echo "🌐 Ermittle dynamische RunPod Proxy-URL..."
POD_ID="${RUNPOD_POD_ID:-}"
if [ -z "$POD_ID" ]; then
  echo "❌ FEHLER: RUNPOD_POD_ID nicht gesetzt – .env nicht geschrieben!"
else
  BASE_URL="https://${POD_ID}-8000.proxy.runpod.net"
  export BASE_URL
  echo "BASE_URL=$BASE_URL" > /workspace/.env
  echo "✅ BASE_URL erfolgreich gesetzt: $BASE_URL"
fi

# ============ 🔧 PYTHONPATH (ohne unbound) ============
# Falls PYTHONPATH leer/unset ist → nur /workspace/app setzen; sonst anhängen.
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}/workspace/app"

# ============ 🔷 JUPYTERLAB THEME ============
mkdir -p /root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension
echo '{ "theme": "JupyterLab Dark" }' \
  > /root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings

# ============ 🔷 JUPYTERLAB (Port 8888) ============
if [ "${JUPYTER:-off}" = "on" ]; then
  echo "🧠 Starte JupyterLab (Port 8888)..."
  nohup jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    --NotebookApp.disable_check_xsrf=True \
    --NotebookApp.notebook_dir='/workspace' \
    --ServerApp.allow_origin='*' \
    > /workspace/jupyter.log 2>&1 &
fi

# ============ 🔷 FASTAPI (Port 8000) ============
if [ "${FASTAPI:-on}" = "on" ]; then
  echo "🚀 Starte zentrale FastAPI (Port 8000)..."
  nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > /workspace/fastapi.log 2>&1 &
else
  echo "⏭️  FASTAPI=off – überspringe FastAPI."
fi

# ============ 🔷 Download/Init (WAN) ============
if [ "${Init:-off}" = "on" ]; then
  echo "🚀 Starte WAN/Think-Init"
  if [ -x /workspace/init.sh ]; then
    nohup bash /workspace/init.sh >/dev/null 2>&1 & disown
  else
    echo "⚠️  /workspace/init.sh nicht gefunden oder nicht ausführbar."
  fi
else
  echo "⏭️  Init=off – überspringe WAN/Think-Download."
fi

# ============ ✅ ABSCHLUSS ============
echo "✅ Dienste wurden gestartet (je nach config). Logs: /workspace/fastapi.log /workspace/jupyter.log"
tail -f /dev/null

