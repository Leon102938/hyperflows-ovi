#!/bin/bash
# init.sh — lädt nur OVI + benötigte WAN-Assets + MMAudio nach /workspace/Ovi/ckpts

set -e

export HF_HOME=/workspace/.cache/hf

# optional: Schalter aus tools.config (falls vorhanden)
source ./tools.config 2>/dev/null || true

# Defaults falls tools.config fehlt
: "${OVI:=on}"          # on/off
: "${WAN_ASSETS:=on}"   # on/off (VAE + T5 + google/* aus Wan2.2 Repo)
: "${MMAUDIO:=on}"      # on/off

STATUS_DIR="/workspace/status"
CKPT_DIR="/workspace/Ovi/ckpts"

ZIMAGE_FLAG_FILE="$STATUS_DIR/zimage_ready"
OVI_FLAG_FILE="$STATUS_DIR/ovi_ready"
WAN_FLAG_FILE="$STATUS_DIR/wan_assets_ready"
MM_FLAG_FILE="$STATUS_DIR/mmaudio_ready"

mkdir -p "$HF_HOME"
mkdir -p "$STATUS_DIR"
mkdir -p "$CKPT_DIR"
mkdir -p "$CKPT_DIR/Ovi"


echo "[zimage] ensure model cache..."
hf download Tongyi-MAI/Z-Image-Turbo --exclude "assets/*" "README.md"
touch "$ZIMAGE_FLAG_FILE"
echo "[zimage] ready."



echo "📁 CKPT_DIR: $CKPT_DIR"
echo "⚙️  Flags: OVI=$OVI | WAN_ASSETS=$WAN_ASSETS | MMAUDIO=$MMAUDIO"

# -------------------------
# 1) OVI fp8 Modell (safetensors)
# -------------------------
if [ "${OVI}" = "on" ]; then
  echo "🧠 Lade OVI fp8 Modell ..."

  OVI_OUT="$CKPT_DIR/Ovi/model_fp8_e4m3fn.safetensors"

  if [ -f "$OVI_OUT" ]; then
    echo "✅ OVI Modell existiert schon: $OVI_OUT"
  else
    # Schnellster Weg: huggingface-cli (kann resume, ist meist stabiler als wget)
    if command -v huggingface-cli >/dev/null 2>&1; then
      echo "➡️  using huggingface-cli download (resume)"
      huggingface-cli download rkfg/Ovi-fp8_quantized \
        --local-dir "$CKPT_DIR/Ovi" \
        --local-dir-use-symlinks False \
        --resume-download \
        --include "model_fp8_e4m3fn.safetensors"
    else
      echo "➡️  huggingface-cli nicht gefunden -> fallback wget"
      wget -O "$OVI_OUT" \
        "https://huggingface.co/rkfg/Ovi-fp8_quantized/resolve/main/model_fp8_e4m3fn.safetensors"
    fi
  fi

  echo "✅ OVI Download fertig."
  touch "$OVI_FLAG_FILE"
else
  echo "⏭️ OVI Download übersprungen (OVI != on)."
fi

# -------------------------
# 2) WAN2.2 Assets: VAE + T5 + google/*
# -------------------------
if [ "${WAN_ASSETS}" = "on" ]; then
  echo "📦 Lade WAN2.2 Assets (VAE + T5 + google/*)..."
  python - <<'PY'
from huggingface_hub import snapshot_download
import os

ckpt_dir = "/workspace/Ovi/ckpts"
wan_dir = os.path.join(ckpt_dir, "Wan2.2-TI2V-5B")
os.makedirs(wan_dir, exist_ok=True)

snapshot_download(
    repo_id="Wan-AI/Wan2.2-TI2V-5B",
    local_dir=wan_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=[
        "google/*",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.2_VAE.pth",
    ],
)
print("✅ WAN assets ok:", wan_dir)
PY
  echo "✅ WAN Assets Download fertig."
  touch "$WAN_FLAG_FILE"
else
  echo "⏭️ WAN Assets übersprungen (WAN_ASSETS != on)."
fi

# -------------------------
# 3) MMAudio Assets
# -------------------------
if [ "${MMAUDIO}" = "on" ]; then
  echo "🎧 Lade MMAudio Assets ..."
  python - <<'PY'
from huggingface_hub import snapshot_download
import os

ckpt_dir = "/workspace/Ovi/ckpts"
mm_dir = os.path.join(ckpt_dir, "MMAudio")
os.makedirs(mm_dir, exist_ok=True)

snapshot_download(
    repo_id="hkchengrex/MMAudio",
    local_dir=mm_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=[
        "ext_weights/best_netG.pt",
        "ext_weights/v1-16.pth",
    ],
)
print("✅ MMAudio ok:", mm_dir)
PY
  echo "✅ MMAudio Download fertig."
  touch "$MM_FLAG_FILE"
else
  echo "⏭️ MMAudio übersprungen (MMAUDIO != on)."
fi


mkdir -p /workspace/status
touch /workspace/status/ovi_ready


echo "🏁 init done."