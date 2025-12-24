# Cleanes RunPod-Base mit CUDA/Torch/Py3.11 vorinstalliert
FROM runpod/pytorch:0.7.0-cu1251-torch260-ubuntu2204

SHELL ["/bin/bash","-lc"]




# Basics & HF-Caches (nur Orte, kein zusätzliches Python/Torch)
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Europe/Berlin \
    HF_HOME=/workspace/.cache/hf \
    TRANSFORMERS_CACHE=/workspace/.cache/hf/transformers \
    HF_HUB_CACHE=/workspace/.cache/hf/hub

WORKDIR /workspace


# Quick sanity check: nutzt diese Python-Env wirklich Torch?
RUN which python && python -V && python -c "import torch; print('torch=', torch.__version__)"


# Nur kleine Tools; KEIN Python/Torch-Reinstall!
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs ffmpeg libsndfile1 libsentencepiece-dev curl wget jq tzdata uuid-runtime \
 && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
 && dpkg-reconfigure -f noninteractive tzdata \
 && git lfs install --system \
 && mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE" \
 && rm -rf /var/lib/apt/lists/*





# 📦 Restliche Python-Deps
# 4) Rest über requirements.txt (einmal!)
COPY requirements.txt /tmp/requirements.txt
RUN python -V && python -m pip -V \
 && python -m pip install --no-cache-dir -r /tmp/requirements.txt



# --- FlashAttention passend zu Torch 2.6 + CUDA12 + ABI (cxx11abi True/False) ---
ARG FLASH_ATTN_VER=2.7.4.post1

RUN python -m pip uninstall -y flash-attn flash_attn || true && \
    ABI=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')") && \
    WHEEL="flash_attn-${FLASH_ATTN_VER}+cu12torch2.6cxx11abi${ABI}-cp310-cp310-linux_x86_64.whl" && \
    wget -nv "https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VER}/${WHEEL}" -O /tmp/flash_attn.whl && \
    python -m pip install --no-cache-dir /tmp/flash_attn.whl && \
    rm -f /tmp/flash_attn.whl && \
    python -c "import flash_attn; print('flash_attn ok:', flash_attn.__version__)"



# Nichts weiter – start.sh kümmert sich um Clone, Modelle, Jupyter etc.
COPY . .
RUN chmod +x /workspace/start.sh
RUN chmod +x /workspace/init.sh
RUN chmod +x /workspace/logs.sh

EXPOSE 8888 8000
CMD ["/bin/bash","-lc","/workspace/start.sh"]
