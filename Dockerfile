# Cleanes RunPod-Base mit CUDA/Torch/Py3.11 vorinstalliert
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Basics & HF-Caches (nur Orte, kein zusätzliches Python/Torch)
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Europe/Berlin \
    HF_HOME=/workspace/.cache/hf \
    TRANSFORMERS_CACHE=/workspace/.cache/hf/transformers \
    HF_HUB_CACHE=/workspace/.cache/hf/hub

WORKDIR /workspace



# Nur kleine Tools; KEIN Python/Torch-Reinstall!
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs ffmpeg libsndfile1 libsentencepiece-dev curl wget jq tzdata uuid-runtime \
 && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
 && dpkg-reconfigure -f noninteractive tzdata \
 && git lfs install --system \
 && mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE" \
 && rm -rf /var/lib/apt/lists/*


# FlashAttention (stable for torch 2.6 + cu12 + cp311 + cxx11abiFALSE)
RUN python3 -m pip uninstall -y flash-attn flash_attn flash_attn_2_cuda || true && \
    rm -rf /usr/local/lib/python3.11/dist-packages/flash_attn* || true && \
    python3 -m pip install --no-deps \
      https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl



# 📦 Restliche Python-Deps
# 4) Rest über requirements.txt (einmal!)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY requirements_heavy.txt /tmp/requirements_heavy.txt
RUN set -eux; mkdir -p /workspace/.tmp && export TMPDIR=/workspace/.tmp; \
     pip install --no-cache-dir -r /tmp/requirements_heavy.txt; \
     rm -rf /workspace/.tmp /root/.cache ~/.cache /tmp/*

# HF CLI + Turbo-Downloader
RUN pip install --no-cache-dir "huggingface_hub[cli]" hf_transfer



# Nichts weiter – start.sh kümmert sich um Clone, Modelle, Jupyter etc.
COPY . .
RUN chmod +x /workspace/start.sh
RUN chmod +x /workspace/init.sh
RUN chmod +x /workspace/logs.sh

EXPOSE 8888 8000
CMD ["/bin/bash","-lc","/workspace/start.sh"]
