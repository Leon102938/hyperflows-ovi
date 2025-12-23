# Cleanes RunPod-Base mit CUDA/Torch/Py3.11 vorinstalliert
FROM runpod/pytorch:0.7.0-cu1251-torch260-ubuntu2204

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





# 📦 Restliche Python-Deps
# 4) Rest über requirements.txt (einmal!)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt




# Nichts weiter – start.sh kümmert sich um Clone, Modelle, Jupyter etc.
COPY . .
RUN chmod +x /workspace/start.sh
RUN chmod +x /workspace/init.sh
RUN chmod +x /workspace/logs.sh

EXPOSE 8888 8000
CMD ["/bin/bash","-lc","/workspace/start.sh"]
