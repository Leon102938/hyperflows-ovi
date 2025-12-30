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



# 📦 Restliche Python-Deps
# 4) Rest über requirements.txt (einmal!)
COPY requirements.txt /tmp/requirements.txt
RUN python -V && python -m pip -V \
 && python -m pip install --no-cache-dir -r /tmp/requirements.txt


# FlashAttention (Python 3.10 / torch 2.6.0+cu124 / CXX11_ABI=False)
RUN python -m pip uninstall -y flash-attn flash_attn || true && \
    python -m pip install --no-cache-dir --no-deps \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"


# =========================
# Z-Image Turbo (separates venv, nutzt denselben HF cache wie OVI)
# =========================
ENV ZIMAGE_VENV=/opt/venvs/zimage

RUN mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_HUB_CACHE}" \
 && python -m venv "${ZIMAGE_VENV}" \
 && "${ZIMAGE_VENV}/bin/pip" install -U pip wheel setuptools \
 && "${ZIMAGE_VENV}/bin/pip" install -U "huggingface_hub>=0.23" transformers accelerate safetensors \
 && "${ZIMAGE_VENV}/bin/pip" install -U git+https://github.com/huggingface/diffusers \
 && "${ZIMAGE_VENV}/bin/python" -c "from diffusers import ZImagePipeline; print('ZImagePipeline OK')"





# Nichts weiter – start.sh kümmert sich um Clone, Modelle, Jupyter etc.
COPY . .
RUN chmod +x /workspace/start.sh
RUN chmod +x /workspace/init.sh
RUN chmod +x /workspace/logs.sh

EXPOSE 8888 8000
CMD ["/bin/bash","-lc","/workspace/start.sh"]
