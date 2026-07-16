# AI Gen API v2 — Image worker (FLUX.2 Klein 9B + face swap)
#
# Build:
#   docker build -f serverless/image/Dockerfile -t <user>/ai-gen-image:latest .
#   docker push   <user>/ai-gen-image:latest
#
# Build context = repo root (so we can COPY workflows.py from one level up).
#
# Models live on the RunPod network volume — start.sh symlinks them in.
# Do NOT bake the 27 GB of weights into the image.

FROM runpod/worker-comfyui:5.4.1-base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    COMFY_ROOT=/comfyui \
    COMFYUI_URL=http://127.0.0.1:8188

# LanPaint custom node (required for FLUX head-swap workflow)
RUN cd /comfyui/custom_nodes \
 && git clone --depth 1 https://github.com/scraed/LanPaint \
 && if [ -f LanPaint/requirements.txt ]; then \
        pip install --no-cache-dir -r LanPaint/requirements.txt; \
    fi

# Python deps used by the handler (runpod-sdk is already in the base image,
# but pin it again here so the build doesn't break if the base bumps it out)
COPY serverless/image/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Shared workflow builders (also used by pod-mode main.py)
COPY workflows.py /app/workflows.py
COPY image_output.py /app/image_output.py

# Face-blocklist filter — imported lazily by the handler when face_filter=true.
# Model weights (~280 MB) are NOT baked into the image — they download once
# on first use and cache on the network volume at /runpod-volume/insightface_models/.
COPY safety.py /app/safety.py
# Logo/flag blocklist filter — imported lazily on first logo_filter=true.
# CLIP ViT-B/32 weights (~150 MB) cache at /runpod-volume/clip_models/.
COPY logo_safety.py /app/logo_safety.py
# Optional text/logo output watermarking. Uses Pillow from the base image.
COPY watermark.py /app/watermark.py
COPY assets/metfone_genai_watermark.png /workspace/assets/metfone_genai_watermark.png
ENV INSIGHTFACE_MODEL_ROOT=/runpod-volume/insightface_models \
    CLIP_MODEL_ROOT=/runpod-volume/clip_models \
    BLOCKLIST_DIR=/runpod-volume/blocklist \
    BLOCKLIST_LOGOS_DIR=/runpod-volume/blocklist_logos \
    BYPASS_LOG=/runpod-volume/face_filter_bypass.log

# Handler + entrypoint
COPY serverless/image/handler.py /app/handler.py
COPY serverless/image/start.sh   /start_serverless.sh
RUN chmod +x /start_serverless.sh

WORKDIR /app

# Override the base image's CMD — start.sh links models, boots ComfyUI in the
# background, then execs the handler so SIGTERM from RunPod reaches Python.
CMD ["/start_serverless.sh"]
