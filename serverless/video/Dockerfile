# AI Gen API v2 — Video worker (LTX-2.3 22B i2v + t2v)
#
# Build:
#   docker build -f serverless/video/Dockerfile -t <user>/ai-gen-video:latest .
#   docker push   <user>/ai-gen-video:latest
#
# Build context = repo root.
#
# Models (~44 GB) live on the RunPod network volume — start.sh symlinks them
# into ComfyUI's models tree. Do NOT bake the weights into the image.

FROM runpod/worker-comfyui:5.4.1-base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    COMFY_ROOT=/comfyui \
    COMFYUI_URL=http://127.0.0.1:8188 \
    VOLUME_OUTPUTS=/runpod-volume/outputs

# ComfyUI-KJNodes — required for the ltx/i2v workflow. Supplies:
#   - ColorMatch (MKL color correction after VAE drift)
#   - ResizeImageMaskNode + ResizeImagesByLongerEdge (input preprocessing)
# ltx/t2v doesn't use these, but i2v is unconditional, so the video worker
# is broken without this node pack.
RUN cd /comfyui/custom_nodes \
 && git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes \
 && if [ -f ComfyUI-KJNodes/requirements.txt ]; then \
        pip install --no-cache-dir -r ComfyUI-KJNodes/requirements.txt; \
    fi

COPY serverless/video/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ffmpeg is needed for the optional output watermark on .mp4 outputs. The
# base image typically already includes it via ComfyUI's video deps; this
# RUN is a defensive no-op if it's already present.
RUN command -v ffmpeg >/dev/null 2>&1 || (apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*)

COPY workflows.py /app/workflows.py
# Optional input-face validation. InsightFace weights stay on the network
# volume and are initialized only when require_detectable_face=true.
COPY safety.py /app/safety.py
# Optional text/logo output watermarking.
COPY watermark.py /app/watermark.py
COPY assets/metfone_genai_watermark.png /workspace/assets/metfone_genai_watermark.png
ENV INSIGHTFACE_MODEL_ROOT=/runpod-volume/insightface_models \
    BLOCKLIST_DIR=/runpod-volume/blocklist \
    BYPASS_LOG=/runpod-volume/face_filter_bypass.log

COPY serverless/video/handler.py /app/handler.py
COPY serverless/video/start.sh   /start_serverless.sh
RUN chmod +x /start_serverless.sh

WORKDIR /app

CMD ["/start_serverless.sh"]
