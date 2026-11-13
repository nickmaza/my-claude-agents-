# footage2robot: web app + API + conversion worker in one image.
# Python 3.11 is pinned because the RLDS exporter's dependencies do not build on 3.13.

FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libegl1 libgles2 libgl1 libglib2.0-0 openssl \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/ backend/
RUN pip install --no-cache-dir "./backend[rlds]"
COPY --from=web /web/dist frontend/dist

ENV FOOTAGE2ROBOT_FRONTEND=/app/frontend/dist \
    FOOTAGE2ROBOT_DATA_DIR=/data \
    FOOTAGE2ROBOT_MODEL_DIR=/opt/models
# Bake the tracking models into the image so containers start without downloads.
RUN python -c "from footage2robot.extract.models import ensure_model; [ensure_model(m) for m in ('pose_full', 'hand', 'interactive_segmenter')]"

# Runs as root on purpose: hosts such as Render mount the persistent disk at /data owned by
# root, and a non-root user could not write jobs to it.
RUN mkdir -p /data
VOLUME /data
EXPOSE 8000
CMD ["footage2robot", "serve", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
