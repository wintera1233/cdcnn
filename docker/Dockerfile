# CDCNN training environment, pinned to the stack of the canonical 2026-09-11 run.
#
# Build:
#   docker build -t cdcnn:cu121 -f docker/Dockerfile .
# Run (repository mounted, GPU attached):
#   docker run --rm --gpus all -v "$PWD":/workspace -w /workspace cdcnn:cu121 \
#     python scripts/run_cdcnn_v6_full.py gpu-smoke --config configs/cdcnn_v6.json
#
# The image carries no project code or data: the repository is mounted at
# /workspace so that run directories, checkpoints and the immutable Dataset stay
# on the host and are never copied into an image layer.
# Bookworm rather than the canonical run's 3.11.2 (a Bullseye image): Bullseye's
# security repository is archived and apt now 404s. The Python patch version
# therefore differs from that run; torch, numpy, pandas and scikit-learn are
# pinned to its exact versions, which is what training determinism depends on.
FROM python:3.11-slim-bookworm

# git: scripts/run_cdcnn_v6_full.py records the revision in run manifests.
# libgomp1: required by scikit-learn and torch CPU kernels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Mirrors the canonical run's interpreter location, which its manifests record
# as /opt/venv/bin/python.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Required by the deterministic CUDA matmul path the protocol enables.
    CUBLAS_WORKSPACE_CONFIG=":4096:8"

COPY docker/requirements-cu121.txt /tmp/requirements-cu121.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements-cu121.txt \
    && rm /tmp/requirements-cu121.txt

# Training is launched explicitly; the image must never start a run by itself.
WORKDIR /workspace
CMD ["python", "-c", "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"]
