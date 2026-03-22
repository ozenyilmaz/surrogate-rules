# Dockerfile
# ----------
# Reproducible container for the surrogate-rules pipeline.
#
# Build:
#   docker build -t surrogate-rules .
#
# Run:
#   docker run --rm \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/outputs:/app/outputs \
#     -v $(pwd)/configs:/app/configs \
#     -e MONGO_URI=mongodb://host.docker.internal:27017 \
#     surrogate-rules \
#     python scripts/run_experiment.py --config configs/mushroom.yaml
#
# NOTE: A valid Gurobi Web Licence Service (WLS) or mounted licence file
# is required.  Set GRB_LICENSE_FILE env var or mount /opt/gurobi/gurobi.lic.

FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps (layer-cached separately from code)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Outputs volume mount point
RUN mkdir -p /app/outputs /app/data

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "scripts/run_experiment.py"]
CMD ["--help"]
