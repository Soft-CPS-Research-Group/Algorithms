# Use a slim Python base image
FROM python:3.10-slim

ARG TARGETARCH

# Set working directory
WORKDIR /app

# Copy only the requirements first to take advantage of caching
COPY requirements.txt .

# Install dependencies with pip, avoid saving cached files to shrink image.
# The amd64 image keeps the CUDA 12.1 PyTorch wheels used by Deucalion/server GPU
# jobs. Those +cu121 wheels are not published for linux/arm64, so the arm64 image
# uses the matching PyPI arm64 PyTorch packages instead.
RUN set -eux; \
    pip install --upgrade pip; \
    if [ "${TARGETARCH:-}" = "arm64" ]; then \
        sed -E \
            -e '/^torch==.*\+cu121$/d' \
            -e '/^torchvision==.*\+cu121$/d' \
            -e '/^torchaudio==.*\+cu121$/d' \
            -e '/^--extra-index-url https:\/\/download\.pytorch\.org\/whl\/cu121$/d' \
            requirements.txt > /tmp/requirements-arm64.txt; \
        pip install --no-cache-dir -r /tmp/requirements-arm64.txt; \
        pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Now copy the rest of your project files
COPY . .

ENV OPEVA_BASE_DIR=/data

# Define the entrypoint for your container
ENTRYPOINT ["python", "run_experiment.py"]
