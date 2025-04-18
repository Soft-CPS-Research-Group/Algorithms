# Dockerfile

FROM python:3.10-slim

# System setup
WORKDIR /app

# Copy all project files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Create entrypoint
ENTRYPOINT ["python", "docker_run.py"]
