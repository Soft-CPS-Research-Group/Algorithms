# Use a slim Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only the requirements first to take advantage of caching
COPY requirements.txt .

# Install dependencies with pip, avoid saving cached files to shrink image
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Now copy the rest of your project files
COPY . .

ENV OPEVA_BASE_DIR=/data

# Define the entrypoint for your container
ENTRYPOINT ["python", "run_experiment.py"]
