"""Serve ONNX models stored as MLflow artifacts via a simple Flask API."""

from __future__ import annotations

import argparse
import os
from typing import List

import mlflow
import numpy as np
from flask import Flask, jsonify, request
from loguru import logger

try:
    import onnxruntime as ort
except Exception as exc:  # pragma: no cover - dependency issue
    raise RuntimeError("onnxruntime is required for serving ONNX models") from exc

app = Flask(__name__)
SESSIONS: List[ort.InferenceSession] = []


def load_models(run_id: str, artifact_path: str) -> None:
    """Download ONNX models from MLflow and create inference sessions."""
    logger.info("Loading ONNX models from run %s", run_id)
    local_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
    for file in sorted(os.listdir(local_dir)):
        if file.endswith(".onnx"):
            session = ort.InferenceSession(os.path.join(local_dir, file))
            SESSIONS.append(session)
            logger.info("Loaded %s", file)
    if not SESSIONS:
        raise RuntimeError("No ONNX models found in the specified artifact path")


@app.post("/predict")
def predict():
    """Run inference for each agent using its ONNX model."""
    payload = request.get_json(force=True)
    observations = payload.get("observations", [])
    actions = []
    for obs, session in zip(observations, SESSIONS):
        inp = np.array(obs, dtype=np.float32)[None, :]
        action = session.run(None, {session.get_inputs()[0].name: inp})[0][0].tolist()
        actions.append(action)
    return jsonify({"actions": actions})


@app.post("/reward")
def reward():  # pragma: no cover - simple passthrough
    data = request.get_json(force=True)
    actions = data.get("actions", [])
    # Placeholder reward: negative sum of absolute actions
    reward_val = -float(np.abs(actions).sum())
    return jsonify({"reward": reward_val})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="MLflow run ID containing ONNX models")
    parser.add_argument(
        "--artifact-path",
        default="onnx_models",
        help="Path of the ONNX artifact within the MLflow run",
    )
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    load_models(args.run_id, args.artifact_path)
    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
