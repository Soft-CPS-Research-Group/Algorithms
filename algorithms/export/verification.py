"""Execute exported files against chronological samples from the evaluated policy."""

import numpy as np

from algorithms.export.portable import confined_path, decode


def onnx_session(path):
    import onnx
    import onnxruntime as ort

    model = onnx.load(str(path), load_external_data=False)
    # External tensors must be packaged explicitly by a supported future contract.
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise ValueError("External ONNX tensor files are not supported by this contract")
    onnx.checker.check_model(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    ), {
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "provider": "CPUExecutionProvider",
        "opsets": {item.domain or "ai.onnx": item.version for item in model.opset_import},
    }


def verify_leaf(metadata, root, samples, settings):
    atol, rtol = settings.get("verification_atol", 1e-5), settings.get("verification_rtol", 1e-4)
    report = {
        "status": "unverified",
        "samples": len(samples),
        "atol": atol,
        "rtol": rtol,
        "maximum_absolute_error": 0.0,
        "artifacts": [],
    }
    if not samples:
        report["reason"] = "no_final_evaluation_samples"
        return report
    if metadata.get("format") == "algorithms_policy":
        from algorithms.export.policy_json import load_policy

        policy = load_policy(confined_path(root, metadata["artifacts"][0]["path"]))
        for sample in samples:
            hook = getattr(policy, "set_episode_context", None)
            if callable(hook):
                hook(episode_step=sample["episode_step"], next_episode_step=None)
            actual = policy.predict(
                decode(sample["observations"]), True, context=decode(sample["context"])
            )
            _compare(actual, decode(sample["output"]), report, atol, rtol)
        report["status"] = "passed"
        return report
    if metadata.get("format") != "onnx" or metadata.get("deployable") is False:
        report["reason"] = metadata.get("runtime_only_reason", "requires_specialised_runtime")
        return report
    for index, artifact in enumerate(metadata.get("artifacts", [])):
        if (
            artifact.get("deployable") is False
            or (artifact.get("config") or {}).get("deployable") is False
        ):
            report["reason"] = "runtime_dependencies_not_embedded"
            return report
        session, runtime = onnx_session(confined_path(root, artifact["path"]))
        inputs = session.get_inputs()
        joint = (artifact.get("config") or {}).get("scope") == "joint"
        if len(inputs) != 1 and not joint:
            report["reason"] = "requires_multi_input_adapter"
            return report
        checked = 0
        for sample in samples:
            observations, expected = decode(sample["observations"]), decode(sample["output"])
            if joint:
                if len(inputs) != len(observations):
                    raise ValueError("Joint policy input cardinality differs from evaluation")
                feed = {
                    item.name: np.asarray(obs, dtype=np.float32).reshape(1, -1)
                    for item, obs in zip(inputs, observations)
                }
                actual = session.run(None, feed)[0][0]
                _compare(actual, expected, report, atol, rtol)
                checked += 1
                continue
            if not isinstance(observations, (list, tuple)) or index >= len(observations):
                report["reason"] = "requires_joint_input_adapter"
                return report
            value = np.asarray(observations[index], dtype=np.float32).reshape(1, -1)
            if (artifact.get("config") or {}).get("input_adapter") == "append_scalar_context":
                signal = decode(sample["context"])
                value = np.concatenate(
                    [value, [[0.0 if signal is None else float(signal)]]], axis=1
                ).astype(np.float32)
            width = inputs[0].shape[-1]
            if isinstance(width, int) and width != value.shape[-1]:
                report["reason"] = "requires_policy_context_or_other_topology"
                return report
            actual = session.run(None, {inputs[0].name: value})[0].reshape(-1)
            _compare(actual, np.asarray(expected[index]).reshape(-1), report, atol, rtol)
            checked += 1
        report["artifacts"].append(
            {
                "path": artifact["path"],
                "samples": checked,
                "runtime": runtime,
                "inputs": [{"name": i.name, "shape": i.shape, "type": i.type} for i in inputs],
                "outputs": [
                    {"name": i.name, "shape": i.shape, "type": i.type}
                    for i in session.get_outputs()
                ],
            }
        )
    report["status"] = "passed"
    return report


def _compare(actual, expected, report, atol, rtol):
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise ValueError("Exported policy has a different output cardinality")
        for a, e in zip(actual, expected):
            _compare(a, e, report, atol, rtol)
        return
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if (
        actual.shape != expected.shape
        or not np.isfinite(actual).all()
        or not np.isfinite(expected).all()
    ):
        raise ValueError("Exported policy shape or finiteness differs from evaluation")
    error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    report["maximum_absolute_error"] = max(report["maximum_absolute_error"], error)
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise ValueError(
            f"Exported policy differs from evaluated policy (max absolute error {error})"
        )
