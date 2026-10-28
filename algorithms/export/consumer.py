"""Reference v2 runtime for verified ONNX/JSON policies and explicit compositions.

Inputs are ordered vectors at the declared raw/encoded boundary. Production
payload-to-vector adapters must implement the environment metadata contract.
"""

from pathlib import Path

import numpy as np

from algorithms.export.portable import confined_path, read_json
from algorithms.export.verification import onnx_session


class BundleRuntime:
    def __init__(self, root, manifest=None, *, validate=True):
        self.root = Path(root)
        self.manifest = manifest or read_json(self.root / "artifact_manifest.json")
        if validate:
            from utils.bundle_validator import validate_bundle_contract

            validate_bundle_contract(self.manifest, self.root)
            if self.manifest.get("manifest_version") != 2:
                raise ValueError(
                    "Legacy bundles need explicit migration and new equivalence evidence"
                )
            if not self.manifest["deployment"]["eligible"]:
                raise ValueError("Bundle is not eligible for this runtime")
            if any(
                node.get("format") == "algorithms_policy"
                for node in self.manifest["execution"]["nodes"]
            ):
                from algorithms import __version__

                version = (
                    self.manifest.get("build", {}).get("versions", {}).get("energaize-algorithms")
                )
                if version != __version__:
                    raise ValueError(
                        f"Executable JSON policy requires Algorithms {version}; installed {__version__}"
                    )
        self.nodes = {node["id"]: node for node in self.manifest["execution"]["nodes"]}
        self.models = {}
        for identifier, node in self.nodes.items():
            if node["kind"] != "agent":
                continue
            artifacts = [
                a for a in self.manifest["agent"]["artifacts"] if a["node_id"] == identifier
            ]
            if node["format"] == "algorithms_policy":
                from algorithms.export.policy_json import load_policy

                self.models[identifier] = load_policy(
                    confined_path(self.root, artifacts[0]["path"])
                )
            elif node["format"] == "onnx":
                self.models[identifier] = [
                    (item, onnx_session(confined_path(self.root, item["path"]))[0])
                    for item in artifacts
                ]
            else:
                raise ValueError(
                    f"No executable adapter for {node['algorithm']} ({node['format']})"
                )

    def predict(
        self,
        observations,
        *,
        raw_observations=None,
        encoded_observations=None,
        episode_step=None,
        context=None,
    ):
        raw = observations if raw_observations is None else raw_observations
        encoded = observations if encoded_observations is None else encoded_observations

        def execute(identifier, raw, encoded, signal):
            node = self.nodes[identifier]
            if node["kind"] == "pipeline":
                for child in node["children"]:
                    signal = execute(child, raw, encoded, signal)
                return signal
            if node["kind"] == "ensemble":
                per_member = isinstance(signal, (list, tuple, np.ndarray)) and len(signal) == len(
                    node["children"]
                )
                return [
                    execute(
                        child,
                        [raw[index]],
                        [encoded[index]],
                        signal[index] if per_member else signal,
                    )[0]
                    for index, child in enumerate(node["children"])
                ]
            values = raw if node["input"] == "raw" else encoded
            model = self.models[identifier]
            if node["format"] == "algorithms_policy":
                hook = getattr(model, "set_episode_context", None)
                if callable(hook):
                    hook(episode_step=episode_step, next_episode_step=None)
                return model.predict(values, True, context=signal)
            outputs = []
            for index, (artifact, session) in enumerate(model):
                inputs = session.get_inputs()
                joint = (artifact.get("config") or {}).get("scope") == "joint"
                rows = values if joint else [values[index]]
                if len(inputs) != len(rows):
                    raise ValueError("Policy input cardinality does not match manifest")
                feed = {
                    item.name: np.asarray(row, dtype=np.float32).reshape(1, -1)
                    for item, row in zip(inputs, rows)
                }
                if (artifact.get("config") or {}).get("input_adapter") == "append_scalar_context":
                    feed[inputs[0].name] = np.concatenate(
                        [feed[inputs[0].name], [[0.0 if signal is None else float(signal)]]], axis=1
                    ).astype(np.float32)
                actual = session.run(None, feed)[0]
                if joint:
                    return actual[0].tolist()
                outputs.append(actual.reshape(-1).tolist())
            return outputs

        result = execute(self.manifest["execution"]["root"], raw, encoded, context)
        bounds = self.manifest.get("environment", {}).get("action_bounds", [])
        if bounds:
            from algorithms.export.portable import decode

            if len(result) != len(bounds):
                raise ValueError("Policy output cardinality differs from action contract")
            submitted = []
            for values, bound in zip(result, bounds):
                bound = bound[0] if isinstance(bound, list) else bound
                low, high = decode(bound["low"]), decode(bound["high"])
                action = np.asarray(values, dtype=float)
                if action.shape != np.asarray(low).shape or not np.isfinite(action).all():
                    raise ValueError("Malformed policy output")
                clipped = np.clip(action, low, high)
                if self.manifest.get("action_policy") == "reject" and not np.array_equal(
                    clipped, action
                ):
                    raise ValueError("Policy output violates action bounds")
                submitted.append(clipped.tolist())
            result = submitted
        return result


def verify_bundle(root):
    from algorithms.export.portable import decode
    from algorithms.export.verification import _compare

    runtime = BundleRuntime(root)
    execution = runtime.manifest["execution"]
    root_node = next(node for node in execution["nodes"] if node["id"] == execution["root"])
    samples = read_json(confined_path(root, root_node["references"]))["samples"]
    report = {"maximum_absolute_error": 0.0, "samples": len(samples)}
    for sample in samples:
        actual = runtime.predict(
            decode(sample["observations"]),
            raw_observations=decode(sample.get("raw_observations")),
            encoded_observations=decode(sample.get("encoded_observations")),
            episode_step=sample["episode_step"],
            context=decode(sample["context"]),
        )
        _compare(
            actual, decode(sample.get("submitted_actions", sample["output"])), report, 1e-5, 1e-4
        )
    if not samples:
        raise ValueError("Bundle has no final-evaluation references")
    report["status"] = "passed"
    return report


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Validate and replay a portable exported policy")
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_bundle(args.bundle), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
