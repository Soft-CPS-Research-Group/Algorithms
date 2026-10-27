"""Portable v2 consumer contract. Integrity is checked before loading any model."""

import json
import math
from pathlib import Path

from algorithms.export.portable import confined_path, decode, read_json
from utils.training_provenance import digest, file_digest

FORMATS = {"onnx", "algorithms_policy", "research_torch", "ti_marl_torch", "policy_description"}


def validate_layout(manifest):
    from utils.bundle_validator import _validate_environment

    environment, topology = manifest["environment"], manifest["topology"]
    _validate_environment(environment)
    count = topology["num_agents"]
    if type(count) is not int or count < 1:
        raise ValueError("Topology requires a positive agent count")
    for key in ("observation_dimensions", "action_dimensions"):
        dimensions = topology[key]
        if len(dimensions) != count or any(type(v) is not int or v < 0 for v in dimensions):
            raise ValueError("Invalid topology dimensions")
    for key in ("observation_names", "encoders", "action_bounds"):
        if len(environment[key]) != count:
            raise ValueError("Environment cardinality differs from topology")
    names = environment["action_names_by_agent"]
    for index, bound in enumerate(environment["action_bounds"]):
        bound = bound[0] if isinstance(bound, list) and len(bound) == 1 else bound
        low, high = decode(bound["low"]), decode(bound["high"])
        action_names = names[str(index)] if isinstance(names, dict) else names[index]
        if not len(low) == len(high) == len(action_names) == topology["action_dimensions"][index]:
            raise ValueError("Action bounds/names differ from topology dimensions")
        if any(math.isnan(a) or math.isnan(b) or a > b for a, b in zip(low, high)):
            raise ValueError("Invalid action bounds")
    if manifest.get("action_policy", "clip") not in {"clip", "reject"}:
        raise ValueError("Unsupported action policy")


def validate_v2(manifest, root):
    json.dumps(manifest, allow_nan=False)
    for field in (
        "metadata",
        "environment",
        "topology",
        "pipeline",
        "algorithm",
        "execution",
        "deployment",
        "evidence",
        "files",
        "provenance",
        "build",
        "evaluation",
        "agent",
    ):
        if field not in manifest:
            raise ValueError(f"Missing v2 manifest field: {field}")
    if manifest.get("contract") != "algorithms.bundle.v2":
        raise ValueError("Unsupported bundle contract")
    if manifest["execution"].get("version") != 1:
        raise ValueError("Unsupported composition version")
    validate_layout(manifest)
    files = manifest["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("A v2 bundle requires a sealed file inventory")
    actual_files = set()
    for path in Path(root).rglob("*"):
        if path.is_symlink():
            raise ValueError("Symlink in bundle inventory")
        if path.is_file() and path != Path(root) / "artifact_manifest.json":
            actual_files.add(path.relative_to(root).as_posix())
    if actual_files != set(files):
        raise ValueError("Bundle inventory has missing or unsealed files")
    for path, expected in files.items():
        if file_digest(confined_path(root, path)) != expected:
            raise ValueError(f"Bundle hash mismatch: {path}")

    def checked(raw):
        path = confined_path(root, raw)
        if raw not in files:
            raise ValueError(f"Unsealed dependency: {raw}")
        return path

    for entry in manifest["evidence"].values():
        if file_digest(checked(entry["path"])) != entry["sha256"]:
            raise ValueError("Evidence hash mismatch")
    identity = manifest["provenance"]
    if identity.get("identity_sha256") != digest(
        {k: v for k, v in identity.items() if k != "identity_sha256"}
    ):
        raise ValueError("Provenance identity mismatch")
    nodes = manifest["execution"]["nodes"]
    if not nodes or any(
        node.get("kind") not in {"pipeline", "ensemble", "agent"} for node in nodes
    ):
        raise ValueError("Invalid composition node kind")
    ids = [node["id"] for node in nodes]
    if len(set(ids)) != len(ids) or manifest["execution"]["root"] not in ids:
        raise ValueError("Ambiguous composition identity")
    if sum(node["parent"] is None for node in nodes) != 1:
        raise ValueError("Composition must have exactly one root")
    node_map = {node["id"]: node for node in nodes}
    if node_map[manifest["execution"]["root"]]["parent"] is not None:
        raise ValueError("Execution root has a parent")
    visited = set()

    def visit(identifier):
        if identifier in visited:
            raise ValueError("Cycle or repeated composition child")
        visited.add(identifier)
        node = node_map[identifier]
        children = node.get("children", [])
        if (node["kind"] == "agent" and children) or (node["kind"] != "agent" and not children):
            raise ValueError("Invalid executable/composition children")
        for child in node.get("children", []):
            if child not in node_map or node_map[child]["parent"] != identifier:
                raise ValueError("Invalid composition parent")
            visit(child)

    visit(manifest["execution"]["root"])
    if visited != set(ids):
        raise ValueError("Unreachable composition node")
    for node in nodes:
        if node["kind"] == "agent":
            report = read_json(checked(node["verification"]))
            samples = read_json(checked(node["references"]))["samples"]
            if node["input"] not in {"raw", "encoded"}:
                raise ValueError("Unsupported observation boundary")
            if node["deployable"] and (
                report["status"] != "passed" or not samples or report["samples"] != len(samples)
            ):
                raise ValueError("Deployable node lacks equivalence evidence")
    artifact_ids = set()
    for artifact in manifest["agent"]["artifacts"]:
        if artifact["artifact_id"] in artifact_ids or artifact["node_id"] not in ids:
            raise ValueError("Ambiguous artifact identity")
        artifact_ids.add(artifact["artifact_id"])
        checked(artifact["path"])
        if artifact["format"] not in FORMATS:
            raise ValueError("Unsupported artifact format")
        node = node_map[artifact["node_id"]]
        if artifact["format"] != node.get("format") or artifact["deployable"] != node.get(
            "deployable"
        ):
            raise ValueError("Artifact and executable node disagree")
        for auxiliary in (artifact.get("config") or {}).get("auxiliary_paths", []):
            checked(auxiliary)
        if (
            artifact["format"] in {"research_torch", "ti_marl_torch", "policy_description"}
            and artifact["deployable"]
        ):
            raise ValueError("Research artifacts cannot claim standalone deployment")
    if {item["node_id"] for item in manifest["agent"]["artifacts"]} != {
        node["id"] for node in nodes if node["kind"] == "agent"
    }:
        raise ValueError("Every executable node needs an artifact")
    alias = manifest["metadata"].get("alias_mapping_path")
    if alias:
        value = read_json(checked(alias))
        if not isinstance(value, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in value.items()
        ):
            raise ValueError("Aliases must be a flat string mapping")
    if manifest["deployment"]["eligible"] and any(
        not node.get("deployable") for node in nodes if node["kind"] == "agent"
    ):
        raise ValueError("Deployment requires all executable nodes to pass")
    eligible = manifest["deployment"]["eligible"]
    if manifest["deployment"]["status"] != ("verified" if eligible else "research"):
        raise ValueError("Deployment status disagrees with eligibility")
    root_node = node_map[manifest["execution"]["root"]]
    if eligible and root_node["kind"] != "agent":
        report = read_json(checked(root_node["verification"]))
        samples = read_json(checked(root_node["references"]))["samples"]
        if report["status"] != "passed" or not samples or report["samples"] != len(samples):
            raise ValueError("Composition lacks whole-policy equivalence evidence")
