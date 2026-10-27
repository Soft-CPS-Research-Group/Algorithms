"""The common export transaction: explicit composition, executable files and evidence."""

from __future__ import annotations

import importlib.metadata
import os
import shutil
import subprocess
import sys
from pathlib import Path

from algorithms.export.portable import confined_path, decode, encode, manifest_values
from algorithms.export.verification import _compare, onnx_session, verify_leaf
from algorithms.runtime.atomic import atomic_json
from utils.training_provenance import digest, file_digest


def inspect_model_files(metadata, directory):
    reports = {}
    for artifact in metadata.get("artifacts", []):
        if (
            artifact.get("format", metadata.get("format")) == "onnx"
            and artifact["path"] not in reports
        ):
            _, details = onnx_session(confined_path(directory, artifact["path"]))
            reports[artifact["path"]] = details
    return reports


def export_tree(unit, root, context):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    nodes, artifacts = [], []
    settings = context["config"].get("bundle") or {}

    def walk(current, relative, parent=None, position=0):
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        identifier = relative or "root"
        children = getattr(current, "stages", None)
        kind = "pipeline" if children else "agent"
        if not children:
            children = getattr(current, "agents", None)
            if children:
                kind = "ensemble"
        node = {
            "id": identifier,
            "parent": parent,
            "position": position,
            "kind": kind,
            "algorithm": getattr(current, "agent_name", type(current).__name__),
        }
        nodes.append(node)
        if children:
            node["children"] = []
            node["references"] = f"{relative + '/' if relative else ''}references.json"
            atomic_json(
                directory / "references.json",
                {
                    "version": "algorithms.policy_references.v1",
                    "samples": getattr(current, "_export_references", []),
                },
            )
            for index, child in enumerate(children):
                child_path = f"{relative + '/' if relative else ''}{'stage' if kind == 'pipeline' else 'agent'}_{index}"
                node["children"].append(child_path)
                walk(child, child_path, identifier, index)
            return
        cap = getattr(current, "capabilities", None)
        node["input"] = getattr(cap, "observations", "undeclared")
        node["input_signal"] = getattr(cap, "input_signal", "undeclared")
        node["output_signal"] = getattr(cap, "output_signal", "undeclared")
        samples = getattr(current, "_export_references", [])
        native_metadata = current.export_artifacts(str(directory), context)
        node["model_runtimes"] = inspect_model_files(native_metadata, directory)
        if getattr(cap, "learning", None) == "none" and hasattr(current, "_export_initial_state"):
            atomic_json(
                directory / "policy.json",
                {
                    "format": "algorithms_policy_v1",
                    "algorithm": current.agent_name,
                    "state": current._export_initial_state,
                },
            )
            metadata = {
                "format": "algorithms_policy",
                "artifacts": [
                    {
                        "path": "policy.json",
                        "agent_index": 0,
                        "format": "algorithms_policy",
                        "config": {
                            "loader": "algorithms.export.policy_json:load_policy",
                            "scope": "joint",
                        },
                    }
                ],
            }
        else:
            metadata = native_metadata
            # Old rule JSON names did not identify executable Python semantics.
            if metadata.get("format") == "rule_based":
                metadata["format"] = "policy_description"
                metadata["deployable"] = False
                for item in metadata.get("artifacts", []):
                    item["format"] = "policy_description"
                    item["deployable"] = False
            if not metadata.get("artifacts"):
                atomic_json(directory / "policy_description.json", encode(metadata))
                metadata = {
                    "format": "policy_description",
                    "deployable": False,
                    "artifacts": [
                        {
                            "path": "policy_description.json",
                            "agent_index": 0,
                            "format": "policy_description",
                            "deployable": False,
                        }
                    ],
                }
        atomic_json(
            directory / "references.json",
            {"version": "algorithms.policy_references.v1", "samples": samples},
        )
        report = verify_leaf(metadata, directory, samples, settings)
        atomic_json(directory / "verification.json", report)
        node["verification"] = f"{relative + '/' if relative else ''}verification.json"
        node["references"] = f"{relative + '/' if relative else ''}references.json"
        node["verification_status"] = report["status"]
        node["reason"] = report.get("reason")
        node["format"] = metadata["format"]
        node["deployable"] = (
            report["status"] == "passed" and metadata.get("deployable") is not False
        )
        node["requirements"] = (
            ["energaize-algorithms"]
            if metadata["format"] == "algorithms_policy"
            else ["onnxruntime"]
            if metadata["format"] == "onnx"
            else ["specialised_runtime"]
        )
        node["topology_scope"] = "exported_layout"
        for local_index, item in enumerate(metadata.get("artifacts", [])):
            artifact = dict(item)
            artifact.update(
                node_id=identifier,
                artifact_id=f"{identifier}:{local_index}",
                deployable=node["deployable"],
            )
            if (
                parent is not None
                and next(n for n in nodes if n["id"] == parent)["kind"] == "ensemble"
            ):
                artifact["member_index"] = position
            artifact["output_signal"] = node["output_signal"]
            artifact["path"] = f"{relative + '/' if relative else ''}{item['path']}"
            cfg = dict(artifact.get("config") or {})
            if "auxiliary_paths" in cfg:
                cfg["auxiliary_paths"] = [
                    f"{relative + '/' if relative else ''}{path}" for path in cfg["auxiliary_paths"]
                ]
            artifact["config"] = cfg
            artifacts.append(artifact)
        # Diagnostic/research sidecars remain inside the sealed inventory even
        # when they are not independently executable artifacts.
        atomic_json(directory / "export_details.json", encode(metadata))

    walk(unit, "")
    leaves = [node for node in nodes if node["kind"] == "agent"]
    verified = bool(leaves) and all(node["deployable"] for node in leaves)
    composed = nodes[0]["kind"] != "agent"
    if verified and composed:
        from algorithms.export.consumer import BundleRuntime

        runtime = BundleRuntime(
            root,
            {"execution": {"root": "root", "nodes": nodes}, "agent": {"artifacts": artifacts}},
            validate=False,
        )
        report = {"maximum_absolute_error": 0.0, "samples": 0, "status": "passed"}
        for sample in getattr(unit, "_export_references", []):
            actual = runtime.predict(
                decode(sample["observations"]),
                raw_observations=decode(sample.get("raw_observations")),
                encoded_observations=decode(sample.get("encoded_observations")),
                episode_step=sample["episode_step"],
                context=decode(sample["context"]),
            )
            _compare(
                actual,
                decode(sample["output"]),
                report,
                settings.get("verification_atol", 1e-5),
                settings.get("verification_rtol", 1e-4),
            )
            report["samples"] += 1
        verified = report["samples"] > 0
        atomic_json(root / "verification.json", report)
        nodes[0]["verification"] = "verification.json"
    deployment = {
        "eligible": verified,
        "status": "verified" if verified else "research",
        "consumer_contract": "algorithms.bundle.v2",
        "requires": sorted({r for n in leaves for r in n["requirements"]}),
        "reasons": [f"{node['id']}: {node['reason']}" for node in leaves if not node["deployable"]],
    }
    if composed:
        deployment["requires"].append("algorithms.composition.v1")
    if settings.get("require_deployable") and not deployment["eligible"]:
        raise ValueError(
            f"Requested deployable bundle, but export is {deployment['status']}: {deployment['reasons']}"
        )
    return {
        "format": "composition" if composed else leaves[0]["format"],
        "artifacts": artifacts,
        "deployable": deployment["eligible"],
        "composition": {"version": 1, "root": "root", "nodes": nodes},
        "deployment": deployment,
    }


def build_identity():
    from algorithms import __version__

    versions = {"python": sys.version.split()[0]}
    for name in (
        "energaize-algorithms",
        "softcpsrecsimulator",
        "torch",
        "numpy",
        "onnx",
        "onnxruntime",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    versions["energaize-algorithms"] = __version__
    import json

    packaged_path = Path(__file__).parents[1] / "_build_info.json"
    packaged = json.loads(packaged_path.read_text()) if packaged_path.is_file() else {}
    commit = os.environ.get("ALGORITHMS_COMMIT") or packaged.get("commit")
    source = (
        "build_environment"
        if os.environ.get("ALGORITHMS_COMMIT")
        else "wheel_metadata"
        if commit
        else "unavailable"
    )
    dirty = packaged.get("dirty")
    if not commit:
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).parents[2],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            source = "source_checkout"
            dirty = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    cwd=Path(__file__).parents[2],
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
            )
        except (OSError, subprocess.CalledProcessError):
            pass
    return {
        "commit": commit,
        "commit_source": source,
        "source_dirty": dirty,
        "image_digest": os.environ.get("ALGORITHMS_IMAGE_DIGEST"),
        "image_digest_status": "injected"
        if os.environ.get("ALGORITHMS_IMAGE_DIGEST")
        else "not_provided_by_launcher",
        "versions": versions,
    }


def complete_manifest(manifest, context, wrapper):
    root, job = context.paths["bundle_dir"], context.paths["job_dir"]
    evidence = root / "evidence"
    evidence.mkdir(exist_ok=True)
    alias = manifest["metadata"].get("alias_mapping_path")
    if alias:
        source = Path(alias).expanduser()
        if not source.is_file():
            source = root / alias
        if not source.is_file():
            raise ValueError(f"Missing configured alias map: {alias}")
        if source.resolve() != (root / "aliases.json").resolve():
            shutil.copyfile(source, root / "aliases.json")
        manifest["metadata"]["alias_mapping_path"] = "aliases.json"
    files = {}
    for source, name in (
        (job / "config.initial.json", "config.initial.json"),
        (context.paths["resolved_config_path"], "config.resolved.yaml"),
        (job / "inputs.json", "inputs.json"),
        (context.paths["results_dir"] / "kpis.json", "kpis.json"),
    ):
        if not source.is_file():
            raise ValueError(f"Missing job provenance file: {source.name}")
        target = evidence / name
        shutil.copyfile(source, target)
        files[name] = {"path": target.relative_to(root).as_posix(), "sha256": file_digest(target)}
    manifest["manifest_version"] = 2
    manifest["contract"] = "algorithms.bundle.v2"
    manifest["algorithm"] = {
        "name": manifest["pipeline"][0]["algorithm"]
        if len(manifest["pipeline"]) == 1
        else "Pipeline",
        "stages": manifest["pipeline"],
    }
    manifest["execution"] = manifest["agent"]["composition"]
    manifest["deployment"] = manifest["agent"]["deployment"]
    manifest["evidence"] = files
    manifest["build"] = build_identity()
    manifest["evaluation"] = getattr(wrapper, "evaluated_policy", None)
    manifest["environment"]["units_status"] = "not_declared_by_simulator_metadata"
    manifest["randomness"] = context.config["runtime"].get("randomness")
    manifest["input_identity"] = context.config["runtime"].get("input_identity")
    manifest["final_topology_sha256"] = digest(manifest["topology"])
    manifest["numeric_encoding"] = {
        "version": 1,
        "$number": ["+inf", "-inf", "nan"],
        "semantics": "unbounded_positive/unbounded_negative/unavailable",
    }
    manifest["action_policy"] = context.config["simulator"].get("action_policy", "clip")
    normalized = manifest_values(manifest)
    manifest.clear()
    manifest.update(normalized)
    return manifest
