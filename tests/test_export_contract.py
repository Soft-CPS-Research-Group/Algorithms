"""Portable contract failures that must be detected before any runtime loads files."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from algorithms.configuration.common import SimulatorExportConfig
from algorithms.export.portable import confined_path, decode, encode, read_json
from algorithms.export.verification import _compare

FIXTURES = Path(__file__).parents[1] / "contracts/bundle/v2/fixtures"


@pytest.mark.parametrize(
    "path",
    [
        "../model.onnx",
        "/tmp/model.onnx",
        "C:/model.onnx",
        "a/../model.onnx",
        "a\\model.onnx",
        "./model.onnx",
        "missing.onnx",
    ],
)
def test_paths_are_portable_confined_existing_files(tmp_path, path):
    with pytest.raises(ValueError):
        confined_path(tmp_path, path)


def test_symlinks_are_rejected_even_when_the_target_is_inside_bundle(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"x")
    (tmp_path / "alias.onnx").symlink_to("model.onnx")
    with pytest.raises(ValueError, match="Symlink"):
        confined_path(tmp_path, "alias.onnx")


def test_numeric_encoding_preserves_unlimited_and_missing_meanings():
    source = {"upper": float("inf"), "lower": -float("inf"), "missing": np.nan, "threshold": 0.0}
    wire = json.dumps(encode(source), allow_nan=False)
    restored = decode(json.loads(wire))
    assert restored["upper"] == float("inf") and restored["lower"] == -float("inf")
    assert np.isnan(restored["missing"]) and restored["threshold"] == 0.0


@pytest.mark.parametrize("text", ['{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}'])
def test_nonstandard_or_ambiguous_json_is_rejected(tmp_path, text):
    path = tmp_path / "payload.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_json(path)


def test_legacy_false_and_null_do_not_enable_bulk_exports_in_normal_profile():
    export = SimulatorExportConfig(
        final_episode_only=False, kpis_final_episode_only=None, timeseries_final_episode_only=False
    )
    assert export.kpis_final_episode_only is export.timeseries_final_episode_only is True
    diagnostic = SimulatorExportConfig(profile="diagnostic", final_episode_only=False)
    assert diagnostic.kpis_final_episode_only is diagnostic.timeseries_final_episode_only is False


def test_mismatched_or_invalid_export_cannot_pass_equivalence():
    for actual in ([0.4], [float("nan")], [0.5, 0.5]):
        with pytest.raises(ValueError):
            _compare(actual, [0.5], {"maximum_absolute_error": 0.0}, 1e-5, 1e-4)


@pytest.mark.parametrize("kind", ["onnx", "policy"])
def test_shared_fixture_remains_executable_after_relocation(tmp_path, kind):
    from algorithms.export.consumer import verify_bundle

    root = tmp_path / "unrelated-location"
    shutil.copytree(FIXTURES / f"valid_{kind}", root)
    assert verify_bundle(root)["samples"] == 3


@pytest.mark.parametrize(
    "case", json.loads((FIXTURES / "invalid_cases.json").read_text()), ids=lambda case: case["name"]
)
def test_shared_invalid_fixtures_are_refused_before_loading(tmp_path, case):
    from utils.bundle_validator import BundleValidationError, validate_bundle_contract

    root = tmp_path / "bundle"
    shutil.copytree(FIXTURES / "valid_onnx", root)
    manifest = read_json(root / "artifact_manifest.json")
    operation = case["operation"]
    if operation == "delete":
        (root / case["path"]).unlink()
    elif operation == "corrupt":
        (root / case["path"]).write_bytes(b"not-the-exported-model")
    elif operation == "artifact_path":
        manifest["agent"]["artifacts"][0]["path"] = case["value"]
    elif operation == "version":
        manifest["manifest_version"] = case["value"]
    elif operation == "format":
        manifest["agent"]["artifacts"][0]["format"] = case["value"]
    elif operation == "duplicate_node":
        manifest["execution"]["nodes"].append(manifest["execution"]["nodes"][0])
    elif operation == "extra_file":
        (root / "unsealed.json").write_text("{}")
    elif operation == "symlink":
        target = tmp_path / "target.onnx"
        (root / case["path"]).replace(target)
        (root / case["path"]).symlink_to(target)
    with pytest.raises(BundleValidationError):
        validate_bundle_contract(manifest, root)


def test_unknown_rule_policy_never_falls_back_to_zero(tmp_path):
    from algorithms.export.policy_json import load_policy

    path = tmp_path / "unknown.json"
    path.write_text(
        json.dumps({"format": "algorithms_policy_v1", "algorithm": "DoesNotExist", "state": {}})
    )
    with pytest.raises(ValueError, match="Unsupported"):
        load_policy(path)


@pytest.mark.parametrize("change", ["count", "bounds", "format", "evidence", "execution_version"])
def test_inconsistent_manifest_semantics_are_rejected(change):
    from utils.bundle_validator import BundleValidationError, validate_bundle_contract

    root = FIXTURES / "valid_onnx"
    manifest = read_json(root / "artifact_manifest.json")
    if change == "count":
        manifest["topology"]["num_agents"] = 2
    elif change == "bounds":
        manifest["environment"]["action_bounds"][0]["low"] = [2.0]
    elif change == "format":
        manifest["execution"]["nodes"][0]["format"] = "algorithms_policy"
    elif change == "evidence":
        manifest["execution"]["nodes"][0]["verification"] = "fixture.json"
    else:
        manifest["execution"]["version"] = 99
    with pytest.raises(BundleValidationError):
        validate_bundle_contract(manifest, root)


def test_known_research_export_is_rejected_during_lightweight_validation(tmp_path):
    from algorithms.configuration.project import validate_config
    from tests.e2e.test_common_job_profiles import config_for

    config = config_for("TIMARL", tmp_path)
    config["outputs"] = {"model_artifacts": "bundle"}
    config["bundle"] = {"require_deployable": True}
    with pytest.raises(ValueError, match="specialised export runtime"):
        validate_config(config)
