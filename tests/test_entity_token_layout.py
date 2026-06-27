"""Tests for ``EntityTokenLayoutBuilder``.

Single-fixture style: most tests share a builder + Building_1 layout built
from the bundled sample payload. Failure-mode tests build their own.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from tests._entity_sample_obs_names import (
    load_sample_observation_names_for_first_building,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    from utils.entity_tokenizer_schema import load_entity_tokenizer_config

    return load_entity_tokenizer_config(
        "configs/tokenizers/entity_default.json"
    )


@pytest.fixture
def builder_and_obs(cfg):
    from algorithms.utils.entity_token_layout import EntityTokenLayoutBuilder

    builder = EntityTokenLayoutBuilder(cfg)
    obs_names = load_sample_observation_names_for_first_building()
    action_names = ["electrical_storage", "electric_vehicle_storage"]
    layout = builder.build("Building_1", obs_names, action_names)
    return builder, obs_names, layout


def _find_segment(layout, family, type_name, instance_substr=None):
    for s in layout.segments:
        if s.family == family and s.type_name == type_name:
            if instance_substr is None or instance_substr in (
                s.instance_id or ""
            ):
                return s
    return None


# ---------------------------------------------------------------------------
# Smoke + happy-path
# ---------------------------------------------------------------------------


def test_dataclasses_constructible():
    from algorithms.utils.entity_token_layout import (
        BuildingTokenLayout,
        NfcExpression,
        TokenSegment,
    )

    seg = TokenSegment(
        family="sro",
        type_name="district_time",
        instance_id="Building_1",
        feature_indices=(0, 1, 2),
        feature_names=(
            "district__month",
            "district__day_type",
            "district__hour",
        ),
    )
    nfc = NfcExpression(
        op="subtract", left_index_in_segment=0, right_index_in_segment=1
    )
    layout = BuildingTokenLayout(
        building_id="Building_1",
        segments=(seg,),
        n_sro=1,
        n_ca=0,
        ca_action_names=(),
        excluded_feature_names=(),
    )
    assert layout.n_sro == 1
    assert seg.derived is None
    assert nfc.op == "subtract"


def test_uses_real_sample_payload(builder_and_obs):
    _, obs_names, layout = builder_and_obs
    classified = set()
    for seg in layout.segments:
        classified.update(seg.feature_indices)
    excluded = len(layout.excluded_feature_names)
    assert len(classified) + excluded == len(obs_names), (
        f"unclassified: total={len(obs_names)} "
        f"classified={len(classified)} excluded={excluded}"
    )
    assert layout.ca_action_names == (
        "electrical_storage",
        "electric_vehicle_storage",
    )


# ---------------------------------------------------------------------------
# ---
# ---------------------------------------------------------------------------


def test_classifies_district_time_to_sro_singleton(builder_and_obs):
    _, obs, layout = builder_and_obs
    seg = _find_segment(layout, "sro", "district_time")
    assert seg is not None
    assert obs.index("district__hour") in seg.feature_indices


def test_classifies_district_pricing_current_separately_from_forecast(
    builder_and_obs,
):
    _, obs, layout = builder_and_obs
    cur = _find_segment(layout, "sro", "district_pricing_current")
    fwd = _find_segment(layout, "sro", "district_pricing_forecast")
    assert cur is not None and fwd is not None
    assert obs.index("district__electricity_pricing") in cur.feature_indices
    forecast_names = [n for n in obs if "electricity_pricing_predicted" in n]
    assert any(obs.index(n) in fwd.feature_indices for n in forecast_names)


def test_classifies_district_carbon_separately_from_pricing(builder_and_obs):
    _, obs, layout = builder_and_obs
    carbon = _find_segment(layout, "sro", "district_carbon")
    cur = _find_segment(layout, "sro", "district_pricing_current")
    fwd = _find_segment(layout, "sro", "district_pricing_forecast")
    assert carbon is not None
    idx = obs.index("district__carbon_intensity")
    assert idx in carbon.feature_indices
    assert idx not in (cur.feature_indices if cur else ())
    assert idx not in (fwd.feature_indices if fwd else ())


def test_classifies_building_storage_state_to_sro(builder_and_obs):
    _, obs, layout = builder_and_obs
    seg = _find_segment(layout, "sro", "building_storage_state")
    assert seg is not None
    assert obs.index("electrical_storage_soc") in seg.feature_indices
    assert obs.index("electrical_storage_soc_ratio") in seg.feature_indices


def test_classifies_per_asset_pv_to_sro(builder_and_obs):
    _, obs, layout = builder_and_obs
    pv_segs = [
        s for s in layout.segments if s.family == "sro" and s.type_name == "pv"
    ]
    assert len(pv_segs) >= 1
    pv_obs = [n for n in obs if n.startswith("pv::")]
    classified = set()
    for s in pv_segs:
        classified.update(s.feature_indices)
    for n in pv_obs:
        assert obs.index(n) in classified


def test_classifies_per_asset_ev_connected_to_sro(builder_and_obs):
    _, _, layout = builder_and_obs
    segs = [
        s
        for s in layout.segments
        if s.family == "sro" and s.type_name == "ev_connected"
    ]
    assert len(segs) >= 1
    for s in segs:
        for n in s.feature_names:
            assert "::connected_ev::" in n


def test_classifies_per_asset_ev_incoming_to_sro(builder_and_obs):
    _, _, layout = builder_and_obs
    segs = [
        s
        for s in layout.segments
        if s.family == "sro" and s.type_name == "ev_incoming"
    ]
    assert len(segs) >= 1
    for s in segs:
        for n in s.feature_names:
            assert "::incoming_ev::" in n


def test_classifies_storage_prefix_to_ca(builder_and_obs):
    _, _, layout = builder_and_obs
    segs = [
        s
        for s in layout.segments
        if s.family == "ca" and s.type_name == "storage"
    ]
    assert len(segs) >= 1
    for s in segs:
        for n in s.feature_names:
            assert n.startswith("storage::")


def test_classifies_charger_prefix_to_ca(builder_and_obs):
    _, _, layout = builder_and_obs
    segs = [
        s
        for s in layout.segments
        if s.family == "ca" and s.type_name == "charger"
    ]
    assert len(segs) >= 1
    for s in segs:
        for n in s.feature_names:
            assert n.startswith("charger::")
            assert "::connected_ev::" not in n
            assert "::incoming_ev::" not in n


# ---------------------------------------------------------------------------
# ---
# ---------------------------------------------------------------------------


def test_nfc_segment_has_two_source_indices_and_subtract_op(builder_and_obs):
    _, obs, layout = builder_and_obs
    nfc = next((s for s in layout.segments if s.family == "nfc"), None)
    assert nfc is not None
    assert nfc.derived is not None
    assert nfc.derived.op == "subtract"
    assert nfc.feature_indices == (
        obs.index("non_shiftable_load"),
        obs.index("solar_generation"),
    )
    assert nfc.derived.left_index_in_segment == 0
    assert nfc.derived.right_index_in_segment == 1


def test_nfc_source_features_not_in_any_sro_group(builder_and_obs):
    _, _, layout = builder_and_obs
    for s in layout.segments:
        if s.family == "sro":
            assert "non_shiftable_load" not in s.feature_names
            assert "solar_generation" not in s.feature_names


# ---------------------------------------------------------------------------
# ---
# ---------------------------------------------------------------------------


def test_excluded_features_dropped_before_classification(builder_and_obs):
    _, _, layout = builder_and_obs
    assert "district__topology_version" in layout.excluded_feature_names
    for s in layout.segments:
        assert "district__topology_version" not in s.feature_names


# ---------------------------------------------------------------------------
# ---
# ---------------------------------------------------------------------------


def test_unmatched_feature_raises(cfg):
    from algorithms.utils.entity_token_layout import EntityTokenLayoutBuilder

    builder = EntityTokenLayoutBuilder(cfg)
    obs = load_sample_observation_names_for_first_building() + [
        "district__some_new_feature"
    ]
    with pytest.raises(ValueError, match="district__some_new_feature"):
        builder.build(
            "Building_1",
            obs,
            ["electrical_storage", "electric_vehicle_storage"],
        )


def test_ambiguous_pattern_raises():
    from algorithms.utils.entity_token_layout import EntityTokenLayoutBuilder
    from utils.entity_tokenizer_schema import EntityTokenizerConfig

    raw = json.loads(
        Path("configs/tokenizers/entity_default.json").read_text()
    )
    raw["sro_types"]["district_time_dup"] = {
        "entity_table": "district",
        "cardinality": "singleton",
        "feature_patterns": ["^district__hour$"],
        "input_dim_fallback": 1,
    }
    cfg = EntityTokenizerConfig.model_validate(raw)
    builder = EntityTokenLayoutBuilder(cfg)
    obs = load_sample_observation_names_for_first_building()
    with pytest.raises(
        ValueError,
        match=r"district__hour.*(district_time|district_time_dup)",
    ):
        builder.build(
            "Building_1",
            obs,
            ["electrical_storage", "electric_vehicle_storage"],
        )


def test_missing_nfc_source_raises(cfg):
    from algorithms.utils.entity_token_layout import EntityTokenLayoutBuilder

    builder = EntityTokenLayoutBuilder(cfg)
    obs = ["non_shiftable_load"]  # solar_generation missing
    with pytest.raises(ValueError, match="solar_generation"):
        builder.build("X", obs, [])


def test_ca_count_mismatch_raises(cfg):
    from algorithms.utils.entity_token_layout import EntityTokenLayoutBuilder

    builder = EntityTokenLayoutBuilder(cfg)
    obs = load_sample_observation_names_for_first_building()
    with pytest.raises(ValueError, match="CA count mismatch"):
        builder.build("Building_1", obs, ["electrical_storage"])


# ---------------------------------------------------------------------------
# ---
# ---------------------------------------------------------------------------


def test_sro_segment_order_follows_config_declaration(builder_and_obs):
    _, _, layout = builder_and_obs
    sros = [s for s in layout.segments if s.family == "sro"]
    declared = [
        "district_time",
        "district_weather_current",
        "district_weather_forecast",
        "district_carbon",
        "district_pricing_current",
        "district_pricing_forecast",
        "district_community_energy",
        "district_community_headroom",
        "district_community_history",
        "district_meta",
        "building_storage_state",
        "building_charging_phase_onehot",
        "building_charging_headroom",
        "building_charging_violation",
        "building_energy_current",
        "building_energy_history",
        "building_meta",
        "pv",
        "ev_connected",
        "ev_incoming",
    ]

    def first_occurrence(seq):
        out, seen = [], set()
        for x in seq:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    seen_in_layout = first_occurrence(s.type_name for s in sros)
    expected = [t for t in declared if t in seen_in_layout]
    assert seen_in_layout == expected


def test_per_asset_sro_segments_sorted_by_instance_id(builder_and_obs):
    _, _, layout = builder_and_obs
    for tname in ("pv", "ev_connected", "ev_incoming"):
        segs = [
            s
            for s in layout.segments
            if s.family == "sro" and s.type_name == tname
        ]
        if len(segs) > 1:
            ids = [s.instance_id for s in segs]
            assert ids == sorted(ids), f"{tname} ids not sorted: {ids}"


def test_segment_overall_order(builder_and_obs):
    _, _, layout = builder_and_obs
    families = [s.family for s in layout.segments]
    n_sro = layout.n_sro
    assert families[:n_sro] == ["sro"] * n_sro
    assert families[n_sro] == "nfc"
    assert families[n_sro + 1 :] == ["ca"] * layout.n_ca


# ---------------------------------------------------------------------------
# ---
# ---------------------------------------------------------------------------


def test_topology_changed_when_names_differ(builder_and_obs):
    builder, obs, _ = builder_and_obs
    new_obs = list(obs) + ["charger::Building_1/charger_99::power_kw"]
    assert (
        builder.topology_changed(
            "Building_1",
            new_obs,
            ["electrical_storage", "electric_vehicle_storage"],
        )
        is True
    )


def test_topology_unchanged_for_identical_names(builder_and_obs):
    builder, obs, _ = builder_and_obs
    assert (
        builder.topology_changed(
            "Building_1",
            obs,
            ["electrical_storage", "electric_vehicle_storage"],
        )
        is False
    )


def test_layout_is_cached(builder_and_obs):
    builder, obs, layout = builder_and_obs
    layout2 = builder.build(
        "Building_1",
        obs,
        ["electrical_storage", "electric_vehicle_storage"],
    )
    assert layout2 is layout


# ---------------------------------------------------------------------------
# ---
# ---------------------------------------------------------------------------


def test_no_external_imports():
    """``algorithms/utils/entity_token_layout.py`` must be portable: only
    stdlib + typing + re. No torch / numpy / pydantic / algorithms.* /
    utils.* imports at any depth."""
    src = Path("algorithms/utils/entity_token_layout.py").read_text()
    tree = ast.parse(src)
    forbidden = ("torch", "numpy", "pydantic", "algorithms.", "utils.")
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                if any(n.name.startswith(p) for p in forbidden):
                    bad.append(n.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if any(mod.startswith(p) for p in forbidden):
                bad.append(mod)
    assert not bad, f"forbidden imports in entity_token_layout.py: {bad}"


def test_coverage_accounting_matches_spec(builder_and_obs):
    """Expected feature counts per singleton SRO type for one building."""
    _, _, layout = builder_and_obs
    counts: dict[str, int] = {}
    for s in layout.segments:
        if s.family == "sro" and s.instance_id == "Building_1":
            counts[s.type_name] = (
                counts.get(s.type_name, 0) + len(s.feature_indices)
            )
    expected = {
        "district_time": 3,
        "district_weather_current": 4,
        "district_weather_forecast": 12,
        "district_carbon": 1,
        "district_pricing_current": 1,
        "district_pricing_forecast": 3,
        "district_community_energy": 12,
        "district_community_headroom": 4,
        "district_community_history": 2,
        "district_meta": 3,
        "building_storage_state": 2,
        "building_charging_phase_onehot": 6,
        "building_charging_headroom": 8,
        "building_charging_violation": 1,
        "building_energy_current": 15,
        "building_energy_history": 4,
    }
    for k, v in expected.items():
        assert counts.get(k) == v, (
            f"{k}: expected {v} features, got {counts.get(k)}"
        )
