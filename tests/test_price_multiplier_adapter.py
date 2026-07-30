from __future__ import annotations

import numpy as np
import pytest

from algorithms.utils.price_multiplier_adapter import (
    CURRENT_PRICE_NAME,
    PREDICTED_PRICE_NAMES,
    ForecastMode,
    PriceMultiplierContext,
    PriceMultiplierObservationAdapter,
    normalize_price_multiplier_context,
    price_feature_bounds_from_metadata,
    price_observation_names_from_metadata,
)


PRICE_NAMES = (CURRENT_PRICE_NAME, *PREDICTED_PRICE_NAMES)


def _layout():
    names = (
        "district__hour",
        CURRENT_PRICE_NAME,
        "net_power_kw",
        PREDICTED_PRICE_NAMES[0],
        "storage::Building_1/electrical_storage::soc",
        PREDICTED_PRICE_NAMES[1],
        PREDICTED_PRICE_NAMES[2],
    )
    low = np.array([0.0, 0.10, -10.0, 0.20, 0.0, 0.30, 0.40], dtype=np.float64)
    high = np.array([23.0, 0.50, 10.0, 0.60, 1.0, 0.90, 1.20], dtype=np.float64)
    observation = np.array([0.25, 0.50, 0.75, 0.25, 0.40, 0.50, 0.75], dtype=np.float32)
    return names, low, high, observation


def _adapter(mode=ForecastMode.REAL_UNMODIFIED):
    names, low, high, _ = _layout()
    return PriceMultiplierObservationAdapter(
        observation_names=names,
        feature_low=low,
        feature_high=high,
        forecast_mode=mode,
    )


@pytest.mark.parametrize(
    ("mode", "context"),
    [
        ("real_unmodified", {"current": 1.0}),
        ("persist_current", {"current": 1.0}),
        (
            "aligned_vector",
            {
                "current": 1.0,
                "forecast_6h": 1.0,
                "forecast_12h": 1.0,
                "forecast_24h": 1.0,
            },
        ),
    ],
)
def test_neutral_multiplier_is_bitwise_noop_and_does_not_mutate(mode, context):
    _, _, _, observation = _layout()
    before = observation.copy()

    transformed, diagnostics = _adapter(mode).transform(observation, context)

    assert transformed is not observation
    assert transformed.dtype == observation.dtype
    assert np.array_equal(transformed, before)
    assert np.array_equal(observation, before)
    assert diagnostics.neutral_noop
    assert diagnostics.clipping_count == 0


def test_real_unmodified_changes_only_current_price_using_its_affine_bounds():
    names, low, high, observation = _layout()
    transformed, diagnostics = _adapter("real_unmodified").transform(
        observation, {"current": 1.5}
    )

    current_index = names.index(CURRENT_PRICE_NAME)
    # encoded .5 -> real .3; multiplied -> .45; encoded -> .875.
    assert transformed[current_index] == pytest.approx(0.875)
    for name in PREDICTED_PRICE_NAMES:
        index = names.index(name)
        assert transformed[index] == observation[index]
    non_price = [index for index, name in enumerate(names) if name not in PRICE_NAMES]
    assert np.array_equal(transformed[non_price], observation[non_price])
    assert diagnostics.real_price_by_feature[CURRENT_PRICE_NAME] == pytest.approx(0.30)
    assert diagnostics.virtual_price_by_feature[CURRENT_PRICE_NAME] == pytest.approx(0.45)
    assert diagnostics.multiplier_by_feature[PREDICTED_PRICE_NAMES[0]] == 1.0


def test_persist_current_applies_one_multiplier_to_current_and_all_forecasts():
    names, _, _, observation = _layout()
    transformed, diagnostics = _adapter("persist_current").transform(
        observation, {"current": 0.5}
    )

    expected = {
        CURRENT_PRICE_NAME: 0.125,  # real .3 -> .15 in [.1, .5]
        PREDICTED_PRICE_NAMES[0]: 0.0,  # real .3 -> .15, clipped to .2
        PREDICTED_PRICE_NAMES[1]: 0.0,  # real .6 -> .3
        PREDICTED_PRICE_NAMES[2]: 0.125,  # real 1.0 -> .5 in [.4, 1.2]
    }
    for name, value in expected.items():
        assert transformed[names.index(name)] == pytest.approx(value)
    assert diagnostics.clipping_count == 1
    assert diagnostics.clipped_features == (PREDICTED_PRICE_NAMES[0],)
    assert diagnostics.virtual_price_unclipped_by_feature[PREDICTED_PRICE_NAMES[0]] == pytest.approx(0.15)
    assert diagnostics.virtual_price_by_feature[PREDICTED_PRICE_NAMES[0]] == pytest.approx(0.20)


def test_aligned_vector_uses_semantic_forecast_context_and_reports_clipping():
    names, _, _, observation = _layout()
    context = {
        "current": 2.0,
        "forecast_6h": 1.5,
        "forecast_12h": 0.5,
        "forecast_24h": 2.0,
    }

    transformed, diagnostics = _adapter("aligned_vector").transform(observation, context)

    assert transformed[names.index(CURRENT_PRICE_NAME)] == pytest.approx(1.0)
    assert transformed[names.index(PREDICTED_PRICE_NAMES[0])] == pytest.approx(0.625)
    assert transformed[names.index(PREDICTED_PRICE_NAMES[1])] == pytest.approx(0.0)
    assert transformed[names.index(PREDICTED_PRICE_NAMES[2])] == pytest.approx(1.0)
    assert diagnostics.clipping_count == 2
    assert diagnostics.clipped_features == (
        CURRENT_PRICE_NAME,
        PREDICTED_PRICE_NAMES[2],
    )
    assert np.array_equal(observation, _layout()[3])


def test_price_bounds_can_be_supplied_as_feature_mappings():
    names, low, high, observation = _layout()
    low_by_name = {name: low[names.index(name)] for name in PRICE_NAMES}
    high_by_name = {name: high[names.index(name)] for name in PRICE_NAMES}
    adapter = PriceMultiplierObservationAdapter(
        observation_names=names,
        feature_low=low_by_name,
        feature_high=high_by_name,
        forecast_mode="real_unmodified",
    )

    transformed, _ = adapter.transform(observation, {"current": 1.5})

    assert transformed[names.index(CURRENT_PRICE_NAME)] == pytest.approx(0.875)


@pytest.mark.parametrize(
    "bad_context",
    [
        {},
        {"current": np.nan},
        {"current": -0.1},
        {"current": 1.0, "forecast_6h": 1.0},
        {"current_multiplier": 1.0},
    ],
)
def test_non_vector_context_validation_is_strict(bad_context):
    with pytest.raises((TypeError, ValueError)):
        _adapter("persist_current").transform(_layout()[3], bad_context)


@pytest.mark.parametrize(
    "bad_context",
    [
        {"current": 1.0},
        {
            "current": 1.0,
            "forecast_6h": 1.0,
            "forecast_12h": 1.0,
            "forecast_24h": 1.0,
            "forecast_48h": 1.0,
        },
        {
            "current": 1.0,
            "forecast_6h": 1.0,
            "forecast_12h": None,
            "forecast_24h": 1.0,
        },
    ],
)
def test_aligned_vector_requires_exact_structured_context(bad_context):
    with pytest.raises((TypeError, ValueError)):
        _adapter("aligned_vector").transform(_layout()[3], bad_context)


def test_direct_context_dataclass_is_validated_for_selected_mode():
    with pytest.raises(ValueError, match="aligned_vector requires"):
        _adapter("aligned_vector").transform(
            _layout()[3], PriceMultiplierContext(current=1.0)
        )

    with pytest.raises(ValueError, match="accepts only the current"):
        _adapter("real_unmodified").transform(
            _layout()[3],
            PriceMultiplierContext(current=1.0, forecast_6h=1.0),
        )


@pytest.mark.parametrize(
    "leaked_name",
    [
        "district__community_net_power_kw",
        "district__forecast_community_net_next_1h_kw",
        "community__net_power_kw",
    ],
)
def test_adapter_rejects_any_community_observation_name(leaked_name):
    names, low, high, _ = _layout()
    with pytest.raises(ValueError, match="community observation"):
        PriceMultiplierObservationAdapter(
            observation_names=(*names, leaked_name),
            feature_low=np.append(low, 0.0),
            feature_high=np.append(high, 1.0),
        )


def test_adapter_discovers_only_exact_price_contract_once():
    names, low, high, _ = _layout()
    legacy_names = tuple(
        "electricity_pricing" if name == CURRENT_PRICE_NAME else name for name in names
    )
    with pytest.raises(ValueError, match="occurred 0 times"):
        PriceMultiplierObservationAdapter(
            observation_names=legacy_names,
            feature_low=low,
            feature_high=high,
        )

    duplicate_names = (*names, CURRENT_PRICE_NAME)
    with pytest.raises(ValueError, match="occurred 2 times"):
        PriceMultiplierObservationAdapter(
            observation_names=duplicate_names,
            feature_low=np.append(low, 0.1),
            feature_high=np.append(high, 0.5),
        )


@pytest.mark.parametrize(
    ("low", "high"),
    [(np.nan, 0.5), (0.1, np.inf), (0.5, 0.5), (0.6, 0.5)],
)
def test_adapter_rejects_non_affine_price_bounds(low, high):
    names, lows, highs, _ = _layout()
    lows = lows.copy()
    highs = highs.copy()
    index = names.index(CURRENT_PRICE_NAME)
    lows[index] = low
    highs[index] = high
    with pytest.raises(ValueError, match="finite affine bounds"):
        PriceMultiplierObservationAdapter(
            observation_names=names,
            feature_low=lows,
            feature_high=highs,
        )


def test_scalar_cc_context_is_normalized_without_changing_mapping_contract():
    assert normalize_price_multiplier_context(None) is None
    assert normalize_price_multiplier_context(np.float32(1.25)) == {"current": 1.25}
    mapping = {"current": 0.75}
    assert normalize_price_multiplier_context(mapping) is mapping
    with pytest.raises(TypeError, match="scalar multiplier"):
        normalize_price_multiplier_context([1.0])


def test_price_bounds_are_resolved_from_wrapper_metadata():
    names, low, high, _ = _layout()
    feature_low, feature_high = price_feature_bounds_from_metadata(
        metadata={
            "raw_observation_names": [list(names)],
            "raw_observation_bounds": [
                {"low": low.tolist(), "high": high.tolist()}
            ],
        },
        agent_index=0,
    )

    assert feature_low == {
        name: pytest.approx(low[names.index(name)]) for name in PRICE_NAMES
    }
    assert feature_high == {
        name: pytest.approx(high[names.index(name)]) for name in PRICE_NAMES
    }


def test_actor_layout_uses_encoded_profile_instead_of_raw_community_superset():
    names, _, _, _ = _layout()
    raw_names = [*names, "district__community_net_power_kw"]

    resolved = price_observation_names_from_metadata(
        metadata={"encoded_observation_names": [list(names)]},
        agent_index=0,
        fallback_observation_names=raw_names,
    )

    assert resolved == list(names)
    PriceMultiplierObservationAdapter(
        observation_names=resolved,
        feature_low={name: 0.0 for name in PRICE_NAMES},
        feature_high={name: 1.0 for name in PRICE_NAMES},
    )


def test_diagnostics_are_serializable_and_expose_real_and_virtual_prices():
    _, _, _, observation = _layout()
    _, diagnostics = _adapter().transform(observation, {"current": 2.0})

    payload = diagnostics.to_dict()

    assert payload["clipping_count"] == 1
    assert payload["real_price_by_feature"][CURRENT_PRICE_NAME] == pytest.approx(0.30)
    assert payload["virtual_price_unclipped_by_feature"][CURRENT_PRICE_NAME] == pytest.approx(0.60)
    assert payload["virtual_price_by_feature"][CURRENT_PRICE_NAME] == pytest.approx(0.50)


def test_adapter_rejects_price_values_outside_encoded_affine_domain():
    observation = _layout()[3].copy()
    observation[1] = 1.01

    with pytest.raises(ValueError, match=r"inside \[0, 1\]"):
        _adapter().transform(observation, {"current": 1.0})


def test_adapter_rejects_nonfinite_observation_and_boolean_multiplier():
    observation = _layout()[3].copy()
    observation[0] = np.nan
    with pytest.raises(ValueError, match="observations must be finite"):
        _adapter().transform(observation, {"current": 1.0})

    with pytest.raises(ValueError, match="not a boolean"):
        _adapter().transform(_layout()[3], {"current": True})
