import importlib
import types
import pandas as pd
import numpy as np
import pytest


def test_module_importable():
    mod = importlib.import_module("scripts.explore_features")
    assert isinstance(mod, types.ModuleType)


def test_has_main():
    from scripts import explore_features
    assert callable(explore_features.main)


@pytest.fixture
def sample_df():
    """Minimal synthetic dataframe matching dataset schema."""
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "seed": rng.integers(100, 105, n),
        "obs_hour": rng.integers(0, 24, n),
        "obs_month": rng.integers(1, 13, n),
        "obs_connected_state": rng.integers(0, 2, n).astype(float),
        "obs_departure_time": rng.integers(6, 22, n).astype(float),
        "obs_required_soc_departure": rng.uniform(0.5, 1.0, n),
        "obs_electrical_vehicle_storage_soc": rng.uniform(0.0, 1.0, n),
        "obs_electrical_storage_soc": np.zeros(n),
        "obs_non_shiftable_load": rng.uniform(0.1, 2.0, n),
        "obs_solar_generation": rng.uniform(0.0, 1.5, n),
        "obs_net_electricity_consumption": rng.uniform(-1.0, 2.0, n),
        "obs_electricity_pricing": rng.uniform(0.1, 0.5, n),
        "obs_electricity_pricing_predicted_1": rng.uniform(0.1, 0.5, n),
        "action_electric_vehicle_storage_charger_5_1": rng.uniform(0, 1, n),
        "action_electrical_storage": np.zeros(n),
        "reward": rng.uniform(-10, 0, n),
    })

def test_section_overview_returns_markdown(sample_df):
    from scripts.explore_features import _section_overview
    fig_path, md = _section_overview(sample_df)
    assert fig_path == ""
    assert "rows" in md.lower() or "dataset" in md.lower()
    assert "action_electrical_storage" in md

def test_section_seed_consistency_creates_figure(sample_df, tmp_path):
    from scripts.explore_features import _section_seed_consistency
    fig_path, md = _section_seed_consistency(sample_df, tmp_path)
    assert (tmp_path / "fig1_seed_consistency.png").exists()
    assert "fig1_seed_consistency.png" in fig_path
    assert "seed" in md.lower()
