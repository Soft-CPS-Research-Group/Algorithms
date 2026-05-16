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

def test_section_temporal_patterns_creates_figure(sample_df, tmp_path):
    from scripts.explore_features import _section_temporal_patterns
    fig_path, md = _section_temporal_patterns(sample_df, tmp_path)
    assert (tmp_path / "fig2_temporal_patterns.png").exists()
    assert "hour" in md.lower()

def test_section_feature_distributions_creates_figure(sample_df, tmp_path):
    from scripts.explore_features import _section_feature_distributions
    fig_path, md = _section_feature_distributions(sample_df, tmp_path)
    assert (tmp_path / "fig3_feature_distributions.png").exists()
    assert "fig3" in fig_path
    assert "distribution" in md.lower()

def test_section_correlation_creates_figure(sample_df, tmp_path):
    from scripts.explore_features import _section_correlation
    fig_path, md = _section_correlation(sample_df, tmp_path)
    assert (tmp_path / "fig4_correlation_matrix.png").exists()
    assert "correlation" in md.lower()

def test_section_mutual_information_creates_figure(sample_df, tmp_path):
    from scripts.explore_features import _section_mutual_information
    fig_path, md = _section_mutual_information(sample_df, tmp_path)
    assert (tmp_path / "fig5_mutual_information.png").exists()
    assert "mutual information" in md.lower()

def test_section_ev_state_patterns_creates_figure(sample_df, tmp_path):
    from scripts.explore_features import _section_ev_state_patterns
    fig_path, md = _section_ev_state_patterns(sample_df, tmp_path)
    assert (tmp_path / "fig6_ev_state_patterns.png").exists()
    assert "soc" in md.lower()


def test_main_produces_all_outputs(tmp_path, monkeypatch):
    """Integration: main() writes doc + 6 figures."""
    import pandas as pd
    import numpy as np
    from scripts import explore_features as ef

    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame({
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

    out_dir = tmp_path / "feature_analysis"
    fig_dir = out_dir / "figures"
    doc_path = out_dir / "feature_analysis.md"

    monkeypatch.setattr(ef, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(ef, "FIGURES_DIR", fig_dir)
    monkeypatch.setattr(ef, "DOC_PATH", doc_path)

    ef.main(df=df)

    figures = list(fig_dir.glob("*.png"))
    assert len(figures) == 6, f"Expected 6 figures, got {len(figures)}: {figures}"
    assert doc_path.exists()
    content = doc_path.read_text()
    for section in ["Dataset overview", "Seed consistency", "Temporal", "distribution",
                    "Correlation", "Mutual information", "EV state", "Derived features"]:
        assert section.lower() in content.lower(), f"Missing section: {section}"
