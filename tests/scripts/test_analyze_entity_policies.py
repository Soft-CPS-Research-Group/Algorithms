# tests/scripts/test_analyze_entity_policies.py
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_15S = str(
    REPO_ROOT
    / "datasets"
    / "citylearn_three_phase_electrical_service_demo_15s_parquet"
    / "schema.json"
)


def _parse(argv):
    import scripts.analyze_entity_policies as m

    return m._build_parser().parse_args(argv)


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


def test_parser_default_schema():
    args = _parse(["--output-dir", "/tmp/x"])
    assert "15s_parquet" in str(args.schema)


def test_parser_requires_output_dir():
    with pytest.raises(SystemExit):
        _parse([])


def test_parser_seed_default():
    args = _parse(["--output-dir", "/tmp/x"])
    assert args.seed == 200


def test_parser_no_models_by_default():
    args = _parse(["--output-dir", "/tmp/x"])
    assert args.iql_root is None
    assert args.cql_root is None


def test_main_accepts_help():
    import scripts.analyze_entity_policies as m

    with pytest.raises(SystemExit) as exc:
        m.main(["--help"])
    assert exc.value.code == 0


# ---------------------------------------------------------------------------
# Rollout logger tests
# ---------------------------------------------------------------------------


def test_logged_rbc_rollout_structure():
    import scripts.analyze_entity_policies as m

    env_kwargs = {"schema_path": SCHEMA_15S, "episode_steps": 3, "offline": True}
    result = m._logged_rbc_rollout(env_seed=200, env_kwargs=env_kwargs)

    assert result["label"] == "RBCSmart"
    assert isinstance(result["steps"], list)
    assert len(result["steps"]) >= 1

    step0 = result["steps"][0]
    for key in ("step", "hour", "price_norm", "district_net_kwh",
                "ev_action_mean", "storage_action_mean"):
        assert key in step0, f"missing key: {key}"

    assert np.isfinite(step0["district_net_kwh"])
    assert "district_kpis" in result
    assert "cost_total" in result["district_kpis"]


# ---------------------------------------------------------------------------
# Figure generation tests
# ---------------------------------------------------------------------------


def _make_synthetic_rollout(label: str, n: int = 240) -> dict:
    return {
        "label": label,
        "env_seed": 200,
        "district_kpis": {
            "cost_total": 7.0,
            "carbon_emissions_total": 5.0,
            "daily_peak_average": 4.0,
            "ramping_average": 100.0,
            "electricity_consumption_total": 6.0,
            "zero_net_energy": -0.5,
        },
        "steps": [
            {
                "step": i,
                "hour": (i * 15) / 3600.0,
                "price_norm": 0.3 + 0.1 * np.sin(i * 0.2),
                "district_net_kwh": 5.0 + np.sin(i * 0.1),
                "ev_action_mean": 0.2 + 0.05 * np.sin(i * 0.3),
                "storage_action_mean": 0.1 * np.cos(i * 0.3),
            }
            for i in range(n)
        ],
    }


def test_generate_figures_creates_all_files(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import scripts.analyze_entity_policies as m

    rollouts = [_make_synthetic_rollout("RBCSmart"), _make_synthetic_rollout("IQL")]
    m._generate_figures(rollouts, tmp_path)

    assert (tmp_path / "fig1_demand_profile.png").exists()
    assert (tmp_path / "fig2_ev_action_by_hour.png").exists()
    assert (tmp_path / "fig3_price_vs_action.png").exists()
    assert (tmp_path / "fig4_kpi_comparison.png").exists()


def test_generate_figures_rbc_only_skips_kpi_comparison(tmp_path):
    """With only RBC, fig4 (requires >1 agent) should not be created."""
    import matplotlib

    matplotlib.use("Agg")
    import scripts.analyze_entity_policies as m

    rollouts = [_make_synthetic_rollout("RBCSmart")]
    m._generate_figures(rollouts, tmp_path)

    assert (tmp_path / "fig1_demand_profile.png").exists()
    assert not (tmp_path / "fig4_kpi_comparison.png").exists()


# ---------------------------------------------------------------------------
# Integration smoke test (RBC only, 3 steps — no model files needed)
# ---------------------------------------------------------------------------


def test_main_rbc_only_smoke(tmp_path):
    import scripts.analyze_entity_policies as m

    rc = m.main([
        "--output-dir", str(tmp_path),
        "--seed", "200",
        "--schema", SCHEMA_15S,
        "--episode-steps", "3",
    ])
    assert rc == 0
    assert (tmp_path / "steps_RBCSmart.csv").exists()
    assert (tmp_path / "fig1_demand_profile.png").exists()
    assert (tmp_path / "kpi_summary.json").exists()
