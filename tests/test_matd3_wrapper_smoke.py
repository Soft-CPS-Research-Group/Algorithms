"""End-to-end wrapper-driven smoke test for AgentTransformerMATD3."""
from __future__ import annotations

from pathlib import Path
import signal

import pytest
import yaml


_TEMPLATE = "configs/templates/rl/transformer_matd3_local.yaml"
_DATASET_SCHEMA = "./datasets/citylearn_three_phase_dynamic_assets_only_demo_15s_parquet/schema.json"
_FAST_DYNAMIC_SCHEMA = "./datasets/citylearn_three_phase_dynamic_assets_only_demo/schema.json"


def _dataset_available() -> bool:
    return Path(_DATASET_SCHEMA).exists()


class _Timeout(Exception):
    pass


def _with_timeout(seconds: int, func):
    def _handler(_signum, _frame):
        raise _Timeout()

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        return func()
    except _Timeout:
        pytest.skip(
            "CityLearnEnv construction exceeded smoke-test timeout while estimating "
            "observation spaces for the dynamic entity dataset."
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@pytest.mark.skipif(not _dataset_available(), reason="dynamic entity dataset not present")
class TestWrapperSmoke:
    def _short_config(self, tmpdir: Path) -> Path:
        """Load the template and shorten the simulation window."""
        with open(_TEMPLATE) as f:
            cfg = yaml.safe_load(f)
        cfg["simulator"]["simulation_start_time_step"] = 0
        cfg["simulator"]["simulation_end_time_step"] = 20
        cfg["simulator"]["episode_time_steps"] = 21
        cfg["simulator"]["episodes"] = 1
        cfg["training"]["steps_between_training_updates"] = 4
        cfg["training"]["target_update_interval"] = 2
        cfg["pipeline"][0]["hyperparameters"]["batch_size"] = 4
        cfg["pipeline"][0]["hyperparameters"]["replay_capacity"] = 5000
        job_dir = tmpdir / "job"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "logs").mkdir(exist_ok=True)
        (job_dir / "results").mkdir(exist_ok=True)
        cfg["runtime"]["job_dir"] = str(job_dir)
        cfg["runtime"]["log_dir"] = str(job_dir / "logs")
        out_path = tmpdir / "config.yaml"
        with open(out_path, "w") as f:
            yaml.safe_dump(cfg, f)
        return out_path

    def _fast_dynamic_config(self, tmpdir: Path) -> Path:
        with open(_TEMPLATE) as f:
            cfg = yaml.safe_load(f)
        cfg["simulator"]["dataset_name"] = "citylearn_three_phase_dynamic_assets_only_demo"
        cfg["simulator"]["dataset_path"] = _FAST_DYNAMIC_SCHEMA
        cfg["simulator"]["simulation_start_time_step"] = 0
        cfg["simulator"]["simulation_end_time_step"] = 2600
        cfg["simulator"]["episode_time_steps"] = 2601
        cfg["simulator"]["episodes"] = 1
        cfg["simulator"]["export"]["mode"] = "none"
        cfg["simulator"]["export"]["export_kpis_on_episode_end"] = False
        cfg["tracking"]["mlflow_enabled"] = False
        cfg["tracking"]["progress_updates_enabled"] = False
        cfg["tracking"]["system_metrics_enabled"] = False
        cfg["tracking"]["log_frequency"] = 100000
        cfg["training"]["steps_between_training_updates"] = 1000000
        cfg["training"]["target_update_interval"] = 0
        cfg["pipeline"][0]["transformer_actor"] = {
            "d_model": 8, "nhead": 2, "num_layers": 1,
            "dim_feedforward": 16, "dropout": 0.0,
        }
        cfg["pipeline"][0]["transformer_critic"] = {
            "d_model": 8, "nhead": 2, "num_layers": 1,
            "dim_feedforward": 16, "dropout": 0.0,
        }
        cfg["pipeline"][0]["hyperparameters"]["batch_size"] = 999999
        cfg["pipeline"][0]["hyperparameters"]["replay_capacity"] = 5000
        cfg["pipeline"][0]["hyperparameters"]["actor_hidden_dim"] = 16
        job_dir = tmpdir / "job_fast"
        job_dir.mkdir(parents=True, exist_ok=True)
        cfg["runtime"]["job_dir"] = str(job_dir)
        cfg["runtime"]["log_dir"] = str(job_dir / "logs")
        out_path = tmpdir / "config_fast.yaml"
        with open(out_path, "w") as f:
            yaml.safe_dump(cfg, f)
        return out_path

    def test_config_validates(self, tmp_path):
        """Config passes schema validation with the new stage type."""
        from utils.config_schema import validate_config
        cfg_path = self._short_config(tmp_path)
        with open(cfg_path) as f:
            cfg = validate_config(yaml.safe_load(f))
        assert cfg.pipeline[0].algorithm == "AgentTransformerMATD3"

    def test_short_run_completes(self, tmp_path):
        """Agent runs through wrapper for the shortened window without crashing."""
        from citylearn.citylearn import CityLearnEnv
        from utils.config_schema import validate_config
        from utils.wrapper_citylearn import Wrapper_CityLearn
        from algorithms.registry import build_execution_unit

        cfg_path = self._short_config(tmp_path)
        with open(cfg_path) as f:
            cfg = validate_config(yaml.safe_load(f))
        cfg_dict = cfg.model_dump()
        agent = build_execution_unit(cfg_dict)
        env = _with_timeout(30, lambda: CityLearnEnv(
            schema=_DATASET_SCHEMA,
            interface="entity",
            topology_mode="dynamic",
            central_agent=False,
            offline=True,
            simulation_start_time_step=0,
            episode_time_steps=21,
        ))
        wrapper = Wrapper_CityLearn(env=env, config=cfg_dict, job_id="matd3-smoke")
        wrapper.set_model(agent)
        wrapper.learn(episodes=1)
        assert agent._replay.total_size >= 1

    def test_export_manifest_actors_only(self, tmp_path):
        """After training, export produces an actor-only manifest."""
        from citylearn.citylearn import CityLearnEnv
        from utils.config_schema import validate_config
        from utils.wrapper_citylearn import Wrapper_CityLearn
        from algorithms.registry import build_execution_unit

        cfg_path = self._short_config(tmp_path)
        with open(cfg_path) as f:
            cfg = validate_config(yaml.safe_load(f))
        cfg_dict = cfg.model_dump()
        agent = build_execution_unit(cfg_dict)
        env = _with_timeout(30, lambda: CityLearnEnv(
            schema=_DATASET_SCHEMA,
            interface="entity",
            topology_mode="dynamic",
            central_agent=False,
            offline=True,
            simulation_start_time_step=0,
            episode_time_steps=21,
        ))
        wrapper = Wrapper_CityLearn(env=env, config=cfg_dict, job_id="matd3-smoke")
        wrapper.set_model(agent)
        wrapper.learn(episodes=1)
        export_dir = tmp_path / "export"
        manifest = agent.export_artifacts(str(export_dir), context={"config": cfg_dict})
        assert manifest["format"] == "onnx"
        assert len(manifest["artifacts"]) >= 1
        for art in manifest["artifacts"]:
            assert "critic" not in art["path"].lower()
            assert (export_dir / art["path"]).exists()
            assert "ca_action_names" in art["config"]
            assert "action_low" in art["config"]
            assert "action_high" in art["config"]

    def test_topology_change_survives_if_dataset_triggers(self, tmp_path):
        """If topology changes in the short window, active replay remains usable."""
        from citylearn.citylearn import CityLearnEnv
        from utils.config_schema import validate_config
        from utils.wrapper_citylearn import Wrapper_CityLearn
        from algorithms.registry import build_execution_unit

        cfg_path = self._short_config(tmp_path)
        with open(cfg_path) as f:
            cfg = validate_config(yaml.safe_load(f))
        cfg_dict = cfg.model_dump()
        agent = build_execution_unit(cfg_dict)
        env = _with_timeout(30, lambda: CityLearnEnv(
            schema=_DATASET_SCHEMA,
            interface="entity",
            topology_mode="dynamic",
            central_agent=False,
            offline=True,
            simulation_start_time_step=0,
            episode_time_steps=21,
        ))
        wrapper = Wrapper_CityLearn(env=env, config=cfg_dict, job_id="matd3-smoke")
        wrapper.set_model(agent)
        wrapper.learn(episodes=1)
        if agent._replay.partition_count > 1:
            assert agent._replay.active_signature is not None
            assert agent._replay.active_partition_size >= 1

    def test_real_dynamic_run_action_counts_change(self, tmp_path):
        """Real wrapper run: topology changes must change per-building action counts."""
        if not Path(_FAST_DYNAMIC_SCHEMA).exists():
            pytest.skip("fast dynamic entity dataset not present")
        from citylearn.citylearn import CityLearnEnv
        from utils.config_schema import validate_config
        from utils.wrapper_citylearn import Wrapper_CityLearn
        from algorithms.registry import build_execution_unit

        cfg_path = self._fast_dynamic_config(tmp_path)
        with open(cfg_path) as f:
            cfg = validate_config(yaml.safe_load(f))
        cfg_dict = cfg.model_dump()
        env = CityLearnEnv(
            schema=_FAST_DYNAMIC_SCHEMA,
            interface="entity",
            topology_mode="dynamic",
            central_agent=False,
            offline=True,
            simulation_start_time_step=0,
            episode_time_steps=2601,
        )
        agent = build_execution_unit(cfg_dict)
        wrapper = Wrapper_CityLearn(env=env, config=cfg_dict, job_id="matd3-action-count-smoke")
        wrapper.set_model(agent)
        records = []
        original_predict = agent.predict

        def logged_predict(observations, deterministic=None, **kwargs):
            actions = original_predict(observations, deterministic=deterministic, **kwargs)
            counts = [len(action) for action in actions]
            records.append(
                {
                    "topology_version": wrapper._entity_topology_version,
                    "action_counts": counts,
                    "total_actions": sum(counts),
                }
            )
            return actions

        agent.predict = logged_predict
        wrapper.learn(episodes=1)

        unique = []
        seen = set()
        for record in records:
            key = (record["topology_version"], tuple(record["action_counts"]))
            if key not in seen:
                seen.add(key)
                unique.append(record)

        assert len(unique) >= 2
        assert len({record["topology_version"] for record in unique}) >= 2
        assert len({record["total_actions"] for record in unique}) >= 2
        assert agent._replay.partition_count >= 2
