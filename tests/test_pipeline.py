"""Unit tests for :mod:`algorithms.pipeline`.

These tests exercise the composite execution units in isolation using a
recording stub that satisfies :class:`ExecutionUnit`. No torch / mlflow
dependencies are required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from algorithms.execution_unit import ExecutionUnit
from algorithms.pipeline import Ensemble, Pipeline


class RecordingUnit(ExecutionUnit):
    """ExecutionUnit stub that captures every call for assertions."""

    def __init__(
        self,
        name: str,
        predict_output: Any = None,
        use_raw_observations: bool = False,
        initial_exploration_done: bool = True,
    ) -> None:
        self.name = name
        self._predict_output = predict_output if predict_output is not None else [[0.0]]
        self.use_raw_observations = use_raw_observations
        self._initial_exploration_done = initial_exploration_done

        self.predict_calls: List[Dict[str, Any]] = []
        self.update_calls: List[Dict[str, Any]] = []
        self.attach_calls: List[Dict[str, Any]] = []
        self.save_calls: List[Dict[str, Any]] = []
        self.load_calls: List[str] = []
        self.export_calls: List[Dict[str, Any]] = []
        self.observation_context_calls: List[Dict[str, Any]] = []
        self.transition_context_calls: List[Dict[str, Any]] = []
        self.topology_transition_calls: List[Dict[str, Any]] = []

    def predict(self, observations, deterministic=None, *, context=None):
        self.predict_calls.append(
            {
                "observations": observations,
                "deterministic": deterministic,
                "context": context,
            }
        )
        return self._predict_output

    def update(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        terminated,
        truncated,
        *,
        update_target_step,
        global_learning_step,
        update_step,
        initial_exploration_done,
    ) -> None:
        self.update_calls.append(
            {
                "observations": observations,
                "actions": actions,
                "rewards": rewards,
                "next_observations": next_observations,
                "terminated": terminated,
                "truncated": truncated,
                "update_target_step": update_target_step,
                "global_learning_step": global_learning_step,
                "update_step": update_step,
                "initial_exploration_done": initial_exploration_done,
            }
        )

    def is_initial_exploration_done(self, global_learning_step: int) -> bool:
        return self._initial_exploration_done

    def attach_environment(self, **kwargs) -> None:
        self.attach_calls.append(kwargs)

    def set_observation_context(self, **kwargs) -> None:
        self.observation_context_calls.append(kwargs)

    def set_transition_context(self, **kwargs) -> None:
        self.transition_context_calls.append(kwargs)

    def record_topology_transition(self, **kwargs) -> None:
        self.topology_transition_calls.append(kwargs)

    def save_checkpoint(self, output_dir: str, step: int) -> Optional[str]:
        self.save_calls.append({"output_dir": output_dir, "step": step})
        return str(Path(output_dir) / f"{self.name}.pth")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        self.load_calls.append(checkpoint_path)

    def export_artifacts(self, output_dir, context=None):
        self.export_calls.append({"output_dir": output_dir, "context": context})
        return {"format": "stub", "name": self.name, "artifacts": []}


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------
class TestPipelinePredict:
    def test_rejects_action_retaining_non_leaf_stage(self) -> None:
        action_retaining = RecordingUnit("action_retaining")
        action_retaining.requires_final_pipeline_stage = True

        with pytest.raises(ValueError, match="must be the final stage"):
            Pipeline([action_retaining, RecordingUnit("leaf")])

    def test_threads_context_top_to_bottom(self) -> None:
        manager = RecordingUnit("manager", predict_output="signal_from_manager")
        leaf = RecordingUnit("leaf", predict_output=[[0.5]])
        pipeline = Pipeline([manager, leaf])

        result = pipeline.predict([[1.0, 2.0]], deterministic=False)

        assert result == [[0.5]]
        assert manager.predict_calls[0]["context"] is None
        assert leaf.predict_calls[0]["context"] == "signal_from_manager"

    def test_initial_context_forwarded_to_first_stage(self) -> None:
        first = RecordingUnit("first", predict_output="out")
        pipeline = Pipeline([first])

        pipeline.predict([[1.0]], context="from_outside")

        assert first.predict_calls[0]["context"] == "from_outside"

    def test_frozen_stage_is_deterministic_while_trainable_stage_explores(self) -> None:
        manager = RecordingUnit("manager", predict_output="price")
        leaf = RecordingUnit("leaf", predict_output=[[0.5]])
        leaf.frozen = True
        pipeline = Pipeline([manager, leaf])

        pipeline.predict([[1.0]], deterministic=False)

        assert manager.predict_calls[0]["deterministic"] is False
        assert leaf.predict_calls[0]["deterministic"] is True

    def test_passes_observations_unchanged_to_each_stage(self) -> None:
        first = RecordingUnit("first", predict_output="ctx")
        second = RecordingUnit("second", predict_output=[[0.1]])
        pipeline = Pipeline([first, second])

        observations = [[1.0, 2.0], [3.0, 4.0]]
        pipeline.predict(observations)

        assert first.predict_calls[0]["observations"] is observations
        assert second.predict_calls[0]["observations"] is observations

    def test_routes_raw_and_encoded_observations_per_stage(self) -> None:
        manager = RecordingUnit("manager", predict_output="price", use_raw_observations=False)
        leaf = RecordingUnit("leaf", predict_output=[[0.2]], use_raw_observations=True)
        pipeline = Pipeline([manager, leaf])
        raw_observations = [["raw"]]
        encoded_observations = [["encoded"]]

        pipeline.set_observation_context(
            raw_observations=raw_observations,
            encoded_observations=encoded_observations,
        )
        result = pipeline.predict(encoded_observations)

        assert result == [[0.2]]
        assert manager.predict_calls[0]["observations"] is encoded_observations
        assert leaf.predict_calls[0]["observations"] is raw_observations
        assert leaf.predict_calls[0]["context"] == "price"

    def test_routes_stage_specific_encoded_observation_profile(self) -> None:
        manager = RecordingUnit("manager", predict_output="price")
        manager.observation_encoding_profile = "cc_level1"
        leaf = RecordingUnit("leaf", predict_output=[[0.2]])
        pipeline = Pipeline([manager, leaf])
        default_encoded = [["building-local"]]
        cc_encoded = [["community"]]

        assert pipeline.required_observation_encoding_profiles() == ["cc_level1"]
        pipeline.set_observation_context(
            raw_observations=[["raw"]],
            encoded_observations=default_encoded,
        )
        pipeline.set_profiled_observation_context(
            {"cc_level1": cc_encoded}
        )
        pipeline.predict(default_encoded)

        assert manager.predict_calls[0]["observations"] is cc_encoded
        assert leaf.predict_calls[0]["observations"] is default_encoded


class TestPipelineUpdate:
    def test_delegates_to_every_stage(self) -> None:
        a = RecordingUnit("a")
        b = RecordingUnit("b")
        pipeline = Pipeline([a, b])

        pipeline.update(
            [[1.0]],
            [[0.5]],
            [0.1],
            [[1.1]],
            terminated=False,
            truncated=False,
            update_target_step=True,
            global_learning_step=42,
            update_step=True,
            initial_exploration_done=True,
        )

        assert len(a.update_calls) == 1
        assert len(b.update_calls) == 1
        assert a.update_calls[0]["global_learning_step"] == 42
        assert b.update_calls[0]["global_learning_step"] == 42

    def test_routes_raw_and_encoded_transitions_per_stage(self) -> None:
        manager = RecordingUnit("manager", use_raw_observations=False)
        leaf = RecordingUnit("leaf", use_raw_observations=True)
        pipeline = Pipeline([manager, leaf])
        raw_observations = [["raw"]]
        raw_next_observations = [["raw_next"]]
        encoded_observations = [["encoded"]]
        encoded_next_observations = [["encoded_next"]]

        pipeline.set_transition_context(
            raw_observations=raw_observations,
            raw_next_observations=raw_next_observations,
            encoded_observations=encoded_observations,
            encoded_next_observations=encoded_next_observations,
        )
        pipeline.update(
            encoded_observations,
            [[0.5]],
            [0.1],
            encoded_next_observations,
            terminated=False,
            truncated=False,
            update_target_step=True,
            global_learning_step=42,
            update_step=True,
            initial_exploration_done=True,
        )

        assert manager.update_calls[0]["observations"] is encoded_observations
        assert manager.update_calls[0]["next_observations"] is encoded_next_observations
        assert leaf.update_calls[0]["observations"] is raw_observations
        assert leaf.update_calls[0]["next_observations"] is raw_next_observations

    def test_routes_stage_specific_encoded_transition_profile(self) -> None:
        manager = RecordingUnit("manager")
        manager.observation_encoding_profile = "cc_level1"
        leaf = RecordingUnit("leaf")
        pipeline = Pipeline([manager, leaf])
        default_encoded = [["building-local"]]
        default_next = [["building-local-next"]]
        cc_encoded = [["community"]]
        cc_next = [["community-next"]]

        pipeline.set_transition_context(
            raw_observations=[["raw"]],
            raw_next_observations=[["raw-next"]],
            encoded_observations=default_encoded,
            encoded_next_observations=default_next,
        )
        pipeline.set_profiled_transition_context(
            profiled_encoded_observations={"cc_level1": cc_encoded},
            profiled_encoded_next_observations={"cc_level1": cc_next},
        )
        pipeline.update(
            default_encoded,
            [[0.5]],
            [0.1],
            default_next,
            terminated=False,
            truncated=False,
            update_target_step=False,
            global_learning_step=1,
            update_step=True,
            initial_exploration_done=True,
        )

        assert manager.update_calls[0]["observations"] is cc_encoded
        assert manager.update_calls[0]["next_observations"] is cc_next
        assert leaf.update_calls[0]["observations"] is default_encoded
        assert leaf.update_calls[0]["next_observations"] is default_next


class TestPipelineLifecycle:
    def test_initial_exploration_requires_all_stages(self) -> None:
        ready = RecordingUnit("ready", initial_exploration_done=True)
        warming = RecordingUnit("warming", initial_exploration_done=False)
        assert Pipeline([ready, ready]).is_initial_exploration_done(10) is True
        assert Pipeline([ready, warming]).is_initial_exploration_done(10) is False

    def test_stage_metrics_are_namespaced_and_training_metrics_consumed(self) -> None:
        manager = RecordingUnit("manager")
        leaf = RecordingUnit("leaf")
        manager.get_diagnostic_metrics = lambda: {"multiplier_mean": 0.9}
        leaf.get_diagnostic_metrics = lambda: {"price_context_active": 1.0}
        manager.consume_latest_training_metrics = lambda: {"update_count": 4.0}
        leaf.consume_latest_training_metrics = lambda: {"actor_loss": -0.2}
        pipeline = Pipeline([manager, leaf])

        assert pipeline.get_diagnostic_metrics() == {
            "Pipeline/stage_0/multiplier_mean": 0.9,
            "Pipeline/stage_1/price_context_active": 1.0,
        }
        assert pipeline.consume_latest_training_metrics() == {
            "Pipeline/stage_0/update_count": 4.0,
            "Pipeline/stage_1/actor_loss": -0.2,
        }

    def test_attach_environment_delegates_to_every_stage(self) -> None:
        a = RecordingUnit("a")
        b = RecordingUnit("b")
        Pipeline([a, b]).attach_environment(
            observation_names=[["x"]],
            action_names=[["y"]],
            action_space=[None],
            observation_space=[None],
            metadata={"k": "v"},
        )
        assert a.attach_calls[0]["metadata"] == {"k": "v"}
        assert b.attach_calls[0]["metadata"] == {"k": "v"}

    def test_use_raw_observations_aggregates_with_any(self) -> None:
        none_raw = RecordingUnit("a", use_raw_observations=False)
        raw = RecordingUnit("b", use_raw_observations=True)
        assert Pipeline([none_raw]).use_raw_observations is False
        assert Pipeline([raw, raw]).use_raw_observations is True
        assert Pipeline([none_raw, raw]).use_raw_observations is False
        assert Pipeline([none_raw, raw]).requires_raw_observation_context is True

    def test_attach_environment_routes_raw_and_encoded_names_per_stage(self) -> None:
        manager = RecordingUnit("manager", use_raw_observations=False)
        leaf = RecordingUnit("leaf", use_raw_observations=True)
        raw_names = [["raw_a"], ["raw_b"]]
        encoded_names = [["encoded_a"], ["encoded_b"]]

        Pipeline([manager, leaf]).attach_environment(
            observation_names=raw_names,
            action_names=[["act_a"], ["act_b"]],
            action_space=["space_a", "space_b"],
            observation_space=["obs_space_a", "obs_space_b"],
            metadata={
                "raw_observation_names": raw_names,
                "encoded_observation_names": encoded_names,
            },
        )

        assert manager.attach_calls[0]["observation_names"] == encoded_names
        assert leaf.attach_calls[0]["observation_names"] == raw_names

    def test_attach_environment_preserves_raw_name_contract_for_encoded_leaf(self) -> None:
        manager = RecordingUnit("manager", use_raw_observations=True)
        leaf_member = RecordingUnit("leaf", use_raw_observations=False)
        leaf_member.requires_raw_observation_context = True
        leaf = Ensemble([leaf_member])
        raw_names = [["raw_a"]]
        encoded_names = [["encoded_a"]]

        Pipeline([manager, leaf]).attach_environment(
            observation_names=raw_names,
            action_names=[["act_a"]],
            action_space=["space_a"],
            observation_space=["obs_space_a"],
            metadata={
                "raw_observation_names": raw_names,
                "encoded_observation_names": encoded_names,
            },
        )

        assert leaf_member.attach_calls[0]["observation_names"] == raw_names

    def test_attach_environment_routes_raw_names_to_warm_start_leaf(self) -> None:
        manager = RecordingUnit("manager", use_raw_observations=False)
        leaf = RecordingUnit("leaf", use_raw_observations=False)
        leaf.warm_start_policy_name = "RBCSmartPolicy"
        raw_names = [["raw_required_soc", "raw_departure_time"]]
        encoded_names = [["encoded_feature"]]

        Pipeline([manager, leaf]).attach_environment(
            observation_names=raw_names,
            action_names=[["electric_vehicle_storage"]],
            action_space=["space"],
            observation_space=["obs_space"],
            metadata={
                "raw_observation_names": raw_names,
                "encoded_observation_names": encoded_names,
            },
        )

        assert manager.attach_calls[0]["observation_names"] == encoded_names
        assert leaf.attach_calls[0]["observation_names"] == raw_names

    def test_attach_environment_routes_profiled_names_to_requesting_stage(self) -> None:
        manager = RecordingUnit("manager")
        manager.observation_encoding_profile = "cc_level1"
        leaf = RecordingUnit("leaf")
        raw_names = [["raw_a"]]
        encoded_names = [["building_a"]]
        cc_names = [["community_a"]]

        Pipeline([manager, leaf]).attach_environment(
            observation_names=raw_names,
            action_names=[["act_a"]],
            action_space=["space_a"],
            observation_space=["obs_space_a"],
            metadata={
                "raw_observation_names": raw_names,
                "encoded_observation_names": encoded_names,
                "profiled_encoded_observation_names": {"cc_level1": cc_names},
            },
        )

        assert manager.attach_calls[0]["observation_names"] == cc_names
        assert leaf.attach_calls[0]["observation_names"] == encoded_names


class TestPipelinePersistence:
    def test_save_creates_subdir_per_stage(self, tmp_path: Path) -> None:
        a = RecordingUnit("a")
        b = RecordingUnit("b")
        Pipeline([a, b]).save_checkpoint(str(tmp_path), step=7)

        assert (tmp_path / "stage_0").is_dir()
        assert (tmp_path / "stage_1").is_dir()
        assert a.save_calls[0]["output_dir"] == str(tmp_path / "stage_0")
        assert b.save_calls[0]["output_dir"] == str(tmp_path / "stage_1")

    def test_save_skips_frozen_pipeline_stage(self, tmp_path: Path) -> None:
        manager = RecordingUnit("manager")
        leaf = RecordingUnit("leaf")
        leaf.frozen = True

        Pipeline([manager, leaf]).save_checkpoint(str(tmp_path), step=7)

        assert manager.save_calls == [
            {"output_dir": str(tmp_path / "stage_0"), "step": 7}
        ]
        assert leaf.save_calls == []
        assert not (tmp_path / "stage_1").exists()

    def test_load_routes_each_stage_subdir(self, tmp_path: Path) -> None:
        a = RecordingUnit("a")
        b = RecordingUnit("b")
        (tmp_path / "stage_0").mkdir()
        (tmp_path / "stage_1").mkdir()

        Pipeline([a, b]).load_checkpoint(str(tmp_path))

        assert a.load_calls == [str(tmp_path / "stage_0")]
        assert b.load_calls == [str(tmp_path / "stage_1")]

    def test_load_skips_missing_subdirs(self, tmp_path: Path) -> None:
        a = RecordingUnit("a")
        b = RecordingUnit("b")
        (tmp_path / "stage_0").mkdir()
        # stage_1 deliberately missing.

        Pipeline([a, b]).load_checkpoint(str(tmp_path))

        assert len(a.load_calls) == 1
        assert b.load_calls == []

    def test_load_stage_checkpoint_routes_only_selected_stage(self, tmp_path: Path) -> None:
        manager = RecordingUnit("manager")
        leaf = RecordingUnit("leaf")
        checkpoint_root = tmp_path / "standalone-leaf"
        checkpoint_root.mkdir()

        Pipeline([manager, leaf]).load_stage_checkpoint(1, str(checkpoint_root))

        assert manager.load_calls == []
        assert leaf.load_calls == [str(checkpoint_root)]

    def test_load_stage_checkpoint_rejects_invalid_index(self, tmp_path: Path) -> None:
        checkpoint_root = tmp_path / "checkpoint"
        checkpoint_root.mkdir()

        with pytest.raises(IndexError, match="outside range"):
            Pipeline([RecordingUnit("only")]).load_stage_checkpoint(
                1, str(checkpoint_root)
            )

    def test_export_aggregates_metadata(self, tmp_path: Path) -> None:
        a = RecordingUnit("a")
        b = RecordingUnit("b")
        metadata = Pipeline([a, b]).export_artifacts(str(tmp_path))
        assert metadata["format"] == "pipeline"
        assert [entry["stage_index"] for entry in metadata["stages"]] == [0, 1]


class TestPipelineConstruction:
    def test_empty_stages_rejected(self) -> None:
        with pytest.raises(ValueError):
            Pipeline([])


# ----------------------------------------------------------------------
# Ensemble
# ----------------------------------------------------------------------
class TestEnsemblePredict:
    def test_each_member_receives_its_observation_slice(self) -> None:
        a = RecordingUnit("a", predict_output=[[0.1]])
        b = RecordingUnit("b", predict_output=[[0.2]])
        ensemble = Ensemble([a, b])

        result = ensemble.predict([[1.0], [2.0]])

        assert result == [[0.1], [0.2]]
        assert a.predict_calls[0]["observations"] == [[1.0]]
        assert b.predict_calls[0]["observations"] == [[2.0]]

    def test_context_broadcast_to_every_member(self) -> None:
        a = RecordingUnit("a", predict_output=[[0.1]])
        b = RecordingUnit("b", predict_output=[[0.2]])
        Ensemble([a, b]).predict([[1.0], [2.0]], context={"signal": 9})
        assert a.predict_calls[0]["context"] == {"signal": 9}
        assert b.predict_calls[0]["context"] == {"signal": 9}

    def test_member_returning_multiple_rows_is_rejected(self) -> None:
        misbehaving = RecordingUnit("oops", predict_output=[[0.1], [0.2]])
        ensemble = Ensemble([misbehaving])
        with pytest.raises(RuntimeError, match="returned 2 rows"):
            ensemble.predict([[1.0]])

    def test_member_returning_non_list_is_passed_through(self) -> None:
        # Non-leaf members may emit a context object (string, dict, tensor).
        # The ensemble must forward these unchanged.
        ctx_emitter = RecordingUnit("ctx", predict_output={"signal": 0.42})
        ensemble = Ensemble([ctx_emitter])
        result = ensemble.predict([[1.0]])
        assert result == [{"signal": 0.42}]


class TestEnsembleUpdate:
    def test_per_agent_slicing(self) -> None:
        a = RecordingUnit("a")
        b = RecordingUnit("b")
        Ensemble([a, b]).update(
            [[1.0], [2.0]],
            [[0.1], [0.2]],
            [0.5, 0.6],
            [[1.1], [2.1]],
            terminated=False,
            truncated=False,
            update_target_step=False,
            global_learning_step=3,
            update_step=True,
            initial_exploration_done=True,
        )
        assert a.update_calls[0]["observations"] == [[1.0]]
        assert a.update_calls[0]["actions"] == [[0.1]]
        assert a.update_calls[0]["rewards"] == [0.5]
        assert b.update_calls[0]["observations"] == [[2.0]]
        assert b.update_calls[0]["actions"] == [[0.2]]
        assert b.update_calls[0]["rewards"] == [0.6]


class TestEnsembleLifecycle:
    def test_topology_transition_rejects_short_actions_before_dispatch(self) -> None:
        a = RecordingUnit("a")
        b = RecordingUnit("b")

        with pytest.raises(RuntimeError, match="actions length"):
            Ensemble([a, b]).record_topology_transition(
                observations=[[1.0], [2.0]],
                actions=[[0.1]],
                rewards=[0.5, 0.6],
                terminated=False,
                truncated=False,
                global_learning_step=1,
            )

        assert a.topology_transition_calls == []
        assert b.topology_transition_calls == []

    def test_context_hooks_route_raw_and_encoded_slices(self) -> None:
        a = RecordingUnit("a")
        b = RecordingUnit("b")
        ensemble = Ensemble([a, b])

        ensemble.set_observation_context(
            raw_observations=[["raw-a"], ["raw-b"]],
            encoded_observations=[["encoded-a"], ["encoded-b"]],
        )
        ensemble.set_transition_context(
            raw_observations=[["raw-a"], ["raw-b"]],
            raw_next_observations=[["next-raw-a"], ["next-raw-b"]],
            encoded_observations=[["encoded-a"], ["encoded-b"]],
            encoded_next_observations=[["next-encoded-a"], ["next-encoded-b"]],
        )

        assert a.observation_context_calls[0] == {
            "raw_observations": [["raw-a"]],
            "encoded_observations": [["encoded-a"]],
        }
        assert b.observation_context_calls[0] == {
            "raw_observations": [["raw-b"]],
            "encoded_observations": [["encoded-b"]],
        }
        assert a.transition_context_calls[0]["raw_next_observations"] == [["next-raw-a"]]
        assert b.transition_context_calls[0]["encoded_next_observations"] == [["next-encoded-b"]]

    def test_warm_start_member_requires_raw_context(self) -> None:
        plain = RecordingUnit("plain")
        guided = RecordingUnit("guided")
        guided.warm_start_policy_name = "RBCSmartPolicy"

        ensemble = Ensemble([plain, guided])
        pipeline = Pipeline([ensemble])

        assert ensemble.requires_raw_observation_context is True
        assert pipeline.requires_raw_observation_context is True

    def test_attach_environment_routes_each_slice(self) -> None:
        a = RecordingUnit("a")
        b = RecordingUnit("b")
        Ensemble([a, b]).attach_environment(
            observation_names=[["o0"], ["o1"]],
            action_names=[["a0"], ["a1"]],
            action_space=["s0", "s1"],
            observation_space=["os0", "os1"],
            metadata={"shared": True},
        )
        assert a.attach_calls[0]["observation_names"] == [["o0"]]
        assert b.attach_calls[0]["observation_names"] == [["o1"]]
        assert a.attach_calls[0]["action_space"] == ["s0"]
        assert b.attach_calls[0]["action_space"] == ["s1"]
        assert a.attach_calls[0]["metadata"] == {"shared": True}

    def test_attach_environment_slices_member_metadata(self) -> None:
        a = RecordingUnit("a")
        b = RecordingUnit("b")

        Ensemble([a, b]).attach_environment(
            observation_names=[["raw_a"], ["raw_b"]],
            action_names=[["act_a"], ["act_b"]],
            action_space=["space_a", "space_b"],
            observation_space=["obs_space_a", "obs_space_b"],
            metadata={
                "building_names": ["Building_1", "Building_2"],
                "raw_observation_names": [["raw_a"], ["raw_b"]],
                "encoded_observation_names": [["encoded_a"], ["encoded_b"]],
                "raw_observation_bounds": [
                    {"low": [0.0], "high": [1.0]},
                    {"low": [-1.0], "high": [2.0]},
                ],
                "shared": "kept",
            },
        )

        assert a.attach_calls[0]["metadata"]["building_names"] == ["Building_1"]
        assert b.attach_calls[0]["metadata"]["building_names"] == ["Building_2"]
        assert a.attach_calls[0]["metadata"]["raw_observation_names"] == [["raw_a"]]
        assert b.attach_calls[0]["metadata"]["raw_observation_names"] == [["raw_b"]]
        assert a.attach_calls[0]["metadata"]["encoded_observation_names"] == [["encoded_a"]]
        assert b.attach_calls[0]["metadata"]["encoded_observation_names"] == [["encoded_b"]]
        assert a.attach_calls[0]["metadata"]["raw_observation_bounds"] == [
            {"low": [0.0], "high": [1.0]}
        ]
        assert b.attach_calls[0]["metadata"]["raw_observation_bounds"] == [
            {"low": [-1.0], "high": [2.0]}
        ]
        assert a.attach_calls[0]["metadata"]["shared"] == "kept"
        assert b.attach_calls[0]["metadata"]["shared"] == "kept"

    def test_initial_exploration_requires_all_members(self) -> None:
        ready = RecordingUnit("ready", initial_exploration_done=True)
        warming = RecordingUnit("warming", initial_exploration_done=False)
        assert Ensemble([ready, ready]).is_initial_exploration_done(0) is True
        assert Ensemble([ready, warming]).is_initial_exploration_done(0) is False

    def test_export_passes_global_agent_index_offset_to_members(self, tmp_path: Path) -> None:
        a = RecordingUnit("a")
        b = RecordingUnit("b")

        Ensemble([a, b]).export_artifacts(str(tmp_path), context={"config": {"bundle": {}}})

        assert a.export_calls[0]["context"]["agent_index_offset"] == 0
        assert b.export_calls[0]["context"]["agent_index_offset"] == 1
        assert a.export_calls[0]["context"]["config"] == {"bundle": {}}
        assert b.export_calls[0]["context"]["config"] == {"bundle": {}}


class TestEnsembleConstruction:
    def test_empty_agents_rejected(self) -> None:
        with pytest.raises(ValueError):
            Ensemble([])


class TestEnsembleAttachEnvironmentSizeMismatch:
    def test_too_few_env_slots_raises(self) -> None:
        a = RecordingUnit("a")
        b = RecordingUnit("b")
        c = RecordingUnit("c")
        ensemble = Ensemble([a, b, c])  # 3 members

        with pytest.raises(ValueError, match="Ensemble size mismatch"):
            ensemble.attach_environment(
                observation_names=[["o0"], ["o1"]],  # only 2 slots
                action_names=[["a0"], ["a1"]],
                action_space=["s0", "s1"],
                observation_space=["os0", "os1"],
            )

    def test_too_many_env_slots_raises(self) -> None:
        a = RecordingUnit("a")
        ensemble = Ensemble([a])  # 1 member

        with pytest.raises(ValueError, match="Ensemble size mismatch"):
            ensemble.attach_environment(
                observation_names=[["o0"], ["o1"], ["o2"]],
                action_names=[["a0"], ["a1"], ["a2"]],
                action_space=["s0", "s1", "s2"],
                observation_space=["os0", "os1", "os2"],
            )

    def test_exact_match_does_not_raise(self) -> None:
        a = RecordingUnit("a")
        b = RecordingUnit("b")
        ensemble = Ensemble([a, b])

        # Should not raise
        ensemble.attach_environment(
            observation_names=[["o0"], ["o1"]],
            action_names=[["a0"], ["a1"]],
            action_space=["s0", "s1"],
            observation_space=["os0", "os1"],
        )
