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

    def test_passes_observations_unchanged_to_each_stage(self) -> None:
        first = RecordingUnit("first", predict_output="ctx")
        second = RecordingUnit("second", predict_output=[[0.1]])
        pipeline = Pipeline([first, second])

        observations = [[1.0, 2.0], [3.0, 4.0]]
        pipeline.predict(observations)

        assert first.predict_calls[0]["observations"] is observations
        assert second.predict_calls[0]["observations"] is observations


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


class TestPipelineLifecycle:
    def test_initial_exploration_requires_all_stages(self) -> None:
        ready = RecordingUnit("ready", initial_exploration_done=True)
        warming = RecordingUnit("warming", initial_exploration_done=False)
        assert Pipeline([ready, ready]).is_initial_exploration_done(10) is True
        assert Pipeline([ready, warming]).is_initial_exploration_done(10) is False

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
        assert Pipeline([none_raw, raw]).use_raw_observations is True


class TestPipelinePersistence:
    def test_save_creates_subdir_per_stage(self, tmp_path: Path) -> None:
        a = RecordingUnit("a")
        b = RecordingUnit("b")
        Pipeline([a, b]).save_checkpoint(str(tmp_path), step=7)

        assert (tmp_path / "stage_0").is_dir()
        assert (tmp_path / "stage_1").is_dir()
        assert a.save_calls[0]["output_dir"] == str(tmp_path / "stage_0")
        assert b.save_calls[0]["output_dir"] == str(tmp_path / "stage_1")

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

    def test_initial_exploration_requires_all_members(self) -> None:
        ready = RecordingUnit("ready", initial_exploration_done=True)
        warming = RecordingUnit("warming", initial_exploration_done=False)
        assert Ensemble([ready, ready]).is_initial_exploration_done(0) is True
        assert Ensemble([ready, warming]).is_initial_exploration_done(0) is False


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
