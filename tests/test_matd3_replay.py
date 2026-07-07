"""Unit tests for the topology-partitioned replay buffer."""
from __future__ import annotations

import numpy as np

from algorithms.utils.matd3_replay import (
    TopologyPartitionedReplay,
    TransitionData,
    LayoutSummary,
    compute_topology_signature,
)


def _make_layout_summary(building_id="B0", n_ca=2, n_sro=3) -> LayoutSummary:
    return LayoutSummary(
        building_id=building_id,
        n_ca=n_ca,
        n_sro=n_sro,
        obs_dim=10,
        action_dim=n_ca,
    )


def _make_transition(
    n_buildings=2, obs_dim=10, n_ca=2, topology_sig="sig_a",
) -> TransitionData:
    return TransitionData(
        observations=[np.random.randn(obs_dim).astype(np.float32) for _ in range(n_buildings)],
        next_observations=[np.random.randn(obs_dim).astype(np.float32) for _ in range(n_buildings)],
        actions=[np.random.randn(n_ca).astype(np.float32) for _ in range(n_buildings)],
        base_actions=[np.random.randn(n_ca).astype(np.float32) for _ in range(n_buildings)],
        next_base_actions=[np.random.randn(n_ca).astype(np.float32) for _ in range(n_buildings)],
        rewards=[float(np.random.randn()) for _ in range(n_buildings)],
        done=False,
        topology_signature=topology_sig,
        layout_summaries=[_make_layout_summary(f"B{i}", n_ca=n_ca) for i in range(n_buildings)],
    )


class TestTopologySignature:
    def test_same_inputs_same_hash(self):
        sig1 = compute_topology_signature(
            building_ids=["B0", "B1"],
            observation_names=[["a", "b"], ["c", "d"]],
            action_names=[["act0"], ["act1"]],
            ca_action_names=[["act0"], ["act1"]],
            per_type_feature_dims={"storage": 5, "pv": 3},
        )
        sig2 = compute_topology_signature(
            building_ids=["B0", "B1"],
            observation_names=[["a", "b"], ["c", "d"]],
            action_names=[["act0"], ["act1"]],
            ca_action_names=[["act0"], ["act1"]],
            per_type_feature_dims={"storage": 5, "pv": 3},
        )
        assert sig1 == sig2

    def test_different_obs_different_hash(self):
        sig1 = compute_topology_signature(
            building_ids=["B0"],
            observation_names=[["a", "b"]],
            action_names=[["act0"]],
            ca_action_names=[["act0"]],
            per_type_feature_dims={"storage": 5},
        )
        sig2 = compute_topology_signature(
            building_ids=["B0"],
            observation_names=[["a", "b", "c"]],
            action_names=[["act0"]],
            ca_action_names=[["act0"]],
            per_type_feature_dims={"storage": 5},
        )
        assert sig1 != sig2

    def test_different_feature_dims_different_hash(self):
        sig1 = compute_topology_signature(
            building_ids=["B0"],
            observation_names=[["a"]],
            action_names=[["act0"]],
            ca_action_names=[["act0"]],
            per_type_feature_dims={"storage": 5},
        )
        sig2 = compute_topology_signature(
            building_ids=["B0"],
            observation_names=[["a"]],
            action_names=[["act0"]],
            ca_action_names=[["act0"]],
            per_type_feature_dims={"storage": 7},
        )
        assert sig1 != sig2

    def test_deterministic(self):
        """Same call produces same result (no random component)."""
        for _ in range(10):
            sig = compute_topology_signature(
                building_ids=["X"],
                observation_names=[["f1", "f2"]],
                action_names=[["a"]],
                ca_action_names=[["a"]],
                per_type_feature_dims={"t": 4},
            )
        assert isinstance(sig, str)
        assert len(sig) > 0


class TestTopologyPartitionedReplay:
    def test_push_and_size(self):
        replay = TopologyPartitionedReplay(capacity=100, batch_size=4)
        replay.set_active_signature("sig_a")
        replay.push(_make_transition(topology_sig="sig_a"))
        assert replay.active_partition_size == 1
        assert replay.total_size == 1

    def test_sample_returns_batch_from_active_only(self):
        replay = TopologyPartitionedReplay(capacity=100, batch_size=4)
        replay.set_active_signature("sig_a")
        for _ in range(10):
            replay.push(_make_transition(topology_sig="sig_a"))
        replay.set_active_signature("sig_b")
        for _ in range(5):
            replay.push(_make_transition(topology_sig="sig_b"))

        batch = replay.sample()
        assert batch is not None
        assert batch.topology_signature == "sig_b"
        assert len(batch.observations[0]) == 4

    def test_sample_returns_none_when_insufficient(self):
        replay = TopologyPartitionedReplay(capacity=100, batch_size=10)
        replay.set_active_signature("sig_a")
        for _ in range(5):
            replay.push(_make_transition(topology_sig="sig_a"))
        assert replay.sample() is None

    def test_eviction_oldest_non_active_first(self):
        replay = TopologyPartitionedReplay(capacity=10, batch_size=2)
        replay.set_active_signature("sig_a")
        for _ in range(5):
            replay.push(_make_transition(topology_sig="sig_a"))
        replay.set_active_signature("sig_b")
        for _ in range(5):
            replay.push(_make_transition(topology_sig="sig_b"))
        assert replay.total_size == 10

        for _ in range(3):
            replay.push(_make_transition(topology_sig="sig_b"))

        assert replay.total_size == 10
        assert replay.partition_size("sig_a") == 2
        assert replay.partition_size("sig_b") == 8

    def test_eviction_ring_buffer_within_active(self):
        replay = TopologyPartitionedReplay(capacity=5, batch_size=2)
        replay.set_active_signature("sig_a")
        for _ in range(8):
            replay.push(_make_transition(topology_sig="sig_a"))
        assert replay.total_size == 5
        assert replay.active_partition_size == 5

    def test_partition_count(self):
        replay = TopologyPartitionedReplay(capacity=100, batch_size=2)
        replay.set_active_signature("sig_a")
        replay.push(_make_transition(topology_sig="sig_a"))
        replay.set_active_signature("sig_b")
        replay.push(_make_transition(topology_sig="sig_b"))
        replay.set_active_signature("sig_c")
        replay.push(_make_transition(topology_sig="sig_c"))
        assert replay.partition_count == 3

    def test_sample_batch_contents_structure(self):
        replay = TopologyPartitionedReplay(capacity=100, batch_size=3)
        replay.set_active_signature("sig_a")
        n_buildings = 2
        for _ in range(10):
            replay.push(_make_transition(n_buildings=n_buildings, topology_sig="sig_a"))

        batch = replay.sample()
        assert batch is not None
        assert len(batch.observations) == n_buildings
        assert len(batch.next_observations) == n_buildings
        assert len(batch.actions) == n_buildings
        assert len(batch.base_actions) == n_buildings
        assert len(batch.next_base_actions) == n_buildings
        assert len(batch.rewards) == n_buildings
        assert batch.observations[0].shape[0] == 3
        assert batch.rewards[0].shape == (3,)
        assert batch.done.shape == (3,)

    def test_set_active_signature_switch(self):
        replay = TopologyPartitionedReplay(capacity=100, batch_size=2)
        replay.set_active_signature("sig_a")
        for _ in range(5):
            replay.push(_make_transition(topology_sig="sig_a"))
        replay.set_active_signature("sig_b")
        for _ in range(5):
            replay.push(_make_transition(topology_sig="sig_b"))
        assert replay.active_signature == "sig_b"
        assert replay.active_partition_size == 5

    def test_checkpoint_state(self):
        replay = TopologyPartitionedReplay(capacity=50, batch_size=4)
        replay.set_active_signature("sig_a")
        for _ in range(10):
            replay.push(_make_transition(topology_sig="sig_a"))

        state = replay.state_dict()
        assert "active_signature" in state
        assert "partitions" in state

        replay2 = TopologyPartitionedReplay(capacity=50, batch_size=4)
        replay2.load_state_dict(state)
        assert replay2.active_signature == "sig_a"
        assert replay2.active_partition_size == 10
