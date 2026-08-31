from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from algorithms.transformer_matd3.types import LayoutSignature


def _signature(n_ca: int = 1) -> LayoutSignature:
    return (
        (
            1,
            n_ca,
            tuple(f"action-{index}" for index in range(n_ca)),
            (
                ("sro", "weather", None, ("temperature",), None),
                (
                    "nfc",
                    "building_nfc",
                    "building-1",
                    ("net_consumption", "solar_generation"),
                    ("subtract", "net_consumption", "solar_generation"),
                ),
                *(
                    ("ca", "storage", f"storage-{index}", ("soc", "power"), None)
                    for index in range(n_ca)
                ),
            ),
            (),
            (("building_nfc", 1), ("storage", 2), ("weather", 2)),
        ),
    )


def _push(buffer, signature: LayoutSignature, value: float) -> None:
    n_ca = signature[0][1]
    buffer.push(
        encoded_obs=[np.full(6, value, dtype=np.float32)],
        next_encoded_obs=[np.full(6, value + 1, dtype=np.float32)],
        actions=[np.full(n_ca, value, dtype=np.float32)],
        reward=np.array([value], dtype=np.float32),
        terminated=False,
        truncated=False,
        layout_signature=signature,
    )


def test_should_sample_only_one_layout_signature() -> None:
    from algorithms.transformer_matd3.replay import SignatureBucketedReplayBuffer

    buffer = SignatureBucketedReplayBuffer(capacity=6, num_agents=1, batch_size=2)
    first_signature = _signature(1)
    second_signature = _signature(2)
    _push(buffer, first_signature, 1)
    _push(buffer, second_signature, 2)
    _push(buffer, first_signature, 3)

    batch = buffer.sample(first_signature, 2)

    assert batch.signature == first_signature
    assert batch.observations[0].shape == (2, 6)
    assert batch.actions[0].shape == (2, 1)
    assert buffer.bucket_size(second_signature) == 1
    assert tuple(buffer.signatures()) == (first_signature, second_signature)


def test_should_evict_oldest_transition_globally() -> None:
    from algorithms.transformer_matd3.replay import SignatureBucketedReplayBuffer

    buffer = SignatureBucketedReplayBuffer(capacity=2, num_agents=1, batch_size=1)
    old_signature = _signature(1)
    current_signature = _signature(2)
    _push(buffer, old_signature, 1)
    _push(buffer, current_signature, 2)
    _push(buffer, current_signature, 3)

    assert buffer.total_size() == 2
    assert buffer.bucket_size(old_signature) == 0
    assert tuple(buffer.signatures()) == (current_signature,)


def test_should_copy_inserted_arrays() -> None:
    from algorithms.transformer_matd3.replay import SignatureBucketedReplayBuffer

    buffer = SignatureBucketedReplayBuffer(capacity=2, num_agents=1, batch_size=1)
    signature = _signature()
    observation = np.ones(6, dtype=np.float32)
    buffer.push(
        encoded_obs=[observation],
        next_encoded_obs=[observation],
        actions=[np.ones(1, dtype=np.float32)],
        reward=np.ones(1, dtype=np.float32),
        terminated=False,
        truncated=False,
        layout_signature=signature,
    )
    observation.fill(99)

    batch = buffer.sample(signature, 1)

    assert np.all(batch.observations[0] == 1)
    assert batch.observations[0].dtype == np.float32


def test_should_round_trip_explicit_base_proposed_and_executed_actions() -> None:
    from algorithms.transformer_matd3.replay import SignatureBucketedReplayBuffer

    buffer = SignatureBucketedReplayBuffer(capacity=2, num_agents=1, batch_size=1)
    signature = _signature()
    buffer.push(
        encoded_obs=[np.zeros(6, dtype=np.float32)],
        next_encoded_obs=[np.ones(6, dtype=np.float32)],
        actions=[np.array([0.7], dtype=np.float32)],
        proposed_actions=[np.array([0.4], dtype=np.float32)],
        executed_actions=[np.array([0.7], dtype=np.float32)],
        base_actions=[np.array([0.1], dtype=np.float32)],
        reward=[0.0],
        terminated=False,
        truncated=False,
        layout_signature=signature,
    )

    transition = buffer.get_state()["transitions"][0]
    batch = buffer.sample(signature, 1)
    assert transition.actions[0].tolist() == pytest.approx([0.7])
    assert transition.proposed_actions[0].tolist() == pytest.approx([0.4])
    assert transition.executed_actions[0].tolist() == pytest.approx([0.7])
    assert transition.base_actions[0].tolist() == pytest.approx([0.1])
    assert batch.proposed_actions[0][0].tolist() == pytest.approx([0.4])
    assert batch.executed_actions[0][0].tolist() == pytest.approx([0.7])
    assert batch.base_actions[0][0].tolist() == pytest.approx([0.1])


def test_should_reject_action_width_mismatch() -> None:
    from algorithms.transformer_matd3.replay import SignatureBucketedReplayBuffer

    buffer = SignatureBucketedReplayBuffer(capacity=2, num_agents=1, batch_size=1)

    with pytest.raises(ValueError, match=r"action\[0\].*expected 2"):
        buffer.push(
            encoded_obs=[np.ones(6, dtype=np.float32)],
            next_encoded_obs=[np.ones(6, dtype=np.float32)],
            actions=[np.ones(1, dtype=np.float32)],
            reward=np.ones(1, dtype=np.float32),
            terminated=False,
            truncated=False,
            layout_signature=_signature(2),
        )


def test_should_reject_optional_field_presence_mismatch_within_bucket() -> None:
    from algorithms.transformer_matd3.replay import SignatureBucketedReplayBuffer

    buffer = SignatureBucketedReplayBuffer(capacity=2, num_agents=1, batch_size=1)
    signature = _signature()
    _push(buffer, signature, 1)

    with pytest.raises(ValueError, match="behavior_actions presence"):
        buffer.push(
            encoded_obs=[np.ones(6, dtype=np.float32)],
            next_encoded_obs=[np.ones(6, dtype=np.float32)],
            actions=[np.ones(1, dtype=np.float32)],
            behavior_actions=[np.ones(1, dtype=np.float32)],
            reward=np.ones(1, dtype=np.float32),
            terminated=False,
            truncated=False,
            layout_signature=signature,
        )


def test_should_restore_complete_buffer_state_and_rng() -> None:
    from algorithms.transformer_matd3.replay import SignatureBucketedReplayBuffer

    signature = _signature()
    source = SignatureBucketedReplayBuffer(capacity=4, num_agents=1, batch_size=2)
    _push(source, signature, 1)
    _push(source, signature, 2)
    _push(source, signature, 3)
    state = source.get_state()
    expected = source.sample(signature, 2)

    restored = SignatureBucketedReplayBuffer(capacity=4, num_agents=1, batch_size=2)
    restored.set_state(state)
    actual = restored.sample(signature, 2)

    assert restored.total_size() == 3
    assert np.array_equal(actual.observations[0], expected.observations[0])


def test_should_round_trip_ordered_buckets_fifo_and_optional_fields() -> None:
    from algorithms.transformer_matd3.replay import SignatureBucketedReplayBuffer

    first_signature = _signature(1)
    second_signature = _signature(2)
    source = SignatureBucketedReplayBuffer(capacity=4, num_agents=1, batch_size=1)
    source.push(
        encoded_obs=[np.full(6, 1, dtype=np.float32)],
        next_encoded_obs=[np.full(6, 2, dtype=np.float32)],
        actions=[np.ones(1, dtype=np.float32)],
        behavior_actions=[np.full(1, 3, dtype=np.float32)],
        next_behavior_actions=[np.full(1, 4, dtype=np.float32)],
        cloning_actions=[np.full(1, 5, dtype=np.float32)],
        reward=np.array([1], dtype=np.float32),
        terminated=False,
        truncated=False,
        layout_signature=first_signature,
    )
    _push(source, second_signature, 2)
    source.push(
        encoded_obs=[np.full(6, 3, dtype=np.float32)],
        next_encoded_obs=[np.full(6, 4, dtype=np.float32)],
        actions=[np.ones(1, dtype=np.float32)],
        behavior_actions=[np.full(1, 6, dtype=np.float32)],
        next_behavior_actions=[np.full(1, 7, dtype=np.float32)],
        cloning_actions=[np.full(1, 8, dtype=np.float32)],
        reward=np.array([3], dtype=np.float32),
        terminated=False,
        truncated=False,
        layout_signature=first_signature,
    )

    state = source.get_state()
    restored = SignatureBucketedReplayBuffer(capacity=4, num_agents=1, batch_size=1)
    restored.set_state(state)
    restored_state = restored.get_state()

    assert restored_state["format"] == "signature_bucketed_v1"
    assert tuple(restored_state["buckets"]) == (first_signature, second_signature)
    assert restored_state["global_fifo"] == (
        (0, first_signature),
        (1, second_signature),
        (2, first_signature),
    )
    assert restored_state["next_sequence_id"] == 3
    assert restored_state["total_size"] == 3
    restored_transition = restored_state["buckets"][first_signature][0]
    assert np.array_equal(
        restored_transition.behavior_actions[0],
        np.full(1, 3, dtype=np.float32),
    )
    assert np.array_equal(
        restored_transition.next_behavior_actions[0],
        np.full(1, 4, dtype=np.float32),
    )
    assert np.array_equal(
        restored_transition.cloning_actions[0],
        np.full(1, 5, dtype=np.float32),
    )


def test_should_validate_bucket_and_fifo_state_atomically() -> None:
    from algorithms.transformer_matd3.replay import SignatureBucketedReplayBuffer

    signature = _signature()
    buffer = SignatureBucketedReplayBuffer(capacity=3, num_agents=1, batch_size=1)
    _push(buffer, signature, 1)
    original = buffer.get_state()
    invalid = dict(original, global_fifo=((99, signature),))

    with pytest.raises(ValueError, match="global_fifo does not match"):
        buffer.set_state(invalid)

    current = buffer.get_state()
    assert current["global_fifo"] == original["global_fifo"]
    assert current["next_sequence_id"] == original["next_sequence_id"]
    assert current["rng_state"] == original["rng_state"]
    assert np.array_equal(
        current["transitions"][0].observations[0],
        original["transitions"][0].observations[0],
    )


def test_should_reject_underfilled_sample_request() -> None:
    from algorithms.transformer_matd3.replay import SignatureBucketedReplayBuffer

    buffer = SignatureBucketedReplayBuffer(capacity=2, num_agents=1, batch_size=2)
    signature = _signature()
    _push(buffer, signature, 1)

    with pytest.raises(ValueError, match="contains 1 transitions"):
        buffer.sample(signature, 2)


def test_should_reject_malformed_nested_signature() -> None:
    from algorithms.transformer_matd3.replay import SignatureBucketedReplayBuffer

    buffer = SignatureBucketedReplayBuffer(capacity=2, num_agents=1, batch_size=1)
    malformed_signature = ((1, 1),)

    with pytest.raises(ValueError, match="building 0 must contain six fields"):
        buffer.bucket_size(malformed_signature)


def test_should_preserve_live_state_when_restored_transition_is_invalid() -> None:
    from algorithms.transformer_matd3.replay import SignatureBucketedReplayBuffer

    signature = _signature()
    buffer = SignatureBucketedReplayBuffer(capacity=3, num_agents=1, batch_size=1)
    _push(buffer, signature, 1)
    original_state = buffer.get_state()
    invalid_transition = replace(
        original_state["transitions"][0],
        actions=(np.ones(2, dtype=np.float32),),
    )
    invalid_state = dict(original_state, transitions=[invalid_transition])

    with pytest.raises(ValueError, match=r"action\[0\].*expected 1"):
        buffer.set_state(invalid_state)

    restored_state = buffer.get_state()
    assert restored_state["next_sequence_id"] == original_state["next_sequence_id"]
    assert np.array_equal(
        restored_state["transitions"][0].observations[0],
        original_state["transitions"][0].observations[0],
    )


def test_should_reject_negative_sequence_counter_for_empty_state() -> None:
    from algorithms.transformer_matd3.replay import SignatureBucketedReplayBuffer

    buffer = SignatureBucketedReplayBuffer(capacity=2, num_agents=1, batch_size=1)
    state = buffer.get_state()
    state["next_sequence_id"] = -1

    with pytest.raises(ValueError, match="next_sequence_id must be non-negative"):
        buffer.set_state(state)


def test_should_reject_unhashable_restored_sequence_id() -> None:
    from algorithms.transformer_matd3.replay import SignatureBucketedReplayBuffer

    signature = _signature()
    buffer = SignatureBucketedReplayBuffer(capacity=2, num_agents=1, batch_size=1)
    _push(buffer, signature, 1)
    state = buffer.get_state()
    state["transitions"] = [
        replace(state["transitions"][0], sequence_id=[]),
    ]

    with pytest.raises(ValueError, match="sequence_id must be non-negative"):
        buffer.set_state(state)


def test_should_reject_unhashable_signature_segment_family() -> None:
    from algorithms.transformer_matd3.replay import SignatureBucketedReplayBuffer

    buffer = SignatureBucketedReplayBuffer(capacity=2, num_agents=1, batch_size=1)
    building = _signature()[0]
    malformed_segment = ([], "weather", None, ("temperature",), None)
    malformed_signature = (
        (building[0], building[1], building[2], (malformed_segment,), (), building[5]),
    )

    with pytest.raises(ValueError, match="segment family must be"):
        buffer.bucket_size(malformed_signature)
