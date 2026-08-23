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
                ("sro", "weather", None),
                ("nfc", "building_nfc", "building-1"),
                *(("ca", "storage", f"storage-{index}") for index in range(n_ca)),
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
    malformed_segment = ([], "weather", None)
    malformed_signature = (
        (building[0], building[1], building[2], (malformed_segment,), (), building[5]),
    )

    with pytest.raises(ValueError, match="segment family must be"):
        buffer.bucket_size(malformed_signature)
