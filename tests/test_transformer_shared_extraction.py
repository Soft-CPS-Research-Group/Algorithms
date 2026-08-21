"""Compatibility tests for the transient Transformer PPO re-export shims."""

from pickle import loads
from types import SimpleNamespace

import torch

from algorithms.transformer_shared.behavior_cloning import (
    BehaviorCloningRegularizer as SharedBehaviorCloningRegularizer,
)
from algorithms.transformer_shared.behavior_cloning import (
    Demonstration as SharedDemonstration,
)
from algorithms.transformer_shared.entity_observation_tokenizer import (
    EntityObservationTokenizer as SharedEntityObservationTokenizer,
)
from algorithms.transformer_shared.entity_observation_tokenizer import (
    TokenizedObservation as SharedTokenizedObservation,
)
from algorithms.transformer_shared.entity_token_layout import (
    BuildingTokenLayout as SharedBuildingTokenLayout,
)
from algorithms.transformer_shared.entity_token_layout import (
    EntityTokenLayoutBuilder as SharedEntityTokenLayoutBuilder,
)
from algorithms.transformer_shared.entity_token_layout import (
    NfcExpression as SharedNfcExpression,
)
from algorithms.transformer_shared.entity_token_layout import (
    TokenSegment as SharedTokenSegment,
)
from algorithms.transformer_shared.transformer_backbone import (
    TransformerBackbone as SharedTransformerBackbone,
)
from algorithms.transformer_shared.value_normalizer import (
    RunningValueNormalizer as SharedRunningValueNormalizer,
)
from algorithms.transformer_ppo.behavior_cloning import (
    BehaviorCloningRegularizer as LegacyBehaviorCloningRegularizer,
)
from algorithms.transformer_ppo.behavior_cloning import (
    Demonstration as LegacyDemonstration,
)
from algorithms.transformer_ppo.entity_observation_tokenizer import (
    EntityObservationTokenizer as LegacyEntityObservationTokenizer,
)
from algorithms.transformer_ppo.entity_observation_tokenizer import (
    TokenizedObservation as LegacyTokenizedObservation,
)
from algorithms.transformer_ppo.entity_token_layout import (
    BuildingTokenLayout as LegacyBuildingTokenLayout,
)
from algorithms.transformer_ppo.entity_token_layout import (
    EntityTokenLayoutBuilder as LegacyEntityTokenLayoutBuilder,
)
from algorithms.transformer_ppo.entity_token_layout import (
    NfcExpression as LegacyNfcExpression,
)
from algorithms.transformer_ppo.entity_token_layout import (
    TokenSegment as LegacyTokenSegment,
)
from algorithms.transformer_ppo.ppo_components import (
    RunningValueNormalizer as LegacyRunningValueNormalizer,
)
from algorithms.transformer_ppo.transformer_backbone import (
    TransformerBackbone as LegacyTransformerBackbone,
)


def test_tppo_shims_reexport_shared_public_types_by_identity():
    pairs = (
        (LegacyBehaviorCloningRegularizer, SharedBehaviorCloningRegularizer),
        (LegacyDemonstration, SharedDemonstration),
        (LegacyEntityObservationTokenizer, SharedEntityObservationTokenizer),
        (LegacyTokenizedObservation, SharedTokenizedObservation),
        (LegacyBuildingTokenLayout, SharedBuildingTokenLayout),
        (LegacyEntityTokenLayoutBuilder, SharedEntityTokenLayoutBuilder),
        (LegacyNfcExpression, SharedNfcExpression),
        (LegacyTokenSegment, SharedTokenSegment),
        (LegacyRunningValueNormalizer, SharedRunningValueNormalizer),
        (LegacyTransformerBackbone, SharedTransformerBackbone),
    )

    assert all(legacy is shared for legacy, shared in pairs)


def test_legacy_serialized_type_globals_resolve_through_shims():
    legacy_globals = {
        "algorithms.transformer_ppo.behavior_cloning": {
            "BehaviorCloningRegularizer": SharedBehaviorCloningRegularizer,
            "Demonstration": SharedDemonstration,
        },
        "algorithms.transformer_ppo.entity_observation_tokenizer": {
            "EntityObservationTokenizer": SharedEntityObservationTokenizer,
            "TokenizedObservation": SharedTokenizedObservation,
        },
        "algorithms.transformer_ppo.entity_token_layout": {
            "BuildingTokenLayout": SharedBuildingTokenLayout,
            "EntityTokenLayoutBuilder": SharedEntityTokenLayoutBuilder,
            "NfcExpression": SharedNfcExpression,
            "TokenSegment": SharedTokenSegment,
        },
        "algorithms.transformer_ppo.ppo_components": {
            "RunningValueNormalizer": SharedRunningValueNormalizer,
        },
        "algorithms.transformer_ppo.transformer_backbone": {
            "TransformerBackbone": SharedTransformerBackbone,
        },
    }

    for module_name, symbols in legacy_globals.items():
        for symbol_name, shared_type in symbols.items():
            payload = f"c{module_name}\n{symbol_name}\n.".encode()
            assert loads(payload) is shared_type


def test_moved_torch_modules_accept_legacy_state_dicts_strictly():
    tokenizer_config = SimpleNamespace(
        nfc=SimpleNamespace(type_name="net_load"),
        ca_types={"storage": object()},
        sro_types={"weather": object()},
    )
    type_input_dims = {"net_load": 1, "storage": 2, "weather": 3}
    legacy_tokenizer = LegacyEntityObservationTokenizer(
        tokenizer_config, d_model=8, type_input_dims=type_input_dims
    )
    shared_tokenizer = SharedEntityObservationTokenizer(
        tokenizer_config, d_model=8, type_input_dims=type_input_dims
    )
    shared_tokenizer.load_state_dict(legacy_tokenizer.state_dict(), strict=True)

    legacy_backbone = LegacyTransformerBackbone(
        d_model=8, nhead=2, num_layers=1, dim_feedforward=16, dropout=0.0
    )
    shared_backbone = SharedTransformerBackbone(
        d_model=8, nhead=2, num_layers=1, dim_feedforward=16, dropout=0.0
    )
    shared_backbone.load_state_dict(legacy_backbone.state_dict(), strict=True)

    for legacy_value, shared_value in zip(
        legacy_tokenizer.state_dict().values(),
        shared_tokenizer.state_dict().values(),
    ):
        torch.testing.assert_close(legacy_value, shared_value)
    for legacy_value, shared_value in zip(
        legacy_backbone.state_dict().values(),
        shared_backbone.state_dict().values(),
    ):
        torch.testing.assert_close(legacy_value, shared_value)
