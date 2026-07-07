"""Unit tests for the global critic token packer."""
from __future__ import annotations

import torch

from algorithms.utils.matd3_global_packer import (
    GlobalTokenPacker,
    PackedGlobalSequence,
    BuildingLayout,
)


def _make_packer(d_model=16, num_token_types=8, max_buildings=4,
                 action_input_mode="final") -> GlobalTokenPacker:
    return GlobalTokenPacker(
        d_model=d_model,
        num_token_types=num_token_types,
        max_buildings=max_buildings,
        action_input_mode=action_input_mode,
    )


def _make_layouts(n_buildings=2, n_sro=2, n_ca=2) -> list[BuildingLayout]:
    """Create synthetic building layouts."""
    return [
        BuildingLayout(
            building_index=b,
            n_sro=n_sro,
            n_nfc=1,
            n_ca=n_ca,
            is_controlled=True,
        )
        for b in range(n_buildings)
    ]


class TestGlobalTokenPacker:
    def test_output_type(self):
        packer = _make_packer()
        layouts = _make_layouts(n_buildings=2, n_sro=2, n_ca=2)
        obs_tokens = [torch.randn(3, 5, 16) for _ in range(2)]
        action_values = [torch.randn(3, 2) for _ in range(2)]

        packed = packer.pack(obs_tokens, action_values, layouts)
        assert isinstance(packed, PackedGlobalSequence)

    def test_output_shapes(self):
        packer = _make_packer()
        layouts = _make_layouts(n_buildings=2, n_sro=3, n_ca=2)
        obs_tokens = [torch.randn(4, 6, 16) for _ in range(2)]
        action_values = [torch.randn(4, 2) for _ in range(2)]

        packed = packer.pack(obs_tokens, action_values, layouts)
        assert packed.global_tokens.shape == (4, 16, 16)
        assert packed.type_ids.shape == (4, 16)
        assert packed.building_ids.shape == (4, 16)
        assert packed.padding_mask.shape == (4, 16)

    def test_variable_building_token_counts(self):
        """Different buildings can have different token counts with padding."""
        packer = _make_packer()
        layouts = [
            BuildingLayout(building_index=0, n_sro=2, n_nfc=1, n_ca=1, is_controlled=True),
            BuildingLayout(building_index=1, n_sro=4, n_nfc=1, n_ca=3, is_controlled=True),
        ]
        obs_tokens = [torch.randn(2, 4, 16), torch.randn(2, 8, 16)]
        action_values = [torch.randn(2, 1), torch.randn(2, 3)]

        packed = packer.pack(obs_tokens, action_values, layouts)
        assert packed.global_tokens.shape == (2, 16, 16)
        assert packed.padding_mask.shape == (2, 16)

    def test_padding_mask_correct(self):
        """Padding positions should be True in mask."""
        packer = _make_packer()
        layouts = _make_layouts(n_buildings=1, n_sro=2, n_ca=2)
        obs_tokens = [torch.randn(1, 5, 16)]
        action_values = [torch.randn(1, 2)]

        packed = packer.pack(obs_tokens, action_values, layouts)
        assert not packed.padding_mask.any()

    def test_action_mode_final(self):
        """Final mode: action token contains 1 scalar projected to d_model."""
        packer = _make_packer(action_input_mode="final")
        layouts = _make_layouts(n_buildings=1, n_sro=1, n_ca=2)
        obs_tokens = [torch.randn(2, 4, 16)]
        action_values = [torch.randn(2, 2)]

        packed = packer.pack(obs_tokens, action_values, layouts)
        assert packed.global_tokens.shape[1] == 6

    def test_action_mode_final_base_delta(self):
        """final_base_delta mode: action token carries 3 scalars."""
        packer = _make_packer(action_input_mode="final_base_delta")
        layouts = _make_layouts(n_buildings=1, n_sro=1, n_ca=2)
        obs_tokens = [torch.randn(2, 4, 16)]
        action_values = [torch.randn(2, 2)]
        base_actions = [torch.randn(2, 2)]

        packed = packer.pack(
            obs_tokens, action_values, layouts,
            base_actions=base_actions,
        )
        assert packed.global_tokens.shape[1] == 6

    def test_action_mode_final_base_delta_normalized(self):
        """Normalized mode: delta is divided by action_span."""
        packer = _make_packer(action_input_mode="final_base_delta_normalized")
        layouts = _make_layouts(n_buildings=1, n_sro=1, n_ca=2)
        obs_tokens = [torch.randn(2, 4, 16)]
        action_values = [torch.randn(2, 2)]
        base_actions = [torch.randn(2, 2)]

        packed = packer.pack(
            obs_tokens, action_values, layouts,
            base_actions=base_actions,
            action_span=2.0,
        )
        assert packed.global_tokens.shape[1] == 6

    def test_controlled_building_indices(self):
        """PackedGlobalSequence reports correct controlled building list."""
        packer = _make_packer()
        layouts = [
            BuildingLayout(building_index=0, n_sro=2, n_nfc=1, n_ca=2, is_controlled=True),
            BuildingLayout(building_index=1, n_sro=2, n_nfc=1, n_ca=0, is_controlled=False),
            BuildingLayout(building_index=2, n_sro=2, n_nfc=1, n_ca=1, is_controlled=True),
        ]
        obs_tokens = [
            torch.randn(1, 5, 16),
            torch.randn(1, 3, 16),
            torch.randn(1, 4, 16),
        ]
        action_values = [torch.randn(1, 2), torch.randn(1, 0), torch.randn(1, 1)]
        packed = packer.pack(obs_tokens, action_values, layouts)
        assert packed.controlled_building_indices == [0, 2]

    def test_building_ids_correct(self):
        """Each token's building_id matches its source building."""
        packer = _make_packer()
        layouts = _make_layouts(n_buildings=2, n_sro=1, n_ca=1)
        obs_tokens = [torch.randn(1, 3, 16), torch.randn(1, 3, 16)]
        action_values = [torch.randn(1, 1), torch.randn(1, 1)]

        packed = packer.pack(obs_tokens, action_values, layouts)
        assert (packed.building_ids[0, :4] == 0).all()
        assert (packed.building_ids[0, 4:] == 1).all()
