"""Export helpers for AgentTransformerPPO."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from loguru import logger

from algorithms.constants import DEFAULT_ONNX_OPSET
from algorithms.utils.observation_tokenizer import ObservationTokenizer
from algorithms.utils.ppo_components import ActorHead
from algorithms.utils.transformer_backbone import TransformerBackbone


class _DeterministicActorExport(nn.Module):
    """Minimal export wrapper for deterministic actor inference."""

    def __init__(self, actor: ActorHead) -> None:
        super().__init__()
        self.actor = actor

    def forward(self, ca_embeddings: torch.Tensor) -> torch.Tensor:
        means = self.actor.mlp(ca_embeddings)
        return torch.tanh(means)


class _DeterministicPolicyExport(nn.Module):
    """End-to-end deterministic policy export module."""

    def __init__(
        self,
        tokenizer: ObservationTokenizer,
        backbone: TransformerBackbone,
        actor: ActorHead,
        marker_registry: Optional[Dict[float, tuple[str, str, Optional[str]]]] = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.backbone = backbone
        self.actor = actor
        self.marker_registry = marker_registry or {}

    def forward(self, encoded_obs: torch.Tensor) -> torch.Tensor:
        tokenized = self.tokenizer(encoded_obs, marker_registry=self.marker_registry)
        backbone_out = self.backbone(
            tokenized.ca_tokens,
            tokenized.sro_tokens,
            tokenized.nfc_token,
        )
        means = self.actor.mlp(backbone_out.ca_embeddings)
        return torch.tanh(means)


class TransformerPPOExportHelper:
    """Helper namespace for export-related operations."""

    @staticmethod
    def build_dummy_observation(agent, agent_index: int) -> torch.Tensor:
        if 0 <= agent_index < len(agent._last_obs) and agent._last_obs[agent_index] is not None:
            cached = agent._last_obs[agent_index].detach().to(agent.device)
            if cached.ndim == 1:
                return cached.unsqueeze(0)
            if cached.ndim == 2:
                return cached[:1]

        marker_values = agent.tokenizer_config.get("marker_values", {})
        ca_base = float(marker_values.get("ca_base", 1000))
        sro_base = float(marker_values.get("sro_base", 2000))
        nfc_marker = float(marker_values.get("nfc", 3001))

        values: List[float] = []

        ca_types = agent.tokenizer_config.get("ca_types", {})
        if ca_types:
            first_ca_spec = next(iter(ca_types.values()))
            ca_dim = int(first_ca_spec.get("input_dim", 1))
            values.append(ca_base + 1.0)
            values.extend([0.0] * max(ca_dim, 0))

        sro_types = agent.tokenizer_config.get("sro_types", {})
        for idx, spec in enumerate(sro_types.values(), start=1):
            sro_dim = int(spec.get("input_dim", 0))
            if sro_dim <= 0:
                continue
            values.append(sro_base + float(idx))
            values.extend([0.0] * sro_dim)

        nfc_dim = int((agent.tokenizer_config.get("nfc", {}) or {}).get("input_dim", 0))
        if nfc_dim > 0:
            values.append(nfc_marker)
            values.extend([0.0] * nfc_dim)

        if not values:
            values = [ca_base + 1.0, 0.0]

        return torch.tensor([values], dtype=torch.float32, device=agent.device)

    @staticmethod
    def export_end_to_end_onnx(agent, export_root: Path) -> List[Dict[str, Any]]:
        num_exports = max(agent._num_buildings, 1)
        onnx_dir = export_root / "onnx_models"
        onnx_dir.mkdir(parents=True, exist_ok=True)

        artifacts: List[Dict[str, Any]] = []

        for agent_index in range(num_exports):
            dummy_obs = TransformerPPOExportHelper.build_dummy_observation(agent, agent_index)
            export_path = onnx_dir / f"agent_{agent_index}.onnx"
            export_module = _DeterministicPolicyExport(
                tokenizer=agent.tokenizer,
                backbone=agent.backbone,
                actor=agent.actor,
                marker_registry=agent._marker_registry_for_building(agent_index),
            ).to(agent.device)
            export_module.eval()

            try:
                previous_fastpath = torch.backends.mha.get_fastpath_enabled()
                torch.backends.mha.set_fastpath_enabled(False)
                try:
                    with torch.no_grad():
                        torch.onnx.export(
                            export_module,
                            dummy_obs,
                            str(export_path),
                            export_params=True,
                            opset_version=max(DEFAULT_ONNX_OPSET, 17),
                            do_constant_folding=True,
                            input_names=[f"observation_agent_{agent_index}"],
                            output_names=[f"action_agent_{agent_index}"],
                            dynamic_axes={
                                f"observation_agent_{agent_index}": {0: "batch_size"},
                                f"action_agent_{agent_index}": {0: "batch_size", 1: "n_ca"},
                            },
                        )
                finally:
                    torch.backends.mha.set_fastpath_enabled(previous_fastpath)

                logger.info(
                    "Exported end-to-end TransformerPPO ONNX for agent {} at {}",
                    agent_index,
                    export_path,
                )
            except Exception as exc:
                logger.warning(
                    "End-to-end ONNX export failed for agent {} ({}); "
                    "falling back to actor-only ONNX export.",
                    agent_index,
                    exc,
                )
                try:
                    actor_fallback = _DeterministicActorExport(agent.actor).to(agent.device).eval()
                    dummy_ca_embeddings = torch.randn(1, 1, agent.d_model, device=agent.device)
                    with torch.no_grad():
                        torch.onnx.export(
                            actor_fallback,
                            dummy_ca_embeddings,
                            str(export_path),
                            export_params=True,
                            opset_version=DEFAULT_ONNX_OPSET,
                            do_constant_folding=True,
                            input_names=[f"ca_embeddings_agent_{agent_index}"],
                            output_names=[f"action_agent_{agent_index}"],
                            dynamic_axes={
                                f"ca_embeddings_agent_{agent_index}": {0: "batch_size", 1: "n_ca"},
                                f"action_agent_{agent_index}": {0: "batch_size", 1: "n_ca"},
                            },
                        )
                    logger.warning(
                        "Exported actor-only fallback ONNX for agent {} at {}",
                        agent_index,
                        export_path,
                    )
                except Exception as fallback_exc:
                    logger.error(
                        "ONNX fallback export also failed for agent {} ({}).",
                        agent_index,
                        fallback_exc,
                    )
                    raise RuntimeError(
                        f"Failed to export ONNX artifact for agent {agent_index}."
                    ) from fallback_exc

            artifacts.append(
                {
                    "agent_index": agent_index,
                    "path": str(export_path.relative_to(export_root)),
                    "format": "onnx",
                    "config": {},
                }
            )

        return artifacts
