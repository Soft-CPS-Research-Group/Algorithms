"""Update-loop helpers for AgentTransformerPPO."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from algorithms.utils.ppo_components import compute_ppo_loss


class TransformerPPOUpdateHelper:
    """Helper namespace for PPO update mechanics."""

    @staticmethod
    def as_done_flags(
        agent,
        value: bool | List[bool] | np.ndarray,
        *,
        name: str,
    ) -> List[bool]:
        if isinstance(value, np.ndarray):
            value = value.tolist()

        if isinstance(value, list):
            if len(value) != agent._num_buildings:
                logger.warning(
                    "{} length ({}) does not match number of buildings ({}); "
                    "broadcasting best-effort.",
                    name,
                    len(value),
                    agent._num_buildings,
                )
                if not value:
                    return [False] * agent._num_buildings
                return [bool(value[min(i, len(value) - 1)]) for i in range(agent._num_buildings)]
            return [bool(v) for v in value]

        return [bool(value)] * agent._num_buildings

    @staticmethod
    def ppo_update(agent, building_idx: int, last_obs: np.ndarray) -> Dict[str, float]:
        buffer = agent.rollout_buffers[building_idx]

        logger.debug(
            "Starting PPO update for building {} with {} buffered transitions.",
            building_idx,
            len(buffer),
        )

        with torch.no_grad():
            last_obs_tensor = torch.tensor(last_obs, dtype=torch.float32, device=agent.device)
            if last_obs_tensor.ndim == 1:
                last_obs_tensor = last_obs_tensor.unsqueeze(0)

            tokenized = agent.tokenizer(
                last_obs_tensor,
                marker_registry=agent._marker_registry_for_building(building_idx),
            )
            backbone_out = agent.backbone(
                tokenized.ca_tokens,
                tokenized.sro_tokens,
                tokenized.nfc_token,
            )
            last_value = agent.critic(backbone_out.pooled)

        buffer.compute_returns_and_advantages(last_value.squeeze())

        all_metrics: Dict[str, List[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
        }

        agent.tokenizer.train()
        agent.backbone.train()
        agent.actor.train()
        agent.critic.train()

        for _ in range(agent.ppo_epochs):
            for batch in buffer.get_batches(agent.minibatch_size):
                tokenized = agent.tokenizer(
                    batch.observations,
                    marker_registry=agent._marker_registry_for_building(building_idx),
                )
                backbone_out = agent.backbone(
                    tokenized.ca_tokens,
                    tokenized.sro_tokens,
                    tokenized.nfc_token,
                )

                _, log_probs_new, _ = agent.actor(
                    backbone_out.ca_embeddings,
                    deterministic=False,
                )
                log_probs_new = log_probs_new.sum(dim=-1)

                values_new = agent.critic(backbone_out.pooled).squeeze(-1)

                loss, batch_metrics = compute_ppo_loss(
                    log_probs_new=log_probs_new,
                    log_probs_old=batch.log_probs,
                    advantages=batch.advantages,
                    values=values_new,
                    returns=batch.returns,
                    clip_eps=agent.clip_eps,
                    value_coeff=agent.value_coeff,
                    entropy_coeff=agent.entropy_coeff,
                )

                agent.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.all_params, agent.max_grad_norm)
                agent.optimizer.step()

                for key, value in batch_metrics.items():
                    all_metrics[key].append(value)

        buffer.clear()

        averaged = {k: sum(v) / len(v) for k, v in all_metrics.items() if v}
        logger.info(
            "Completed PPO update for building {} with metrics {}",
            building_idx,
            averaged,
        )
        return averaged
