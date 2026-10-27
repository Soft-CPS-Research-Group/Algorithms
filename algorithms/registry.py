"""Lazy agent registry and the single shared execution-unit builder."""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from copy import deepcopy
from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict, List

from loguru import logger

from algorithms.catalogue import AGENT_SPECS
from algorithms.contracts import AgentSpec

if TYPE_CHECKING:
    from algorithms.execution_unit import ExecutionUnit


def import_symbol(path: str) -> Any:
    module, name = path.split(":", 1)
    return getattr(import_module(module), name)


class LazyAgentRegistry(MutableMapping):
    """Mapping compatibility for callers, with imports deferred until lookup."""

    def __init__(self) -> None:
        self._loaded: dict[str, Any] = {}

    def __getitem__(self, name: str) -> Any:
        if name not in self._loaded:
            self._loaded[name] = import_symbol(AGENT_SPECS[name].implementation)
        return self._loaded[name]

    def __setitem__(self, name: str, value: Any) -> None:
        self._loaded[name] = value

    def __delitem__(self, name: str) -> None:
        self._loaded.pop(name, None)
        AGENT_SPECS.pop(name, None)
        ENCODED_OBSERVATION_ALGORITHMS.discard(name)

    def __iter__(self) -> Iterator[str]:
        return iter(dict.fromkeys([*AGENT_SPECS, *self._loaded]))

    def __len__(self) -> int:
        return len(set(AGENT_SPECS) | set(self._loaded))

    def __contains__(self, name: object) -> bool:
        return name in AGENT_SPECS or name in self._loaded


ALGORITHM_REGISTRY = LazyAgentRegistry()
PLACEHOLDER_ALGORITHMS = {"SingleAgentRL"}
ENCODED_OBSERVATION_ALGORITHMS = set(
    name for name, spec in AGENT_SPECS.items() if spec.capabilities.observations == "encoded"
)


def agent_spec(name: str) -> AgentSpec:
    try:
        return AGENT_SPECS[name]
    except KeyError as error:
        raise ValueError(build_unsupported_algorithm_message(name)) from error


def register_agent(spec: AgentSpec) -> None:
    """Register implementation, parameter schema and capabilities together."""
    if spec.name in AGENT_SPECS or spec.name in PLACEHOLDER_ALGORITHMS:
        raise ValueError(f"Algorithm {spec.name!r} is already registered")
    for path in (spec.implementation, spec.configuration):
        if ":" not in path:
            raise ValueError("Implementation and configuration use module:Symbol paths")
    AGENT_SPECS[spec.name] = spec
    if spec.capabilities.observations == "encoded":
        ENCODED_OBSERVATION_ALGORITHMS.add(spec.name)


def supported_algorithms() -> List[str]:
    """Return registered algorithm names sorted for stable error messages."""
    return sorted(ALGORITHM_REGISTRY.keys())


def placeholder_algorithms() -> List[str]:
    """Return known schema placeholders that are not yet runtime implementations."""
    return sorted(PLACEHOLDER_ALGORITHMS)


def is_algorithm_supported(name: str | None) -> bool:
    """Check whether ``name`` is backed by a runtime implementation."""
    return bool(name) and name in ALGORITHM_REGISTRY


def build_unsupported_algorithm_message(name: str | None) -> str:
    """Build a clear fail-fast message for unsupported/placeholder algorithms."""
    supported = ", ".join(supported_algorithms()) or "none"
    placeholders = ", ".join(placeholder_algorithms()) or "none"

    if not name:
        return (
            "Algorithm name is required in configuration. "
            f"Supported algorithms: {supported}. "
            f"Known placeholders: {placeholders}."
        )

    if name in PLACEHOLDER_ALGORITHMS:
        return (
            f"Algorithm '{name}' is a schema placeholder and has no runtime implementation yet. "
            f"Supported algorithms: {supported}. "
            f"Known placeholders: {placeholders}."
        )

    return (
        f"Unsupported algorithm '{name}'. "
        f"Supported algorithms: {supported}. "
        f"Known placeholders: {placeholders}."
    )


def _stage_to_agent_view(
    global_config: Dict[str, Any], stage_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """Synthesise an agent-facing config view for a single pipeline stage.

    Existing agent constructors read ``self.config["algorithm"][...]`` as
    historical convention. Rather than rewriting every agent in this
    branch, the builder copies the global config and substitutes a stage
    slice under ``algorithm`` so each agent sees the same shape it has
    always seen. The migration to per-stage config objects can happen
    independently in a follow-up, without changing this builder.
    """
    agent_view = dict(global_config)
    algorithm_block: Dict[str, Any] = {
        "name": stage_cfg["algorithm"],
        "hyperparameters": stage_cfg.get("hyperparameters", {}) or {},
    }
    for optional_key in (
        "networks",
        "replay_buffer",
        "exploration",
        "policy",
        "tokenizer",
        "tokenizer_config_path",
        "transformer",
        "behavior_cloning",
    ):
        if optional_key in stage_cfg and stage_cfg[optional_key] is not None:
            algorithm_block[optional_key] = stage_cfg[optional_key]
    agent_view["algorithm"] = algorithm_block
    return agent_view


def _single_agent_member_view(
    agent_view: Dict[str, Any],
    *,
    member_index: int,
) -> Dict[str, Any]:
    """Return a one-slot config view for a strict single-agent ensemble member.

    Runtime topology is derived for the complete CityLearn environment before
    execution units are built.  A strict single-agent implementation must not
    accidentally construct one actor/critic per building from that global
    topology.  This helper slices the derived dimensions to the member's slot
    and gives every learner a stable seed offset.
    """
    member_view = deepcopy(agent_view)
    topology = dict(member_view.get("topology") or {})

    for key in ("observation_dimensions", "action_dimensions", "action_space"):
        values = topology.get(key)
        if values is None:
            continue
        if not isinstance(values, (list, tuple)):
            raise ValueError(
                f"topology.{key} must be a list when building a strict single-agent ensemble."
            )
        if member_index >= len(values):
            raise ValueError(
                f"Cannot build single-agent member {member_index}: topology.{key} "
                f"contains only {len(values)} slot(s)."
            )
        topology[key] = [deepcopy(values[member_index])]

    topology["num_agents"] = 1
    member_view["topology"] = topology

    training = dict(member_view.get("training") or {})
    base_seed = int(training.get("seed", 22))
    training["seed"] = base_seed + int(member_index)
    member_view["training"] = training

    runtime = dict(member_view.get("runtime") or {})
    runtime["member_index"] = int(member_index)
    member_view["runtime"] = runtime
    return member_view


def build_execution_unit(config: Dict[str, Any]) -> ExecutionUnit:
    """Instantiate the model the wrapper drives.

    Reads ``config['pipeline']`` (an ordered list of stage descriptions)
    and produces:

    * a single :class:`BaseAgent` when the pipeline has exactly one
      stage with ``count == 1`` (current default — backwards compatible
      with the historical single-agent flow),
    * an :class:`Ensemble` for a single stage with ``count > 1``,
    * a :class:`Pipeline` of stages otherwise (each entry being either
      a single agent or an :class:`Ensemble` when ``count > 1``).

    Adding a new hierarchy level is purely a configuration change — no
    code change here, in the wrapper, or in any agent class.
    """
    from algorithms.pipeline import Ensemble, Pipeline

    pipeline_cfg = config.get("pipeline") or []
    if not pipeline_cfg:
        message = build_unsupported_algorithm_message(None)
        logger.error(message)
        raise ValueError(message)

    from algorithms.runtime.randomness import RandomStreams, derived_seed, seeded

    def construct(agent_cls, view, stage_index, member_index):
        view = deepcopy(view)
        root_seed = int((config.get("training") or {}).get("seed", 22))
        seed = derived_seed(root_seed, "stage", stage_index, "member", member_index)
        view.setdefault("training", {})["seed"] = seed
        with seeded(seed):
            unit = agent_cls(config=view)
        unit.random_streams = RandomStreams(seed)
        unit.runtime_identity = {"stage": stage_index, "member": member_index, "seed": seed}
        return unit

    stages: List[ExecutionUnit] = []
    for stage_index, stage_cfg in enumerate(pipeline_cfg):
        algorithm_name = stage_cfg.get("algorithm")
        if not is_algorithm_supported(algorithm_name):
            message = build_unsupported_algorithm_message(algorithm_name)
            logger.error(message)
            raise ValueError(message)

        agent_cls = ALGORITHM_REGISTRY[algorithm_name]
        agent_view = _stage_to_agent_view(config, stage_cfg)
        count = int(stage_cfg.get("count", 1) or 1)
        if count < 1:
            raise ValueError(f"Stage '{algorithm_name}' has count={count}; must be >= 1.")
        if algorithm_name in {"AgentTransformerPPO", "AgentTransformerMATD3"} and count != 1:
            raise ValueError(f"{algorithm_name} pipeline stages require count=1")

        frozen = bool(stage_cfg.get("frozen", False))

        if count == 1:
            unit = construct(agent_cls, agent_view, stage_index, 0)
            unit.agent_name = algorithm_name
            if algorithm_name in AGENT_SPECS:
                unit.capabilities = AGENT_SPECS[algorithm_name].capabilities
            unit.frozen = frozen
            stages.append(unit)
        else:
            if bool(getattr(agent_cls, "single_agent_only", False)):
                members = [
                    construct(
                        agent_cls,
                        _single_agent_member_view(agent_view, member_index=member_index),
                        stage_index,
                        member_index,
                    )
                    for member_index in range(count)
                ]
            else:
                members = [
                    construct(agent_cls, agent_view, stage_index, member) for member in range(count)
                ]
            ensemble = Ensemble(members)
            for member in members:
                member.agent_name = algorithm_name
                if algorithm_name in AGENT_SPECS:
                    member.capabilities = AGENT_SPECS[algorithm_name].capabilities
            if algorithm_name in AGENT_SPECS:
                ensemble.capabilities = AGENT_SPECS[algorithm_name].capabilities
            ensemble.frozen = frozen
            stages.append(ensemble)

    if len(stages) == 1:
        return stages[0]
    return Pipeline(stages)
