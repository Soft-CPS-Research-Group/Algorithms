"""Price-signal adapter for community-blind building-local policies.

The adapter changes only the price coordinates in an already min-max encoded
observation.  In particular, it does not append coordinator state and it
rejects observation layouts that contain community features.  This keeps a
local policy's observation dimension and semantics stable when a coordinator
communicates through a virtual retail price.

Price coordinates use the raw feature bounds supplied by the environment::

    raw = low + encoded * (high - low)
    virtual = raw * multiplier
    encoded_virtual = clip((virtual - low) / (high - low), 0, 1)

The neutral context is handled as an explicit no-op.  The returned array is a
copy (the caller's array is never mutated), but its values and dtype are
bitwise-identical to the input.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


CURRENT_PRICE_NAME = "district__electricity_pricing"
PREDICTED_PRICE_NAMES = (
    "district__electricity_pricing_predicted_1",
    "district__electricity_pricing_predicted_2",
    "district__electricity_pricing_predicted_3",
)
PRICE_NAMES = (CURRENT_PRICE_NAME, *PREDICTED_PRICE_NAMES)

CURRENT_CONTEXT_KEY = "current"
FORECAST_CONTEXT_KEYS = ("forecast_6h", "forecast_12h", "forecast_24h")
PRICE_CONTEXT_KEYS = (CURRENT_CONTEXT_KEY, *FORECAST_CONTEXT_KEYS)


class ForecastMode(str, Enum):
    """How the local policy's three price forecasts are virtualized."""

    REAL_UNMODIFIED = "real_unmodified"
    ALIGNED_VECTOR = "aligned_vector"
    PERSIST_CURRENT = "persist_current"


@dataclass(frozen=True)
class PriceMultiplierContext:
    """Validated coordinator signal for one local observation.

    ``forecast_*`` values are required only by ``aligned_vector``.  Use
    :meth:`from_mapping` instead of constructing the dataclass directly when
    consuming an external/pipeline payload; it rejects missing, unused and
    misspelled fields.
    """

    current: float
    forecast_6h: float | None = None
    forecast_12h: float | None = None
    forecast_24h: float | None = None

    @classmethod
    def from_mapping(
        cls,
        context: Mapping[str, Any],
        *,
        forecast_mode: ForecastMode | str,
    ) -> "PriceMultiplierContext":
        if not isinstance(context, Mapping):
            raise TypeError("Price multiplier context must be a mapping.")

        mode = _parse_forecast_mode(forecast_mode)
        required = (
            set(PRICE_CONTEXT_KEYS)
            if mode is ForecastMode.ALIGNED_VECTOR
            else {CURRENT_CONTEXT_KEY}
        )
        supplied = set(context)
        missing = required - supplied
        extra = supplied - required
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append(f"missing keys {sorted(missing, key=str)}")
            if extra:
                details.append(f"unexpected keys {sorted(extra, key=str)}")
            raise ValueError(
                f"Invalid {mode.value} price multiplier context: "
                + "; ".join(details)
                + "."
            )

        values = {key: _validate_multiplier(context[key], key) for key in required}
        return cls(
            current=values[CURRENT_CONTEXT_KEY],
            forecast_6h=values.get("forecast_6h"),
            forecast_12h=values.get("forecast_12h"),
            forecast_24h=values.get("forecast_24h"),
        )

    def multipliers(self, forecast_mode: ForecastMode | str) -> Mapping[str, float]:
        """Return one multiplier for each discovered price feature."""

        mode = _parse_forecast_mode(forecast_mode)
        current = _validate_multiplier(self.current, CURRENT_CONTEXT_KEY)
        forecast_values = (
            self.forecast_6h,
            self.forecast_12h,
            self.forecast_24h,
        )
        if mode is not ForecastMode.ALIGNED_VECTOR and any(
            value is not None for value in forecast_values
        ):
            raise ValueError(
                f"{mode.value} accepts only the current multiplier; forecast multipliers "
                "would be unused."
            )
        if mode is ForecastMode.REAL_UNMODIFIED:
            values = (current, 1.0, 1.0, 1.0)
        elif mode is ForecastMode.PERSIST_CURRENT:
            values = (current, current, current, current)
        else:
            if any(value is None for value in forecast_values):
                raise ValueError(
                    "aligned_vector requires current, forecast_6h, forecast_12h and "
                    "forecast_24h multipliers."
                )
            values = (
                current,
                *(
                    _validate_multiplier(value, key)
                    for key, value in zip(FORECAST_CONTEXT_KEYS, forecast_values)
                ),
            )
        return MappingProxyType(dict(zip(PRICE_NAMES, values)))


@dataclass(frozen=True)
class PriceMultiplierDiagnostics:
    """Auditable real-to-virtual price transformation details."""

    forecast_mode: str
    multiplier_by_feature: Mapping[str, float]
    real_price_by_feature: Mapping[str, float]
    virtual_price_unclipped_by_feature: Mapping[str, float]
    virtual_price_by_feature: Mapping[str, float]
    encoded_price_by_feature: Mapping[str, float]
    clipped_features: tuple[str, ...]
    clipping_count: int
    neutral_noop: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "forecast_mode": self.forecast_mode,
            "multiplier_by_feature": dict(self.multiplier_by_feature),
            "real_price_by_feature": dict(self.real_price_by_feature),
            "virtual_price_unclipped_by_feature": dict(
                self.virtual_price_unclipped_by_feature
            ),
            "virtual_price_by_feature": dict(self.virtual_price_by_feature),
            "encoded_price_by_feature": dict(self.encoded_price_by_feature),
            "clipped_features": list(self.clipped_features),
            "clipping_count": self.clipping_count,
            "neutral_noop": self.neutral_noop,
        }


class PriceMultiplierObservationAdapter:
    """Apply a virtual price context to one actor observation.

    Strict-local validation remains the default used by PPO/TD3 leaves.  A
    centralized-training multi-agent policy may already have community
    coordinates in its established observation contract; setting
    ``require_strict_local=False`` preserves those coordinates while still
    changing only the four exact price features.
    """

    def __init__(
        self,
        *,
        observation_names: Sequence[str],
        feature_low: Sequence[float] | Mapping[str, float],
        feature_high: Sequence[float] | Mapping[str, float],
        forecast_mode: ForecastMode | str = ForecastMode.REAL_UNMODIFIED,
        require_strict_local: bool = True,
    ) -> None:
        self.observation_names = tuple(str(name) for name in observation_names)
        if not self.observation_names:
            raise ValueError("observation_names must not be empty.")
        self.forecast_mode = _parse_forecast_mode(forecast_mode)
        self.require_strict_local = bool(require_strict_local)
        if self.require_strict_local:
            self._validate_strict_local_names()
        self.price_indices = self._discover_price_indices()
        self._low = self._resolve_bounds(feature_low, "feature_low")
        self._high = self._resolve_bounds(feature_high, "feature_high")
        self._validate_price_bounds()

    def transform(
        self,
        observation: Sequence[float] | np.ndarray,
        context: PriceMultiplierContext | Mapping[str, Any],
    ) -> tuple[np.ndarray, PriceMultiplierDiagnostics]:
        """Return a transformed copy and diagnostics; never mutate ``observation``."""

        values = np.asarray(observation)
        if values.ndim != 1 or values.size != len(self.observation_names):
            raise ValueError(
                "Observation must be a 1-D vector aligned with observation_names: "
                f"expected {len(self.observation_names)}, got shape {values.shape}."
            )
        if not np.issubdtype(values.dtype, np.floating):
            raise TypeError("Price multiplier observations must use a floating dtype.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Price multiplier observations must be finite.")

        parsed_context = self._parse_context(context)
        multipliers = parsed_context.multipliers(self.forecast_mode)
        output = values.copy()

        real_prices: dict[str, float] = {}
        virtual_unclipped: dict[str, float] = {}
        virtual_prices: dict[str, float] = {}
        encoded_prices: dict[str, float] = {}
        clipped: list[str] = []

        for name in PRICE_NAMES:
            index = self.price_indices[name]
            encoded = float(values[index])
            if not np.isfinite(encoded):
                raise ValueError(f"Encoded price feature {name!r} must be finite.")
            if encoded < 0.0 or encoded > 1.0:
                raise ValueError(
                    f"Encoded price feature {name!r} must be inside [0, 1]; got {encoded}."
                )
            low = float(self._low[index])
            high = float(self._high[index])
            real = low + encoded * (high - low)
            candidate = real * multipliers[name]
            virtual = float(np.clip(candidate, low, high))
            encoded_virtual = float(np.clip((virtual - low) / (high - low), 0.0, 1.0))

            real_prices[name] = real
            virtual_unclipped[name] = candidate
            virtual_prices[name] = virtual
            encoded_prices[name] = encoded if multipliers[name] == 1.0 else encoded_virtual
            if candidate < low or candidate > high:
                clipped.append(name)

        neutral = all(multiplier == 1.0 for multiplier in multipliers.values())
        if not neutral:
            for name, index in self.price_indices.items():
                # Do not rewrite unmodified forecast coordinates.  Besides
                # avoiding rounding, this makes real_unmodified semantically
                # exact even for a lower precision observation dtype.
                if multipliers[name] != 1.0:
                    output[index] = encoded_prices[name]

        diagnostics = PriceMultiplierDiagnostics(
            forecast_mode=self.forecast_mode.value,
            multiplier_by_feature=_read_only_mapping(multipliers),
            real_price_by_feature=_read_only_mapping(real_prices),
            virtual_price_unclipped_by_feature=_read_only_mapping(virtual_unclipped),
            virtual_price_by_feature=_read_only_mapping(virtual_prices),
            encoded_price_by_feature=_read_only_mapping(encoded_prices),
            clipped_features=tuple(clipped),
            clipping_count=len(clipped),
            neutral_noop=neutral,
        )
        return output, diagnostics

    def _parse_context(
        self, context: PriceMultiplierContext | Mapping[str, Any]
    ) -> PriceMultiplierContext:
        if isinstance(context, PriceMultiplierContext):
            # Calling multipliers performs mode-specific validation even for a
            # directly constructed dataclass.
            context.multipliers(self.forecast_mode)
            return context
        return PriceMultiplierContext.from_mapping(
            context, forecast_mode=self.forecast_mode
        )

    def _validate_strict_local_names(self) -> None:
        leaked = sorted(
            name for name in self.observation_names if "community" in name.casefold()
        )
        if leaked:
            raise ValueError(
                "Strict-local price adaptation rejects community observation features: "
                f"{leaked}."
            )

    def _discover_price_indices(self) -> dict[str, int]:
        indices: dict[str, int] = {}
        for name in PRICE_NAMES:
            matches = [
                index
                for index, candidate in enumerate(self.observation_names)
                if candidate == name
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Strict-local price layout requires each exact price feature once; "
                    f"{name!r} occurred {len(matches)} times."
                )
            indices[name] = matches[0]
        return indices

    def _resolve_bounds(
        self,
        bounds: Sequence[float] | Mapping[str, float],
        label: str,
    ) -> np.ndarray:
        if isinstance(bounds, Mapping):
            missing = [name for name in PRICE_NAMES if name not in bounds]
            if missing:
                raise ValueError(f"{label} is missing price features {missing}.")
            result = np.full(len(self.observation_names), np.nan, dtype=np.float64)
            for name, index in self.price_indices.items():
                result[index] = _finite_float(bounds[name], f"{label}[{name!r}]")
            return result

        result = np.asarray(bounds, dtype=np.float64)
        if result.ndim != 1 or result.size != len(self.observation_names):
            raise ValueError(
                f"{label} must align with observation_names: expected "
                f"{len(self.observation_names)}, got shape {result.shape}."
            )
        return result.copy()

    def _validate_price_bounds(self) -> None:
        for name, index in self.price_indices.items():
            low = float(self._low[index])
            high = float(self._high[index])
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                raise ValueError(
                    f"Price feature {name!r} requires finite affine bounds with high > low; "
                    f"got low={low}, high={high}."
                )


def _parse_forecast_mode(value: ForecastMode | str) -> ForecastMode:
    try:
        return value if isinstance(value, ForecastMode) else ForecastMode(str(value))
    except ValueError as exc:
        choices = ", ".join(mode.value for mode in ForecastMode)
        raise ValueError(f"Unknown forecast_mode {value!r}; expected one of: {choices}.") from exc


def _validate_multiplier(value: Any, key: str) -> float:
    parsed = _finite_float(value, f"price multiplier {key!r}")
    if parsed < 0.0:
        raise ValueError(f"Price multiplier {key!r} must be non-negative.")
    return parsed


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a finite number, not a boolean.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number.") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{label} must be a finite number.")
    return parsed


def _read_only_mapping(values: Mapping[str, float]) -> Mapping[str, float]:
    return MappingProxyType({str(key): float(value) for key, value in values.items()})


def normalize_price_multiplier_context(context: Any) -> Mapping[str, Any] | None:
    """Accept the scalar signal emitted by current CC stages or a rich mapping."""

    if context is None:
        return None
    if isinstance(context, Mapping):
        return context
    if isinstance(context, np.ndarray) and context.ndim == 0:
        context = context.item()
    if isinstance(context, (int, float, np.integer, np.floating)) and not isinstance(
        context,
        (bool, np.bool_),
    ):
        return {CURRENT_CONTEXT_KEY: _validate_multiplier(context, CURRENT_CONTEXT_KEY)}
    raise TypeError(
        "Local price context must be a scalar multiplier, a structured mapping, or None."
    )


def normalize_price_multiplier_contexts(
    context: Any,
    *,
    num_agents: int,
) -> list[Mapping[str, Any] | None]:
    """Broadcast a CC-L1 scalar or distribute a CC-L2 per-agent vector.

    A scalar/structured mapping is broadcast to every actor.  A one-dimensional
    sequence must contain exactly one scalar or structured mapping per actor.
    This explicit shape check prevents a malformed CC-L2 vector from silently
    becoming a shared context.
    """

    count = int(num_agents)
    if count < 1:
        raise ValueError("num_agents must be >= 1 for local price conditioning.")
    if context is None:
        return [None for _ in range(count)]

    if isinstance(context, np.ndarray):
        if context.ndim == 0:
            context = context.item()
        elif context.ndim == 1:
            values = context.tolist()
            if len(values) != count:
                raise ValueError(
                    "Per-agent price multiplier vector length must match num_agents: "
                    f"expected {count}, got {len(values)}."
                )
            return [normalize_price_multiplier_context(value) for value in values]
        else:
            raise ValueError(
                "Per-agent price multiplier context must be one-dimensional."
            )

    if isinstance(context, (list, tuple)):
        values = list(context)
        if len(values) != count:
            raise ValueError(
                "Per-agent price multiplier vector length must match num_agents: "
                f"expected {count}, got {len(values)}."
            )
        return [normalize_price_multiplier_context(value) for value in values]

    shared = normalize_price_multiplier_context(context)
    return [shared for _ in range(count)]


def price_feature_bounds_from_metadata(
    *,
    metadata: Mapping[str, Any] | None,
    agent_index: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Resolve raw affine price bounds attached by the CityLearn wrapper."""

    metadata = dict(metadata or {})
    raw_names_groups = metadata.get("raw_observation_names")
    raw_bounds_groups = metadata.get("raw_observation_bounds")
    if not isinstance(raw_names_groups, (list, tuple)) or not isinstance(
        raw_bounds_groups,
        (list, tuple),
    ):
        raise ValueError(
            "Local price conditioning requires raw_observation_names and "
            "raw_observation_bounds metadata."
        )
    if agent_index < 0 or agent_index >= len(raw_names_groups) or agent_index >= len(
        raw_bounds_groups
    ):
        raise ValueError(f"Missing price-bound metadata for agent index {agent_index}.")

    names = [str(name) for name in raw_names_groups[agent_index]]
    bounds = raw_bounds_groups[agent_index]
    if not isinstance(bounds, Mapping):
        raise ValueError("Each raw_observation_bounds entry must be a mapping.")
    low = np.asarray(bounds.get("low"), dtype=np.float64).reshape(-1)
    high = np.asarray(bounds.get("high"), dtype=np.float64).reshape(-1)
    if low.size != len(names) or high.size != len(names):
        raise ValueError("Raw observation bounds must align with raw observation names.")
    indices = {name: index for index, name in enumerate(names)}
    missing = [name for name in PRICE_NAMES if name not in indices]
    if missing:
        raise ValueError(f"Raw observation metadata is missing price features {missing}.")
    return (
        {name: float(low[indices[name]]) for name in PRICE_NAMES},
        {name: float(high[indices[name]]) for name in PRICE_NAMES},
    )


def price_observation_names_from_metadata(
    *,
    metadata: Mapping[str, Any] | None,
    agent_index: int,
    fallback_observation_names: Sequence[str],
) -> list[str]:
    """Resolve the names of the vector that is actually passed to the actor.

    Entity-mode agents are attached with the raw CityLearn layout because
    safety/teacher components need it, while their actors consume the encoded
    profile.  Price conditioning must therefore validate and index the encoded
    view, not the simulator's superset of raw community telemetry.
    """

    metadata = dict(metadata or {})
    encoded_groups = metadata.get("encoded_observation_names")
    if isinstance(encoded_groups, (list, tuple)):
        if agent_index < 0 or agent_index >= len(encoded_groups):
            raise ValueError(
                f"Missing encoded observation-name metadata for agent index {agent_index}."
            )
        names = encoded_groups[agent_index]
        if not isinstance(names, (list, tuple)):
            raise ValueError("Each encoded_observation_names entry must be a sequence.")
        return [str(name) for name in names]

    return [str(name) for name in fallback_observation_names]


__all__ = [
    "CURRENT_CONTEXT_KEY",
    "CURRENT_PRICE_NAME",
    "FORECAST_CONTEXT_KEYS",
    "ForecastMode",
    "PREDICTED_PRICE_NAMES",
    "PRICE_CONTEXT_KEYS",
    "PRICE_NAMES",
    "PriceMultiplierContext",
    "PriceMultiplierDiagnostics",
    "PriceMultiplierObservationAdapter",
    "normalize_price_multiplier_context",
    "normalize_price_multiplier_contexts",
    "price_feature_bounds_from_metadata",
    "price_observation_names_from_metadata",
]
