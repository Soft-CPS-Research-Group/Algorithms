"""Shared community-market settlement math.

The equations mirror CityLearn's aggregate-building market settlement so
reward functions can optimise the same economic quantity later reported by
``community_settled_cost_total_eur``.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return float(default)
    return parsed


def allocate_weighted_import_share(
    imports: Sequence[float],
    traded_energy: float,
    weights: Optional[Sequence[float]] = None,
) -> List[float]:
    """Allocate local energy to importers using weights and demand caps."""

    normalized_imports = [max(_safe_float(value), 0.0) for value in imports]
    allocations = [0.0 for _ in normalized_imports]
    remaining = max(_safe_float(traded_energy), 0.0)
    normalized_weights = (
        [1.0 for _ in normalized_imports]
        if weights is None
        else [max(_safe_float(value), 0.0) for value in weights]
    )
    if len(normalized_weights) != len(normalized_imports):
        raise ValueError("Community settlement weights must align with member imports")

    eps = 1.0e-9
    while remaining > eps:
        needs = [
            max(imported - allocated, 0.0)
            for imported, allocated in zip(normalized_imports, allocations)
        ]
        active_indexes = [index for index, need in enumerate(needs) if need > eps]
        if not active_indexes:
            break

        active_weight_sum = sum(normalized_weights[index] for index in active_indexes)
        granted = [0.0 for _ in normalized_imports]
        for index in active_indexes:
            share = (
                remaining / float(len(active_indexes))
                if active_weight_sum <= eps
                else remaining * (normalized_weights[index] / active_weight_sum)
            )
            granted[index] = min(share, needs[index])

        granted_total = sum(granted)
        if granted_total <= eps:
            break
        allocations = [
            allocated + grant for allocated, grant in zip(allocations, granted)
        ]
        remaining -= granted_total

    return allocations


def community_settlement_components(
    observations: Sequence[Mapping[str, Any]],
    *,
    local_price_ratio: float = 0.8,
    grid_export_price: float = 0.0,
    import_member_weights: Optional[Sequence[float]] = None,
) -> tuple[List[Mapping[str, float]], Mapping[str, float]]:
    """Return member rows and district totals for one settled time step."""

    ratio = min(max(_safe_float(local_price_ratio, 0.8), 0.0), 1.0)
    export_price = max(_safe_float(grid_export_price, 0.0), 0.0)
    net_values = [
        _safe_float(observation.get("net_electricity_consumption"), 0.0)
        for observation in observations
    ]
    imports = [max(value, 0.0) for value in net_values]
    exports = [max(-value, 0.0) for value in net_values]
    prices = [
        max(_safe_float(observation.get("electricity_pricing"), 0.0), 0.0)
        for observation in observations
    ]

    total_import = sum(imports)
    total_export = sum(exports)
    traded_energy = min(total_import, total_export)
    local_imports = (
        allocate_weighted_import_share(
            imports,
            traded_energy,
            import_member_weights,
        )
        if total_import > 0.0 and traded_energy > 0.0
        else [0.0 for _ in imports]
    )
    local_exports = (
        [exported * (traded_energy / total_export) for exported in exports]
        if total_export > 0.0 and traded_energy > 0.0
        else [0.0 for _ in exports]
    )

    rows: List[Mapping[str, float]] = []
    total_cost = 0.0
    total_grid_import = 0.0
    total_grid_export = 0.0
    for imported, exported, price, local_import, local_export in zip(
        imports,
        exports,
        prices,
        local_imports,
        local_exports,
    ):
        local_price = ratio * price
        grid_import = max(imported - local_import, 0.0)
        grid_export = max(exported - local_export, 0.0)
        settlement_cost = (
            grid_import * price
            + local_import * local_price
            - local_export * local_price
            - grid_export * export_price
        )
        total_cost += settlement_cost
        total_grid_import += grid_import
        total_grid_export += grid_export
        rows.append(
            {
                "community_settlement_cost": settlement_cost,
                "community_local_import_energy": local_import,
                "community_local_export_energy": local_export,
                "community_grid_import_energy": grid_import,
                "community_grid_export_energy": grid_export,
                "community_local_price": local_price,
            }
        )

    totals = {
        "community_settlement_cost_total": total_cost,
        "community_local_traded_energy": traded_energy,
        "community_grid_import_after_settlement": total_grid_import,
        "community_grid_export_after_settlement": total_grid_export,
        "community_local_price_ratio": ratio,
        "community_grid_export_price": export_price,
    }
    return rows, totals


__all__ = [
    "allocate_weighted_import_share",
    "community_settlement_components",
]
