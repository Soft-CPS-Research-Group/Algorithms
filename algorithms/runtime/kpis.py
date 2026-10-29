"""Versioned final-evaluation scorecard, separate from simulator diagnostic CSVs."""

import csv
import math
from importlib.metadata import PackageNotFoundError, version

from algorithms.runtime.atomic import atomic_json
from utils.training_provenance import file_digest

UNITS = {
    "kwh": "kWh",
    "kw": "kW",
    "eur": "EUR",
    "ratio": "1",
    "percent": "%",
    "hours": "h",
    "count": "count",
    "c": "degC",
    "kg": "kg",
    "kgco2": "kgCO2",
}


def baseline_reference(enabled):
    """Name the simulator counterfactual; it is not Algorithms' NormalPolicy."""
    if not enabled:
        return None
    try:
        simulator_version = version("softcpsrecsimulator")
    except PackageNotFoundError:
        simulator_version = None
    from citylearn.agents.baseline import BusinessAsUsualAgent

    definition = getattr(BusinessAsUsualAgent, "DEFINITION_VERSION", "deadline-spread-v1")
    immediate = definition == "immediate-full-charge-v1"
    return {
        "implementation": "citylearn.agents.baseline.BusinessAsUsualAgent",
        "package": "softcpsrecsimulator",
        "version": simulator_version,
        "definition": definition,
        "equivalent_to_normal_policy": False,
        "description": (
            "Simulator native counterfactual requests immediate EV charging to 100%. "
            "It shares the operational BAU objective with Algorithms NormalPolicy; "
            "the implementations and configurable limits are not identical."
        )
        if immediate
        else (
            "Simulator native counterfactual, with deadline-spread EV charging. "
            "Algorithms NormalPolicy instead requests immediate charging. "
            "Compare separate jobs with identical inputs for a NormalPolicy baseline."
        ),
    }


def _kpi_rows(path):
    if path.suffix == ".parquet":
        import pandas as pd

        frame = pd.read_parquet(path)
        if not len(frame.columns) or frame.columns[0] != "KPI":
            raise ValueError("Unrecognised simulator KPI columns")
        yield from frame.to_dict(orient="records")
    else:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or reader.fieldnames[0] != "KPI":
                raise ValueError("Unrecognised simulator KPI columns")
            yield from reader


def write_scorecard(context, wrapper):
    paths = context.paths
    export = context.config["simulator"]["export"]
    evaluated = getattr(wrapper, "evaluated_policy", None)
    candidates = sorted(
        path
        for suffix in ("csv", "parquet")
        for path in paths["simulation_data_dir"].rglob(f"exported_kpis.{suffix}")
    )
    requested = export.get("export_kpis_on_episode_end", True)
    reason = "not_requested" if not requested else "no_final_evaluation" if not evaluated else None
    if requested and evaluated and len(candidates) != 1:
        raise ValueError(f"Expected exactly one final KPI export, found {len(candidates)}")
    records, sources = [], []
    if reason is None:
        for path in candidates:
            sources.append(
                {"path": path.relative_to(paths["job_dir"]).as_posix(), "sha256": file_digest(path)}
            )
            for row in _kpi_rows(path):
                name = row.pop("KPI")
                for entity, raw in row.items():
                    scope = "community" if entity == "District" else "building"
                    if name.startswith("district_") != (scope == "community"):
                        continue
                    try:
                        value = float(raw)
                        valid = math.isfinite(value)
                        missing = None if valid else "simulator_nonfinite"
                    except (TypeError, ValueError):
                        value, valid, missing = None, False, "simulator_unavailable"
                    unit = UNITS.get(name.rsplit("_", 1)[-1])
                    records.append(
                        {
                            "name": name,
                            "entity": entity,
                            "level": scope,
                            "value": value if valid else None,
                            "valid": valid,
                            "missing_reason": missing,
                            "unit": unit,
                            "unit_source": "simulator_metric_suffix" if unit else "undeclared",
                        }
                    )
    result = {
        "schema_version": "algorithms.kpis.v1",
        "eligible_for_scorecard": reason is None,
        "missing_reason": reason,
        "evaluation": evaluated,
        "sources": sources,
        "baseline_calculation_enabled": export.get("include_business_as_usual", True),
        "baseline_reference": baseline_reference(export.get("include_business_as_usual", True)),
        "metrics": records,
    }
    atomic_json(paths["results_dir"] / "kpis.json", result)
    return result
