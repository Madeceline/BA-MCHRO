"""Deterministic, leakage-free features for static and time-expanded SND data."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


# The order is part of the model schema.  Do not reorder without bumping the
# schema version written alongside a fitted model.
FEATURE_NAMES = (
    "physical_nodes",
    "expanded_nodes",
    "arcs",
    "arc_density",
    "commodities",
    "total_demand",
    "mean_demand",
    "max_demand",
    "demand_cv",
    "total_capacity",
    "capacity_to_demand",
    "min_capacity_to_demand",
    "mean_variable_cost",
    "mean_fixed_cost",
    "fixed_cost_ratio",
    "time_periods",
    "service_arc_count",
    "wait_arc_count",
)

FEATURE_SCHEMA_VERSION = "snd-instance-features-v1"


def _as_data(instance: Any) -> Mapping[str, Any]:
    if hasattr(instance, "to_dict"):
        instance = instance.to_dict()
    if not isinstance(instance, Mapping):
        raise TypeError("instance must be a mapping or expose to_dict()")
    return instance


def _safe(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("instance feature is not finite")
    return value


def instance_features(instance: Any) -> dict[str, float]:
    """Return fixed-schema numeric features without using performance labels.

    Only the instance itself is inspected.  In particular, reference values,
    solver outputs and evaluation status are never read, so these features can
    safely be used before a program is executed.
    """

    data = _as_data(instance)
    nodes = list(data.get("nodes", ()))
    arcs = list(data.get("arcs", ()))
    commodities = list(data.get("commodities", ()))
    if not nodes or not arcs or not commodities:
        raise ValueError("instance must contain non-empty nodes, arcs and commodities")

    metadata = data.get("metadata") or {}
    expansion = data.get("time_expansion") or {}
    physical_nodes = int(metadata.get("physical_node_count", len(expansion.get("physical_nodes", ()))
                          or len(nodes)))
    expanded_nodes = len(nodes)
    node_denominator = max(physical_nodes * max(physical_nodes - 1, 1), 1)
    arc_density = len(expansion.get("physical_arcs", ())) / node_denominator if expansion else len(arcs) / node_denominator

    demands = [_safe(item["demand"]) for item in commodities]
    capacities = [_safe(item["capacity"]) for item in arcs]
    variable_costs = [_safe(item["variable_cost"]) for item in arcs]
    fixed_costs = [_safe(item["fixed_cost"]) for item in arcs]
    total_demand = sum(demands)
    mean_demand = total_demand / len(demands)
    mean_capacity = sum(capacities) / len(capacities)
    demand_variance = sum((value - mean_demand) ** 2 for value in demands) / len(demands)
    total_capacity = sum(capacities)
    total_fixed = sum(fixed_costs)
    total_variable = sum(variable_costs)
    service_arc_count = len(arcs)
    wait_arc_count = 0
    periods = int(metadata.get("time_periods", 0) or 0)
    if expansion:
        schedule = expansion.get("arc_schedule", ())
        wait_arc_count = sum(item.get("kind") == "wait" for item in schedule)
        service_arc_count = len(arcs) - wait_arc_count
        periods = int(metadata.get("time_periods", max(len(expansion.get("time_points", ())) - 1, 0)))

    features = {
        "physical_nodes": float(physical_nodes),
        "expanded_nodes": float(expanded_nodes),
        "arcs": float(len(arcs)),
        "arc_density": float(arc_density),
        "commodities": float(len(commodities)),
        "total_demand": float(total_demand),
        "mean_demand": float(mean_demand),
        "max_demand": float(max(demands)),
        "demand_cv": float(math.sqrt(demand_variance) / mean_demand if mean_demand else 0.0),
        "total_capacity": float(total_capacity),
        "capacity_to_demand": float(total_capacity / total_demand if total_demand else 0.0),
        "min_capacity_to_demand": float(min(capacities) / mean_demand if mean_demand else 0.0),
        "mean_variable_cost": float(sum(variable_costs) / len(variable_costs)),
        "mean_fixed_cost": float(sum(fixed_costs) / len(fixed_costs)),
        "fixed_cost_ratio": float(total_fixed / (total_fixed + total_variable)
                                   if total_fixed + total_variable else 0.0),
        "time_periods": float(periods),
        "service_arc_count": float(service_arc_count),
        "wait_arc_count": float(wait_arc_count),
    }
    if tuple(features) != FEATURE_NAMES:
        raise RuntimeError("feature schema order changed unexpectedly")
    return {key: _safe(value) for key, value in features.items()}
