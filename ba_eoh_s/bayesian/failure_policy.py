"""Training-only reliability gates and explicitly labelled failure penalties.

These scores are selection decisions, never successful observations or game
payoffs. Deployment continues to use the original independent checks.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from ba_eoh_s.bayesian.posterior import JointLossSamples, target_weights


def failure_kind(reason: str | None) -> str:
    value = str(reason or "unknown").lower()
    if "timeout" in value or "timed out" in value or "超时" in value:
        return "timeout"
    if "syntax" in value:
        return "syntax"
    if "code_limit" in value or "源码超过" in value:
        return "code_limit"
    if "line_limit" in value or "行数预算" in value:
        return "line_limit"
    if "invalid_solution" in value or "infeasible" in value or "不可行" in value:
        return "invalid_solution"
    if "runtime" in value or "nameerror" in value or "typeerror" in value:
        return "runtime"
    return "evaluation_error"


def selection_evidence(model: Any, program: Any, instances: Sequence[Any],
                       weights: Sequence[float] | None = None) -> dict:
    """Use observed distinct instances from the current weighted target.

    Historical records remain in the model/ledger. Explicit hashes identify
    current representatives; feature-only offline targets use all real history
    with equal weights because their generator provenance is unavailable.
    """
    importance = target_weights(weights, len(instances))
    hashes = [x.get("instance_hash") if isinstance(x, Mapping) else None for x in instances]
    target = ({str(h): float(w) for h, w in zip(hashes, importance)}
              if all(hashes) and len(set(hashes)) == len(hashes) else None)
    return model.selection_evidence(program, instance_weights=target)


def rejection_reason(evidence: Mapping, config: Any) -> str | None:
    successes = evidence["successful_observations"]
    failures = evidence["failed_observations"]
    count = evidence.get("distinct_instances", successes + failures)
    kinds = evidence.get("failure_types", {})
    if kinds.get("syntax", 0) or kinds.get("code_limit", 0):
        return "invalid_source"
    if failures:
        if count < config.failure_rate_min_instances:
            return "insufficient_reliability_observations"
        rate = evidence.get("weighted_failure_rate", failures / count)
        if rate > config.max_failure_rate + 1e-12:
            return "training_failure_rate_exceeded"
    if successes < config.min_observations:
        return "insufficient_real_observations"
    return None


def penalize_failures(bank: JointLossSamples, evidence: Mapping[str, Mapping],
                      margin: float) -> JointLossSamples:
    """Overlay KNOWN failed cells, using the original CPM severity rule.

    Successful values/predictions stay unchanged. A failed cell has no loss
    observation; its decision score is worst nonfailed loss + severity*margin.
    This does not fit a new feasibility model or invent successful evaluations.
    """
    failures = {pid: row.get("failure_instances", {}) for pid, row in evidence.items()}
    if not any(failures.values()):
        return bank
    values = bank.losses.copy()
    for j, instance_hash in enumerate(bank.instance_hashes):
        bad = [i for i, pid in enumerate(bank.program_ids) if instance_hash in failures.get(pid, {})]
        if not bad:
            continue
        good = [i for i in range(len(bank.program_ids)) if i not in bad]
        worst = values[:, good, j].max(axis=1) if good else np.zeros(len(values))
        for i in bad:
            kind = failures[bank.program_ids[i]][instance_hash]
            severity = {"line_limit": 1., "timeout": 2.}.get(kind, 3.)
            values[:, i, j] = worst + severity * margin
    return replace(bank, losses=values)
