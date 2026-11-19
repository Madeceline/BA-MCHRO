"""Auditable, de-duplicated program--instance observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from typing import Any, Literal
from pathlib import Path

from ba_eoh_s.bayesian.features import FEATURE_SCHEMA_VERSION, instance_features

EvaluationSplit = Literal["training", "validation", "test"]
ObservationSource = Literal["real_execution", "cache_reuse", "posterior_prediction"]
_ALLOWED_SPLITS = {"training", "validation", "test"}
_ALLOWED_SOURCES = {"real_execution", "cache_reuse", "posterior_prediction"}
_FEASIBLE_STATUSES = {"ok", "feasible", "success"}


def _hash_instance(instance: Mapping[str, Any]) -> str:
    payload = json.dumps(instance, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _finite_or_none(value: Any, name: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite or None")
    return result


@dataclass(frozen=True, slots=True)
class EvaluationObservation:
    """One deduplicated observation, with explicit source and split labels.

    ``observed_performance`` is a loss (lower is better) and is intentionally
    ``None`` for an execution failure.  A failure remains usable for the
    separate feasibility model without being fabricated into a continuous loss.
    """

    program_id: str
    code_hash: str
    player: str
    generator_id: str | None
    generator_direction: str | None
    instance_hash: str
    instance_features: dict[str, float]
    seed: int
    sample_index: int | None
    evaluation_split: EvaluationSplit
    observed_performance: float | None
    solver_utility: float | None
    feasibility_status: str
    failure_reason: str | None
    reference_kind: str | None
    reference_value: float | None
    evaluation_time: float | None
    cache_hit: bool
    observation_source: ObservationSource
    configuration_signature: str = "default"
    feature_schema_version: str = FEATURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.program_id or not self.code_hash or not self.instance_hash:
            raise ValueError("program_id, code_hash and instance_hash are required")
        if self.evaluation_split not in _ALLOWED_SPLITS:
            raise ValueError(f"invalid evaluation_split: {self.evaluation_split}")
        if self.observation_source not in _ALLOWED_SOURCES:
            raise ValueError(f"invalid observation_source: {self.observation_source}")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if self.sample_index is not None and (type(self.sample_index) is not int or self.sample_index < 0):
            raise ValueError("sample_index must be a non-negative integer or None")
        if not self.feasibility_status:
            raise ValueError("feasibility_status cannot be empty")
        if not self.feature_schema_version:
            raise ValueError("feature_schema_version cannot be empty")
        if not isinstance(self.cache_hit, bool):
            raise ValueError("cache_hit must be bool")
        _finite_or_none(self.observed_performance, "observed_performance")
        _finite_or_none(self.solver_utility, "solver_utility")
        elapsed = _finite_or_none(self.evaluation_time, "evaluation_time")
        if elapsed is not None and elapsed < 0:
            raise ValueError("evaluation_time must be non-negative")
        if any(not math.isfinite(float(value)) for value in self.instance_features.values()):
            raise ValueError("instance_features must be finite")

    @property
    def feasible(self) -> bool:
        return self.feasibility_status in _FEASIBLE_STATUSES and self.observed_performance is not None

    @property
    def key(self) -> tuple[str, str, str, int, str]:
        """Stable uniqueness key for deterministic program--instance runs."""
        return (self.code_hash, self.instance_hash, self.evaluation_split,
                self.seed, self.configuration_signature)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "EvaluationObservation":
        return cls(**dict(row))


class ObservationStore:
    """Content-addressed observation ledger with cache-aware accounting.

    Adding the same deterministic run twice does not increase the effective
    sample count.  A later real execution may replace an earlier cache-only
    placeholder; conflicting real observations are rejected.
    """

    def __init__(self, observations: Iterable[EvaluationObservation] = ()) -> None:
        self._records: dict[tuple[str, str, str, int, str], EvaluationObservation] = {}
        self.extend(observations)

    def add(self, observation: EvaluationObservation | Mapping[str, Any]) -> bool:
        if not isinstance(observation, EvaluationObservation):
            observation = EvaluationObservation.from_dict(observation)
        previous = self._records.get(observation.key)
        if previous is None:
            self._records[observation.key] = observation
            return True
        if previous.observation_source != "real_execution" and observation.observation_source == "real_execution":
            self._records[observation.key] = observation
            return True
        if previous.observation_source == "real_execution" and observation.observation_source == "real_execution":
            if previous.to_dict() != observation.to_dict():
                raise ValueError(f"conflicting real observations for key={observation.key}")
        return False

    def extend(self, observations: Iterable[EvaluationObservation | Mapping[str, Any]]) -> int:
        return sum(self.add(observation) for observation in observations)

    def records(self, *, include_cache: bool = True) -> list[EvaluationObservation]:
        rows = list(self._records.values())
        if include_cache:
            return rows
        return [row for row in rows if row.observation_source == "real_execution"]

    def effective_records(self, *, split: str | None = None) -> list[EvaluationObservation]:
        rows = [row for row in self._records.values() if row.observation_source == "real_execution"]
        return [row for row in rows if split is None or row.evaluation_split == split]

    def by_program(self, program_id: str, *, effective_only: bool = True) -> list[EvaluationObservation]:
        rows = self.effective_records() if effective_only else self.records()
        return [row for row in rows if row.program_id == program_id or row.code_hash == program_id]

    def by_instance(self, instance_hash: str, *, effective_only: bool = True) -> list[EvaluationObservation]:
        rows = self.effective_records() if effective_only else self.records()
        return [row for row in rows if row.instance_hash == instance_hash]

    @property
    def actual_evaluation_count(self) -> int:
        return sum(row.observation_source == "real_execution" and not row.cache_hit
                   for row in self._records.values())

    @property
    def cache_hit_count(self) -> int:
        return sum(row.cache_hit or row.observation_source == "cache_reuse"
                   for row in self._records.values())

    @property
    def unique_count(self) -> int:
        return len(self._records)

    def to_list(self) -> list[dict[str, Any]]:
        return [row.to_dict() for row in self.records()]

    @classmethod
    def from_dicts(cls, rows: Iterable[Mapping[str, Any]]) -> "ObservationStore":
        return cls(EvaluationObservation.from_dict(row) for row in rows)

    def to_jsonl(self, path: str | Path) -> None:
        """Persist the deduplicated ledger without touching evaluator archives."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for row in self.to_list():
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "ObservationStore":
        rows = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(EvaluationObservation.from_dict(json.loads(line)))
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid observation JSONL at line {line_number}") from exc
        return cls(rows)

    def count_by_split(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self._records.values():
            counts[row.evaluation_split] = counts.get(row.evaluation_split, 0) + 1
        return counts

    def count_by_source(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self._records.values():
            counts[row.observation_source] = counts.get(row.observation_source, 0) + 1
        return counts

    def real_observations(self, *, split: str | None = None) -> list[EvaluationObservation]:
        return self.effective_records(split=split)


def _program_fields(program: Any) -> tuple[str, str, str]:
    if isinstance(program, str):
        return program, program, "solver"
    if isinstance(program, Mapping):
        program_id = str(program.get("id", program.get("program_id", program.get("code_hash", ""))))
        code_hash = str(program.get("code_hash", program_id))
        player = str(program.get("player", "solver"))
        return program_id, code_hash, player
    program_id = str(getattr(program, "id", getattr(program, "code_hash", "")))
    code_hash = str(getattr(program, "code_hash", program_id))
    player = str(getattr(program, "player", "solver"))
    return program_id, code_hash, player


def observation_from_evaluation(
    program: Any,
    instance: Any,
    result: Mapping[str, Any],
    *,
    evaluation_split: EvaluationSplit,
    seed: int,
    sample_index: int | None = None,
    generator_id: str | None = None,
    generator_direction: str | None = None,
    cache_hit: bool = False,
    observation_source: ObservationSource = "real_execution",
    configuration_signature: str = "default",
) -> EvaluationObservation:
    """Adapt an existing evaluator row without changing evaluator behavior."""

    program_id, code_hash, player = _program_fields(program)
    data = instance.to_dict() if hasattr(instance, "to_dict") else instance
    if not isinstance(data, Mapping):
        raise TypeError("instance must be a mapping or expose to_dict()")
    status = str(result.get("status", "unknown"))
    feasible = status in _FEASIBLE_STATUSES and result.get("feasible", True) is not False
    observed = result.get("gap") if feasible else None
    utility = result.get("solver_utility")
    failure = None if feasible else str(result.get("failure_type", result.get("error", status)))
    elapsed = result.get("wall_time", result.get("elapsed", result.get("solve_time")))
    return EvaluationObservation(
        program_id=program_id,
        code_hash=code_hash,
        player=player,
        generator_id=generator_id,
        generator_direction=generator_direction,
        instance_hash=_hash_instance(data),
        instance_features=instance_features(data),
        seed=seed,
        sample_index=sample_index,
        evaluation_split=evaluation_split,
        observed_performance=_finite_or_none(observed, "observed_performance"),
        solver_utility=_finite_or_none(utility, "solver_utility"),
        feasibility_status="feasible" if feasible else status,
        failure_reason=failure,
        reference_kind=result.get("reference_kind"),
        reference_value=_finite_or_none(result.get("reference_value"), "reference_value"),
        evaluation_time=_finite_or_none(elapsed, "evaluation_time"),
        cache_hit=cache_hit,
        observation_source=observation_source,
        configuration_signature=configuration_signature,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
    )
