"""Reliability gate, paired follow-ups and failure-penalty provenance."""
from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from ba_eoh_s.bayesian import (ActiveEvaluationConfig, BayesianActiveEvaluator,
    BayesianCPMConfig, BayesianPerformanceModel, EvaluationObservation, EvaluationOutcome,
    ObservationStore, bayesian_cpm_select)
from ba_eoh_s.bayesian.failure_policy import (failure_kind, penalize_failures,
                                                    rejection_reason, selection_evidence)
from m_asro.problems.snd.evaluation import digest
from tests.test_active_evaluation import pool
from tests.test_bayesian import observation
from tests import test_bayesian_integration as integration


def representatives(n):
    return [dict(instance_hash=f"instance-{i}", instance_features={"x": float(i)}) for i in range(n)]


def fitted(n, failed, *, program="p", kind="timeout"):
    rows = []
    for i in range(n):
        row = observation(program, i, None if i in failed else .1,
                          status="solver_failure" if i in failed else "feasible")
        if i in failed:
            row["failure_reason"] = kind
        rows.append(row)
    return BayesianPerformanceModel(feature_names=("x",)).fit(rows), rows


class FollowupBackend:
    configuration_signature = "default"

    def __init__(self):
        self.cache, self.calls = {}, []

    def peek(self, program, instance):
        row = self.cache.get((program, instance.instance_hash))
        return replace(row, cache_hit=True) if row else None

    def evaluate(self, program, instance):
        cached = self.peek(program, instance)
        if cached:
            return EvaluationOutcome(cached, False, True)
        self.calls.append((program, instance.instance_hash))
        i = instance.sample_index
        fail = program == "p" and i == 0
        row = EvaluationObservation.from_dict(observation(program, i, None if fail else .1,
                                             status="timeout" if fail else "feasible"))
        row = replace(row, seed=instance.seed, sample_index=i, instance_features=instance.features)
        self.cache[(program, instance.instance_hash)] = row
        return EvaluationOutcome(row, True, False)


class TestFailurePolicy(unittest.TestCase):
    def test_107_successes_two_timeouts_remain_eligible(self):
        model, _ = fitted(109, {0, 1})
        selected, log = bayesian_cpm_select(["p"], model, representatives(109), 1,
                                            config=BayesianCPMConfig(n_draws=16))
        self.assertEqual(selected, ["p"])
        evidence = log["candidate_evidence"]["p"]
        self.assertEqual((evidence["successful_observations"], evidence["failed_observations"]), (107, 2))
        self.assertAlmostEqual(evidence["weighted_failure_rate"], 2/109)
        self.assertEqual(evidence["failure_types"], {"timeout": 2})
        self.assertTrue(np.isfinite(log["expected_cpi"]))
        self.assertEqual(model.posterior_parameters("p")["observation_count"], 107)

    def test_early_failure_is_pending_not_final_reliability_rejection(self):
        model, _ = fitted(2, {1})
        evidence = selection_evidence(model, "p", representatives(2))
        self.assertEqual(rejection_reason(evidence, BayesianCPMConfig()),
                         "insufficient_reliability_observations")
        model, _ = fitted(10, {1})
        evidence = selection_evidence(model, "p", representatives(10))
        self.assertEqual(rejection_reason(evidence, BayesianCPMConfig()), "training_failure_rate_exceeded")

    def test_threshold_and_minimum_are_configurable_with_inclusive_boundary(self):
        model, _ = fitted(20, {1})
        evidence = selection_evidence(model, "p", representatives(20))
        self.assertIsNone(rejection_reason(evidence, BayesianCPMConfig()))
        self.assertEqual(rejection_reason(evidence, BayesianCPMConfig(max_failure_rate=.04)),
                         "training_failure_rate_exceeded")
        model, _ = fitted(2, {1})
        self.assertIsNone(rejection_reason(selection_evidence(model, "p", representatives(2)),
            BayesianCPMConfig(min_observations=1, failure_rate_min_instances=2, max_failure_rate=.5)))
        for kwargs in (dict(failure_rate_min_instances=0), dict(max_failure_rate=-.1),
                       dict(max_failure_rate=float("nan")), dict(failure_margin=0)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BayesianCPMConfig(**kwargs)

    def test_weighted_rate_uses_current_target_and_keeps_history(self):
        model, rows = fitted(20, {0})
        items = representatives(20)
        weighted = selection_evidence(model, "p", items, [10] + [1]*19)
        self.assertAlmostEqual(weighted["weighted_failure_rate"], 10/29)
        self.assertEqual(rejection_reason(weighted, BayesianCPMConfig()), "training_failure_rate_exceeded")
        restricted = selection_evidence(model, "p", items[1:])
        self.assertEqual(restricted["failed_observations"], 0)
        self.assertEqual(model.selection_evidence("p")["failed_observations"], 1)
        self.assertEqual(len(model._records), len(rows))

    def test_same_instance_cache_seeds_configs_do_not_inflate_trials(self):
        model, rows = fitted(10, {0})
        duplicates = [dict(row, seed=99, cache_hit=True) for row in rows]
        duplicates += [dict(row, configuration_signature="second") for row in rows]
        duplicates += [observation("p", 500, None, status="timeout", split=split)
                       for split in ("validation", "test")]
        duplicates += [observation("p", 999, 0, source="posterior_prediction")]
        model.update(duplicates)
        evidence = model.selection_evidence("p")
        self.assertEqual(evidence["distinct_instances"], 10)
        self.assertEqual(evidence["failed_observations"], 1)
        self.assertAlmostEqual(evidence["weighted_failure_rate"], .1)

    def test_failed_cells_get_separate_penalty_not_predicted_success(self):
        for kind, expected in (("timeout", 2.), ("runtime", 3.), ("invalid_solution", 3.)):
            model, _ = fitted(20, {0}, kind=kind)
            items = representatives(20)
            bank = model.joint_posterior_predictive(["p"], items, 8)
            evidence = selection_evidence(model, "p", items)
            scored = penalize_failures(bank, {"p": evidence}, 1.)
            np.testing.assert_array_equal(scored.losses[:, 0, 0], expected)
            np.testing.assert_array_equal(scored.losses[:, 0, 1:], .1)
            self.assertEqual(model.posterior_parameters("p")["observation_count"], 19)
            self.assertIsNone(model._records[("p", "instance-0", "training", "default")]["observed_performance"])
        for reason, expected in (("TimeoutError: stopped", "timeout"), ("runtime", "runtime"),
                                 ("invalid_solution", "invalid_solution"), ("code_limit", "code_limit"),
                                 ("SyntaxError: invalid", "syntax")):
            self.assertEqual(failure_kind(reason), expected)

    def test_early_failure_gets_budgeted_paired_followups(self):
        backend = FollowupBackend()
        store = ObservationStore()
        config = ActiveEvaluationConfig(total_budget=40, cost_mode="fixed", progress_output=False)
        cpm = BayesianCPMConfig(n_draws=16)
        def run(budget):
            return BayesianActiveEvaluator(BayesianPerformanceModel(feature_names=("x",)), backend,
                config=replace(config, total_budget=budget), cpm_config=cpm, store=store).run(["p", "q"], pool(20), 2)
        selected, log = run(40)
        self.assertNotIn("p", selected)  # 1/10 > 5%, only after follow-up evidence.
        self.assertTrue(any(e["event"] == "failure_followup" for e in log["events"]))
        self.assertEqual(log["cpm"]["candidate_evidence"]["p"]["distinct_instances"], 10)
        self.assertEqual(log["cpm"]["excluded_candidates"]["p"]["reason"], "training_failure_rate_exceeded")
        self.assertEqual(len(backend.calls), len(set(backend.calls)))
        self.assertLessEqual(len(backend.calls), 40)
        self.assertEqual(log["actual_evaluation_count"], len(backend.calls))
        failed = [r for r in store.real_observations(split="training") if not r.feasible]
        self.assertEqual(len(failed), 1)
        self.assertIsNone(failed[0].observed_performance)
        before = len(backend.calls)
        selected, log = run(0)
        self.assertNotIn("p", selected)
        self.assertEqual(log["actual_evaluation_count"], 0)
        self.assertEqual(len(backend.calls), before)

    def test_followups_respect_small_budget_and_wall_limit(self):
        backend = FollowupBackend()
        _, log = BayesianActiveEvaluator(BayesianPerformanceModel(feature_names=("x",)), backend,
            config=ActiveEvaluationConfig(total_budget=5, progress_output=False),
            cpm_config=BayesianCPMConfig(n_draws=16)).run(["p", "q"], pool(20), 2)
        self.assertLessEqual(len(backend.calls), 5)
        self.assertEqual(len(set(backend.calls)), len(backend.calls))
        self.assertEqual(log["cpm"]["excluded_candidates"]["p"]["reason"],
                         "insufficient_reliability_observations")


class TestRateAudit(unittest.TestCase):
    def test_single_audit_failure_under_five_percent_is_retained_without_forged_rows(self):
        # Reuse setup helpers without inheriting/re-running their test methods.
        case = integration.TestBayesianOracleContracts()
        case.setUp()
        self.addCleanup(case.doCleanups)
        from m_asro.problems.snd.generator import SNDInstance
        case.data = [SNDInstance(3, .7, .5, seed=100+i).to_dict() for i in range(20)]
        case.labels = [{"source": "base", "hash": digest(x)} for x in case.data]
        session = case.session(information_threshold=1e100, min_exploration_fraction=0.)
        bad_hash = digest(case.data[1])
        def evaluate(program, data):
            row = ({"status": "solver_failure", "gap": None, "failure_type": "timeout", "wall_time": .01}
                   if digest(data) == bad_hash else case.observed(program, data))
            session.evaluator.evaluations[program.code_hash, digest(data)] = row
            return row
        with patch.object(session.evaluator, "evaluate", side_effect=evaluate):
            selected, vectors = session.select(case.programs[:1], 0)
        self.assertEqual(selected, case.programs[:1])
        self.assertEqual(session.actual, 20)
        self.assertEqual(session.counts["audit"], 19)
        record = session.selections[-1]
        self.assertEqual(record["audit_excluded_ids"], [])
        self.assertEqual(record["training_failure_count"], {case.programs[0].id: 1})
        self.assertAlmostEqual(vectors[case.programs[0].code_hash][1], 2.)
        bad = session.backend.peek(selected[0], session.pool[1])
        self.assertFalse(bad.feasible)
        self.assertIsNone(bad.observed_performance)
        self.assertEqual(session.evaluator.evaluations[selected[0].code_hash, bad_hash]["status"], "solver_failure")


if __name__ == "__main__":
    unittest.main()
