"""Exact finite-bank checks plus conjugate joint-posterior integration tests."""
from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from m_asro.problems.snd.baselines import BASE_SOLVER
from ba_eoh_s.bayesian import (BayesianCPMConfig, BayesianPerformanceModel,
                                     JointLossSamples, bayesian_cpm_select,
                                     posterior_summary, select_portfolio)
from ba_eoh_s.eoh_s import cpi, cpm_select
from tests.test_bayesian import observation


class FixedPosterior:
    """An exactly enumerable posterior, not model-generated real observations."""
    training_split = "training"

    def __init__(self, bank):
        self.bank, self.calls = bank, 0

    def selection_evidence(self, program, **kwargs):
        return {"successful_observations": 2, "failed_observations": 0}

    def joint_posterior_predictive(self, programs, instances, n_draws=256, **kwargs):
        self.calls += 1
        return self.bank


class TestPosteriorCPI(unittest.TestCase):
    def test_exact_instance_minimum_then_expectation(self):
        # E[min] is zero; min(E) is one.  These must not be interchanged.
        bank = JointLossSamples(("a", "b"), ("x", "y"),
                                np.array([[[0, 2], [2, 0]], [[2, 0], [0, 2]]]), 7, "exact")
        np.testing.assert_array_equal(bank.cpi_samples(["a", "b"]), [0, 0])
        np.testing.assert_array_equal(bank.cpi_samples(["a"], [3, 1]), [.5, 1.5])
        np.testing.assert_array_equal(bank.marginal_gain_samples("b", ["a"], [3, 1]), [.5, 1.5])
        np.testing.assert_array_equal(bank.marginal_gain_samples("a", ["a"]), [0, 0])
        with self.assertRaises(ValueError):
            bank.marginal_gain_samples("a", [])

    def test_exact_interval_probability_and_mc_error(self):
        stat = posterior_summary(np.array([0., 0., 1., 1.]), .5, epsilon=.5)
        self.assertEqual(stat["mean"], .5)
        self.assertEqual(stat["credible_interval"], [0, 1])
        self.assertEqual(stat["improvement_probability"], .5)
        self.assertAlmostEqual(stat["variance"], 1/3)
        self.assertEqual(posterior_summary(np.array([.5, .5]), epsilon=.5)["improvement_probability"], 0)

    def test_invalid_bank_and_weights_fail(self):
        for values in (np.ones((2, 2)), np.full((2, 1, 1), np.nan)):
            with self.assertRaises(ValueError):
                JointLossSamples(("a",), ("x",), values, 7, "invalid")
        bank = JointLossSamples(("a",), ("x", "y"), np.ones((2, 1, 2)), 7, "valid")
        for weights in ([0, 0], [-1, 2], [1], [np.nan, 1]):
            with self.assertRaises(ValueError):
                bank.cpi_samples(["a"], weights)


class TestBayesianCPM(unittest.TestCase):
    def test_known_complementarity_matches_original_on_point_posterior(self):
        vectors = {"a": [1, 7], "b": [8, 0], "redundant": [1.2, 7.2], "general": [5, 5]}
        bank = JointLossSamples(tuple(vectors), ("x", "y"),
                                np.repeat(np.array([list(vectors.values())]), 8, axis=0), 7, "exact")
        model = FixedPosterior(bank)
        result, log = bayesian_cpm_select(list(vectors), model, [{"x": -1}, {"x": 1}], 2,
                                         config=BayesianCPMConfig(n_draws=8, epsilon=.1))
        original, _ = cpm_select(list(vectors), vectors, 2)
        self.assertEqual(result, original)
        self.assertEqual(result, ["a", "b"])
        self.assertEqual(log["expected_cpi"], .5)
        self.assertEqual(log["cpi_interval"], [.5, .5])
        self.assertEqual(log["steps"][1]["marginal_gain_mean"], 3.5)
        self.assertEqual(log["steps"][1]["improvement_probability"], 1)
        self.assertIsNone(log["steps"][0]["marginal_gain_mean"])
        self.assertIsNone(log["empirical_cpi"])
        self.assertEqual(model.calls, 1)
        json.dumps(log, allow_nan=False)

    def test_weighted_point_posterior_matches_original(self):
        vectors = {"a": [0, 6], "b": [4, 0], "c": [2, 2]}
        bank = JointLossSamples(tuple(vectors), ("x", "y"),
                                np.repeat(np.array([list(vectors.values())]), 8, axis=0), 0, "exact")
        result, log = bayesian_cpm_select(list(vectors), FixedPosterior(bank), [{}, {}], 1, [9, 1],
                                         config=BayesianCPMConfig(n_draws=8))
        original, _ = cpm_select(list(vectors), vectors, 1, [9, 1])
        self.assertEqual(result, original)
        self.assertAlmostEqual(log["expected_cpi"], cpi(vectors, result, [9, 1]))

    def test_greedy_gains_monotonic_and_duplicate_candidates_do_not_expand(self):
        vectors = {"a": [1, 7], "b": [8, 0], "c": [4, 4]}
        bank = JointLossSamples(tuple(vectors), ("x", "y"),
                                np.repeat(np.array([list(vectors.values())]), 8, axis=0), 0, "exact")
        result, log = bayesian_cpm_select(["a", "b", "c", "a"], FixedPosterior(bank), [{}, {}], 9,
                                         config=BayesianCPMConfig(n_draws=8))
        self.assertEqual(len(result), 3)
        self.assertEqual(log["status"], "insufficient_eligible_candidates")
        path = [step["posterior_cpi"]["mean"] for step in log["steps"]]
        self.assertTrue(all(a >= b for a, b in zip(path, path[1:])))

    def test_nondegenerate_posterior_gain_statistics_use_one_bank(self):
        losses = np.array([[[1, 7], [8, 0]], [[1, 7], [8, 2]],
                           [[1, 7], [8, 4]], [[1, 7], [8, 7]]])
        bank = JointLossSamples(("a", "b"), ("x", "y"), losses, 0, "exact")
        selected, log = bayesian_cpm_select(["a", "b"], FixedPosterior(bank), [{}, {}], 2,
                                           config=BayesianCPMConfig(n_draws=4, epsilon=2.))
        self.assertEqual(selected, ["a", "b"])
        stat = log["steps"][1]["marginal_gain"]
        self.assertEqual(stat["mean"], (3.5 + 2.5 + 1.5 + 0) / 4)
        self.assertEqual(stat["improvement_probability"], .5)
        self.assertGreater(stat["std"], 0)
        self.assertGreater(log["cpi_uncertainty"], 0)

    def test_original_dispatch_is_identical_and_does_not_use_model(self):
        vectors = {"a": [1, 3], "b": [3, 1]}
        expected = cpm_select(list(vectors), vectors, 2)
        self.assertEqual(expected, select_portfolio(list(vectors), 2, vectors=vectors))
        with self.assertRaises(ValueError):
            select_portfolio(list(vectors), 2, cpm_method="unknown")

    def test_cold_failures_test_labels_and_cache_are_separate(self):
        rows = [observation("ok", i, 2.) for i in [0, 1]]
        rows += [observation("fail", i, 1.) for i in [0, 1]]
        rows += [observation("fail", 2, None, status="timeout"),
                 observation("ok", 2, None, status="timeout", split="test"),
                 observation("cold", 0, -10), observation("cache", 0, -10, source="cache_reuse")]
        model = BayesianPerformanceModel(feature_names=("x",)).fit(rows)
        selected, log = bayesian_cpm_select(["ok", "fail", "cold", "cache"], model, [{"x": 0}], 3)
        self.assertEqual(selected, ["ok"])
        self.assertEqual(log["excluded_candidates"]["fail"]["reason"], "insufficient_reliability_observations")
        self.assertEqual(log["excluded_candidates"]["cold"]["reason"], "insufficient_real_observations")
        self.assertEqual(model.selection_evidence("ok")["failed_observations"], 0)

    def test_no_eligible_and_no_validation_test_selection(self):
        model = BayesianPerformanceModel(feature_names=("x",))
        with patch.object(model, "joint_posterior_predictive", side_effect=AssertionError("must not draw")):
            selected, log = bayesian_cpm_select(["cold"], model, [{"x": 0}], 1)
        self.assertEqual(selected, [])
        self.assertEqual(log["status"], "no_eligible_candidates")
        for split in ["validation", "test"]:
            with self.assertRaises(ValueError):
                bayesian_cpm_select(["cold"], model, [{"x": 0, "evaluation_split": split}], 1)
        model.fit([], evaluation_split="validation")
        with self.assertRaises(ValueError):
            bayesian_cpm_select([], model, [{"x": 0}], 1)

    def test_warm_start_and_composite_guards(self):
        base = dict(player="solver", code=BASE_SOLVER, operator="LS", error=None, status="parsed",
                    member_ids=None, mixture_weights=None)
        programs = [SimpleNamespace(**base, id="copy", code_hash="copy"),
                    SimpleNamespace(**{**base, "code": "", "operator": "base"}, id="base", code_hash="base"),
                    SimpleNamespace(**{**base, "code": "", "member_ids": ["copy"]}, id="mixed", code_hash="mixed")]
        model = BayesianPerformanceModel(feature_names=("x",)).fit(
            [observation(p.id, i, 1) for p in programs for i in [0, 1]])
        result, log = bayesian_cpm_select(programs, model, [{"x": 0}], 2)
        self.assertFalse(result)
        self.assertEqual(log["excluded_candidates"]["copy"]["reason"], "warm_start_copy")
        self.assertEqual(log["excluded_candidates"]["mixed"]["reason"], "requires_pure_candidate")


class TestJointPosterior(unittest.TestCase):
    def test_known_deterministic_cells_are_not_noisy_reruns(self):
        model = BayesianPerformanceModel(feature_names=("x",)).fit([observation("a", 0, 2)])
        bank = model.joint_posterior_predictive(["a"], [
            {"x": 0, "instance_hash": "instance-0"},
            {"x": 1, "instance_hash": "future"}], 32)
        np.testing.assert_array_equal(bank.losses[:, 0, 0], np.full(32, 2.))
        self.assertGreater(bank.losses[:, 0, 1].std(), 0)
        self.assertEqual(bank.conditioned_observed_cells, 1)

    def test_sparse_unpaired_observations_predict_on_common_representatives(self):
        rows = [observation("a", 0, 2), observation("a", 2, 3),
                observation("b", 1, 4), observation("b", 3, 1)]
        model = BayesianPerformanceModel(feature_names=("x",)).fit(rows)
        selected, log = bayesian_cpm_select(["a", "b"], model, [{"x": .5}, {"x": 1.5}], 2,
                                           config=BayesianCPMConfig(n_draws=64))
        self.assertEqual(len(selected), 2)
        self.assertEqual(log["conditioned_observed_cells"], 0)
        self.assertGreater(log["cpi_uncertainty"], 0)
        self.assertEqual(log["observation_source"], "posterior_prediction")

    def test_joint_sample_covariance_matches_conjugate_parameters(self):
        rows = [observation(p, i, 2 + i) for p in ["a", "b"] for i in [0, 1]]
        model = BayesianPerformanceModel(feature_names=("x",), observation_noise=.25).fit(rows)
        bank = model.joint_posterior_predictive(["a", "b"], [{"x": .25}, {"x": .75}], 40000,
                                               random_seed=17, include_noise=False)
        design = np.array([[1, -.5], [1, .5]])
        parameters = model.posterior_parameters("a")
        expected = design @ parameters["covariance"] @ design.T
        np.testing.assert_allclose(np.cov(bank.losses[:, 0, :], rowvar=False), expected, rtol=.04, atol=.002)
        cross = np.corrcoef(bank.losses[:, 0, 0], bank.losses[:, 1, 0])[0, 1]
        self.assertLess(abs(cross), .025)  # No fabricated perfect cross-program correlation.

    def test_repeated_instance_same_seed_and_candidate_order(self):
        model = BayesianPerformanceModel(feature_names=("x",)).fit(
            [observation(p, i, 2+i) for p in ["a", "b"] for i in [0, 1]])
        xs = [{"x": .5, "instance_hash": "same"}] * 2
        a = model.joint_posterior_predictive(["a", "b"], xs, 32, random_seed=5)
        b = model.joint_posterior_predictive(["b", "a"], xs, 32, random_seed=5)
        np.testing.assert_array_equal(a.losses[:, 0], b.losses[:, 1])
        np.testing.assert_array_equal(a.losses[:, 0, 0], a.losses[:, 0, 1])

    def test_sample_seed_does_not_create_independent_likelihood(self):
        model = BayesianPerformanceModel(feature_names=("x",)).fit(
            [observation("a", 0, 1, seed=i) for i in range(5)])
        self.assertEqual(model.selection_evidence("a")["successful_observations"], 1)

    def test_real_model_recovers_complementary_pair(self):
        def loss(pid, x):
            return {"left": 4+3*x, "right": 4.3-3.3*x, "general": 5., "redundant": 4.2+3*x}[pid]
        ids = ["left", "right", "general", "redundant"]
        rows = [observation(pid, float(x), loss(pid, x)) for pid in ids for x in np.linspace(-1, 1, 12)]
        model = BayesianPerformanceModel(feature_names=("x",), observation_noise=.0001).fit(rows)
        reps = [{"x": float(x)} for x in np.linspace(-1, 1, 24)]
        selected, log = bayesian_cpm_select(ids, model, reps, 2, config=BayesianCPMConfig(n_draws=4096))
        self.assertEqual(set(selected), {"left", "right"})
        self.assertGreater(log["steps"][1]["improvement_probability"], .99)
        self.assertLess(log["expected_cpi"], 3)
        repeated, other = bayesian_cpm_select(ids[::-1], model, reps, 2,
                                             config=BayesianCPMConfig(n_draws=4096))
        self.assertEqual(repeated, selected)
        self.assertEqual(other["cpi_interval"], log["cpi_interval"])


if __name__ == "__main__":
    unittest.main()
