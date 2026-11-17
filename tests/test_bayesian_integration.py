"""BA Solver Oracle contracts and small isolated offline/replay regression."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from m_asro import MultiAgentSNDSolver
from m_asro.problems.snd.baselines import BASE_SOLVER, BASE_GENERATOR
from ba_eoh_s.bayesian import BayesianOracleConfig, ObservationStore, OracleBudgetError
from ba_eoh_s.bayesian.oracle import SolverOracleSession
from m_asro.problems.snd.evaluation import SNDEvaluator, digest
from m_asro.runtime.performance import TimingRecorder
from m_asro.llm.replay import ReplayClient
from m_asro.experiments.storage import ProgramArchive
from m_asro.problems.snd.generator import SNDInstance, SNDParameterBounds


def small_config(**overrides):
    settings = dict(cpm_method="bayesian", active_evaluation=True, evaluation_budget=64,
                    min_observations=1, initial_per_program=1, n_draws=32, cost_mode="fixed")
    return BayesianOracleConfig(**(settings | overrides))


class BootstrapClient:
    """Pickleable offline responses: invalid I1, then a valid LS/CS child."""
    def __init__(self, reject_all=False):
        self.reject_all = reject_all

    def generate(self, prompt, temperature=1., n=1):
        code = BASE_GENERATOR
        if "PLAYER=solver\n" in prompt:
            code = ("def solve_snd(instance):\n    return (\n"
                    if self.reject_all or "## Operator-Specific Prompt\nI1" in prompt else
                    BASE_SOLVER.replace("1.0 * arc", "0.2 * arc"))
        return [f"EXPLANATION: offline bootstrap test\n```python\n{code}\n```" for _ in range(n)]


class TestBayesianOracleContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.solver = MultiAgentSNDSolver(SNDInstance(3, .7, .5, seed=7),
            generator_directions=("cost",), population_size=2, eoh_framework="eoh-s",
            response_mode="mpsro", bayesian_config=small_config(), progress_output=False,
            make_plots=False, reference_method="heuristic", test_instance_count=1)
        self.solver.root = Path(self.tmp.name)
        self.solver.archive = ProgramArchive(self.solver.root)
        self.solver.timing = TimingRecorder(self.solver.archive)
        self.solver.evaluator = SNDEvaluator(self.solver.base, self.solver.archive,
            self.solver.utility, *self.solver.reference_args,
            reference_config=self.solver.reference_config, timing=self.solver.timing)
        self.solver._bayesian_store = ObservationStore()
        self.programs = [self.solver.archive.add(player="solver", role="solver", explanation="frozen",
            code=BASE_SOLVER.replace("1.0 * arc", f"{factor} * arc"), parents=[],
            iteration=1, eoh_iteration=0, operator="I1") for factor in (.2, 2.)]
        self.data = [self.solver.base, SNDInstance(3, .7, .5, seed=97).to_dict(),
                     SNDInstance(3, .7, .5, seed=98).to_dict()]
        self.labels = [{"source": "base", "hash": digest(x)} for x in self.data]

    def session(self, **config):
        self.solver.bayesian_config = small_config(**config)
        return SolverOracleSession(self.solver, 1, self.data, None, self.labels)

    def warm_start(self, **changes):
        fields = dict(player="solver", role="solver", explanation="training control", code=BASE_SOLVER,
                      parents=[], iteration=0, eoh_iteration=0, operator="base")
        return self.solver.archive.add(**(fields | changes))

    @staticmethod
    def observed(program, data):
        # Explicit toy evaluator rows for budget-only tests; real isolated
        # feasibility and payoff integration is tested below, not claimed here.
        gap = .1 if "0.2 * arc" in program.code else .2
        return {"status": "ok", "gap": gap, "wall_time": .01,
                "metrics": {k: gap for k in ("cost", "path", "transport", "fixed", "congestion", "work")},
                "reference_kind": "toy_contract_only", "reference_value": 1.}

    def stub(self, session):
        def evaluate(program, data):
            key = program.code_hash, digest(data)
            row = self.observed(program, data)
            session.evaluator.evaluations[key] = row
            return row
        return patch.object(session.evaluator, "evaluate", side_effect=evaluate)

    def test_private_cache_snapshot_is_not_shared_with_generator_threads(self):
        session = self.session()
        self.assertEqual([x.seed for x in session.pool], [7, 97, 98])
        self.assertIsNot(session.evaluator, self.solver.evaluator)
        self.assertIsNot(session.evaluator.evaluations, self.solver.evaluator.evaluations)
        self.assertIs(session.evaluator.utility, self.solver.utility)
        self.assertIs(session.evaluator.timing, self.solver.timing)
        self.assertIs(session.evaluator._process_slots, self.solver.evaluator._process_slots)
        self.assertEqual(session.evaluator.reference_signature, self.solver.evaluator.reference_signature)
        row = self.observed(self.programs[0], self.data[0])
        key = self.programs[0].code_hash, digest(self.data[0])
        session.evaluator.evaluations[key] = row
        self.assertNotIn(key, self.solver.evaluator.evaluations)
        session.publish()
        self.assertEqual(self.solver.evaluator.evaluations[key], row)

    def test_budget_shared_across_stages_includes_full_real_audit(self):
        session = self.session(evaluation_budget=14, initial_per_program=1,
                               information_threshold=1e100, min_exploration_fraction=0.)
        with self.stub(session):
            first, vectors = session.select(self.programs[:1], 0)
            second, vectors = session.select(first+self.programs[1:], 1)
        report = session.finish()
        self.assertLessEqual(report["actual_evaluation_count"], 14)
        self.assertEqual(sum(report["evaluation_count_by_phase"].values()), report["actual_evaluation_count"])
        self.assertGreater(report["evaluation_count_by_phase"]["audit"], 0)
        self.assertTrue(all(len(v) == 3 for v in vectors.values()))
        self.assertTrue(all(np.isfinite(v).all() for v in vectors.values()))
        self.assertTrue(all(r.observation_source == "real_execution" and r.solver_utility is not None
                            for r in self.solver._bayesian_store.records()))
        self.assertEqual(self.solver._bayesian_store.unique_count, report["actual_evaluation_count"])

    def test_insufficient_budget_is_fatal_without_free_retry_or_warm_start(self):
        session = self.session(evaluation_budget=0)
        with self.stub(session), self.assertRaises(OracleBudgetError):
            session.select(self.programs, 0)
        self.assertEqual(session.actual, 0)
        self.assertTrue((self.solver.root / "bayesian_failures.jsonl").is_file())

    def test_bayesian_only_budget_is_not_ignored(self):
        session = self.session(active_evaluation=False, evaluation_budget=1)
        with self.stub(session), self.assertRaises(OracleBudgetError):
            session.select(self.programs, 0)
        self.assertEqual(session.actual, 1)

    def test_next_outer_response_reuses_real_cache_without_new_samples(self):
        first = self.session(active_evaluation=False)
        with self.stub(first):
            first.select(self.programs, 0)
        first.finish()
        count = self.solver._bayesian_store.unique_count
        second = self.session(active_evaluation=False, evaluation_budget=0)
        with patch.object(second.evaluator, "evaluate", side_effect=AssertionError("cached cell executed")):
            second.select(self.programs, 0)
        self.assertEqual(second.actual, 0)
        self.assertEqual(second.restored_cache_count, count)
        self.assertEqual(self.solver._bayesian_store.unique_count, count)
        self.assertEqual(second.model.observation_count, count)

    def test_expired_session_refuses_new_real_execution(self):
        session = self.session(active_evaluation=False, max_wall_time=1.)
        session.started -= 2.
        with self.stub(session), self.assertRaises(OracleBudgetError):
            session.select(self.programs, 0)
        self.assertEqual(session.actual, 0)

    def test_active_only_original_cpm_uses_common_actual_columns(self):
        session = self.session(cpm_method="original", evaluation_budget=24)
        with self.stub(session):
            selected, _ = session.select(self.programs, 0)
        log = session.selections[-1]
        selection = log["selection"]
        self.assertEqual(selection["method"], "cpm")
        self.assertTrue(selection["observed_common_columns"])
        self.assertEqual(log["portfolio_cpi_source"], "real_complete_training_audit")
        self.assertAlmostEqual(log["portfolio_cpi"], .1)
        self.assertEqual(len(selected), 2)

    def test_heldout_overlap_rejected_before_any_execution(self):
        self.solver.tests.append(self.data[1])
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.session()

    def test_frozen_pre_response_cache_excludes_late_generator_evidence(self):
        self.solver._bayesian_cache_snapshot = ({}, {})
        key = self.programs[0].code_hash, digest(self.data[0])
        self.solver.evaluator.evaluations[key] = self.observed(self.programs[0], self.data[0])
        session = self.session()
        self.assertNotIn(key, session.evaluator.evaluations)
        self.assertIsNone(session.backend.peek(self.programs[0], session.pool[0]))

    def test_warm_start_copy_is_training_only_never_a_bayesian_selection(self):
        copied = self.warm_start(code="# renamed\n"+BASE_SOLVER, operator="I1")
        session = self.session()
        with self.stub(session):
            parents, vectors = session.select([copied], 0)
        self.assertEqual(parents, [copied])
        self.assertEqual(session.actual, len(self.data))
        self.assertTrue(np.isfinite(vectors[copied.code_hash]).all())
        log = session.selections[-1]
        self.assertEqual(log["selected_ids"], [])
        self.assertIsNone(log["portfolio_cpi"])
        self.assertEqual(log["temporary_warm_start_ids"], [copied.id])
        self.assertEqual(self.solver._without_warm_start(parents), [])

    def test_warm_start_retires_only_after_k_successfully_audited_programs(self):
        warm = self.warm_start()
        session = self.session()
        with self.stub(session):
            first, _ = session.select([warm], 0)
            second, _ = session.select(first+self.programs[:1], 1)
            self.assertIn(warm, second)
            self.assertFalse(getattr(self.solver, "_warm_start_retired", False))
            third, _ = session.select(second+self.programs[1:], 1)
        self.assertEqual(len(third), 2)
        self.assertNotIn(warm, third)
        self.assertTrue(self.solver._warm_start_retired)
        self.assertEqual(session.actual, 9)  # Three distinct sources x three instances.
        self.assertEqual(self.solver._bayesian_store.unique_count, 9)
        with self.stub(session), self.assertRaises(OracleBudgetError):
            session.select([warm], 1)
        events = [json.loads(s) for s in (self.solver.root/"events.jsonl").read_text().splitlines()]
        retired = [e for e in events if e["event"] == "warm_start_retired"]
        self.assertEqual(len(retired), 1)
        self.assertEqual(set(retired[0]["non_warm_start_ids"]), {p.id for p in self.programs})

    def test_failed_candidate_does_not_count_towards_retirement(self):
        warm = self.warm_start()
        session = self.session(active_evaluation=False)
        def evaluate(program, data):
            row = (self.observed(program, data) if program != self.programs[1] else
                   {"status": "solver_failure", "gap": None, "failure_type": "timeout", "wall_time": .01})
            session.evaluator.evaluations[program.code_hash, digest(data)] = row
            return row
        with patch.object(session.evaluator, "evaluate", side_effect=evaluate):
            parents, _ = session.select([warm]+self.programs, 0)
        self.assertEqual(parents, [self.programs[0], warm])
        self.assertFalse(getattr(self.solver, "_warm_start_retired", False))
        self.assertEqual(session.selections[-1]["selected_ids"], [self.programs[0].id])
        self.assertAlmostEqual(session.selections[-1]["portfolio_cpi"], .1)

    def test_single_slot_capacity_and_retirement_survive_new_oracle(self):
        self.solver.K = 1
        warm = self.warm_start()
        first = self.session()
        with self.stub(first):
            parents, _ = first.select([warm], 0)
            self.assertEqual(parents, [warm])
            parents, _ = first.select(parents+self.programs[:1], 1)
        self.assertEqual(parents, self.programs[:1])
        self.assertTrue(self.solver._warm_start_retired)
        first.finish()
        second = self.session(evaluation_budget=0)
        with patch.object(second.evaluator, "evaluate", side_effect=AssertionError("cached row executed")):
            parents, _ = second.select([warm]+self.programs[:1], 0)
        self.assertEqual(parents, self.programs[:1])
        self.assertEqual(second.actual, 0)
        self.assertEqual(second.selections[-1]["temporary_warm_start_ids"], [])

    def test_failed_control_can_be_parent_without_becoming_a_valid_observation(self):
        warm = self.warm_start()
        session = self.session()
        def evaluate(program, data):
            row = {"status": "solver_failure", "gap": None, "failure_type": "line_limit", "wall_time": .01}
            session.evaluator.evaluations[program.code_hash, digest(data)] = row
            return row
        with patch.object(session.evaluator, "evaluate", side_effect=evaluate):
            parents, vectors = session.select([warm], 0)
        self.assertEqual(parents, [warm])
        self.assertTrue(np.isfinite(vectors[warm.code_hash]).all())
        self.assertTrue(all(not r.feasible and r.observed_performance is None
                            for r in self.solver._bayesian_store.records()))
        self.assertIsNone(session.selections[-1]["portfolio_cpi"])
        self.assertEqual(session.selections[-1]["warm_start_control_failure_count"], {warm.id: 3})

    def test_temporary_control_is_not_free_or_used_after_budget_exhaustion(self):
        session = self.session(evaluation_budget=0)
        with self.stub(session), self.assertRaises(OracleBudgetError):
            session.select([self.warm_start()], 0)
        self.assertEqual(session.actual, 0)

    def test_active_only_can_keep_control_separate_from_original_cpm(self):
        warm = self.warm_start()
        session = self.session(cpm_method="original")
        with self.stub(session):
            parents, _ = session.select([warm]+self.programs[:1], 0)
        self.assertEqual(parents, [self.programs[0], warm])
        self.assertAlmostEqual(session.selections[-1]["portfolio_cpi"], .1)
        self.assertEqual(session.selections[-1]["selected_ids"], [self.programs[0].id])

    def test_audit_failure_allows_training_to_continue_with_control(self):
        warm = self.warm_start()
        session = self.session(information_threshold=1e100, min_exploration_fraction=0.)
        def evaluate(program, data):
            row = (self.observed(program, data) if program == warm or digest(data) == digest(self.data[0]) else
                   {"status": "solver_failure", "gap": None, "failure_type": "timeout", "wall_time": .01})
            session.evaluator.evaluations[program.code_hash, digest(data)] = row
            return row
        with patch.object(session.evaluator, "evaluate", side_effect=evaluate):
            parents, _ = session.select([warm]+self.programs[:1], 0)
        self.assertEqual(parents, [warm])
        self.assertEqual(session.actual, 6)
        self.assertEqual(session.selections[-1]["audit_excluded_ids"], [self.programs[0].id])
        self.assertIsNone(session.selections[-1]["portfolio_cpi"])

    def test_failure_revealed_in_audit_cannot_be_selected_or_retried_free(self):
        session = self.session(information_threshold=1e100, min_exploration_fraction=0.)
        def evaluate(program, data):
            key = program.code_hash, digest(data)
            row = (self.observed(program, data) if digest(data) == digest(self.data[0]) else
                   {"status": "solver_failure", "gap": None, "failure_type": "timeout", "wall_time": .01})
            session.evaluator.evaluations[key] = row
            return row
        with patch.object(session.evaluator, "evaluate", side_effect=evaluate), self.assertRaises(OracleBudgetError):
            session.select(self.programs[:1], 0)
        self.assertEqual(session.actual, 3)
        self.assertEqual(session.counts["audit"], 2)
        self.assertEqual(session.selections, [])

    def test_fixed_cost_replay_is_independent_of_timing_fields(self):
        from ba_eoh_s.bayesian import ActiveEvaluationConfig, BayesianActiveEvaluator, BayesianPerformanceModel
        from tests.test_active_evaluation import DeterministicBackend, pool
        outputs = []
        for seconds in (.01, 100.):
            backend = DeterministicBackend()
            evaluate = backend.evaluate
            def timed(p, x):
                row = evaluate(p, x)
                return replace(row, observation=replace(row.observation, evaluation_time=seconds))
            backend.evaluate = timed
            active = BayesianActiveEvaluator(BayesianPerformanceModel(feature_names=("x",)), backend,
                config=ActiveEvaluationConfig(total_budget=15, cost_mode="fixed", progress_output=False))
            selected, report = active.run(["a", "b", "c"], pool(), 2)
            outputs.append((selected, backend.calls))
        self.assertEqual(outputs[0], outputs[1])

    def test_config_requires_mpsro_eohs_and_valid_settings(self):
        for kwargs in ({"cpm_method": "bad"}, {"active_evaluation": 1}, {"evaluation_budget": -1},
                       {"cost_mode": "bad"}, {"min_observations": 3, "initial_per_program": 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                small_config(**kwargs)
        with self.assertRaises(ValueError):
            MultiAgentSNDSolver(self.solver.base, bayesian_config=small_config())
        # Explicit disabled settings work with the original EoH and pure mode.
        old = MultiAgentSNDSolver(self.solver.base, bayesian_config=BayesianOracleConfig())
        self.assertNotIn("bayesian", old.config)


class TestBayesianOfflineIntegration(unittest.TestCase):
    def bootstrap_solver(self, directory, **overrides):
        kwargs = dict(generator_directions=("cost",), outer_iterations=1, eoh_iterations=2,
            population_size=2, n_candidates=1, instances_per_generator=2, seed=7,
            eoh_framework="eoh-s", response_mode="mpsro", parallel_workers=1, response_workers=1,
            generator_operators=("M1",), test_instance_count=1, validation_instance_count=1,
            progress_output=False, make_plots=False, reference_method="heuristic", reference_restarts=1,
            stagnation_restarts=0, bayesian_config=small_config(), llm_client=BootstrapClient())
        return MultiAgentSNDSolver(SNDInstance(3, .7, .5, seed=7), output_dir=directory,
                                  **(kwargs | overrides))

    def test_rejected_initial_candidate_recovers_and_exactly_replays(self):
        with TemporaryDirectory() as directory:
            solver = self.bootstrap_solver(directory)
            result = solver.solve("bootstrap recovery")
            records = result.data["bayesian"]["oracles"][0]["selections"]
            self.assertEqual(records[0]["status"], "temporary_warm_start_training_only")
            self.assertIsNone(records[0]["portfolio_cpi"])
            self.assertEqual(records[-1]["qualified_non_warm_count"], 1)
            self.assertTrue(records[-1]["temporary_warm_start_ids"])
            self.assertTrue(any(solver._is_warm_start(p) for p in solver._eohs_population))
            self.assertTrue(all(not solver._contains_warm_start(p) for p in solver._response_candidates["solver"]))
            solver_calls = [r for r in solver.archive.calls if r["player"] == "solver"]
            self.assertEqual(solver_calls[0]["operator"], "I1")
            self.assertEqual(solver_calls[1]["operator"], "LS")
            self.assertTrue(result.data["constraint_satisfaction"]["feasible"])
            chosen = next(p for p in solver.archive.programs if p.id == result.data["selected_solver"]["id"])
            self.assertFalse(solver._contains_warm_start(chosen))
            summary = solver._eohs_sets["solver"]
            warm_hashes = {p.code_hash for p in solver.archive.programs if solver._is_warm_start(p)}
            self.assertFalse(warm_hashes & set(summary["performance_vectors"]))
            replay = self.bootstrap_solver(directory,
                llm_client=ReplayClient(result.output_dir/"llm_calls.jsonl")).solve("bootstrap recovery")
            self.assertEqual(result.data["meta"], replay.data["meta"])
            self.assertEqual(result.data["selected_solver"]["code_hash"], replay.data["selected_solver"]["code_hash"])

    def test_all_rejected_candidates_finish_finite_search_without_export(self):
        for ls_enabled in (True, False):
            with self.subTest(ls_enabled=ls_enabled), TemporaryDirectory() as directory:
                solver = self.bootstrap_solver(directory, llm_client=BootstrapClient(reject_all=True),
                                               ls_enabled=ls_enabled)
                with self.assertRaisesRegex(RuntimeError, "非暖启动"):
                    solver.solve("bounded bootstrap failure")
                calls = [r for r in solver.archive.calls if r["player"] == "solver"]
                self.assertEqual(len(calls), 3)  # Initial I1 plus exactly R recovery rounds.
                self.assertEqual([r["operator"] for r in calls],
                                 ["I1", "LS", "LS"] if ls_enabled else ["I1", "I1", "I1"])
                report = solver._bayesian_reports[0]
                self.assertTrue(all(r["portfolio_cpi"] is None for r in report["selections"]))
                # Repeated stages reuse control evidence, not new executions.
                self.assertEqual(report["actual_evaluation_count"], 0)  # Initial game already cached warm control.
                self.assertEqual(solver._bayesian_store.unique_count,
                                 len({r.instance_hash for r in solver._bayesian_store.records()}))
                self.assertFalse((solver.root/"final_solver").exists())
                self.assertFalse((solver.root/"result.json").exists())
                self.assertTrue((solver.root/"selection_status.json").exists())

    def test_offline_and_exact_replay_keep_payoff_expectation_and_splits(self):
        with TemporaryDirectory() as directory:
            kwargs = dict(generator_directions=("cost",), outer_iterations=1, eoh_iterations=1,
                population_size=2, n_candidates=2, instances_per_generator=2, seed=7,
                eoh_framework="eoh-s", response_mode="mpsro", parallel_workers=2, response_workers=2,
                generator_operators=("M1",), test_instance_count=1, validation_instance_count=1,
                progress_output=False, make_plots=False, reference_method="heuristic", reference_restarts=1,
                stagnation_restarts=0, bayesian_config=small_config())
            first_solver = MultiAgentSNDSolver(SNDInstance(3, .7, .5, seed=7), output_dir=directory, **kwargs)
            first = first_solver.solve("stage5 deterministic integration")
            replay = MultiAgentSNDSolver(SNDInstance(3, .7, .5, seed=7), output_dir=directory,
                llm_client=ReplayClient(first.output_dir/"llm_calls.jsonl"), **kwargs).solve("stage5 deterministic integration")
            self.assertEqual(first.data["meta"], replay.data["meta"])
            self.assertEqual(first.data["selected_solver"]["code_hash"], replay.data["selected_solver"]["code_hash"])
            self.assertEqual(first.data["bayesian"]["portfolio_cpi"], replay.data["bayesian"]["portfolio_cpi"])
            self.assertEqual(first.data["llm_summary"]["successful_calls"], replay.data["llm_summary"]["successful_calls"])
            metrics = first.data["bayesian"]["oracles"][0]["response_metrics"]
            self.assertAlmostEqual(metrics["mixed_response_utility"], sum(
                r["weight"]*r["utility"] for r in metrics["members"]))
            self.assertNotEqual(metrics["portfolio_cpi"], metrics["mixed_response_utility"])
            index = first_solver.evaluator.program_index
            for item in [p for p in first_solver.archive.programs if p.player == "solver" and p.mixture_weights]:
                for opponent in [p for p in first_solver.archive.programs if p.player == "cost"]:
                    actual = first_solver.evaluator.pair(item, opponent)
                    rows = [first_solver.evaluator.pair(index[mid], opponent) for mid in item.member_ids]
                    expected = [sum(w*r[j] for w, r in zip(item.mixture_weights, rows)) for j in range(2)]
                    np.testing.assert_allclose(actual, expected)
            records = first_solver._bayesian_store.records()
            self.assertTrue(all(r.evaluation_split == "training" for r in records))
            self.assertFalse({r.instance_hash for r in records} &
                             {digest(x) for x in [*first_solver.tests, *first_solver.validation]})
            self.assertTrue(all(not first_solver._contains_warm_start(index[mid])
                                for mid in first.data["selected_solver_set"]))
            self.assertTrue(first.data["constraint_satisfaction"]["feasible"])
            self.assertTrue(all(i["test_evaluations"] is None for i in first.data["iterations"]))
            audit = first.data["bayesian"]["holdout"]
            self.assertTrue(audit["frozen_before_test"])
            self.assertEqual(audit, replay.data["bayesian"]["holdout"])
            self.assertAlmostEqual(audit["portfolio_cpi"], min(
                first_solver.evaluator.evaluate(index[mid], first_solver.tests[0])["gap"]
                for mid in first.data["selected_solver_set"]))

    def test_disabled_switches_replay_the_original_path(self):
        with TemporaryDirectory() as directory:
            kwargs = dict(generator_directions=("cost",), outer_iterations=1, eoh_iterations=1,
                population_size=2, n_candidates=1, instances_per_generator=1, seed=7,
                eoh_framework="eoh-s", response_mode="mpsro", parallel_workers=1,
                generator_operators=("M1",), test_instance_count=1, progress_output=False,
                make_plots=False, reference_method="heuristic", reference_restarts=1)
            base = SNDInstance(3, .7, .5, seed=7)
            first = MultiAgentSNDSolver(base, output_dir=directory, **kwargs).solve("legacy replay")
            with patch("m_asro.core.solver.SolverOracleSession", side_effect=AssertionError("disabled called Bayesian")):
                second = MultiAgentSNDSolver(base, output_dir=directory, **kwargs,
                    bayesian_config=BayesianOracleConfig(),
                    llm_client=ReplayClient(first.output_dir/"llm_calls.jsonl")).solve("legacy replay")
            self.assertEqual(first.data["meta"], second.data["meta"])
            self.assertEqual(first.data["objective"], second.data["objective"])
            self.assertEqual(first.data["selected_solver"]["code_hash"], second.data["selected_solver"]["code_hash"])
            self.assertNotIn("bayesian", second.data)
            self.assertNotIn("bayesian", second.data["config"])
            self.assertFalse((second.output_dir/"bayesian_observations.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
