"""Bounded operator generation, snapshot semantics and exact offline replay."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from unittest.mock import patch

from m_asro import MultiAgentSNDSolver
from ba_eoh_s.bayesian import BayesianOracleConfig
from m_asro.problems.snd.evaluation import digest
from m_asro.runtime.performance import TimingRecorder, ordered_map
from m_asro.llm.replay import ReplayClient
from m_asro.experiments.storage import ProgramArchive
from m_asro.problems.snd.generator import SNDInstance, SNDParameterBounds


class TestOperatorGeneration(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def solver(self, folder="run", **changes):
        kwargs = dict(generator_directions=("cost",), population_size=2,
                      operator_generation_workers=2, llm_max_concurrency=2,
                      progress_output=False, make_plots=False, llm_retries=0,
                      test_instance_count=1, parallel_workers=1)
        kwargs.update(changes)
        solver = MultiAgentSNDSolver(SNDInstance(3, .7, .5, seed=7), **kwargs)
        solver.root = self.root / folder
        solver.archive = ProgramArchive(solver.root)
        solver.timing = TimingRecorder(solver.archive)
        solver.user_prompt = "operator concurrency test"
        return solver

    def test_config_validation_and_legacy_defaults(self):
        for kwargs in ({"operator_generation_workers": 0},
                       {"operator_generation_workers": True},
                       {"llm_max_concurrency": 0}, {"llm_max_concurrency": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.solver(**kwargs)
        solver = self.solver(operator_generation_workers=1, llm_max_concurrency=None)
        self.assertEqual(solver.llm_max_concurrency, 1)
        self.assertEqual(solver.config["llm_seed_scheme"], "legacy_global_call_index")
        self.assertEqual(solver.config["operator_generation_schedule"], "legacy")
        solver = self.solver(operator_generation_workers=3, response_workers=2,
                             llm_max_concurrency=None)
        self.assertEqual(solver.llm_max_concurrency, 6)

    def test_batch_order_seed_and_partial_retry_ignore_completion_order(self):
        def run(folder, reverse):
            solver = self.solver(folder, llm_retries=1)
            barrier = threading.Barrier(2)
            attempts = {}
            lock = threading.Lock()
            def llm(kind, arguments, timeout):
                _, prompt, _, n, seed = arguments
                self.assertEqual(n, 1)
                with lock:
                    attempt = attempts.get(prompt, 0)
                    attempts[prompt] = attempt + 1
                if attempt == 0:
                    barrier.wait(5)
                time.sleep(.02 if (prompt == "CS") != reverse else 0)
                if prompt == "CS" and attempt == 0:
                    # Keep a valid partial response despite count mismatch.
                    return {"result": ["def solve_snd(instance):\n    return 1", 123]}
                return {"result": [f"def solve_snd(instance):\n    return {seed}"]}
            with patch.object(solver, "_generation_prompt", side_effect=lambda _, op, *a: op), \
                    patch("m_asro.core.solver.isolated_call", side_effect=llm):
                with solver.archive.scope("t001_solver"):
                    result = solver._generate_batch("solver", 1, 1, {}, {},
                        [("CS", [], None), ("LS", [], None)])
            rows = [(p.id, p.code_hash, p.operator) for p in result]
            calls = {(c["operator"], c["attempt"]): c["seed"] for c in solver.archive.calls}
            self.assertEqual([p.operator for p in result], ["CS", "CS", "LS"])
            self.assertEqual(len(set(calls.values())), 3)
            self.assertEqual(sorted(c["index"] for c in solver.archive.calls), [0, 1, 2])
            self.assertEqual(sum(c["status"] == "failed" for c in solver.archive.calls), 1)
            return rows, calls
        self.assertEqual(run("normal", False), run("reverse", True))

    def test_global_limit_includes_nested_players_and_failed_retries(self):
        solver = self.solver(response_workers=2, operator_generation_workers=3, llm_retries=1)
        barrier = threading.Barrier(2)
        lock = threading.Lock()
        active = peak = admissions = 0
        seen = set()
        def llm(kind, arguments, timeout):
            nonlocal active, peak, admissions
            prompt = arguments[1]
            with lock:
                active += 1
                peak = max(peak, active)
                admissions += 1
                first_two = admissions <= 2
                retry = prompt in seen
                seen.add(prompt)
            try:
                if first_two:
                    barrier.wait(5)
                time.sleep(.01)
                if not retry:
                    raise RuntimeError("transient failure")
                return {"result": ["def generate_snd(base_instance, seed):\n    return base_instance"]}
            finally:
                with lock:
                    active -= 1
        def player_search(player):
            with solver.archive.scope(f"t001_{player}"):
                return solver._generate_batch(player, 1, 1, {}, {},
                    [(op, [], None) for op in ("M1", "M2", "M3")])
        with patch.object(solver, "_generation_prompt", side_effect=lambda p, op, *a: p+op), \
                patch("m_asro.core.solver.isolated_call", side_effect=llm):
            # ordered_map marks the outer player threads as already active.
            results = ordered_map(player_search, ("cost", "path"), 2)
        self.assertEqual(peak, 2)
        self.assertEqual([len(result) for result in results], [3, 3])
        self.assertEqual(len(solver.archive.calls), 12)
        self.assertEqual(len({c["seed"] for c in solver.archive.calls}), 12)
        self.assertEqual(len({p.id for p in solver.archive.programs}), 6)
        self.assertTrue(any(c["queue_seconds"] > .005 for c in solver.archive.calls))
        # A failed request must release its slot before retrying.
        self.assertTrue(solver._llm_slots.acquire(blocking=False))
        solver._llm_slots.release()

    def test_eoh_batches_use_old_parents_and_select_once_per_round(self):
        solver = self.solver()
        def program(tag):
            return solver.archive.add(player="solver", role="solver", explanation=tag,
                code=f"def solve_snd(instance):\n    return {tag!r}", parents=[],
                iteration=1, eoh_iteration=0, operator="I1")
        initial = [program("first"), program("second")]
        generated = program("later")
        old_ids = {p.id for p in initial}
        def batch(player, t, r, pools, meta, jobs):
            self.assertEqual([op for op, _, _ in jobs], ["E1", "E2", "M1", "M2", "M3"])
            self.assertTrue(all(p.id in old_ids for _, parents, _ in jobs for p in parents))
            return [generated]
        from ba_eoh_s.clustering import select_representatives
        with patch.object(solver, "_generate", return_value=initial), \
                patch.object(solver, "_generate_batch", side_effect=batch), \
                patch.object(solver, "_fitness", side_effect=lambda p, *a: 1 if p == generated else 0), \
                patch("m_asro.core.solver.select_representatives", wraps=select_representatives) as rank:
            result = solver._search_eoh_original("solver", 1, {"solver": []}, {})
        self.assertEqual(rank.call_count, 2)  # Initialization, then whole batch.
        self.assertEqual(result, generated)

    def test_one_failed_operator_keeps_others_with_single_request_slot(self):
        solver = self.solver(llm_max_concurrency=1, llm_retries=1)
        def llm(kind, arguments, timeout):
            if arguments[1] == "CS":
                raise RuntimeError("operator failed")
            return {"result": ["def solve_snd(instance):\n    return 1"]}
        with patch.object(solver, "_generation_prompt", side_effect=lambda _, op, *a: op), \
                patch("m_asro.core.solver.isolated_call", side_effect=llm):
            result = solver._generate_batch("solver", 1, 1, {}, {},
                [("CS", [], None), ("LS", [], None)])
        self.assertEqual([p.operator for p in result], ["LS"])
        self.assertEqual(sum(c["status"] == "failed" for c in solver.archive.calls), 2)

    def test_eohs_cs_ls_share_vectors_and_select_after_batch(self):
        solver = self.solver(eoh_framework="eoh-s", response_mode="mpsro")
        initial = [solver.archive.add(player="solver", role="solver", explanation=str(i),
            code=f"def solve_snd(instance):\n    return {i}", parents=[], iteration=1,
            eoh_iteration=0, operator="I1") for i in (1, 2)]
        def batch(player, t, r, pools, meta, jobs):
            self.assertEqual([op for op, _, _ in jobs], ["CS", "LS"])
            self.assertEqual({p.id for p in jobs[0][1]}, {p.id for p in initial})
            self.assertIn(jobs[1][1][0], initial)
            self.assertEqual(jobs[0][2]["cpi"], jobs[1][2]["cpi"])
            return []
        with patch.object(solver, "_generate", return_value=initial), \
                patch.object(solver, "_generate_batch", side_effect=batch), \
                patch.object(solver, "_eohs_scenarios", return_value=(
                    [solver.base], [1.0], [{"source": "base"}])), \
                patch.object(solver, "_cpi_vectors", side_effect=lambda ps, *a: {
                    p.code_hash: [float(i)] for i, p in enumerate(ps)}), \
                patch.object(solver, "_fitness", return_value=0.):
            solver._search_eohs("solver", 1, {"solver": []}, {})
        self.assertEqual(solver._response_candidates["solver"], initial)

    def test_offline_spawn_and_exact_replay_eoh_eohs_and_bayesian(self):
        for framework, bayesian in (("eoh", False), ("eoh-s", False), ("eoh-s", True)):
            with self.subTest(framework=framework, bayesian=bayesian):
                kwargs = dict(generator_directions=("cost",), outer_iterations=1,
                    eoh_iterations=1, population_size=2, n_candidates=2,
                    instances_per_generator=2, seed=7, eoh_framework=framework,
                    response_mode="mpsro", response_workers=2, parallel_workers=2,
                    operator_generation_workers=2, llm_max_concurrency=2,
                    generator_operators=("M1", "M2"), progress_output=False,
                    make_plots=False, reference_method="heuristic", reference_restarts=1,
                    test_instance_count=1, validation_instance_count=1,
                    stagnation_restarts=0, instance_bounds=SNDParameterBounds(
                        nodes=(3, 4), coverage=(.7, .8), fixed_cost=(.4, .7)))
                if bayesian:
                    kwargs["bayesian_config"] = BayesianOracleConfig(cpm_method="bayesian",
                        active_evaluation=True, evaluation_budget=100, n_draws=16)
                first_solver = MultiAgentSNDSolver(SNDInstance(3, .7, .5, seed=7),
                    output_dir=self.root / "offline", **kwargs)
                first = first_solver.solve("parallel operator integration")
                second = MultiAgentSNDSolver(SNDInstance(3, .7, .5, seed=7),
                    llm_client=ReplayClient(first.output_dir / "llm_calls.jsonl"),
                    output_dir=self.root / "replay", **kwargs).solve("parallel operator integration")
                self.assertEqual(first.data["selected_solver"], second.data["selected_solver"])
                self.assertEqual(first.data["meta"], second.data["meta"])
                self.assertEqual(first.data["objective"], second.data["objective"])
                self.assertEqual(first.data["llm_summary"], second.data["llm_summary"])
                self.assertTrue(first.data["constraint_satisfaction"]["feasible"])
                self.assertEqual(first.data["config"]["llm_max_concurrency"], 2)
                self.assertTrue(all(i["test_evaluations"] is None for i in first.data["iterations"]))
                self.assertTrue(all(not first_solver._is_warm_start(p)
                    for p in first_solver._response_candidates["solver"]))
                if bayesian:
                    metric = first.data["bayesian"]["oracles"][0]["response_metrics"]
                    self.assertAlmostEqual(metric["mixed_response_utility"], sum(
                        r["weight"] * r["utility"] for r in metric["members"]))
                    self.assertEqual(first.data["bayesian"]["portfolio_cpi"],
                                     second.data["bayesian"]["portfolio_cpi"])
                    self.assertFalse({r.instance_hash for r in first_solver._bayesian_store.records()}
                        & {digest(x) for x in [*first_solver.tests, *first_solver.validation]})


if __name__ == "__main__":
    unittest.main()
