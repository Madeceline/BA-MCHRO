"""Export a real, validated portfolio after a handled training interruption.

This module does not checkpoint or resume training. It never reads holdout
instances, changes Oracle budgets, or fills missing payoffs with predictions.
"""
from __future__ import annotations

import math
import os
import time
from typing import Any, TYPE_CHECKING
import uuid

from ba_eoh_s.eoh_s import cpi
from snd.evaluation import digest, validate_solution
from m_asro.runtime.execution import execute, validate_code
from m_asro.runtime.storage import Program, write_json

if TYPE_CHECKING:
    from m_asro.core.solver import MultiAgentSNDSolver


def remember_portfolio(solver: MultiAgentSNDSolver, members: list[Program],
                       scenarios: list[dict], weights: list[float] | None,
                       iteration: int, stage: int) -> None:
    """Keep selections, including pending audits, separate from search state."""
    unique = {p.code_hash: p for p in solver._without_warm_start(members)
              if p.error is None and p.member_ids is None}
    if unique:
        solver._interruption_portfolios.append({
            "members": list(unique.values()), "scenarios": list(scenarios),
            "weights": list(weights) if weights is not None else None,
            "iteration": iteration, "eoh_iteration": stage})


def _message(exc: BaseException) -> str:
    message = f"{type(exc).__name__}: {exc}"
    for key in ("LLM_API_KEY", "API_KEY", "OPENAI_API_KEY"):
        secret = os.environ.get(key)
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message


def _deployment_check(solver: MultiAgentSNDSolver, program: Program, data: dict,
                      split: str, checks: list[dict]) -> dict | None:
    """Reuse real results, or independently execute without requesting a reference.

    Additional runs are labelled deployment checks OUTSIDE the training Oracle
    budget, logged individually, and never used to update the Bayesian model.
    Failed cached executions are not retried to obtain a more favourable label.
    """
    key = (program.code_hash, digest(data))
    started = time.perf_counter()
    cache_hit = key in solver.evaluator.evaluations
    record = {"program_id": program.id, "instance_hash": key[1],
              "evaluation_split": split, "cache_hit": cache_hit,
              "executed": False, "budget_scope": "interruption_deployment_check"}
    try:
        limits = solver.utility
        validate_code(program.code, "solve_snd(instance)",
                      max_code_lines=limits.max_code_lines,
                      max_code_chars=limits.max_code_chars, max_ast_nodes=limits.max_ast_nodes)
        if cache_hit:
            result = solver.evaluator.evaluations[key]
            if result.get("status") != "ok":
                raise ValueError(result.get("error", "cached candidate execution failed"))
        else:
            record["executed"] = True
            execution = execute(program.code, "solve_snd", data,
                timeout=solver.reference_args[3],
                line_limit=solver.evaluator._execution_line_limit(data),
                max_code_lines=limits.max_code_lines,
                max_code_chars=limits.max_code_chars, max_ast_nodes=limits.max_ast_nodes)
            result = execution["result"]
        objective, constraints = validate_solution(data, result)
        if not constraints["feasible"]:
            raise ValueError("中断部署检查：候选返回不可行方案")
        record["status"] = "ok"
        return {"objective": objective, "arc_open": result["arc_open"],
                "flows": result["flows"], "constraint_satisfaction": constraints}
    except Exception as exc:
        record.update(status="failed", error=_message(exc))
        return None
    finally:
        record["wall_time"] = time.perf_counter() - started
        checks.append(record)
        solver.archive.append("interruption_deployment_checks.jsonl", record)


def export_interrupted(solver: MultiAgentSNDSolver, exc: BaseException) -> dict | None:
    """Return a partial result, or None when no non-baseline set can be exported."""
    interruption: dict[str, Any] = {
        "reason": _message(exc), "exception_type": type(exc).__name__,
        "training_status": "interrupted", "resumable": False,
        "export_status": "pending", "deployment_checks": []}
    status_path = solver.root / "interruption.json"
    write_json(status_path, interruption)
    # Exceptions after a complete export (e.g. plots) must preserve that result.
    result_path = solver.root / "result.json"
    if result_path.exists() and (solver.root / "final_solver" / "solver.py").exists():
        import json
        final = json.loads(result_path.read_text(encoding="utf-8"))
        interruption["export_status"] = "existing_final_preserved"
        final.update(training_status="interrupted", interruption=interruption)
        write_json(status_path, interruption)
        write_json(result_path, final)
        solver._report(f"运行中断，已保留最终程序：{solver.root / 'final_solver' / 'solver.py'}")
        return final

    checks = interruption["deployment_checks"]
    checked: dict[str, dict | None] = {}
    rejected: list[dict] = []
    selected_members: list[Program] = []
    base_results: dict[str, dict] = {}
    source = None
    # The newest selected portfolio takes priority. If its audit was not
    # completed, consider only fully observed members, then older selections.
    for snapshot in reversed(solver._interruption_portfolios):
        vectors = {}
        for program in solver._without_warm_start(snapshot["members"]):
            rows = [solver.evaluator.evaluations.get((program.code_hash, digest(x)))
                    for x in snapshot["scenarios"]]
            if not rows or any(row is None or row.get("status") != "ok"
                               or row.get("gap") is None or not math.isfinite(row["gap"])
                               for row in rows):
                rejected.append({"program_id": program.id, "reason": "incomplete_or_failed_real_audit"})
                continue
            if program.code_hash not in checked:
                base = _deployment_check(solver, program, solver.base, "base", checks)
                valid = base is not None
                if valid:
                    for data in solver.validation:
                        if _deployment_check(solver, program, data, "validation", checks) is None:
                            valid = False
                            break
                checked[program.code_hash] = base if valid else None
            base = checked[program.code_hash]
            if base is None:
                rejected.append({"program_id": program.id, "reason": "deployment_check_failed"})
                continue
            selected_members.append(program)
            base_results[program.id] = base
            vectors[program.code_hash] = [row["gap"] for row in rows]
        if selected_members:
            source = snapshot
            break
    interruption["excluded_members"] = rejected
    if not selected_members:
        interruption["export_status"] = "no_validated_non_warm_portfolio"
        write_json(status_path, interruption)
        solver._report("训练中断：尚无通过真实审计及部署检查的非暖启动互补集合，无法导出；保留原始异常。")
        return None

    mixed = solver.response_mode == "mpsro"
    weights = [1.0 / len(selected_members)] * len(selected_members) if mixed else None
    selected = solver.archive.add(player="solver", role="solver",
        explanation="训练中断时导出的当前互补集合；" +
                    ("沿用 M-PSRO 等权混合部署。" if mixed else "逐成员验解并选择最低目标可行结果。"),
        code="", parents=selected_members, members=selected_members,
        mixture_weights=weights, iteration=source["iteration"],
        eoh_iteration=source["eoh_iteration"], operator="INTERRUPTED_PORTFOLIO")
    solver.evaluator.program_index.update({p.id: p for p in solver.archive.programs})
    chosen = (solver.evaluator._draw_member(selected, [solver.seed, selected.code_hash, digest(solver.base)])
              if mixed else min(selected_members, key=lambda p: base_results[p.id]["objective"]))
    solution = base_results[chosen.id]
    reference = solver.evaluator.references.get(digest(solver.base), {})
    reference_value = reference.get("value")
    measured_gap = ((solution["objective"] - reference_value) / (abs(reference_value) if reference_value else 1.)
                    if reference_value is not None else None)
    # Stage in a separate directory so even a partial failed prior export is
    # preserved. The standard final_solver location is used when it is free.
    staged = solver.root / ("interrupted_solver_" + uuid.uuid4().hex[:8])
    solver._export(selected, folder=staged)
    folder = solver.root / "final_solver"
    if not folder.exists():
        staged.rename(folder)
    else:
        folder = staged
    interruption.update(export_status="exported", source_iteration=source["iteration"],
        source_eoh_iteration=source["eoh_iteration"], member_ids=selected.member_ids,
        additional_execution_count=sum(row["executed"] for row in checks),
        cache_hit_count=sum(row["cache_hit"] for row in checks))
    relative_folder = folder.relative_to(solver.root).as_posix()
    elapsed = time.perf_counter() - solver._solve_started
    final = {
        "schema": "multi-agent-snd-v1", "config": solver.config,
        "user_prompt": solver.user_prompt, "training_status": "interrupted",
        "interruption": interruption, "objective": solution["objective"],
        "optimal": False, "gap": measured_gap, "reference_value": reference_value,
        "reference_kind": reference.get("kind"), "solve_time": elapsed,
        "total_wall_time": elapsed, "arc_open": solution["arc_open"], "flows": solution["flows"],
        "constraint_satisfaction": solution["constraint_satisfaction"],
        "solver_status": "heuristic_feasible", "solver_used": "multi_agent_generated_heuristic",
        "selected_solver": selected.to_dict(), "selected_solver_set": selected.member_ids,
        "deployment_kind": "mixed" if mixed else "ensemble",
        "deployment_selection": {"rule": "interrupted_current_real_audited_portfolio",
                                 "selected_id": selected.id, "validation_required": True},
        "selected_solver_distribution": None,
        "portfolio_cpi": cpi(vectors, list(vectors), source["weights"]),
        "portfolio_cpi_source": "real_complete_training_audit",
        "mixed_response_utility": None,
        "mixed_response_utility_status": "not_recomputed_on_interruption",
        "instance_format": solver.base.get("format", "static-snd-v2"),
        "time_expansion": solver.base.get("time_expansion"),
        "players": list(solver.players), "programs": solver.archive.catalog(),
        "iterations": getattr(solver, "_interruption_history", []),
        "eohs": solver._eohs_sets, "test_evaluations": None, "test_summary": None,
        "test_success_rate": None, "test_mean_gap": None,
        "diagnostic_scope": {"test": "not_run_by_interruption_handler",
                             "training_completed": False, "resumable": False},
        "timing_summary": solver.timing.summary(),
        "artifacts": {"result": "result.json", "interruption": "interruption.json",
                      "final_solver": relative_folder + "/solver.py",
                      "runtime_config": relative_folder + "/runtime_config.json",
                      "program_archive": "programs.jsonl",
                      "deployment_checks": "interruption_deployment_checks.jsonl"}}
    if solver.bayesian_config.enabled:
        final["bayesian"] = {"config": solver.config["bayesian"],
                             "oracles": solver._bayesian_reports,
                             "payoff_source": "original_real_pair_no_posterior_payoffs"}
    write_json(status_path, interruption)
    write_json(result_path, final)
    write_json(solver.root / "timing_summary.json", final["timing_summary"])
    write_json(solver.root / "lineage.json", {"selected_id": selected.id, "programs": solver.archive.catalog()})
    write_json(solver.output_base / "latest_run.json", {"output_dir": str(solver.root), "result": str(result_path)})
    solver.archive.write_browser(selected.id)
    solver._report(f"训练已中断，导出 {len(selected_members)} 个非暖启动集合成员：{folder / 'solver.py'}")
    solver._report(f"中断原因：{interruption['reason']}")
    return final
