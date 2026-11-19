"""Player-specific objectives and EoH operator prompt construction."""
from __future__ import annotations

import json
import math

from eoh.operators import (OPERATOR_TEMPLATES, render_parents, _COMMON_NOTE,
                        _SOLVER_EXPECTED_OUTPUT, _GENERATOR_EXPECTED_OUTPUT)
from m_asro.runtime.storage import Program
from m_asro.core.utilities import UtilityConfig


def training_search_context(scenarios: list[dict], weights: list[float] | None,
                            population: list[Program], vectors: dict,
                            evaluator) -> dict:
    """Summarize training inputs and existing evaluations without executing code.

    Selection losses may include failure penalties. Only cached gaps are
    reported as measured performance; absent cache entries remain unobserved.
    """
    from ba_eoh_s.bayesian.features import instance_features
    from snd.evaluation import digest

    count = len(scenarios)
    mass = list(weights) if weights is not None else [1.0] * count
    if len(mass) != count or any(not math.isfinite(w) or w < 0 for w in mass):
        raise ValueError("Training weights must match scenarios and be nonnegative")
    total = sum(mass)
    if count and total <= 0:
        raise ValueError("Training weights must have positive total mass")
    hashes = [digest(data) for data in scenarios]
    with evaluator._cache_lock:
        cached_rows = {
            (p.code_hash, i): dict(row)
            for i, key in enumerate(hashes) for p in population
            if (row := evaluator.evaluations.get((p.code_hash, key))) is not None
        }
    instances = []
    for i, data in enumerate(scenarios):
        evidence = {}
        for p in population:
            row = cached_rows.get((p.code_hash, i))
            evidence[p.id] = ({
                "source": "real_evaluation", "status": row["status"],
                "gap": row.get("gap"), "failure_type": row.get("failure_type"),
                "executed_lines": row.get("lines"),
            } if row is not None else {"source": "unavailable", "status": "unobserved"})
        instances.append({
            "instance_index": i, "weight": mass[i] / total,
            "features": instance_features(data),
            "portfolio_selection_loss": min(
                (vectors[p.code_hash][i] for p in population), default=None),
            "evidence": evidence,
        })
    return {"split": "training", "instances": instances,
            "loss_note": "Selection losses may include failure penalties; cached gaps are real observations."}


def build_prompt(player: str, operator: str, parents: list[Program], user_prompt: str,
                 utility: UtilityConfig, opponents: dict,
                 eoh_s_context: dict | None = None,
                 temporal: bool = False) -> str:
    """五段 prompt；对手策略、效用定义及用户引导全部进入调用日志。"""
    template = OPERATOR_TEMPLATES[operator]
    if len(parents) != template.parent_count:
        raise ValueError("父代数量不符合算子定义")
    schema = ("输入字段 nodes、arcs、commodities；弧含 id/tail/head/capacity/fixed_cost/variable_cost；"
              "商品含 id/origin/destination/demand。静态有向、可拆分多商品弧流模型。"
              "求解方案满足逐商品流量守恒、共享弧容量及开启约束；目标是固定费用+运输费用。")
    if temporal:
        schema += ("当前实例是时间展开网络：nodes 是物理节点-时刻的整数编码，arcs 已包含服务弧和等待弧，"
                   "time_expansion 保存时间映射、弧时刻表及商品释放/截止时间；不要合并时间节点。")
    # 空上下文保留原 EoH prompt 的两个换行，确保历史 replay 完全兼容。
    context_text = ""
    if eoh_s_context and operator in {"CS", "LS"}:
        # 将实例级性能向量和 CPI 显式给 LLM，令其能针对短板生成候选。
        context_text = ("\n## EoH-S Complementarity Context\n"
                        "性能值越小越好。请优先改善以下实例级短板。\n"
                        f"{json.dumps(eoh_s_context, ensure_ascii=False, sort_keys=True)}\n")
    temporal_note = ("时间展开实例还必须保留 time_expansion 的时刻映射与 arc_schedule 一致；"
                     "生成器可扰动需求值，但必须同步更新 commodity_windows.demand。\n"
                     if temporal else "")
    expected_output = _SOLVER_EXPECTED_OUTPUT if player == "solver" else _GENERATOR_EXPECTED_OUTPUT
    if temporal and player != "solver":
        expected_output += ("\n时间展开实例必须同时更新 commodities[*].demand 与 "
                            "time_expansion.commodity_windows[*].demand；不得改动释放/截止时间。")
    efficiency_note = (
        "\n程序质量要求：避免对全部弧/节点的反复全表扫描，优先建立邻接表和复用中间结果；"
        "循环必须保证进度或有显式终止条件。运行时限制的是实际执行的 Python 行数，"
        "不是源码行数；超限的循环程序会被判失败。"
        f"源码上限为 {utility.max_code_lines or '不限'} 行、"
        f"{utility.max_code_chars or '不限'} 字符、"
        f"{utility.max_ast_nodes or '不限'} 个 AST 节点。"
    )
    return (f"## Task Description\nPLAYER={player}\n{schema}\n{utility.description(player)}\n"
            f"用户引导：{user_prompt}\n对手及当前边际概率：{json.dumps(opponents, ensure_ascii=False)}\n\n"
            f"{context_text}## Operator-Specific Prompt\n{operator} – {template.name}\n{template.prompt}\n\n"
            f"## Expected Output\n{expected_output}\n\n"
            f"## Note\n{_COMMON_NOTE}\n"
            f"生成器不得改变节点、弧及商品的顺序、ID和端点。{efficiency_note}\n\n"
            f"{temporal_note}"
            f"## Parent Heuristics\n{render_parents(operator, parents)}")
