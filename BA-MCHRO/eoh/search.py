"""Original EoH response search; caller supplies game fitness and generation.

The sequential and batched schedules preserve their original selection points.
"""
from __future__ import annotations

from eoh.clustering import select_representatives
from eoh.operators import OPERATOR_TEMPLATES, EVOLUTION_OPERATOR_CODES
from m_asro.runtime.storage import Program

def search(self, player: str, t: int, pools: dict, meta: dict) -> Program:
    """原有 EoH 搜索实现，保留 Top-K/文本聚类兼容行为。"""
    # 混合响应不能当作变异父代；取出其纯程序支持集继续进化。
    archive_index = {p.id: p for p in self.archive.programs}
    population = [archive_index[mid] for policy in pools[player]
                  for mid in (policy.member_ids if policy.mixture_weights is not None
                              else [policy.id]) if archive_index[mid].member_ids is None]
    scores: dict[str, float] = {}

    def rank(candidates: list[Program]) -> list[Program]:
        unique = {}
        for p in candidates:
            if p.code_hash not in scores:
                scores[p.code_hash] = self._fitness(p, pools, meta)
            self.archive.append("events.jsonl", {"event": "fitness", "program_id": p.id,
                                                  "iteration": t, "fitness": scores[p.code_hash]})
            unique.setdefault(p.code_hash, p)
        candidate_values = list(unique.values())
        eligible = self._training_candidates(candidate_values, player)
        selected, diagnostic = select_representatives(
            eligible, scores, self.K,
            mode=self.eoh_selection, method=self.cluster_method,
            min_clusters=self.min_clusters)
        diagnostic["warm_start_candidates"] = sum(
            1 for candidate in candidate_values if self._contains_warm_start(candidate))
        diagnostic["warm_start_excluded"] = sum(
            1 for candidate in candidate_values if candidate not in eligible)
        self.archive.append("events.jsonl", {
            "event": "population_selection", "player": player, "iteration": t,
            "selection": diagnostic,
        })
        return selected

    population = rank(population + self._generate(player, "I1", [], t, 0, pools, meta))
    for r in range(1, self.R + 1):
        operators = self.generator_operators if player != "solver" else EVOLUTION_OPERATOR_CODES
        jobs: list[tuple[str, list[Program], dict | None]] = []
        for operator in operators:
            count = OPERATOR_TEMPLATES[operator].parent_count
            if len(population) < count:
                self.archive.append("events.jsonl", {"event": "skipped_operator", "player": player,
                    "iteration": t, "eoh_iteration": r, "operator": operator, "reason": "insufficient_distinct_parents"})
                continue
            rng = (self._player_rng[player] if self.response_workers > 1
                   or self.operator_generation_workers > 1 else self.rng)
            parents = rng.sample(population, count)
            if self.operator_generation_workers > 1:
                # No rank/update inside the loop: all parents are chosen
                # from this round's frozen population, including skips.
                jobs.append((operator, parents, None))
            else:
                population = rank(population + self._generate(player, operator, parents, t, r, pools, meta))
        if self.operator_generation_workers > 1:
            generated = self._generate_batch(player, t, r, pools, meta, jobs)
            population = rank(population + generated)
        self._report(f"  {player}：EoH {r}/{self.R} 完成，最佳效用 {scores[population[0].code_hash]:.6f}")
    # M-PSRO 从本地候选池筛选混合支持集；原模式仍只取首名。
    if player == "solver":
        original_population = list(population)
        population = self._without_warm_start(population)
        if not population:
            message = ("EoH 未生成可部署的非暖启动求解器程序；"
                       "暖启动仅用于初始化和训练诊断，禁止作为最终输出")
            self._write_selection_status(error=message, pools=pools, candidates=original_population)
            raise RuntimeError(message)
    self._response_candidates[player] = list(population)
    return population[0]

