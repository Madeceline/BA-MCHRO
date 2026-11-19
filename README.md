# BA-MCHRO

Bayesian active evaluation and complementary heuristic coevolution for service network design (SND).

BA-MCHRO maintains one Solver and up to six directional Generator players in a general-sum empirical game. M-PSRO expands their strategy pools using LLM program search; a coarse correlated equilibrium (CCE) supplies the meta-strategy. Bayesian modeling, active evaluation and Bayesian-CPM operate inside the Solver response oracle. Generator search, player utilities and the CCE linear program retain their original definitions.

## Source layout

```text
BA-MCHRO/
  snd/        SND instances, constraints, evaluation, references and deployment
  m_asro/     M-PSRO, CCE, utilities, process execution, storage and diagnostics
  eoh/        EoH search, clustering, LLM clients, operators, prompts and replay
  ba-eoh-s/   Complementary search, Bayesian model, CPM and active evaluation
```

The import names are `snd`, `m_asro`, `eoh` and `ba_eoh_s`. `setup.py` maps the four source directories to those packages; `ba-eoh-s` is the physical directory name, while Python imports use underscores. This repository distributes runtime sources only. Demos, tests, plotting tools, datasets, experiment measurements and historical generated algorithms are not included.

## Installation

Python 3.12 or later is required. From the repository root:

```bash
uv sync --locked
```

Alternatively:

```bash
python -m pip install -e .
```

SciPy/HiGHS is the default LP and MIP backend. Gurobi, CPLEX and OR-Tools are optional reference backends and require separate installation. The dependency bounds and lockfile are retained from the full project. They also contain optional-workflow libraries used there; publishing this subset does not re-resolve those dependencies.

## Offline run

Save the following as a Python script and execute it. The `__main__` guard is required for spawned processes, especially on Windows.

```python
from m_asro import MultiAgentSNDSolver
from snd.generator import SNDInstance
from ba_eoh_s.bayesian import BayesianOracleConfig


def main():
    solver = MultiAgentSNDSolver(
        SNDInstance(3, .7, .5, seed=7),
        generator_directions=("cost",),
        outer_iterations=1,
        eoh_iterations=1,
        population_size=2,
        instances_per_generator=1,
        validation_instance_count=1,
        test_instance_count=1,
        eoh_framework="eoh-s",
        response_mode="mpsro",
        make_plots=False,
        bayesian_config=BayesianOracleConfig(
            cpm_method="bayesian",
            active_evaluation=True,
            evaluation_budget=128,
            n_draws=64,
            min_observations=1,
        ),
    )
    result = solver.solve("Minimize fixed and transport costs subject to flow and capacity constraints")
    print(result.objective, result.output_dir)


if __name__ == "__main__":
    main()
```

The default generation client is deterministic and offline: it does not contact an LLM. The example verifies the execution path rather than demonstrating discovery quality. Runs are written to `experiments/runs/` by default; pass `output_dir` to choose another location. Plotting is disabled by default in this core-only distribution. `make_plots=True` requires the separate plotting tools from the full project.

## API-backed generation

Pass an explicit `LiteLLMClient` as `llm_client` to enable model calls. Configure credentials in process environment variables or an untracked local environment file. Never commit API keys.

```python
from m_asro import LiteLLMClient

client = LiteLLMClient(model="openai/<provider-model-id>")
# Pass llm_client=client to MultiAgentSNDSolver.
```

Use the provider-qualified model identifier required by LiteLLM. An OpenAI-compatible endpoint can be configured with `OPENAI_BASE_URL` and `OPENAI_API_KEY`. API execution consumes the account's inference quota. `response_workers`, `operator_generation_workers` and `llm_max_concurrency` bound player searches, operator generation and in-flight LLM requests, respectively. Operator instructions are in `BA-MCHRO/eoh/operator_prompts.txt`.

## Bayesian configurations

| Variant | `cpm_method` | `active_evaluation` |
| --- | --- | --- |
| Original evaluation and selection | `original` | `False` |
| Bayesian-CPM | `bayesian` | `False` |
| Active evaluation | `original` | `True` |
| Combined BA-MCHRO | `bayesian` | `True` |

Pass these settings through `BayesianOracleConfig`. Disabling both additions retains the original path. `evaluation_budget` counts all real Solver-oracle attempts, including initialization, paired controls, failures and final member audits. Cache hits do not add independent likelihood samples. Generator search, empirical payoff-table updates, validation and held-out testing are outside that oracle budget. `max_wall_time` limits admission of new evaluations; individual executions retain their configured timeout.

The performance model uses independent per-program conjugate Gaussian linear regressions over instance features. A joint parameter draw preserves dependence across instances for each program. Model residual variance represents approximation error, not randomness in a deterministic program's repeated output. The model does not learn cross-program residual correlation.

For a complementary portfolio H, CPI is the expected per-instance minimum loss. Bayesian-CPM estimates its posterior, credible intervals, marginal gains and improvement probabilities. Active evaluation uses complementarity-aware UCB with paired real evaluations; uniform and random acquisition are also available. Nested EVI is not implemented.

M-PSRO mixed responses instead use member-weighted expected utility. CPI is not a mixed-strategy payoff. Empirical payoff tables use real evaluations, not posterior predictions. Training, validation and test observations remain isolated. Warm starts are initialization aids and cannot enter exported Solver responses.

## Execution and evidence

The SND model has binary arc-open decisions and nonnegative splittable commodity flows, shared capacities and flow conservation. The objective is fixed opening cost plus transport cost. Generated `solve_snd(instance)` programs return actual opening and flow decisions. The evaluator independently recomputes cost and checks feasibility. Reference values may be proven MIP optima or feasible incumbents; a gap against an incumbent is not a certified optimality gap.

Candidate checks cover Python syntax and source size. Separate-process execution enforces wall-clock and Python line-event budgets; it is not a security container for hostile code. Handled interruption can export an available verified portfolio but does not implement resumable training. Deployment uses the existing validation and fallback rules.

Runs record configurations, sources and lineage, LLM calls, real evaluations, timings, meta-strategies and exports. Preserve the runtime configuration and source hashes when reporting experiments. Exact replay requires the same prompts and search configuration used for recording. The finite empirical CCE and bounded program searches do not imply equilibrium convergence over all programs. No general quality improvement or training speedup is asserted by this source release.
