# BA-MCHRO

Bayesian active evaluation and complementary heuristic coevolution for service network design (SND).

M-ASRO maintains a Solver player and directional Generator players in a general-sum empirical game. A coarse correlated equilibrium (CCE) defines the meta-strategy. BA-EoH-S supplies Bayesian performance modeling, complementary portfolio selection and paired active evaluation inside the Solver response oracle.

## Structure

```text
m_asro/
  core/             M-PSRO controller, CCE, utilities and interruption handling
  llm/              Clients, replay, parsing and operator prompts
  runtime/          Isolated execution, concurrency, progress and timing
  problems/snd/     Problem adapter, generation, reference solvers and deployment
  experiments/      Program storage and evaluation diagnostics
ba_eoh_s/
  eoh_s.py          CPI, CPM, complementary search and local search
  clustering.py     Population representative selection
  bayesian/         Observations, conjugate regression, posterior CPM and acquisition
tests/              Core unit and offline/replay integration tests
```

The package organization follows [ASRO](https://github.com/Kexy24/ASRO/tree/main/asro). This repository retains its own Solver–Generator game, utilities and CCE formulation. Compatibility packages (`multi_agent_snd`, `snd_generator`, `snd_solver`, `snd_visualization`) preserve imports used by previously exported programs.

This distribution contains core source and mathematical documentation. Demo scripts, benchmark drivers, instance datasets, training results, exported trained algorithms and credentials are excluded. The SND generation implementation is included because the training controller depends on it.

## Installation

Python 3.12 or later is required.

```bash
uv sync --extra test
```

Alternatively:

```bash
python -m pip install -e ".[test]"
```

SciPy/HiGHS provides the default MIP and LP backends. Gurobi, CPLEX and OR-Tools are optional reference backends and require separate installation.

## Use

Create a Python file with a main guard so isolated processes can start on Windows:

```python
from m_asro import MultiAgentSNDSolver
from m_asro.problems.snd.generator import SNDInstance
from ba_eoh_s.bayesian import BayesianOracleConfig

def main():
    solver = MultiAgentSNDSolver(
        SNDInstance(4, .6, .5, seed=7),
        eoh_framework="eoh-s",
        response_mode="mpsro",
        bayesian_config=BayesianOracleConfig(
            cpm_method="bayesian",
            active_evaluation=True,
            evaluation_budget=256,
            n_draws=256,
        ),
        make_plots=False,
    )
    result = solver.solve("Minimize cost subject to capacity and flow conservation")
    print(result.objective, result.output_dir)

if __name__ == "__main__":
    main()
```

The default generation client is offline. API execution requires an explicit `LiteLLMClient`:

```python
from m_asro.llm.client import LiteLLMClient

client = LiteLLMClient(model="openai/<provider-model-name>")
# Pass llm_client=client to MultiAgentSNDSolver.
```

Configure `LLM_API_KEY` and `LLM_BASE_URL` through the process environment or a local `llmapi.env`. OpenAI-compatible environment names are also supported. Never commit credentials. API training incurs provider charges.

## Configuration

| Setting | Role |
| --- | --- |
| `outer_iterations` | Empirical-game expansion rounds |
| `eoh_iterations`, `population_size` | Response-search depth and local portfolio capacity |
| `warm_start_algorithms` | Initialization programs, excluded from final Solver exports |
| `instances_per_generator` | Training samples per Generator program |
| `validation_instance_count`, `test_instance_count` | Validation and held-out sample counts |
| `strategy_pool_limit`, `max_profiles` | Strategy-pool and CCE joint-profile limits |
| `response_workers` | Concurrent player response searches |
| `operator_generation_workers`, `llm_max_concurrency` | Operator generation and shared LLM concurrency |
| `candidate_timeout`, `execution_line_limit` | Candidate execution budgets |
| `reference_method`, `time_limit`, `reference_timeout` | Reference selection and solver/process budgets |

`BayesianOracleConfig` controls Solver-oracle behavior:

| Variant | `cpm_method` | `active_evaluation` |
| --- | --- | --- |
| Original | `original` | `False` |
| Bayesian-CPM | `bayesian` | `False` |
| Active evaluation | `original` | `True` |
| Combined | `bayesian` | `True` |

Both Bayesian options default to disabled. Acquisition supports `ucb`, `uniform` and `random`; nested EVI is not implemented. `evaluation_budget` counts real Solver-oracle attempts, including initial controls, failures, paired completion and member audits. Cache hits and posterior predictions are tracked separately. `max_wall_time` limits admission of additional evaluations; empirical-game updates, Generator search, validation and held-out testing have separate costs.

## Mathematical boundaries

- CCE constraints, player utilities, Generator search and member-weighted mixed payoffs retain their definitions. Responses search against one common previous strategy/meta-strategy snapshot.
- Portfolio CPI aggregates the per-instance minimum loss. Mixed-response utility uses member weights. These are different quantities.
- The performance model uses per-program conjugate linear Gaussian regression. Deterministic cache duplicates add no likelihood terms; model uncertainty describes unobserved performance rather than random program output.
- Posterior predictions guide acquisition and selection, never replace real empirical-game payoffs. Validation and held-out results do not update the training posterior.
- Source checks enforce Python syntax and size limits. Separate processes enforce execution budgets, and solutions undergo independent feasibility validation. Process isolation is not a security sandbox for hostile code.
- Warm starts may serve as temporary parents and controls, but cannot enter final Solver response support. Handled interruption attempts to export an available validated non-warm portfolio; training is not resumable.

The linear model has fixed residual variance and independent program regressions. Feature misspecification and finite representative samples limit calibration. Finite LLM searches do not establish best responses over all programs, and CCE describes the finite empirical game. No general speed, quality or convergence guarantee follows from these mechanisms.

## Validation

```bash
python -m pytest -q
```

The tests cover conjugate posterior algebra, seeded sampling, cold starts, complementarity, paired budget admission, cache deduplication, failure evidence, split isolation, weighted mixed utilities, CCE, warm-start exclusion, interruption and offline/replay training. Optional archive-dependent checks require local fixtures; this source release contains no measured benchmark results.

## Documentation

- [Mathematical model and algorithm structure](项目数学原理与算法结构.md)
- [Game definitions](博弈详解.md)
