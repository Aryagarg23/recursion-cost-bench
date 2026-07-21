# recursion-cost-bench

A local, empirical dataset generator for a question that turns out to be surprisingly
hard to answer by guessing: **how many tokens does it take a coding agent to actually
verify a piece of code is correct, as a function of how complex that code's control
flow is?**

This is a small piece of a larger effort (with Kaustubh) to predict token cost for a
programming task given a codebase, a scope, and a problem -- the same shape of decision
you'd make instructing an agent like this one. Rather than asking an LLM to *guess* a
token count up front (an easy but shallow baseline), this repo generates real, measured
cost data by actually running a local model against a controlled ladder of tasks and
reading the true token counts off the wire.

## The core idea

Big-O notation describes how an algorithm's cost scales with input size. This project
asks the same question about an LLM verifying that algorithm: does token cost scale
with the code's structural complexity (recursion depth, branching factor), or with
something else entirely (whether the model can execute code, how large the test input
happens to be, how it's asked to phrase its verification)?

To isolate that, every task in this dataset is a **synthetic, deliberately nonsensical
recursive function** -- not a real algorithm like Fibonacci, but a function with the
same *shape* (linear recursion, binary-tree recursion, triple-branching recursion,
mixed/parity-conditional recursion) built from randomized coefficients. Semantics don't
matter and are intentionally arbitrary; only structure (recursion pattern, branching
factor, base cases, input domain) is controlled. Every function terminates by
construction -- each recursive call strictly decreases `n` toward a base case.

## Two-tier task structure

A single ground-truth function produces two linked, separately-gradeable tasks:

**Worker tier (generation).** A local model (via vLLM) is shown the function and asked
to write a test file for it -- nothing else. It never sees a pre-written test or a
"correct" answer to imitate; it has to decide what "correct" means and prove it. Its
only output is code. Retried up to 4 times, feeding back the failure, until its tests
actually pass against the real function (or it gives up).

**Grading (mechanical, not self-reported).** This is the part that keeps the dataset
honest. "The model's tests pass" is not enough on its own -- a model under cost pressure
could pass by writing a single vacuous assertion. So every generated test file is also
run against 2-3 **seeded mutants** of the ground-truth function (an off-by-one base
case, a flipped operator, a shifted constant -- each verified at generation time to
actually produce different output somewhere in the input domain). A good test suite
fails on the mutants; a vacuous one doesn't notice. That gives a `mutation_score`
alongside plain pass/fail.

**Reviewer tier (adversarial verification).** After a task converges, a second call asks
the model to try to break its own work: construct an alternative, wrong implementation
that would still pass the tests it just wrote. Its claim is never trusted directly --
we mechanically verify the alternative really does differ in behavior from the ground
truth, then actually run the original tests against it. If they still pass, the reviewer
found a real gap in its own test suite (`reviewer_gap_confirmed: true`). If the tests
catch it, the reviewer's suspicion didn't hold up. This tier has its own token cost,
separate from the worker's.

Nothing in either tier is graded by asking a model whether it did a good job.
Everything is graded by execution.

## Two run modes -- the actual experiment

- **`noexec`** -- the worker is a bare chat completion. No code execution. It has to
  hand-compute expected values for a function's behavior, the same way you would if
  you had to review code without a REPL.
- **`exec`** -- the worker gets a `run_python` tool via native OpenAI-style
  tool-calling, so it can check real behavior before committing to an answer.

Both modes use the *exact same task-generation seeds* (`make_task(idx)` is
deterministic), so a `noexec` run and an `exec` run are a paired comparison, not two
unrelated samples. Early results: execution access raises convergence a lot (no more
failures from bad mental arithmetic on large-`n` linear recursion) -- but in a small
pilot, tasks that converged with execution scored much worse on `mutation_score` than
the no-execution tasks did, and the reviewer found a real gap almost every time. Fast
convergence and rigorous verification are not the same thing, and this dataset is built
to tell them apart.

## Repo layout

```
scripts/
  generate_tasks.py   ground-truth function generator (5 recursion templates,
                      randomized coefficients, mutation generation + liveness check)
  harness.py          mechanical grading: run untrusted code in an isolated,
                      timeout-bounded subprocess -- never trust a self-report
  llm_client.py       thin client for the local vLLM OpenAI-compatible server,
                      logs telemetry in the same shape ag23-llm already uses
  run_pipeline.py     the time-boxed loop: generate -> worker -> grade -> mutate
                      -> review, for either mode
```

`data/` and `data_exec/` (gitignored -- regenerate them, don't expect them checked in)
hold, per run: `tasks.jsonl` (ground truth + mutants), `attempts.jsonl` (one row per
LLM call, telemetry-shaped), `tasks_results.jsonl` (rolled-up per-task outcome),
`review.jsonl` (reviewer tier), plus `preview_10rows.csv` and `final_dataset.csv`
flattened exports and a `summary.json`.

## Dataset columns (flattened CSV export)

`task_id`, `execution_enabled`, `function_name`, `recursion_pattern`,
`branching_factor`, `expected_big_o`, `difficulty_tier`, `input_domain`,
`iterations_attempted` (outer retry rounds), `iterations_to_converge`,
`num_llm_calls` (total API calls, including tool round-trips), `num_tool_calls`,
`converged`, `total_tokens`, `mutation_caught`, `mutation_total`, `mutation_score`,
`reviewer_claimed_gap`, `reviewer_valid_claim`, `reviewer_gap_confirmed`,
`reviewer_tokens`.

## Running it

Requires a local OpenAI-compatible model server (this was built and run against a
local vLLM instance serving `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ` on
`localhost:8000`) and Python with `httpx` installed.

```bash
python scripts/run_pipeline.py <budget_seconds>          # noexec mode (default)
python scripts/run_pipeline.py <budget_seconds> exec     # execution-enabled mode
```

Runs until the time budget elapses (or 500 tasks, whichever first), saving a 10-row
preview checkpoint the first time that many tasks complete, and a final export when
the budget runs out.
