"""Generate a genuinely fresh, never-before-seen batch of tasks (different idx
range, so different random coefficients/n_max draws -- zero overlap with the
original 256-task dataset) and run them through the SAME no-execution worker
pipeline as scripts/run_pipeline.py, to sanity-check that
scripts/baseline_regression.py's r~0.68 finding holds on data the model/features
were never fit against, not just on a random split of the same 256 rows.

Skips mutation testing and the reviewer tier on purpose -- this script only needs
task features + actual_total_tokens (the worker's real cost) to test regression
generalization, not the full rigor-auditing pipeline.

Resume-safe: checks data_holdout/tasks_results.jsonl for already-done task_ids
before each task, so it's safe to call repeatedly in short time-boxed chunks
(this env's tool calls have an effective ~50-60s ceiling; background nohup jobs
were observed to die silently mid-session, so foreground chunking is the
reliable pattern here).

Usage: python3 scripts/generate_holdout.py [start_idx] [count] [budget_seconds]
  start_idx default 10000 (guaranteed non-overlapping -- original run only used
  idx 1-256), count default 80, budget_seconds default 45.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
from generate_tasks import make_task
from run_pipeline import run_worker

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def append_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=str) + "\n")

def main():
    start_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    budget_seconds = int(sys.argv[3]) if len(sys.argv) > 3 else 45
    data_dir = os.path.join(REPO, "data_holdout")
    os.makedirs(data_dir, exist_ok=True)
    tasks_path = os.path.join(data_dir, "tasks.jsonl")
    results_path = os.path.join(data_dir, "tasks_results.jsonl")

    import llm_client
    llm_client.DATA_DIR = data_dir
    llm_client.TELEMETRY_PATH = os.path.join(data_dir, "telemetry.jsonl")

    done_ids = set(r["task_id"] for r in load_jsonl(results_path))
    print(f"already done: {len(done_ids)}", flush=True)

    deadline = time.time() + budget_seconds
    n_done_this_run = 0
    for i in range(count):
        idx = start_idx + i
        if time.time() > deadline:
            print("time budget hit, stopping this invocation", flush=True)
            break
        try:
            task = make_task(idx)
        except Exception as e:
            print(f"idx={idx} task-gen failed: {e!r}", flush=True)
            continue
        if task["task_id"] in done_ids:
            continue
        attempts, converged, test_code, _num_tool_calls, outer_used, _transcripts = run_worker(task)
        total_tokens = sum(a.get("total_tokens", 0) for a in attempts)
        append_jsonl(tasks_path, task)
        result = {
            "task_id": task["task_id"], "recursion_pattern": task["recursion_pattern"],
            "branching_factor": task["branching_factor"], "expected_big_o": task["expected_big_o"],
            "difficulty_tier": task["difficulty_tier"], "input_domain": task["input_domain"],
            "converged": converged, "iterations_attempted": outer_used, "total_tokens": total_tokens,
        }
        append_jsonl(results_path, result)
        done_ids.add(task["task_id"])
        n_done_this_run += 1
        print(f"{task['task_id']} converged={converged} iters={outer_used} tokens={total_tokens}", flush=True)
    print(f"DONE this invocation: {n_done_this_run} new tasks, total so far: {len(done_ids)}", flush=True)

if __name__ == "__main__":
    main()
