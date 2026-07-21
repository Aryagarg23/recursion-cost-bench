"""v2 of the blind-guess experiment: give the local swarm model the SAME 4
structural features the tabular baseline uses (recursion_pattern, branching_factor,
expected_big_o, input_domain/n_max) PLUS room to reason step-by-step before
answering, instead of the v1 forced bare-integer/no-context/no-reasoning setup
(scripts/predict_tokens.py). Still no execution, still no access to the real
outcome -- this is the fair rematch: same information + same reasoning latitude
as scripts/baseline_regression.py's tabular model, to see whether an LLM given a
fair shot can compete with r~0.68, or whether v1's near-zero r was really about
elicitation and not a caclulation deep-fault.

Run from repo root: python3 scripts/predict_tokens_v2.py [data_dir] [limit]
Writes data/predictions_v2.jsonl (same resume-safe append pattern as v1).
"""
import sys, os, re, json, time
sys.path.insert(0, os.path.dirname(__file__))
from llm_client import call_llm

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

FINAL_RE = re.compile(r"FINAL:\s*(-?\d[\d,]*)")

def predict_prompt(task):
    return (
        "An AI coding agent is about to be given the Python function below and asked "
        "to write a test file that verifies it is correct (plain `assert`-based tests, "
        "no pytest fixtures, no code execution available -- it must hand-compute "
        "expected values). It will be retried up to 4 times if its tests don't "
        "actually pass against the real function, each retry seeing the previous "
        "failure as feedback.\n\n"
        f"Function:\n```python\n{task['function_source']}```\n\n"
        f"Recursion pattern: {task['recursion_pattern']} (branching factor {task['branching_factor']})\n"
        f"Expected complexity: {task['expected_big_o']}\n"
        f"The function will realistically be tested on inputs in the range {task['input_domain']}.\n\n"
        "Think step by step about how error-prone hand-computing the correct expected "
        "values would be for this function at realistic test inputs, and therefore how "
        "many of the 4 retries it will likely need. Keep your reasoning concise (a few "
        "sentences), then end your answer with a line of the exact form:\nFINAL: <integer>\n"
        "where <integer> is your estimate of TOTAL tokens (prompt+completion, summed "
        "across every retry) this will cost."
    )

def predict_one(task, idx, total):
    prompt = predict_prompt(task)
    resp = call_llm(prompt, role="predictor_v2", goal=task["problem_statement"],
                     project="recursion-cost-bench", iteration=1, max_tokens=1024, temperature=0.4)
    text = (resp["text"] or "").strip()
    m = FINAL_RE.search(text.replace(",", ""))
    predicted = int(m.group(1).replace(",", "")) if m else None
    finished = predicted is not None
    print(f"[{idx}/{total}] {task['task_id']} predicted={predicted!r} finished={finished} "
          f"resp_tokens={resp['telemetry_row']['total_tokens']}", flush=True)
    return {
        "task_id": task["task_id"],
        "predicted_tokens": predicted,
        "finished": finished,
        "raw_response": text[:800],
        "prediction_cost_tokens": resp["telemetry_row"]["total_tokens"],
        "ok": resp["ok"],
    }

def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "data")
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    tasks = load_jsonl(os.path.join(data_dir, "tasks.jsonl"))
    seen = {}
    for t in tasks:
        seen.setdefault(t["task_id"], t)
    tasks = list(seen.values())
    if limit:
        tasks = tasks[:limit]
    out_path = os.path.join(data_dir, "predictions_v2.jsonl")
    done_ids = set()
    if os.path.exists(out_path):
        for row in load_jsonl(out_path):
            done_ids.add(row["task_id"])
    remaining = [t for t in tasks if t["task_id"] not in done_ids]
    print(f"total={len(tasks)} already_done={len(done_ids)} remaining={len(remaining)}", flush=True)
    with open(out_path, "a") as f:
        for i, task in enumerate(remaining, 1):
            row = predict_one(task, i, len(remaining))
            f.write(json.dumps(row) + "\n")
            f.flush()
    print("PREDICTIONS_V2_COMPLETE", flush=True)

if __name__ == "__main__":
    main()
