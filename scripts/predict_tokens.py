"""Ask the local swarm model to *guess* how many tokens the worker-tier task would
cost, given only the same information the worker itself saw (the function source +
the task framing) -- no access to what actually happened. Logged separately from the
real run so it can be compared against measured total_tokens without contaminating it.
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

INT_RE = re.compile(r"-?\d[\d,]*")

def predict_prompt(task):
    return (
        "An AI coding agent is about to be given the Python function below and asked "
        "to write a test file that verifies it is correct (plain `assert`-based tests, "
        "no pytest fixtures). It will be retried up to 4 times if its tests don't "
        "actually pass against the real function, each retry seeing the previous "
        "failure.\n\n"
        f"Function:\n```python\n{task['function_source']}```\n\n"
        "Estimate the TOTAL number of tokens (prompt + completion combined, summed "
        "across every attempt/retry it takes) this will cost. Output ONLY a single "
        "integer. No explanation, no words, no punctuation other than the digits."
    )

def predict_one(task, idx, total):
    prompt = predict_prompt(task)
    resp = call_llm(prompt, role="predictor", goal=task["problem_statement"],
                     project="recursion-cost-bench", iteration=1, max_tokens=40, temperature=0.4)
    text = (resp["text"] or "").strip()
    m = INT_RE.search(text.replace(",", ""))
    predicted = int(m.group().replace(",", "")) if m else None
    print(f"[{idx}/{total}] {task['task_id']} predicted={predicted!r} raw={text[:60]!r}", flush=True)
    return {
        "task_id": task["task_id"],
        "predicted_tokens": predicted,
        "raw_response": text[:300],
        "prediction_cost_tokens": resp["telemetry_row"]["total_tokens"],
        "ok": resp["ok"],
    }

def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "data")
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    tasks = load_jsonl(os.path.join(data_dir, "tasks.jsonl"))
    # de-dupe: tasks.jsonl can have retried idx offsets (make_task's own retry-on-dead-mutant
    # path), but task_id is the true unique key -- keep first occurrence.
    seen = {}
    for t in tasks:
        seen.setdefault(t["task_id"], t)
    tasks = list(seen.values())
    if limit:
        tasks = tasks[:limit]
    out_path = os.path.join(data_dir, "predictions.jsonl")
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
    print("PREDICTIONS_COMPLETE", flush=True)

if __name__ == "__main__":
    main()
