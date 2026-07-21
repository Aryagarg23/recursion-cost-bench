"""Main time-boxed loop: generate a recursive-function task, have the local swarm
(vLLM worker) write tests for it, grade mechanically (real function + seeded
mutants), then have the swarm adversarially try to break its own tests (reviewer
side). Saves a 10-row preview checkpoint and a final checkpoint at the time budget.

Two modes, selected by the 2nd CLI arg:
  noexec (default) -- worker is a bare chat completion, no code execution.
  exec             -- worker gets a `run_python` OpenAI-style tool so it can check
                       real behavior instead of hand-computing it. Reviewer stays
                       execution-free in both modes (kept as a control).
Both modes use the exact same task-generation seeds (make_task(idx)) so runs are
a paired comparison, and every result row is tagged execution_enabled so the two
datasets merge cleanly.
"""
import sys, os, time, json, re, csv, traceback
sys.path.insert(0, os.path.dirname(__file__))
from generate_tasks import make_task
from llm_client import call_llm, call_llm_tools
from harness import grade, eval_outputs, run_snippet

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAX_WORKER_ITERS = 4
MAX_TOOL_ROUNDS = 6
PROJECT = "recursion-cost-bench"

RUN_PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Execute a python snippet and return its stdout. The target function is "
            "already defined and callable by name in each snippet. Each call is "
            "independent -- variables do not persist between calls, so make every "
            "snippet self-contained (e.g. include its own print statement)."
        ),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Self-contained python code to run."}},
            "required": ["code"],
        },
    },
}

def log(msg, log_path):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")

def append_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=str) + "\n")

CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

def extract_code(text):
    m = CODE_RE.search(text or "")
    return m.group(1) if m else (text or "")

def worker_prompt(task, prior_failure=None):
    p = (
        "You will be given a Python function. Write a test file that verifies it is "
        "correct. Output ONLY a single python code block containing test functions "
        "named test_*, using plain `assert` statements. No pytest imports, no fixtures, "
        "no classes, no print statements, no explanation text outside the code block. "
        f"The function is importable via: from solution import {task['function_name']}\n\n"
        f"Function source:\n```python\n{task['function_source']}```\n"
    )
    if prior_failure:
        p += f"\nYour previous attempt failed with this result:\n{prior_failure}\nFix the test file and output the corrected version."
    return p

def worker_prompt_exec(task, prior_failure=None):
    p = (
        "You will be given a Python function. Write a test file that verifies it is "
        "correct. You have a run_python tool available -- use it to check the "
        "function's real output on a few inputs before finalizing, rather than "
        "guessing expected values by hand. "
        "When ready, give your FINAL answer as a single python code block containing "
        "ONLY test functions named test_*, using plain `assert` statements (no pytest "
        "imports, no fixtures, no classes, no print statements, no explanation text "
        "outside the code block). Do not call the tool once you are giving your final "
        f"answer.\nThe function is importable via: from solution import {task['function_name']}\n\n"
        f"Function source:\n```python\n{task['function_source']}```\n"
    )
    if prior_failure:
        p += f"\nYour previous final answer failed with this result:\n{prior_failure}\nFix it."
    return p

def run_worker(task):
    """No-execution worker: bare chat completion, retried up to MAX_WORKER_ITERS times."""
    attempts = []
    transcripts = []
    prior_failure = None
    converged = False
    test_code = None
    outer_used = 0
    for i in range(1, MAX_WORKER_ITERS + 1):
        outer_used = i
        prompt = worker_prompt(task, prior_failure)
        resp = call_llm(prompt, role="worker", goal=task["problem_statement"], project=PROJECT, iteration=i)
        attempts.append(resp["telemetry_row"])
        if not resp["ok"]:
            prior_failure = f"LLM call failed: {resp['error']}"
            transcripts.append({"outer_iteration": i, "messages": [{"role": "user", "content": prompt}],
                                 "verdict": {"call_error": resp["error"]}})
            continue
        code = extract_code(resp["text"])
        verdict = grade(task["function_source"], code)
        all_pass = bool(verdict) and all(v == "pass" for v in verdict.values()) and not any(
            k in verdict for k in ("no_tests", "import_error", "timeout", "crash", "parse_error"))
        transcripts.append({"outer_iteration": i,
                             "messages": [{"role": "user", "content": prompt},
                                          {"role": "assistant", "content": resp["text"]}],
                             "verdict": verdict})
        if all_pass:
            converged, test_code = True, code
            break
        prior_failure = json.dumps(verdict)[:1500]
    return attempts, converged, test_code, 0, outer_used, transcripts

def run_worker_exec(task):
    """Execution-enabled worker: gets a run_python tool via native OpenAI-style
    tool-calling, in a multi-turn loop, still wrapped in the same outer retry-on-
    failure structure as run_worker."""
    attempts = []
    prior_failure = None
    converged = False
    test_code = None
    tool_calls_used = 0
    outer_used = 0
    transcripts = []
    for outer in range(1, MAX_WORKER_ITERS + 1):
        outer_used = outer
        messages = [{"role": "user", "content": worker_prompt_exec(task, prior_failure)}]
        final_text = None
        call_failed = False
        for _round in range(MAX_TOOL_ROUNDS):
            resp = call_llm_tools(messages, role="worker", goal=task["problem_statement"],
                                   project=PROJECT, iteration=outer, tools=[RUN_PYTHON_TOOL])
            attempts.append(resp["telemetry_row"])
            if not resp["ok"] or resp["message"] is None:
                call_failed = True
                break
            msg = resp["message"]
            messages.append(msg)
            tcs = msg.get("tool_calls") or []
            if tcs:
                tool_calls_used += len(tcs)
                for tc in tcs:
                    try:
                        args = json.loads(tc["function"]["arguments"] or "{}")
                    except Exception:
                        args = {}
                    output = run_snippet(task["function_source"], args.get("code", ""))
                    messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": output})
                continue
            final_text = msg.get("content") or ""
            break
        if call_failed:
            prior_failure = f"LLM call failed: {resp.get('error')}"
            transcripts.append({"outer_iteration": outer, "messages": messages,
                                 "verdict": {"call_error": resp.get("error")}})
            continue
        if final_text is None:
            prior_failure = "exhausted tool-call rounds without a final answer"
            transcripts.append({"outer_iteration": outer, "messages": messages,
                                 "verdict": {"exhausted_tool_rounds": True}})
            continue
        code = extract_code(final_text)
        verdict = grade(task["function_source"], code)
        all_pass = bool(verdict) and all(v == "pass" for v in verdict.values()) and not any(
            k in verdict for k in ("no_tests", "import_error", "timeout", "crash", "parse_error"))
        transcripts.append({"outer_iteration": outer, "messages": messages, "verdict": verdict})
        if all_pass:
            converged, test_code = True, code
            break
        prior_failure = json.dumps(verdict)[:1500]
    return attempts, converged, test_code, tool_calls_used, outer_used, transcripts

def run_mutation(task, test_code):
    caught, total, details = 0, len(task["mutants"]), []
    for m in task["mutants"]:
        v = grade(m["source"], test_code)
        is_diag = any(k in v for k in ("no_tests", "import_error", "timeout", "crash", "parse_error"))
        did_catch = (not is_diag) and any(val != "pass" for val in v.values())
        caught += 1 if did_catch else 0
        details.append({"mutant_id": m["id"], "caught": did_catch})
    return caught, total, details

def run_reviewer(task, test_code, mutation_caught, mutation_total):
    """Reviewer stays execution-free in both modes -- a control so only the worker
    side of the comparison changes."""
    prompt = (
        f"Here is a Python function and a test file someone wrote for it.\n\n"
        f"Function:\n```python\n{task['function_source']}```\n\n"
        f"Tests:\n```python\n{test_code}```\n\n"
        f"Mechanical check: these tests pass on the real function above, and catch "
        f"{mutation_caught}/{mutation_total} of a set of seeded bugs.\n\n"
        f"Your job: try to write ONE alternative, WRONG implementation of "
        f"{task['function_name']} (same name and signature) that behaves differently "
        f"from the correct one on at least one input, but would still PASS all the "
        f"given tests. Output ONLY the alternative function as a single python code "
        f"block, or output exactly NONE if you can't find one. No explanation."
    )
    resp = call_llm(prompt, role="reviewer", goal=task["problem_statement"], project=PROJECT,
                     iteration=1, temperature=0.5)
    text = (resp["text"] or "").strip()
    code = extract_code(text)
    claimed = bool(code.strip()) and "NONE" not in text.upper()[:20]
    valid_claim, gap_confirmed = False, False
    if claimed:
        n_max = int(task["input_domain"].split("-")[-1])
        inputs = list(range(0, n_max + 1))
        real_out = eval_outputs(task["function_source"], task["function_name"], inputs)
        alt_out = eval_outputs(code, task["function_name"], inputs)
        if alt_out is not None and real_out is not None and alt_out != real_out:
            valid_claim = True
            v = grade(code, test_code)
            is_diag = any(k in v for k in ("no_tests", "import_error", "timeout", "crash", "parse_error"))
            gap_confirmed = (not is_diag) and bool(v) and all(val == "pass" for val in v.values())
    return {
        "task_id": task["task_id"], "reviewer_tokens": resp["telemetry_row"]["total_tokens"],
        "reviewer_claimed_gap": claimed, "reviewer_valid_claim": valid_claim,
        "reviewer_gap_confirmed": gap_confirmed,
        "alt_code": code if claimed else None,
        "raw_reviewer_text": text,
    }

CSV_FIELDS = ["task_id", "execution_enabled", "function_name", "recursion_pattern", "branching_factor",
              "expected_big_o", "difficulty_tier", "input_domain", "iterations_attempted",
              "iterations_to_converge", "num_llm_calls", "num_tool_calls", "converged", "total_tokens",
              "mutation_caught", "mutation_total", "mutation_score", "reviewer_claimed_gap",
              "reviewer_valid_claim", "reviewer_gap_confirmed", "reviewer_tokens"]

def export_csv(results, reviews, tasks_by_id, path):
    reviews_by_task = {r["task_id"]: r for r in reviews}
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in results:
            t = tasks_by_id.get(r["task_id"], {})
            rev = reviews_by_task.get(r["task_id"], {})
            w.writerow({
                "task_id": r["task_id"], "execution_enabled": r.get("execution_enabled"),
                "function_name": t.get("function_name"), "recursion_pattern": t.get("recursion_pattern"),
                "branching_factor": t.get("branching_factor"), "expected_big_o": t.get("expected_big_o"),
                "difficulty_tier": t.get("difficulty_tier"), "input_domain": t.get("input_domain"),
                "iterations_attempted": r.get("iterations_attempted"),
                "iterations_to_converge": r.get("iterations_to_converge"),
                "num_llm_calls": r.get("num_llm_calls"),
                "num_tool_calls": r.get("num_tool_calls"),
                "converged": r.get("converged"), "total_tokens": r.get("total_tokens"),
                "mutation_caught": r.get("mutation_caught"), "mutation_total": r.get("mutation_total"),
                "mutation_score": r.get("mutation_score"),
                "reviewer_claimed_gap": rev.get("reviewer_claimed_gap"),
                "reviewer_valid_claim": rev.get("reviewer_valid_claim"),
                "reviewer_gap_confirmed": rev.get("reviewer_gap_confirmed"),
                "reviewer_tokens": rev.get("reviewer_tokens"),
            })

def main():
    budget_seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 1800
    mode = sys.argv[2] if len(sys.argv) > 2 else "noexec"
    execution_enabled = (mode == "exec")
    data_dir = os.path.join(REPO, "data_exec" if execution_enabled else "data")
    log_dir = os.path.join(REPO, "logs_exec" if execution_enabled else "logs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    import llm_client
    llm_client.DATA_DIR = data_dir
    llm_client.TELEMETRY_PATH = os.path.join(data_dir, "telemetry.jsonl")

    tasks_path = os.path.join(data_dir, "tasks.jsonl")
    attempts_path = os.path.join(data_dir, "attempts.jsonl")
    results_path = os.path.join(data_dir, "tasks_results.jsonl")
    review_path = os.path.join(data_dir, "review.jsonl")
    transcripts_path = os.path.join(data_dir, "transcripts.jsonl")
    log_path = os.path.join(log_dir, "run.log")

    worker_fn = run_worker_exec if execution_enabled else run_worker
    start = time.time()
    deadline = start + budget_seconds
    idx = 0
    completed, reviews, tasks_by_id = [], [], {}
    preview_saved = False
    log(f"=== run_pipeline starting, budget={budget_seconds}s mode={mode} ===", log_path)
    while time.time() < deadline and idx < 500:
        idx += 1
        try:
            task = make_task(idx)
        except Exception as e:
            log(f"task-gen failed at idx={idx}: {e!r}", log_path)
            continue
        tasks_by_id[task["task_id"]] = task
        append_jsonl(tasks_path, task)
        log(f"{task['task_id']} ({task['recursion_pattern']}, {task['expected_big_o']}, "
            f"{task['difficulty_tier']}) starting", log_path)
        try:
            attempts, converged, test_code, num_tool_calls, outer_used, transcripts = worker_fn(task)
        except Exception as e:
            log(f"{task['task_id']} worker crashed: {e!r}\n{traceback.format_exc()[-800:]}", log_path)
            continue
        for a in attempts:
            append_jsonl(attempts_path, a)
        for t in transcripts:
            append_jsonl(transcripts_path, {"task_id": task["task_id"], "execution_enabled": execution_enabled, **t})
        total_tokens = sum(a.get("total_tokens", 0) for a in attempts)
        result = {
            "task_id": task["task_id"], "execution_enabled": execution_enabled,
            "iterations_attempted": outer_used,
            "iterations_to_converge": outer_used if converged else None,
            "num_llm_calls": len(attempts),
            "num_tool_calls": num_tool_calls, "total_tokens": total_tokens, "converged": converged,
            "final_test_code": test_code,
        }
        if converged:
            try:
                caught, total, details = run_mutation(task, test_code)
                result["mutation_caught"], result["mutation_total"] = caught, total
                result["mutation_score"] = f"{caught}/{total}"
                review = run_reviewer(task, test_code, caught, total)
                reviews.append(review)
                append_jsonl(review_path, review)
            except Exception as e:
                log(f"{task['task_id']} mutation/review crashed: {e!r}", log_path)
        append_jsonl(results_path, result)
        completed.append(result)
        log(f"{task['task_id']} done: converged={converged} iters={len(attempts)} "
            f"tool_calls={num_tool_calls} tokens={total_tokens}", log_path)
        if not preview_saved and len(completed) >= 10:
            export_csv(completed[:10], reviews, tasks_by_id, os.path.join(data_dir, "preview_10rows.csv"))
            log(f"PREVIEW_SAVED: {data_dir}/preview_10rows.csv (10 rows)", log_path)
            preview_saved = True
    export_csv(completed, reviews, tasks_by_id, os.path.join(data_dir, "final_dataset.csv"))
    summary = {
        "mode": mode, "execution_enabled": execution_enabled,
        "total_tasks": len(completed), "converged_tasks": sum(1 for r in completed if r["converged"]),
        "total_tokens": sum(r["total_tokens"] for r in completed),
        "total_tool_calls": sum(r.get("num_tool_calls", 0) for r in completed),
        "wall_clock_s": round(time.time() - start, 1), "budget_seconds": budget_seconds,
    }
    with open(os.path.join(data_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"RUN_COMPLETE: {json.dumps(summary)}", log_path)

if __name__ == "__main__":
    main()
