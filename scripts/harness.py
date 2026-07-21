"""Mechanical grading: run untrusted (LLM-authored) code in an isolated subprocess
with a timeout. Never trust a self-report -- everything here is graded by execution.
"""
import subprocess, sys, tempfile, os, json

DRIVER = '''
import json, sys
try:
    import test_module
except Exception as e:
    print(json.dumps({"import_error": repr(e)}))
    sys.exit(0)
fns = [(n, getattr(test_module, n)) for n in dir(test_module) if n.startswith("test_") and callable(getattr(test_module, n))]
if not fns:
    print(json.dumps({"no_tests": True}))
    sys.exit(0)
results = {}
for name, fn in fns:
    try:
        fn()
        results[name] = "pass"
    except Exception as e:
        results[name] = "fail:" + repr(e)
print(json.dumps(results))
'''

def grade(solution_source: str, test_source: str, timeout=10):
    """Run `test_source` (must define test_* functions using `from solution import X`)
    against `solution_source`. Returns a dict of {test_name: "pass"|"fail:..."} or a
    diagnostic dict ({"no_tests": True} / {"import_error": ...} / {"timeout": True})."""
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "solution.py"), "w") as f:
            f.write(solution_source)
        with open(os.path.join(d, "test_module.py"), "w") as f:
            f.write(test_source or "")
        with open(os.path.join(d, "driver.py"), "w") as f:
            f.write(DRIVER)
        try:
            p = subprocess.run([sys.executable, "driver.py"], cwd=d,
                                capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"timeout": True}
        out = p.stdout.strip().splitlines()
        if not out:
            return {"crash": True, "stderr": p.stderr[-1500:]}
        try:
            return json.loads(out[-1])
        except Exception:
            return {"parse_error": True, "stdout": p.stdout[-1500:], "stderr": p.stderr[-1500:]}

def eval_outputs(source: str, func_name: str, inputs, timeout=15):
    """Run `source` in an isolated subprocess, call func_name(n) for each n in inputs,
    return the list of repr(result) strings, or None if the source itself crashes."""
    calls = "\n".join(
        f'try:\n    _r.append(repr({func_name}({n})))\nexcept Exception as e:\n    _r.append("EXC:" + repr(e))'
        for n in inputs
    )
    script = source + f"\n_r = []\n{calls}\nimport json\nprint(json.dumps(_r))\n"
    try:
        p = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=timeout)
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return None

def run_snippet(function_source: str, snippet: str, timeout=10):
    """Execute `function_source` followed by an arbitrary `snippet` in one isolated
    subprocess (fresh each call -- no state persists between calls). Used as the
    run_python tool's implementation: lets a model check real behavior instead of
    hand-computing it. Returns captured stdout (plus a stderr tail on error)."""
    script = function_source + "\n" + (snippet or "") + "\n"
    try:
        p = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=timeout)
        out = p.stdout
        if p.returncode != 0:
            out += "\n[stderr]\n" + p.stderr[-1000:]
        out = out.strip()
        return out[-3000:] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "[error] execution timed out"
    except Exception as e:
        return f"[error] {e!r}"
