"""Ground-truth recursive function generator. Every function is deterministic and
terminates by construction (each recursive call strictly decreases n toward a base
case). Semantics are arbitrary/nonsensical on purpose -- only structural properties
(recursion pattern, branching factor, depth) are controlled, so difficulty tracks
Big-O shape, not "is this a real algorithm."""
import random
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from harness import eval_outputs

TEMPLATES = ["linear", "tail", "binary", "triple", "mixed"]

N_MAX_BY_TEMPLATE = {
    "linear": [50, 120, 250],
    "tail":   [50, 120, 250],
    "binary": [10, 14, 18],
    "triple": [6, 8, 9],
    "mixed":  [10, 16, 20],
}

def _coeffs(rng):
    return {
        "a": rng.randint(2, 9), "b": rng.randint(1, 5),
        "c0": rng.randint(0, 5), "c1": rng.randint(1, 6), "c2": rng.randint(1, 6),
    }

def render(template, fn, rng):
    c = _coeffs(rng)
    if template == "linear":
        src = f"def {fn}(n):\n    if n <= 0:\n        return {c['c0']}\n    return {fn}(n - 1) + n * {c['a']} + {c['b']}\n"
        mutants = [
            ("base_off_by_one", src.replace("n <= 0", "n < 0")),
            ("op_flip", src.replace(f"+ n * {c['a']}", f"- n * {c['a']}")),
            ("const_shift", src.replace(f"return {c['c0']}\n", f"return {c['c0']+1}\n")),
        ]
        return src, mutants, "linear", 1, 1, "O(n)"
    if template == "tail":
        src = f"def {fn}(n, acc={c['c0']}):\n    if n <= 0:\n        return acc\n    return {fn}(n - 1, acc + n * {c['a']})\n"
        mutants = [
            ("base_off_by_one", src.replace("n <= 0", "n < 0")),
            ("op_flip", src.replace(f"acc + n * {c['a']}", f"acc - n * {c['a']}")),
            ("const_shift", src.replace(f"acc={c['c0']}", f"acc={c['c0']+1}")),
        ]
        return src, mutants, "tail", 1, 1, "O(n)"
    if template == "binary":
        src = (f"def {fn}(n):\n    if n <= 0:\n        return {c['c0']}\n    if n == 1:\n        return {c['c1']}\n"
               f"    return {fn}(n - 1) + {fn}(n - 2)\n")
        mutants = [
            ("base_shift", src.replace("n == 1", "n == 2")),
            ("op_flip", src.replace(f"{fn}(n - 1) + {fn}(n - 2)", f"{fn}(n - 1) - {fn}(n - 2)")),
            ("const_shift", src.replace(f"return {c['c1']}\n", f"return {c['c1']+1}\n")),
        ]
        return src, mutants, "binary-tree", 2, 2, "O(phi^n)"
    if template == "triple":
        src = (f"def {fn}(n):\n    if n <= 0:\n        return {c['c0']}\n    if n == 1:\n        return {c['c1']}\n"
               f"    if n == 2:\n        return {c['c2']}\n    return {fn}(n - 1) + {fn}(n - 2) + {fn}(n - 3)\n")
        mutants = [
            ("base_shift", src.replace("n == 2", "n == 3")),
            ("op_flip", src.replace(f"{fn}(n - 2) + {fn}(n - 3)", f"{fn}(n - 2) - {fn}(n - 3)")),
            ("const_shift", src.replace(f"return {c['c2']}\n", f"return {c['c2']+1}\n")),
        ]
        return src, mutants, "triple-branch", 3, 3, "O(3^n)-ish"
    # mixed
    src = (f"def {fn}(n):\n    if n <= 0:\n        return {c['c0']}\n    if n == 1:\n        return {c['c1']}\n"
           f"    if n % 2 == 0:\n        return {fn}(n - 1) - {fn}(n - 2)\n    return {fn}(n - 1) + n\n")
    mutants = [
        ("parity_flip", src.replace("n % 2 == 0", "n % 2 == 1")),
        ("op_flip", src.replace(f"{fn}(n - 1) - {fn}(n - 2)", f"{fn}(n - 1) + {fn}(n - 2)")),
        ("odd_branch_flip", src.replace(f"{fn}(n - 1) + n", f"{fn}(n - 1) - n")),
    ]
    return src, mutants, "mixed-parity", 2, 2, "O(phi^n) worst-case"

def make_task(idx, _depth=0):
    if _depth > 5:
        raise RuntimeError(f"could not generate a live task near idx={idx}")
    rng = random.Random(1000 + idx)
    template = TEMPLATES[idx % len(TEMPLATES)]
    n_max = rng.choice(N_MAX_BY_TEMPLATE[template])
    fn_name = f"f_{idx:05d}"
    src, mutant_specs, pattern, branching, base_cases, big_o = render(template, fn_name, rng)
    inputs = list(range(0, n_max + 1))
    ref = eval_outputs(src, fn_name, inputs)
    if ref is None:
        return make_task(idx + 1000, _depth + 1)
    live_mutants = []
    for mid, msrc in mutant_specs:
        mref = eval_outputs(msrc, fn_name, inputs)
        if mref is not None and mref != ref:
            live_mutants.append({"id": mid, "source": msrc})
    if not live_mutants:
        return make_task(idx + 2000, _depth + 1)
    if branching >= 3:
        difficulty = "high"
    elif branching == 2:
        difficulty = "high" if n_max >= 16 else "medium"
    else:  # branching == 1 (linear/tail) -- difficulty here comes from having to
        # mentally trace many steps of arithmetic with no code execution available
        difficulty = "low" if n_max <= 20 else ("medium" if n_max <= 100 else "high")
    problem_statement = (
        f"Write tests for the Python function `{fn_name}` below. Tests must "
        f"meaningfully exercise its recursive behavior, not just call it once."
    )
    return {
        "task_id": f"rf-{idx:05d}",
        "function_name": fn_name,
        "function_source": src,
        "recursion_pattern": pattern,
        "branching_factor": branching,
        "base_case_count": base_cases,
        "input_domain": f"n: int, 0-{n_max}",
        "expected_big_o": big_o,
        "difficulty_tier": difficulty,
        "problem_statement": problem_statement,
        "mutants": live_mutants,
    }

if __name__ == "__main__":
    import json
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    for i in range(n):
        t = make_task(i)
        print(json.dumps({k: v for k, v in t.items() if k != "mutants"} | {"num_mutants": len(t["mutants"])}))
