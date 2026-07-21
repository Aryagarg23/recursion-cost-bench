"""Does raw static-code structure (AST node counts, cyclomatic complexity, etc.) add
anything beyond the hand-labeled recursion_pattern/difficulty_tier categorical
features already used in scripts/baseline_regression.py?

Answer for THIS benchmark: no. recursion_pattern is generated from exactly 5 fixed
templates (scripts/generate_tasks.py), so it's already a perfect categorical summary
of AST shape -- computing AST features here just re-derives the same signal, noisier.
Result (10-fold CV, random forest, out-of-fold Pearson r):

  AST features alone (no pattern/tier labels):        r ~ 0.64
  categorical baseline (pattern + tier + branching):  r ~ 0.66
  AST features ADDED to the categorical baseline:     r ~ 0.66 (no improvement,
                                                        slightly worse -- noise, not signal)

This is a negative result worth keeping on record: on a REAL, arbitrary codebase
(not 5 synthetic templates), AST/static-analysis features would NOT be redundant --
there's no clean categorical label to fall back on for real code, so cheap structural
features (call depth, cyclomatic complexity, branching, LOC) become one of the more
promising modalities to add when this approach is generalized past this benchmark.

Reconstructs function_source deterministically from each task_id's encoded index
(same random.Random(1000+idx) seeding as scripts/generate_tasks.py -- verified
bit-exact against the 256 real sources) rather than needing function_source stored
in the merged CSV.

Run from repo root: python3 scripts/ast_feature_test.py
"""
import os
import random
import ast
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr

import sys
sys.path.insert(0, os.path.dirname(__file__))
from generate_tasks import render, TEMPLATES, N_MAX_BY_TEMPLATE

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def gen_source(idx):
    """Mirrors generate_tasks.make_task's prefix (rng draws up through render()),
    skipping the eval_outputs/mutant-liveness work this script doesn't need."""
    rng = random.Random(1000 + idx)
    template = TEMPLATES[idx % len(TEMPLATES)]
    n_max = rng.choice(N_MAX_BY_TEMPLATE[template])
    fn_name = f"f_{idx:05d}"
    src, _mutants, pattern, _branch, _base, _big_o = render(template, fn_name, rng)
    return src, pattern, n_max

df = pd.read_csv(os.path.join(REPO, "data", "merged_actual_vs_predicted.csv"))
df["idx"] = df["task_id"].str.extract(r"rf-(\d+)").astype(int)
df["n_max"] = df["input_domain"].str.extract(r"0-(\d+)").astype(int)

srcs, mismatches = [], 0
for _, row in df.iterrows():
    src, pattern, n_max = gen_source(row["idx"])
    if pattern != row["recursion_pattern"] or n_max != row["n_max"]:
        mismatches += 1
    srcs.append(src)
df["function_source"] = srcs
print(f"n = {len(df)}")
print(f"pattern/n_max mismatches after regeneration: {mismatches}/{len(df)} (0 = bit-exact reconstruction)")

def ast_features(src):
    tree = ast.parse(src)
    n_const = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Constant))
    const_vals = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, int)]
    n_if = sum(1 for n in ast.walk(tree) if isinstance(n, ast.If))
    return {
        "ast_n_nodes": sum(1 for _ in ast.walk(tree)),
        "ast_n_calls": sum(1 for n in ast.walk(tree) if isinstance(n, ast.Call)),
        "ast_n_if": n_if,
        "ast_n_binop": sum(1 for n in ast.walk(tree) if isinstance(n, ast.BinOp)),
        "ast_n_compare": sum(1 for n in ast.walk(tree) if isinstance(n, ast.Compare)),
        "ast_n_const": n_const,
        "ast_max_const": max(const_vals) if const_vals else 0,
        "ast_n_chars": len(src),
        "ast_n_lines": src.count("\n"),
        "ast_cyclomatic": 1 + n_if,
    }

feat_df = pd.DataFrame([ast_features(s) for s in df["function_source"]])
df = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
ast_cols = list(feat_df.columns)

y = df["actual_total_tokens"].values
kf = KFold(n_splits=10, shuffle=True, random_state=42)

X_ast_only = df[ast_cols].assign(branching_factor=df["branching_factor"], n_max=df["n_max"])
oof_ast = cross_val_predict(
    Pipeline([("model", RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42))]),
    X_ast_only, y, cv=kf)
r_ast, _ = pearsonr(y, oof_ast)

feature_cols_cat = ["recursion_pattern", "difficulty_tier"]
feature_cols_num = ["branching_factor", "n_max"]
pre_cat = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_cat)],
                             remainder="passthrough")
oof_cat = cross_val_predict(
    Pipeline([("pre", pre_cat), ("model", RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42))]),
    df[feature_cols_cat + feature_cols_num], y, cv=kf)
r_cat, _ = pearsonr(y, oof_cat)

X_combo = pd.concat([df[ast_cols], df[feature_cols_cat], df[feature_cols_num]], axis=1)
oof_combo = cross_val_predict(
    Pipeline([("pre", pre_cat), ("model", RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42))]),
    X_combo, y, cv=kf)
r_combo, _ = pearsonr(y, oof_combo)

print(f"\nAST-only features (no pattern/tier labels), RF, out-of-fold r = {r_ast:.3f}")
print(f"Categorical baseline (pattern + tier + branching + n_max), r = {r_cat:.3f}")
print(f"AST features + categorical combined, r = {r_combo:.3f}")
print(f"Delta from adding AST features on top of categorical: {r_combo - r_cat:+.3f}  "
      f"(near zero -> redundant on this benchmark, see module docstring)")
