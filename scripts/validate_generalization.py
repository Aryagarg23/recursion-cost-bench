"""Two independent checks that scripts/baseline_regression.py's r~0.68 isn't an
artifact of small-n overfitting or leaky cross-validation:

1. Explicit 80/20 train/test split (not just 10-fold CV) -- fit only on 80% of
   the original 256 tasks, score only on the untouched 20%, so there is a single
   unambiguous held-out set rather than a rotation.
2. A genuinely FRESH holdout: fit on ALL 256 original tasks, then score against
   data_holdout/ -- a separate batch of 75 tasks generated later with a
   different idx range (10000+, guaranteed zero overlap) via
   scripts/generate_holdout.py, run through the real no-execution worker
   pipeline (scripts/run_pipeline.py's run_worker), same as the original data.
   This is the strongest test available: different random coefficients, a
   different run, a different point in time -- if the finding survives this,
   it's not fit to quirks of the original 256 rows.

Run from repo root: python3 scripts/validate_generalization.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr, spearmanr

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_features(df):
    df = df.copy()
    df["n_max"] = df["input_domain"].str.extract(r"0-(\d+)").astype(int)
    return df

feature_cols_cat = ["recursion_pattern", "difficulty_tier"]
feature_cols_num = ["branching_factor", "n_max"]
pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_cat)],
                         remainder="passthrough")

# ---- Check 1: explicit 80/20 split on the original 256 ----
orig = load_features(pd.read_csv(os.path.join(REPO, "data", "merged_actual_vs_predicted.csv")))
X = orig[feature_cols_cat + feature_cols_num]
y = orig["actual_total_tokens"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"[Check 1] explicit 80/20 split: train n={len(X_train)}  test n={len(X_test)} "
      f"(held out, zero overlap, never touched during fit)")
for name, model in [("linear_regression", LinearRegression()),
                     ("random_forest", RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42))]:
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)
    r_train, _ = pearsonr(y_train, pipe.predict(X_train))
    r_test, _ = pearsonr(y_test, pipe.predict(X_test))
    print(f"  {name:20s}  train r={r_train:.3f}  HELD-OUT TEST r={r_test:.3f}")

# ---- Check 2: fresh, never-before-seen holdout batch ----
holdout_path = os.path.join(REPO, "data_holdout", "holdout_merged.csv")
if not os.path.exists(holdout_path):
    print(f"\n[Check 2] skipped -- {holdout_path} not found "
          f"(run scripts/generate_holdout.py first)")
else:
    fresh = load_features(pd.read_csv(holdout_path))
    X_fresh = fresh[feature_cols_cat + feature_cols_num]
    y_fresh = fresh["actual_total_tokens"]
    print(f"\n[Check 2] fresh holdout: train on ALL {len(orig)} original tasks, "
          f"test on {len(fresh)} brand-new tasks (different idx range, different run)")
    for name, model in [("linear_regression", LinearRegression()),
                         ("random_forest", RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42))]:
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(X, y)
        pred = pipe.predict(X_fresh)
        r, p = pearsonr(y_fresh, pred)
        sr, sp = spearmanr(y_fresh, pred)
        mae = np.mean(np.abs(y_fresh - pred))
        print(f"  {name:20s}  Pearson r={r:.3f} (p={p:.3g})  Spearman r={sr:.3f}  MAE={mae:.0f}")

print(f"\nFor reference: scripts/baseline_regression.py's original 10-fold CV "
      f"out-of-fold r was 0.675 on the same 256 tasks.")
