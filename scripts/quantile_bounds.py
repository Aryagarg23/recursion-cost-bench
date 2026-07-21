"""Predict a range instead of a point: gradient-boosted quantile regression on the
same 4 pre-task features as scripts/baseline_regression.py (recursion_pattern,
branching_factor, n_max, difficulty_tier), fitting a lower bound (5th percentile),
an upper bound (95th percentile), and a median -- all 10-fold cross-validated so the
reported coverage is honest (out-of-fold, not fit-then-score-on-training-data).

This follows directly from scripts/baseline_regression.py's finding that a point
estimate (r~0.68) already beats the LLM's blind guess (r~-0.10) by a wide margin --
the natural next question is whether a calibrated range is a more honest way to
express "how much will this cost" than a single number, given the underlying
variance differs a lot by task type (see per-pattern coverage/width breakdown below).

Run from repo root: python3 scripts/quantile_bounds.py
Outputs written to analysis_output/ (gitignored, same convention as the other
analysis scripts in this repo).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(REPO, "analysis_output")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Arya's presentation-identity tokens (light/paper mode) ----
GROUND = "#eae4d6"
INK = "#282215"
HAIR = "#c6b99f"
BLUE = "#3b42db"
HOT = "#e85b30"

plt.rcParams.update({
    "figure.facecolor": GROUND, "axes.facecolor": GROUND, "savefig.facecolor": GROUND,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": HAIR, "grid.color": HAIR, "grid.alpha": 0.55, "axes.grid": True,
    "grid.linewidth": 0.6, "axes.linewidth": 1.0, "font.family": "sans-serif", "font.size": 11,
})

df = pd.read_csv(os.path.join(REPO, "data", "merged_actual_vs_predicted.csv"))
df["n_max"] = df["input_domain"].str.extract(r"0-(\d+)").astype(int)
feature_cols_cat = ["recursion_pattern", "difficulty_tier"]
feature_cols_num = ["branching_factor", "n_max"]
X = df[feature_cols_cat + feature_cols_num]
y = df["actual_total_tokens"].values
pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_cat)],
                         remainder="passthrough")

def make_quantile_model(alpha):
    return Pipeline([("pre", pre), ("model", GradientBoostingRegressor(
        loss="quantile", alpha=alpha, n_estimators=200, max_depth=3,
        min_samples_leaf=10, random_state=42))])

# [0.05, 0.95] chosen over [0.10, 0.90] after checking calibration directly --
# [0.10,0.90] under-covers (73% empirical vs 80% nominal); [0.05,0.95] comes out
# close to its 90% target (88.7% empirical), so that's the honest one to ship.
LOWER_Q, UPPER_Q = 0.05, 0.95
kf = KFold(n_splits=10, shuffle=True, random_state=42)
lower_pred = cross_val_predict(make_quantile_model(LOWER_Q), X, y, cv=kf)
upper_pred = cross_val_predict(make_quantile_model(UPPER_Q), X, y, cv=kf)
median_pred = cross_val_predict(make_quantile_model(0.5), X, y, cv=kf)

# independently-fit quantile models can cross (lower > upper) on a few rows -- clip
crossed = int((lower_pred > upper_pred).sum())
lo = np.minimum(lower_pred, upper_pred)
hi = np.maximum(lower_pred, upper_pred)

coverage = np.mean((y >= lo) & (y <= hi))
width = hi - lo
print(f"n = {len(df)}")
print(f"quantile crossing fixed by clipping: {crossed}/{len(y)} rows")
print(f"target [{LOWER_Q:.0%},{UPPER_Q:.0%}] nominal={UPPER_Q-LOWER_Q:.0%}  "
      f"empirical coverage={coverage:.1%}  mean width={width.mean():.0f} tokens "
      f"({width.mean()/y.mean():.0%} of mean actual)")

df["lower_pred"] = lo
df["upper_pred"] = hi
df["median_pred"] = median_pred
df["in_interval"] = (y >= lo) & (y <= hi)
df.to_csv(os.path.join(OUT_DIR, "predictions_with_bounds.csv"), index=False)

print("\nCoverage + width by recursion_pattern (variance differs a lot by type --\n"
      "tail is tight and well-calibrated, mixed-parity needs a much wider band):")
for pat, sub in df.groupby("recursion_pattern"):
    print(f"  {pat:15s} n={len(sub):3d}  coverage={sub['in_interval'].mean():.1%}  "
          f"mean_width={(sub['upper_pred']-sub['lower_pred']).mean():.0f}")

# ---- Figure: sorted BY PREDICTED MEDIAN (not actual) so the band reads as a
# smooth corridor -- actual then scatters around/outside it rather than making
# the band itself look jagged (sorting by actual instead makes the band jump
# wildly, since neighboring ranks then belong to unrelated task types whose
# bounds have nothing to do with each other).
sorted_df = df.sort_values("median_pred").reset_index(drop=True)
rank = np.arange(len(sorted_df))

fig, ax = plt.subplots(figsize=(9, 6.4), dpi=180)
ax.fill_between(rank, sorted_df["lower_pred"], sorted_df["upper_pred"],
                 color=BLUE, alpha=0.18, linewidth=0,
                 label=f"predicted {UPPER_Q-LOWER_Q:.0%} interval [{LOWER_Q:.0%},{UPPER_Q:.0%}]")
ax.plot(rank, sorted_df["median_pred"], color=BLUE, linewidth=1.6, label="predicted median")

inside = sorted_df[sorted_df["in_interval"]]
outside = sorted_df[~sorted_df["in_interval"]]
ax.scatter(inside.index, inside["actual_total_tokens"], color=INK, s=18, alpha=0.65,
           label="actual (inside predicted band)")
ax.scatter(outside.index, outside["actual_total_tokens"], color=HOT, s=26, zorder=5,
           label=f"actual (outside predicted band, n={len(outside)})")

ax.set_xlabel("tasks, ranked by predicted median (ascending)")
ax.set_ylabel("tokens")
ax.set_title(
    f"Predicting a range beats predicting a point\n"
    f"n={len(df)} tasks — {coverage:.0%} of actual costs fall inside the predicted "
    f"[{LOWER_Q:.0%},{UPPER_Q:.0%}] band (target {UPPER_Q-LOWER_Q:.0%})",
    fontsize=11.5, color=INK, loc="left"
)
leg = ax.legend(frameon=True, loc="upper left", fontsize=8.5, labelcolor=INK,
                 facecolor=GROUND, edgecolor=HAIR, framealpha=0.95)
leg.get_frame().set_linewidth(0.8)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "quantile_bounds_sorted.png"))
plt.close(fig)

print("\nsaved to analysis_output/: predictions_with_bounds.csv, quantile_bounds_sorted.png")
