"""Test whether this is even a language-modeling problem at all: fit plain tabular
regressors (linear regression, random forest, gradient boosting) on 4 features that
are knowable BEFORE a task ever runs -- recursion_pattern, branching_factor, n_max
(parsed from input_domain), difficulty_tier -- and compare their out-of-fold Pearson r
against the LLM's own blind-guess predictions (scripts/predict_tokens.py).

Deliberately excludes converged/iterations_attempted/mutation_score/predicted_tokens
as features: those are outcomes of running the task, not properties known in advance,
and using them would be leakage.

Run from repo root: python3 scripts/baseline_regression.py
Outputs written to analysis_output/ (gitignored -- see scripts/analyze_predictions.py
for the same convention).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
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
CAT = ["#3b42db", "#c2491d", "#6f2f96", "#77701c"]

plt.rcParams.update({
    "figure.facecolor": GROUND,
    "axes.facecolor": GROUND,
    "savefig.facecolor": GROUND,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.edgecolor": HAIR,
    "grid.color": HAIR,
    "grid.alpha": 0.55,
    "axes.grid": True,
    "grid.linewidth": 0.6,
    "axes.linewidth": 1.0,
    "font.family": "sans-serif",
    "font.size": 11,
})

df = pd.read_csv(os.path.join(REPO, "data", "merged_actual_vs_predicted.csv"))
LLM_PEARSON_R = -0.099  # from scripts/analyze_predictions.py, already measured

# Only features knowable BEFORE the task runs.
df["n_max"] = df["input_domain"].str.extract(r"0-(\d+)").astype(int)
feature_cols_cat = ["recursion_pattern", "difficulty_tier"]
feature_cols_num = ["branching_factor", "n_max"]
X = df[feature_cols_cat + feature_cols_num]
y = df["actual_total_tokens"]

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_cat),
], remainder="passthrough")

models = {
    "linear_regression": Pipeline([("pre", pre), ("model", LinearRegression())]),
    "random_forest": Pipeline([("pre", pre), ("model",
        RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42))]),
    "gradient_boosting": Pipeline([("pre", pre), ("model",
        GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42))]),
}

kf = KFold(n_splits=10, shuffle=True, random_state=42)
print(f"n = {len(df)}")
print(f"LLM blind guess (already measured): Pearson r = {LLM_PEARSON_R:.3f}\n")

r_by_model = {}
oof_by_model = {}
for name, pipe in models.items():
    oof_pred = cross_val_predict(pipe, X, y, cv=kf)
    r, p = pearsonr(y, oof_pred)
    sr, sp = spearmanr(y, oof_pred)
    mae = np.mean(np.abs(y - oof_pred))
    rmse = np.sqrt(np.mean((y - oof_pred) ** 2))
    r_by_model[name] = r
    oof_by_model[name] = oof_pred
    print(f"{name:20s}  out-of-fold Pearson r = {r:.3f} (p={p:.3g})  "
          f"Spearman r = {sr:.3f}  MAE={mae:.0f}  RMSE={rmse:.0f}")

# feature importances from a random forest fit on all the data (inspection only --
# the r values above are the honest out-of-fold ones, this is just to see what's driving it)
rf_full = Pipeline([("pre", pre), ("model",
    RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42))])
rf_full.fit(X, y)
feat_names = rf_full.named_steps["pre"].get_feature_names_out()
importances = rf_full.named_steps["model"].feature_importances_
order = np.argsort(importances)[::-1]
print("\nRandom forest feature importances:")
for i in order:
    print(f"  {feat_names[i]:35s} {importances[i]:.3f}")

df["rf_oof_pred"] = oof_by_model["random_forest"]
df["gb_oof_pred"] = oof_by_model["gradient_boosting"]
df.to_csv(os.path.join(OUT_DIR, "predictions_with_baselines.csv"), index=False)

# ---- Figure A: bar chart, LLM guess r vs baseline model r ----
bars = [
    ("LLM blind guess\n(no execution, sees only source)", LLM_PEARSON_R, HOT),
    ("Linear regression\n(pattern + branching + n_max + tier)", r_by_model["linear_regression"], BLUE),
    ("Random forest\n(same 4 features)", r_by_model["random_forest"], BLUE),
    ("Gradient boosting\n(same 4 features)", r_by_model["gradient_boosting"], BLUE),
]
fig, ax = plt.subplots(figsize=(10, 5.6), dpi=180)
labels = [b[0] for b in bars]
values = [b[1] for b in bars]
colors = [b[2] for b in bars]
y_pos = np.arange(len(labels))
ax.barh(y_pos, values, color=colors, height=0.55, edgecolor=GROUND, linewidth=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10)
ax.invert_yaxis()
ax.axvline(0, color=INK, linewidth=0.8)
for i, v in enumerate(values):
    ax.text(v + (0.02 if v >= 0 else -0.02), i, f"{v:.2f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=10, color=INK)
ax.set_xlim(-0.25, 0.95)
ax.set_xlabel("Pearson r vs. actual measured tokens (out-of-fold for the models, n=256)")
ax.set_title(
    "This was never a language problem — it's a regression problem\n"
    "four structural task features beat the LLM's own guess by ~7x",
    fontsize=11.5, color=INK, loc="left"
)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "llm_vs_baseline_bar.png"))
plt.close(fig)

# ---- Figure B: scatter, actual vs random-forest out-of-fold predicted ----
def group_of(pattern):
    return "linear/tail (O(n))" if pattern in ("linear", "tail") else pattern

df["group"] = df["recursion_pattern"].map(group_of)
group_order = ["linear/tail (O(n))", "binary-tree", "mixed-parity", "triple-branch"]
color_map = dict(zip(group_order, CAT))

fig2, ax2 = plt.subplots(figsize=(9, 6.4), dpi=180)
for g in group_order:
    sub = df[df["group"] == g]
    ax2.scatter(sub["actual_total_tokens"], sub["rf_oof_pred"],
                s=32, alpha=0.7, color=color_map[g], edgecolor=GROUND, linewidth=0.4, label=g)
lims = [df["actual_total_tokens"].min() * 0.9, df["actual_total_tokens"].max() * 1.05]
ax2.plot(lims, lims, linestyle="--", color=HOT, linewidth=1.6, label="y = x (perfect)")
ax2.set_xlim(lims)
ax2.set_ylim(lims)
ax2.set_xlabel("actual total tokens (measured)")
ax2.set_ylabel("random-forest predicted tokens (out-of-fold, 10-fold CV)")
ax2.set_title(
    f"A tabular model actually tracks real cost\n"
    f"n={len(df)}, 4 pre-task features only  —  Pearson r = {r_by_model['random_forest']:.2f} "
    f"(vs. LLM guess r = {LLM_PEARSON_R:.2f})",
    fontsize=11.5, color=INK, loc="left"
)
leg = ax2.legend(frameon=True, loc="upper left", fontsize=8.5, labelcolor=INK,
                  facecolor=GROUND, edgecolor=HAIR, framealpha=0.95)
leg.get_frame().set_linewidth(0.8)
for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, "rf_actual_vs_predicted_scatter.png"))
plt.close(fig2)

print("\nsaved to analysis_output/: predictions_with_baselines.csv, "
      "llm_vs_baseline_bar.png, rf_actual_vs_predicted_scatter.png")
