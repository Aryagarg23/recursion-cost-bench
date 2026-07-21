"""Compare the v2 (full-context + reasoning) LLM predictions against v1 (bare
guess, no context) and against the tabular regression baseline
(scripts/baseline_regression.py). All three predict the same target
(actual_total_tokens) for the same 256 tasks.

Run from repo root: python3 scripts/analyze_v2_predictions.py
Reads data/merged_v1_v2_actual.csv (built by joining tasks.jsonl + tasks_results.jsonl
+ predictions.jsonl + predictions_v2.jsonl by task_id).
Outputs written to analysis_output/ (gitignored, same convention as the other
analysis scripts in this repo).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(REPO, "analysis_output")
os.makedirs(OUT_DIR, exist_ok=True)

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

df = pd.read_csv(os.path.join(REPO, "data", "merged_v1_v2_actual.csv"))
print(f"n = {len(df)}")

r1, p1 = pearsonr(df["actual_total_tokens"], df["predicted_v1_bare"])
r2, p2 = pearsonr(df["actual_total_tokens"], df["predicted_v2_rich"])
sr1, sp1 = spearmanr(df["actual_total_tokens"], df["predicted_v1_bare"])
sr2, sp2 = spearmanr(df["actual_total_tokens"], df["predicted_v2_rich"])

REGRESSION_R = 0.675  # scripts/baseline_regression.py, linear regression, 10-fold CV

print(f"v1 bare guess (no context, no reasoning, 40 tokens):      "
      f"Pearson r={r1:.3f} (p={p1:.3g})  Spearman r={sr1:.3f}")
print(f"v2 rich context + reasoning (~400 tokens, same features   "
      f"Pearson r={r2:.3f} (p={p2:.3g})  Spearman r={sr2:.3f}")
print(f"    as scripts/baseline_regression.py):")
print(f"tabular regression (no LLM at all):                        Pearson r={REGRESSION_R:.3f}")

print(f"\nv2 finished (reached a FINAL: line) rate: {df['v2_finished'].mean():.1%}")
print(f"v2 predicted range: {df['predicted_v2_rich'].min()}-{df['predicted_v2_rich'].max()}  "
      f"(actual range: {df['actual_total_tokens'].min()}-{df['actual_total_tokens'].max()})")
print(f"v2 mean guess: {df['predicted_v2_rich'].mean():.0f}  actual mean: {df['actual_total_tokens'].mean():.0f}  "
      f"-> v2 underestimates true scale by ~{df['actual_total_tokens'].mean()/df['predicted_v2_rich'].mean():.1f}x")
print("\nv2 predicted_tokens value counts (top 10):")
print(df["predicted_v2_rich"].value_counts().head(10))

bars = [
    ("LLM: no context, no reasoning\n(forced bare integer, 40 tokens)", r1, HOT),
    ("LLM: full context + reasoning\n(same features as regression, ~400 tokens)", r2, HOT),
    ("Tabular regression\n(4 structural features, no LLM)", REGRESSION_R, BLUE),
]
fig, ax = plt.subplots(figsize=(10, 5.0), dpi=180)
labels = [b[0] for b in bars]
values = [b[1] for b in bars]
colors = [b[2] for b in bars]
y_pos = np.arange(len(labels))
ax.barh(y_pos, values, color=colors, height=0.5, edgecolor=GROUND, linewidth=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10)
ax.invert_yaxis()
ax.axvline(0, color=INK, linewidth=0.8)
for i, v in enumerate(values):
    ax.text(v + (0.02 if v >= 0 else -0.02), i, f"{v:.2f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=10, color=INK)
ax.set_xlim(-0.25, 0.85)
ax.set_xlabel("Pearson r vs. actual measured tokens (n=256)")
ax.set_title(
    "Giving the LLM full context and room to reason didn't help\n"
    "its own token-cost guess stays at zero correlation either way",
    fontsize=11.5, color=INK, loc="left"
)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "llm_context_vs_reasoning_bar.png"))
plt.close(fig)

print("\nsaved to analysis_output/: llm_context_vs_reasoning_bar.png")
