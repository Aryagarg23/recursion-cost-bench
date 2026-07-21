"""Compare LLM-predicted token cost (blind guess, no execution) against the real
measured token cost from the completed no-execution swarm run. Produces a scatter
plot, a correlation matrix, and prints summary stats.

Run from repo root: python3 scripts/analyze_predictions.py
Outputs are written to analysis_output/ (gitignored -- regenerate locally; the
rendered PNGs were delivered to Arya directly rather than committed as binaries).
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr, spearmanr

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(REPO, "analysis_output")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Arya's presentation-identity tokens (light/paper mode) ----
GROUND = "#eae4d6"
INK = "#282215"
HAIR = "#c6b99f"
BLUE = "#3b42db"
HOT = "#e85b30"
CAT = ["#3b42db", "#c2491d", "#6f2f96", "#77701c"]  # fixed categorical order

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
df["converged_bin"] = df["converged"].astype(str).str.strip().eq("True").astype(int)

def group_of(pattern):
    if pattern in ("linear", "tail"):
        return "linear/tail (O(n))"
    if pattern == "binary-tree":
        return "binary-tree"
    if pattern == "mixed-parity":
        return "mixed-parity"
    if pattern == "triple-branch":
        return "triple-branch"
    return "other"

df["group"] = df["recursion_pattern"].map(group_of)
group_order = ["linear/tail (O(n))", "binary-tree", "mixed-parity", "triple-branch"]
color_map = dict(zip(group_order, CAT))

# ---- correlations ----
pear_r, pear_p = pearsonr(df["actual_total_tokens"], df["predicted_tokens"])
spear_r, spear_p = spearmanr(df["actual_total_tokens"], df["predicted_tokens"])
print(f"Pearson r  (actual vs predicted): {pear_r:.3f}  (p={pear_p:.3g})")
print(f"Spearman r (actual vs predicted): {spear_r:.3f}  (p={spear_p:.3g})")
print(f"n = {len(df)}")
print(df.groupby("group")["actual_total_tokens"].agg(["count", "mean", "std", "min", "max"]))
print(df["predicted_tokens"].value_counts())

# ---- Figure 1: scatter, actual vs predicted ----
# predicted_tokens collapses to just 2 discrete values (1200 / 1500), so a raw
# scatter stacks dozens of points exactly on top of one another -- add a small
# fixed-seed jitter in y so overlap density is actually visible, and lower alpha
# so stacked points read darker instead of just occluding each other.
rng = np.random.default_rng(42)
fig, ax = plt.subplots(figsize=(9, 6.4), dpi=180)
for g in group_order:
    sub = df[df["group"] == g]
    y_jittered = sub["predicted_tokens"] + rng.uniform(-22, 22, size=len(sub))
    ax.scatter(sub["actual_total_tokens"], y_jittered,
               s=30, alpha=0.45, color=color_map[g], edgecolor=GROUND, linewidth=0.4, label=g)

lims = [min(df["actual_total_tokens"].min(), df["predicted_tokens"].min()) * 0.9,
        max(df["actual_total_tokens"].max(), df["predicted_tokens"].max()) * 1.05]
ax.plot(lims, lims, linestyle="--", color=HOT, linewidth=1.6, label="y = x (perfect guess)")

# actual best-fit line: what the LLM's guesses really do as actual cost rises
slope, intercept = np.polyfit(df["actual_total_tokens"], df["predicted_tokens"], 1)
fit_x = np.array(lims)
fit_y = slope * fit_x + intercept
ax.plot(fit_x, fit_y, linestyle="-", color=BLUE, linewidth=2.0,
        label=f"actual fit (slope = {slope:.3f}, r = {pear_r:.2f})")

ax.set_xlim(lims)
ax.set_ylim(200, 1700)

ax.set_xlabel("actual total tokens (measured, real swarm run)")
ax.set_ylabel("predicted tokens (LLM guess, no execution; jittered ±22 to show overlap density)")
ax.set_title(
    f"Does an LLM's upfront token guess track what a task actually costs?\n"
    f"n={len(df)} recursive-function test-writing tasks  —  Pearson r = {pear_r:.2f}\n"
    f"Spearman r = {spear_r:.2f} (neither significant, p > 0.1)",
    fontsize=11.5, color=INK, loc="left"
)
# legend placed in the empty lower-right quadrant (no data there) with a solid
# backing box, so it never sits on top of the dense y=1200/1500 rows.
leg = ax.legend(frameon=True, loc="lower right", fontsize=8.5, labelcolor=INK,
                 facecolor=GROUND, edgecolor=HAIR, framealpha=0.95)
leg.get_frame().set_linewidth(0.8)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "actual_vs_predicted_scatter.png"))
plt.close(fig)

# ---- Figure 2: correlation matrix heatmap ----
num_cols = {
    "actual_tokens": df["actual_total_tokens"],
    "predicted_tokens": df["predicted_tokens"],
    "branching_factor": df["branching_factor"],
    "converged": df["converged_bin"],
    "iterations_attempted": df["iterations_attempted"],
    "prediction_cost_tokens": df["prediction_cost_tokens"],
}
corr_df = pd.DataFrame(num_cols).corr(method="pearson")
labels = list(corr_df.columns)

fig2, ax2 = plt.subplots(figsize=(6.6, 5.8), dpi=180)
diverging = LinearSegmentedColormap.from_list("garg_diverging", [BLUE, GROUND, "#c2491d"])
im = ax2.imshow(corr_df.values, cmap=diverging, vmin=-1, vmax=1)
ax2.set_xticks(range(len(labels)))
ax2.set_yticks(range(len(labels)))
ax2.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
ax2.set_yticklabels(labels, fontsize=9)
ax2.grid(False)
for i in range(len(labels)):
    for j in range(len(labels)):
        v = corr_df.values[i, j]
        txt_color = GROUND if abs(v) > 0.6 else INK
        ax2.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5, color=txt_color)
for spine in ax2.spines.values():
    spine.set_visible(False)
cbar = fig2.colorbar(im, ax=ax2, shrink=0.85)
cbar.ax.tick_params(colors=INK, labelsize=8)
cbar.outline.set_visible(False)
ax2.set_title("Correlation matrix — real outcome vs. task features",
              fontsize=11.5, color=INK, loc="left")
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, "correlation_matrix.png"))
plt.close(fig2)

corr_df.to_csv(os.path.join(OUT_DIR, "correlation_matrix.csv"))
print("\nsaved to analysis_output/: actual_vs_predicted_scatter.png, correlation_matrix.png, correlation_matrix.csv")
