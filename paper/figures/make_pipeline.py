"""Pipeline figure of the manuscript (Figure 1): vertical flow with wrapped text; contribution steps shaded."""
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9})
steps = [
    ("Training data and black box model", "Reference model f fitted on the training split; hyperparameters tuned on the tuning split; the model is fixed for every explanation method", False),
    ("Attribution pool", "B bootstrap resamples x M model classes refitted with logged seeds; feature level Shapley values per member with a fixed background sample", True),
    ("Fuzzy concepts", "Fuzzy c means on the correlation profiles of the features (consensus over resamples); candidates over K and the fuzzifier; a priori grouping as a further candidate; membership matrix U", True),
    ("Fuzzy attributions", "Aggregation of Shapley values by U (Eq. 3); trapezoidal fuzzy numbers over the pool (Eq. 4); sign confidence, disagreement index and type 2 upgrade; linguistic labels (Eq. 7)", True),
    ("Pareto selection", "Objectives of Eq. 9 for every configuration (partition, sparsity, alpha): fidelity loss, complexity, instability; Pareto front, knee point default, selection membership (Eq. 10)", True),
    ("Outputs and evaluation", "Linguistic concept explanation, global fuzzy ranking, front and defuzzified feature vector; synthetic truth, benchmark and questionnaire phases; Table 1 metrics and statistical tests", False),
]
fig, ax = plt.subplots(figsize=(6.3, 7.4))
ax.set_xlim(0, 10); ax.set_ylim(0, len(steps) * 1.25 + 0.2); ax.axis("off")
h = 1.0
for i, (title, detail, contrib) in enumerate(steps):
    y = (len(steps) - 1 - i) * 1.25 + 0.2
    face = "#d9d9d9" if contrib else "#f7f7f7"
    box = FancyBboxPatch((0.4, y), 9.3, h, boxstyle="round,pad=0.02,rounding_size=0.12", linewidth=1.0, edgecolor="#333333", facecolor=face)
    ax.add_patch(box)
    ax.text(0.65, y + h - 0.12, title, ha="left", va="top", fontsize=10)
    ax.text(0.65, y + h - 0.42, "\n".join(textwrap.wrap(detail, 80)), ha="left", va="top", fontsize=8.0, linespacing=1.15)
    if i < len(steps) - 1:
        ax.add_patch(FancyArrowPatch((5.0, y), (5.0, y - 0.23), arrowstyle="-|>", mutation_scale=12, linewidth=1.0, color="#333333"))
ax.text(9.4, len(steps) * 1.25 + 0.05, "shaded: proposed contribution", ha="right", va="bottom", fontsize=8, color="#333333")
fig.tight_layout()
fig.savefig("pipeline.png", dpi=300)
print("saved")
