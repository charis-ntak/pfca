"""Merge the per family Phase A result folders of the reduced study into one run folder (results/phase_a_reduced_all).

The additive family ran under results/phase_a_reduced with three seeds before the run was split per family with
two seeds; only the seeds 0 and 1 are kept so that every cell carries the same number of replications.
"""
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["phase_a_reduced", "phase_a_reduced_interaction", "phase_a_reduced_redundancy"]
out = ROOT / "results" / "phase_a_reduced_all"
out.mkdir(parents=True, exist_ok=True)
frames, per_instance = [], []
for name in SOURCES:
    folder = ROOT / "results" / name
    if (folder / "metrics.csv").exists():
        frames.append(pd.read_csv(folder / "metrics.csv"))
    if (folder / "per_instance.csv").exists():
        per_instance.append(pd.read_csv(folder / "per_instance.csv"))
df = pd.concat(frames, ignore_index=True)
df = df[df["seed"] < 2]
df.to_csv(out / "metrics.csv", index=False)
if per_instance:
    pi = pd.concat(per_instance, ignore_index=True)
    pi[pi["seed"] < 2].to_csv(out / "per_instance.csv", index=False)
for name in ("config.yaml", "environment.json"):
    src = ROOT / "results" / SOURCES[0] / name
    if src.exists():
        shutil.copy(src, out / name)
print(f"merged {len(df)} rows: " + ", ".join(f"{f} {n}" for f, n in df.groupby('family').size().items()))
