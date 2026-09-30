"""Evaluation module: synthetic generators, metrics, baselines, statistics and datasets."""

from pfca.evaluation.synthetic import SyntheticProblem, make_synthetic
from pfca.evaluation import metrics, baselines, statistics, datasets, budget

__all__ = ["SyntheticProblem", "make_synthetic", "metrics", "baselines", "statistics", "datasets", "budget"]
