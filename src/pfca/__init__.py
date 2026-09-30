"""Pareto optimal fuzzy concept attribution (PFCA).

A post hoc explainability method that attributes model predictions to fuzzy
concepts formed from correlated features, represents the uncertainty of every
attribution as a fuzzy number with linguistic labels, and selects the reported
explanation from a Pareto front that trades off fidelity, simplicity and
stability.

Author: Charis Ntakolia, Hellenic Air Force Academy.
"""

from pfca.attribution import AttributionEngine, AttributionResult, default_model_classes
from pfca.concepts import ConceptFormer, ConceptPartition, fuzzy_c_means
from pfca.explainer import PFCAExplainer, PFCAExplanation
from pfca.fuzzification import (
    IntervalType2FuzzyNumber,
    LinguisticLabelSet,
    TrapezoidalFuzzyNumber,
    fuzzify,
)
from pfca.selection import ExplanationConfiguration, ParetoSelector, knee_point, pareto_mask

__version__ = "0.1.0"
__author__ = "Charis Ntakolia"

__all__ = [
    "AttributionEngine",
    "AttributionResult",
    "ConceptFormer",
    "ConceptPartition",
    "ExplanationConfiguration",
    "IntervalType2FuzzyNumber",
    "LinguisticLabelSet",
    "PFCAExplainer",
    "PFCAExplanation",
    "ParetoSelector",
    "TrapezoidalFuzzyNumber",
    "default_model_classes",
    "fuzzify",
    "fuzzy_c_means",
    "knee_point",
    "pareto_mask",
]
