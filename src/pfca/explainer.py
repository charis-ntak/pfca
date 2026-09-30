"""The PFCA explainer: a scikit learn style interface tying the modules together.

Usage
-----
>>> explainer = PFCAExplainer(n_resamples=50, random_state=0).fit(X_train, y_train)
>>> explanation = explainer.explain(X_test[:200])
>>> explanation.describe(0)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from pfca.attribution import AttributionEngine, AttributionResult, aggregate_to_concepts, redistribute_to_features
from pfca.concepts import ConceptFormer, ConceptPartition, identity_partition
from pfca.fuzzification import (
    IntervalType2FuzzyNumber,
    LinguisticLabelSet,
    TrapezoidalFuzzyNumber,
    centroid_array,
    disagreement_index,
    fuzzify_array,
    fuzzy_ranking,
    possibility_matrix,
    sign_confidence,
    standardization_scale,
    tfn_from_quantiles,
)
from pfca.selection import (
    ExplanationConfiguration,
    ParetoSelector,
    PartitionPool,
    SelectionResult,
    concept_selection_membership,
    retained_concepts,
)


@dataclass
class PFCAExplanation:
    """Outputs of PFCA for a set of explained instances (Section 3.6 of the guide)."""

    partition: ConceptPartition
    configuration: ExplanationConfiguration
    retained: np.ndarray
    activity: np.ndarray
    quantiles: np.ndarray
    concept_pool: np.ndarray
    sign_confidence: np.ndarray
    disagreement: np.ndarray
    type2_mask: np.ndarray
    global_quantiles: np.ndarray
    possibility: np.ndarray
    ranking: np.ndarray
    dominance: np.ndarray
    linguistic: np.ndarray
    label_set: LinguisticLabelSet
    scale: float
    feature_selection_membership: np.ndarray
    concept_selection_membership: np.ndarray
    selection: SelectionResult
    attribution: AttributionResult
    reference_output: np.ndarray
    disagreement_threshold: float
    fuzzy_shape: str = "trapezoidal"
    support_percentiles: tuple[float, float] = (5.0, 95.0)
    core_percentiles: tuple[float, float] = (25.0, 75.0)
    pools: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    @property
    def n_explain(self) -> int:
        return int(self.quantiles.shape[0])

    @property
    def n_concepts(self) -> int:
        return int(self.quantiles.shape[1])

    @property
    def concept_names(self) -> list[str]:
        return self.partition.concept_names()

    @property
    def membership(self) -> np.ndarray:
        return self.partition.U

    @property
    def concept_centroids(self) -> np.ndarray:
        """Defuzzified concept attribution (centroid of the trapezoid), shape (n_explain, K)."""
        return centroid_array(self.quantiles)

    @property
    def concept_medians(self) -> np.ndarray:
        return self.quantiles[..., 2]

    def feature_attribution(self, retained_only: bool = False) -> np.ndarray:
        """Defuzzified feature level vector obtained by redistributing concept centroids."""
        C = self.concept_centroids.copy()
        if retained_only:
            mask = np.zeros(self.n_concepts, dtype=bool)
            mask[self.retained] = True
            C[:, ~mask] = 0.0
        return redistribute_to_features(C, self.partition.U)

    def fuzzy_attribution(self, instance: int, concept: int):
        """Fuzzy attribution of one instance and concept; type 2 when the model classes disagree."""
        if self.type2_mask[instance, concept]:
            return self.type2_number(instance, concept)
        return tfn_from_quantiles(self.quantiles[instance, concept])

    def type2_number(self, instance: int, concept: int) -> IntervalType2FuzzyNumber:
        B, M = self.attribution.n_resamples, self.attribution.n_model_classes
        pool = self.concept_pool.reshape(B, M, self.n_explain, self.n_concepts)
        members = []
        for m in range(M):
            q = fuzzify_array(pool[:, m, instance, concept], axis=0, shape=self.fuzzy_shape, support_percentiles=self.support_percentiles, core_percentiles=self.core_percentiles)
            members.append(tfn_from_quantiles(q))
        return IntervalType2FuzzyNumber(members)

    def global_fuzzy_importance(self) -> list[TrapezoidalFuzzyNumber]:
        return [tfn_from_quantiles(q) for q in self.global_quantiles]

    def best_labels(self) -> tuple[np.ndarray, np.ndarray]:
        """Most compatible label index and degree for every instance and concept."""
        idx = np.argmax(self.linguistic, axis=-1)
        deg = np.take_along_axis(self.linguistic, idx[..., None], axis=-1)[..., 0]
        return idx, deg

    def instance_frame(self, instance: int) -> pd.DataFrame:
        """Table with one row per concept for an explained instance."""
        names = self.concept_names
        idx, deg = self.best_labels()
        rows = []
        for k in range(self.n_concepts):
            F = tfn_from_quantiles(self.quantiles[instance, k])
            rows.append(
                {
                    "concept": names[k],
                    "retained": bool(k in set(self.retained.tolist())),
                    "centroid": F.centroid(),
                    "median": float(self.quantiles[instance, k, 2]),
                    "support_low": F.a,
                    "core_low": F.b,
                    "core_high": F.c,
                    "support_high": F.d,
                    "sign_confidence": float(self.sign_confidence[instance, k]),
                    "disagreement": float(self.disagreement[instance, k]),
                    "type2": bool(self.type2_mask[instance, k]),
                    "label": self.label_set.names[int(idx[instance, k])],
                    "label_degree": float(deg[instance, k]),
                    "selection_membership": float(self.concept_selection_membership[k]),
                }
            )
        df = pd.DataFrame(rows)
        return df.reindex(df["centroid"].abs().sort_values(ascending=False).index).reset_index(drop=True)

    def global_frame(self) -> pd.DataFrame:
        names = self.concept_names
        G = self.global_fuzzy_importance()
        rank_pos = np.empty(self.n_concepts, dtype=int)
        rank_pos[self.ranking] = np.arange(1, self.n_concepts + 1)
        rows = []
        for k in range(self.n_concepts):
            rows.append(
                {
                    "concept": names[k],
                    "rank": int(rank_pos[k]),
                    "retained": bool(k in set(self.retained.tolist())),
                    "importance_centroid": G[k].centroid(),
                    "importance_support_low": G[k].a,
                    "importance_support_high": G[k].d,
                    "dominance": float(self.dominance[k]),
                    "activity": float(self.activity[k]),
                    "fuzziness": float(self.partition.fuzziness[k]),
                    "selection_membership": float(self.concept_selection_membership[k]),
                }
            )
        return pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)

    def describe(self, instance: int, retained_only: bool = True) -> str:
        """Linguistic explanation of one instance."""
        lines = [f"Instance {instance}: model output {self.reference_output[instance]:.4g}"]
        cfg = self.configuration
        lines.append(f"Configuration: K={cfg.n_concepts}, fuzzifier={cfg.fuzzifier}, sparsity={cfg.sparsity}, alpha={cfg.alpha:.2f}")
        idx, deg = self.best_labels()
        order = np.argsort(-np.abs(self.concept_centroids[instance]))
        names = self.concept_names
        retained = set(self.retained.tolist())
        for k in order:
            if retained_only and k not in retained:
                continue
            F = self.fuzzy_attribution(instance, k)
            if isinstance(F, IntervalType2FuzzyNumber):
                lo, hi = F.centroid_interval()
                prof = self.label_set.profile_type2(F, self.scale)
                label = max(prof, key=lambda n: prof[n][1])
                lines.append(
                    f"  {names[k]}: {label} (compatibility {prof[label][0]:.2f} to {prof[label][1]:.2f}, model classes disagree, "
                    f"disagreement {self.disagreement[instance, k]:.2f}), centroid between {lo:+.3g} and {hi:+.3g}, "
                    f"sign confidence {self.sign_confidence[instance, k]:.2f}, selection membership {self.concept_selection_membership[k]:.2f}"
                )
            else:
                lines.append(
                    f"  {names[k]}: {self.label_set.names[int(idx[instance, k])]} (compatibility {deg[instance, k]:.2f}), "
                    f"centroid {F.centroid():+.3g}, support [{F.a:+.3g}, {F.d:+.3g}], core [{F.b:+.3g}, {F.c:+.3g}], "
                    f"sign confidence {self.sign_confidence[instance, k]:.2f}, disagreement {self.disagreement[instance, k]:.2f}, "
                    f"selection membership {self.concept_selection_membership[k]:.2f}"
                )
        return "\n".join(lines)

    def front(self) -> pd.DataFrame:
        return self.selection.front

    def to_dict(self) -> dict:
        """JSON serializable summary (without the raw pool)."""
        return {
            "configuration": self.configuration.as_dict(),
            "retained": [int(k) for k in self.retained],
            "concept_names": self.concept_names,
            "membership": self.partition.U.tolist(),
            "quantiles": self.quantiles.tolist(),
            "sign_confidence": self.sign_confidence.tolist(),
            "disagreement": self.disagreement.tolist(),
            "type2": self.type2_mask.tolist(),
            "global_quantiles": self.global_quantiles.tolist(),
            "possibility": self.possibility.tolist(),
            "ranking": [int(k) for k in self.ranking],
            "labels": self.label_set.names,
            "linguistic": self.linguistic.tolist(),
            "scale": self.scale,
            "feature_selection_membership": self.feature_selection_membership.tolist(),
            "concept_selection_membership": self.concept_selection_membership.tolist(),
            "front": json.loads(self.selection.front.to_json(orient="records")),
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf8") as fh:
            json.dump(self.to_dict(), fh)


class PFCAExplainer(BaseEstimator):
    """Pareto optimal fuzzy concept attribution.

    Parameters
    ----------
    model_classes : list of estimators or (name, estimator) pairs, optional
        Model classes used to quantify model dependence. The first is the
        reference model class. None uses gradient boosting, random forest and a
        multilayer perceptron.
    n_resamples : int
        Number of bootstrap resamples B (resample 0 is the original data).
    background_size : int or None
        Background sample size for the Shapley computation.
    explainer : {'auto', 'tree', 'linear', 'kernel', 'exact', 'permutation'}
    n_concepts_grid : iterable of int
    fuzzifier_grid : iterable of float
    concept_distance : {'correlation', 'loading'}
    correlation : {'pearson', 'spearman'}
    apriori_groups : dict or list, optional
        Known grouping of the features, included as a candidate partition.
    partition_source : {'data', 'apriori', 'both'}
        Which candidate partitions enter the selection.
    consensus : bool
        Consensus fuzzy c means over resamples of the instances.
    crisp_concepts : bool
        Ablation: harden the concept partitions.
    fuzzy_shape : {'trapezoidal', 'triangular'}
    support_percentiles, core_percentiles : pairs of floats
    disagreement_threshold : float
        Disagreement index above which an attribution is upgraded to type 2.
    linguistic_labels : LinguisticLabelSet, optional
    linguistic_scale : {'mean_abs', 'output_sd'} or float
    alpha_grid : iterable of float
    sparsity_grid : iterable of int or None
    solver : {'grid', 'nsga2', 'fixed'}
        'fixed' evaluates only ``fixed_configuration`` (ablation).
    fixed_configuration : dict, optional
        Keys n_concepts, fuzzifier, sparsity, alpha (or partition='apriori').
    knee_method : {'utopia', 'hyperplane'}
    surrogate : {'linear', 'tree'}
    task : {'auto', 'regression', 'classification'}
    target_class : int
    n_jobs : int
    random_state : int or None
    """

    def __init__(
        self,
        model_classes=None,
        n_resamples: int = 50,
        background_size: int | None = 100,
        explainer: str = "auto",
        n_concepts_grid: Sequence[int] = (2, 3, 4, 5, 6),
        fuzzifier_grid: Sequence[float] = (1.5, 2.0, 2.5),
        concept_distance: str = "correlation",
        correlation: str = "pearson",
        apriori_groups=None,
        partition_source: str = "both",
        consensus: bool = False,
        crisp_concepts: bool = False,
        fuzzy_shape: str = "trapezoidal",
        support_percentiles: tuple[float, float] = (5.0, 95.0),
        core_percentiles: tuple[float, float] = (25.0, 75.0),
        disagreement_threshold: float = 0.5,
        linguistic_labels: LinguisticLabelSet | None = None,
        linguistic_scale="mean_abs",
        alpha_grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
        sparsity_grid: Sequence[int] | None = None,
        solver: str = "grid",
        fixed_configuration: dict | None = None,
        knee_method: str = "utopia",
        surrogate: str = "linear",
        cv: int = 5,
        nsga_pop_size: int = 40,
        nsga_generations: int = 30,
        task: str = "auto",
        target_class: int = 1,
        kernel_nsamples="auto",
        permutation_max_evals="auto",
        n_jobs: int = 1,
        random_state: int | None = None,
        verbose: int = 0,
    ):
        self.model_classes = model_classes
        self.n_resamples = n_resamples
        self.background_size = background_size
        self.explainer = explainer
        self.n_concepts_grid = n_concepts_grid
        self.fuzzifier_grid = fuzzifier_grid
        self.concept_distance = concept_distance
        self.correlation = correlation
        self.apriori_groups = apriori_groups
        self.partition_source = partition_source
        self.consensus = consensus
        self.crisp_concepts = crisp_concepts
        self.fuzzy_shape = fuzzy_shape
        self.support_percentiles = support_percentiles
        self.core_percentiles = core_percentiles
        self.disagreement_threshold = disagreement_threshold
        self.linguistic_labels = linguistic_labels
        self.linguistic_scale = linguistic_scale
        self.alpha_grid = alpha_grid
        self.sparsity_grid = sparsity_grid
        self.solver = solver
        self.fixed_configuration = fixed_configuration
        self.knee_method = knee_method
        self.surrogate = surrogate
        self.cv = cv
        self.nsga_pop_size = nsga_pop_size
        self.nsga_generations = nsga_generations
        self.task = task
        self.target_class = target_class
        self.kernel_nsamples = kernel_nsamples
        self.permutation_max_evals = permutation_max_evals
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------
    def fit(self, X, y, feature_names: Sequence[str] | None = None):
        """Fit the pool of models and form the candidate concept partitions."""
        if feature_names is None and hasattr(X, "columns"):
            feature_names = [str(c) for c in X.columns]
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.feature_names_ = list(feature_names) if feature_names is not None else [f"x{j}" for j in range(X.shape[1])]
        self.engine_ = AttributionEngine(
            model_classes=self.model_classes,
            n_resamples=self.n_resamples,
            background_size=self.background_size,
            explainer=self.explainer,
            task=self.task,
            target_class=self.target_class,
            kernel_nsamples=self.kernel_nsamples,
            permutation_max_evals=self.permutation_max_evals,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
        ).fit(X, y)
        self.task_ = self.engine_.task_
        self._fit_partitions(X)
        return self

    @property
    def reference_model_(self):
        return self.engine_.reference_model_

    # ------------------------------------------------------------------
    def _partition_pool(self, partition: ConceptPartition, attr: AttributionResult) -> PartitionPool:
        concept = aggregate_to_concepts(attr.values, partition.U)
        B, M, n, K = concept.shape
        pool = concept.reshape(B * M, n, K)
        quantiles = fuzzify_array(pool, axis=0, shape=self.fuzzy_shape, support_percentiles=self.support_percentiles, core_percentiles=self.core_percentiles)
        global_imp = np.abs(pool).mean(axis=1)
        global_quantiles = fuzzify_array(global_imp, axis=0, shape=self.fuzzy_shape, support_percentiles=self.support_percentiles, core_percentiles=self.core_percentiles)
        return PartitionPool(partition, pool, quantiles, global_quantiles)

    @classmethod
    def from_engine(cls, engine: AttributionEngine, feature_names: Sequence[str] | None = None, **kwargs) -> "PFCAExplainer":
        """Build an explainer around an already fitted attribution engine (shares the pool of models)."""
        obj = cls(model_classes=engine.model_classes, n_resamples=engine.n_resamples, background_size=engine.background_size, explainer=engine.explainer, task=engine.task, target_class=engine.target_class, n_jobs=engine.n_jobs, random_state=engine.random_state, **kwargs)
        X = engine.X_train_
        obj.engine_ = engine
        obj.task_ = engine.task_
        obj.feature_names_ = list(feature_names) if feature_names is not None else [f"x{j}" for j in range(X.shape[1])]
        obj._fit_partitions(X)
        return obj

    def _fit_partitions(self, X: np.ndarray) -> None:
        d = X.shape[1]
        grid = [int(k) for k in self.n_concepts_grid if 1 <= int(k) < d]
        if not grid and self.partition_source != "apriori":
            grid = [d - 1] if d > 1 else []
        if self.partition_source == "apriori" and self.apriori_groups is None:
            raise ValueError("partition_source='apriori' requires apriori_groups.")
        self.former_ = ConceptFormer(
            n_concepts_grid=grid if self.partition_source != "apriori" else (),
            fuzzifier_grid=self.fuzzifier_grid,
            distance=self.concept_distance,
            correlation=self.correlation,
            apriori_groups=self.apriori_groups if self.partition_source in ("apriori", "both") else None,
            consensus=self.consensus,
            crisp=self.crisp_concepts,
            random_state=self.random_state,
        )
        if d == 1:
            self.former_.n_concepts_grid = ()
        self.former_.fit(X, self.feature_names_)
        if d == 1 and not self.former_.candidates_:
            self.former_.candidates_ = [identity_partition(1, self.feature_names_)]
        self.X_train_ = X
        self.n_features_in_ = d

    def explain(self, X_explain) -> PFCAExplanation:
        """Compute the fuzzy concept explanation of the given instances."""
        X_explain = np.asarray(X_explain, dtype=float)
        attr = self.engine_.explain(X_explain)
        return self.explain_with(attr)

    def explain_with(self, attr: AttributionResult) -> PFCAExplanation:
        """Build the explanation from a precomputed pool of feature level attributions."""
        output = attr.reference_output
        selector = ParetoSelector(
            alpha_grid=self.alpha_grid,
            sparsity_grid=self.sparsity_grid,
            solver=self.solver,
            knee_method=self.knee_method,
            surrogate=self.surrogate,
            cv=self.cv,
            nsga_pop_size=self.nsga_pop_size,
            nsga_generations=self.nsga_generations,
            random_state=self.random_state if self.random_state is not None else 0,
        )
        pools: dict[str, PartitionPool] = {}

        def get_pool(partition: ConceptPartition) -> PartitionPool:
            if partition.name not in pools:
                pools[partition.name] = self._partition_pool(partition, attr)
            return pools[partition.name]

        if self.solver == "grid":
            for part in self.former_.candidates_:
                get_pool(part)
            selection = selector.run_grid(pools, output)
        elif self.solver == "fixed":
            if not self.fixed_configuration:
                raise ValueError("solver='fixed' requires fixed_configuration.")
            fc = dict(self.fixed_configuration)
            if fc.get("partition") == "apriori":
                part = self.former_.apriori_partition()
            else:
                part = self.former_.partition(int(fc["n_concepts"]), fc.get("fuzzifier", 2.0))
            pp = get_pool(part)
            selector.alpha_grid = (float(fc.get("alpha", 0.0)),)
            selector.sparsity_grid = (int(fc.get("sparsity", part.n_concepts)),)
            selection = selector.run_grid({part.name: pp}, output)
        elif self.solver == "nsga2":
            fcm_grid = [int(k) for k in self.n_concepts_grid if 1 <= int(k) < self.n_features_in_]
            extra = ("apriori",) if (self.apriori_groups is not None and self.partition_source in ("apriori", "both")) else ()

            def provider(K, m, source):
                if source == "apriori":
                    return get_pool(self.former_.apriori_partition())
                return get_pool(self.former_.partition(int(K), None if self.concept_distance == "loading" else float(m)))

            if self.partition_source == "apriori":
                selection = selector.run_grid({"apriori": get_pool(self.former_.apriori_partition())}, output)
            else:
                selection = selector.run_nsga2(provider, (min(fcm_grid), max(fcm_grid)), tuple(float(m) for m in self.fuzzifier_grid), output, extra)
        else:
            raise ValueError("solver must be 'grid', 'nsga2' or 'fixed'")
        return self._build_explanation(selection, selection.knee, pools, attr)

    def _build_explanation(self, selection: SelectionResult, index: int, pools: dict, attr: AttributionResult) -> PFCAExplanation:
        cfg = selection.configurations[index]
        pp = pools[cfg.partition]
        partition = pp.partition
        B, M, n, d = attr.values.shape
        K = partition.n_concepts
        concept = pp.pool.reshape(B, M, n, K)
        retained, activity, _ = retained_concepts(pp.quantiles, pp.global_quantiles, cfg.alpha, cfg.sparsity)
        sc = sign_confidence(pp.pool, axis=0)
        dis = disagreement_index(concept)
        type2 = dis > self.disagreement_threshold
        label_set = self.linguistic_labels if self.linguistic_labels is not None else LinguisticLabelSet()
        scale = standardization_scale(concept[0, 0], self.linguistic_scale, attr.reference_output)
        linguistic = label_set.profile_array(pp.quantiles, scale)
        G = [tfn_from_quantiles(q) for q in pp.global_quantiles]
        poss = possibility_matrix(G)
        ranking, dominance = fuzzy_ranking(G)
        csm = concept_selection_membership(partition.U, selection.feature_selection_membership)
        return PFCAExplanation(
            partition=partition,
            configuration=cfg,
            retained=retained,
            activity=activity,
            quantiles=pp.quantiles,
            concept_pool=pp.pool,
            sign_confidence=sc,
            disagreement=dis,
            type2_mask=type2,
            global_quantiles=pp.global_quantiles,
            possibility=poss,
            ranking=ranking,
            dominance=dominance,
            linguistic=linguistic,
            label_set=label_set,
            scale=scale,
            feature_selection_membership=selection.feature_selection_membership,
            concept_selection_membership=csm,
            selection=selection,
            attribution=attr,
            reference_output=attr.reference_output,
            disagreement_threshold=self.disagreement_threshold,
            fuzzy_shape=self.fuzzy_shape,
            support_percentiles=tuple(self.support_percentiles),
            core_percentiles=tuple(self.core_percentiles),
            pools=pools,
        )

    def explanation_at(self, explanation: PFCAExplanation, index: int) -> PFCAExplanation:
        """Rebuild the explanation for another configuration of the evaluated table (for example another front member)."""
        return self._build_explanation(explanation.selection, int(index), explanation.pools, explanation.attribution)

    def fit_explain(self, X, y, X_explain, feature_names: Sequence[str] | None = None) -> PFCAExplanation:
        return self.fit(X, y, feature_names).explain(X_explain)
