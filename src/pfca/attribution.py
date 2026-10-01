"""Feature level Shapley attribution over bootstrap resamples and model classes.

This module implements Steps 1 and 2 of the PFCA algorithm. The reference
model is fitted on the training data, the training data is resampled B times,
every model class is refitted on every resample, and feature level Shapley
values are computed for the instances to be explained with a background sample
drawn from the resample. Resample index 0 is always the original training data,
so that with B = 1 and M = 1 the pool reduces to the Shapley values of the
reference model, which is required for the exact reduction to SHAP.
"""

from __future__ import annotations

import numbers
import warnings
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import type_of_target

from pfca.utils import make_rng, spawn_seeds

# Model classes explained with TreeSHAP. The histogram based gradient boosting
# estimators of scikit learn are deliberately absent: shap evaluates their trees
# on the raw split thresholds rather than on the binned values used by the
# model, so local accuracy fails for them, and they are explained through the
# model agnostic branches instead.
TREE_TYPES = (
    "GradientBoostingRegressor",
    "GradientBoostingClassifier",
    "RandomForestRegressor",
    "RandomForestClassifier",
    "ExtraTreesRegressor",
    "ExtraTreesClassifier",
    "DecisionTreeRegressor",
    "DecisionTreeClassifier",
    "XGBRegressor",
    "XGBClassifier",
    "LGBMRegressor",
    "LGBMClassifier",
)
LINEAR_TYPES = ("LinearRegression", "Ridge", "Lasso", "ElasticNet", "BayesianRidge")


def default_model_classes(task: str, random_state: int | None = 0) -> list[tuple[str, BaseEstimator]]:
    """Return the default set of model classes used to quantify model dependence.

    The first entry is the reference model class. Gradient boosting, random
    forest and a multilayer perceptron are used, as suggested in the study
    design guide.
    """
    if task == "regression":
        return [
            ("gradient_boosting", GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=random_state)),
            ("random_forest", RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=random_state, n_jobs=1)),
            (
                "mlp",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800, early_stopping=True, random_state=random_state)),
                    ]
                ),
            ),
        ]
    if task == "classification":
        return [
            ("gradient_boosting", GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=random_state)),
            ("random_forest", RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=random_state, n_jobs=1)),
            (
                "mlp",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800, early_stopping=True, random_state=random_state)),
                    ]
                ),
            ),
        ]
    raise ValueError("task must be 'regression' or 'classification'")


def _final_estimator(model):
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model


def _estimator_task(model) -> str | None:
    """Task implied by an estimator: 'classification', 'regression' or None when unknown.

    scikit learn's is_classifier sees through pipelines. Objects that are not
    scikit learn estimators and do not expose ``_estimator_type`` give None,
    and the task is then inferred from the target values.
    """
    try:
        return "classification" if is_classifier(model) else "regression"
    except Exception:
        pass
    kind = getattr(_final_estimator(model), "_estimator_type", None)
    if kind is None:
        return None
    return "classification" if kind == "classifier" else "regression"


def _class_column(classes, target_class) -> int:
    """Column of predict_proba that corresponds to ``target_class``.

    ``target_class`` is matched first as one of the class labels; otherwise it
    must be an integer index into the sorted classes. A ValueError naming the
    available classes is raised when neither interpretation applies.
    """
    classes = np.asarray(list(classes)).tolist()
    for i, c in enumerate(classes):
        try:
            if bool(c == target_class):
                return i
        except Exception:
            continue
    if isinstance(target_class, numbers.Integral) and 0 <= int(target_class) < len(classes):
        return int(target_class)
    raise ValueError(f"target_class {target_class!r} is neither one of the classes {classes} nor an index into them.")


def output_function(model, task: str, target_class=1) -> Callable[[np.ndarray], np.ndarray]:
    """Return the scalar function that is explained.

    Regressors are explained through predict; classifiers through the
    predicted probability of the target class, which is given either as one of
    the class labels of the fitted estimator or as an index into its sorted
    classes.
    """
    if task == "classification":
        col = _class_column(getattr(_final_estimator(model), "classes_", [0, 1]), target_class)

        def g(Z):
            return np.asarray(model.predict_proba(np.asarray(Z)))[:, col]

        return g

    def g_reg(Z):
        return np.asarray(model.predict(np.asarray(Z)), dtype=float)

    return g_reg


def choose_explainer(model, n_features: int, requested: str = "auto") -> str:
    """Resolve the 'auto' explainer choice for a fitted model.

    Estimators listed in TREE_TYPES are explained with TreeSHAP, except a
    GradientBoostingClassifier with more than two classes, which shap supports
    only for binary problems and which therefore takes the model agnostic
    branch. Linear models use the linear explainer. Every other model is
    explained exactly up to 10 features and by permutation sampling above.
    """
    if requested != "auto":
        return requested
    est = _final_estimator(model)
    name = type(est).__name__
    if not isinstance(model, Pipeline):
        if name in TREE_TYPES:
            multiclass_gbc = name == "GradientBoostingClassifier" and len(getattr(est, "classes_", ())) > 2
            if not multiclass_gbc:
                return "tree"
        elif name in LINEAR_TYPES:
            return "linear"
    if n_features <= 10:
        return "exact"
    return "permutation"


def shapley_values(
    model,
    X_explain: np.ndarray,
    background: np.ndarray,
    task: str,
    explainer: str = "auto",
    target_class=1,
    kernel_nsamples: int | str = "auto",
    permutation_max_evals: int | str = "auto",
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Compute feature level Shapley values for X_explain.

    Returns the matrix of Shapley values with shape (n_explain, n_features) and
    the expected value of the explained output over the background sample. All
    estimators use the interventional (marginal) expectation with respect to
    the whole background sample, which is handed to shap as an explicit masker
    so that no subsampling takes place, and the efficiency property holds with
    respect to the mean background output. The kernel estimator runs without
    the l1 feature preselection of shap, so that it targets the full Shapley
    vector, and it is seeded from ``seed`` through the global numpy state,
    which is restored afterwards, because shap's KernelExplainer draws its
    coalitions from that state.
    """
    import shap

    X_explain = np.asarray(X_explain, dtype=float)
    background = np.asarray(background, dtype=float)
    d = X_explain.shape[1]
    method = choose_explainer(model, d, explainer)
    g = output_function(model, task, target_class)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "tree":
            masker = shap.maskers.Independent(background, max_samples=background.shape[0])
            kwargs = {"data": masker, "feature_perturbation": "interventional"}
            if task == "classification":
                kwargs["model_output"] = "probability"
            expl = shap.TreeExplainer(model, **kwargs)
            values = np.asarray(expl.shap_values(X_explain, check_additivity=False))
            expected = np.asarray(expl.expected_value, dtype=float).ravel()
            if task == "classification":
                classes = list(getattr(_final_estimator(model), "classes_", [0, 1]))
                col = _class_column(classes, target_class)
                if values.ndim == 3:
                    values = values[:, :, col]
                    expected = expected[col : col + 1]
                elif len(classes) == 2 and col == 0:
                    # binary gradient boosting: shap returns the probability of classes_[1]
                    values = -values
                    expected = 1.0 - expected
            return values, float(expected[-1])
        if method == "linear":
            expl = shap.LinearExplainer(model, background)
            values = np.asarray(expl.shap_values(X_explain))
            return values, float(np.ravel(expl.expected_value)[-1])
        if method == "kernel":
            expl = shap.KernelExplainer(g, background)
            state = np.random.get_state()
            np.random.seed(int(seed) % (2**32))
            try:
                values = np.asarray(expl.shap_values(X_explain, nsamples=kernel_nsamples, l1_reg=0, silent=True))
            finally:
                np.random.set_state(state)
            return values, float(np.ravel(expl.expected_value)[-1])
        if method == "exact":
            masker = shap.maskers.Independent(background, max_samples=background.shape[0])
            expl = shap.ExactExplainer(g, masker)
            res = expl(X_explain)
            return np.asarray(res.values), float(np.ravel(res.base_values)[0])
        if method == "permutation":
            masker = shap.maskers.Independent(background, max_samples=background.shape[0])
            expl = shap.PermutationExplainer(g, masker, seed=seed)
            max_evals = permutation_max_evals if permutation_max_evals != "auto" else max(2 * d + 1, 500)
            res = expl(X_explain, max_evals=max_evals)
            return np.asarray(res.values), float(np.ravel(res.base_values)[0])
    raise ValueError(f"Unknown explainer '{method}'.")


@dataclass
class AttributionResult:
    """Pool of feature level attributions.

    Attributes
    ----------
    values : ndarray of shape (B, M, n_explain, n_features)
        Feature level Shapley values for every resample b and model class m.
    expected : ndarray of shape (B, M)
        Expected output over the background sample of every pool member.
    outputs : ndarray of shape (B, M, n_explain)
        Output of every refitted model on the explained instances.
    reference_output : ndarray of shape (n_explain,)
        Output of the reference model (b = 0, m = 0) on the explained instances.
    model_class_names : list of str
    resample_indices : list of ndarray
        Row indices of the training data used by every resample; index 0 is the
        identity (original data).
    seeds : ndarray of shape (B, M)
        Integer seeds used for every pool member, for logging.
    classes : ndarray or None
        Sorted class labels of a classification target; None for regression.
    explained_class : label or None
        Class whose probability the pool explains; None for regression.
    """

    values: np.ndarray
    expected: np.ndarray
    outputs: np.ndarray
    reference_output: np.ndarray
    model_class_names: list[str]
    resample_indices: list[np.ndarray] = field(default_factory=list)
    seeds: np.ndarray | None = None
    classes: np.ndarray | None = None
    explained_class: int | str | None = None

    @property
    def n_resamples(self) -> int:
        return self.values.shape[0]

    @property
    def n_model_classes(self) -> int:
        return self.values.shape[1]

    @property
    def n_explain(self) -> int:
        return self.values.shape[2]

    @property
    def n_features(self) -> int:
        return self.values.shape[3]

    def pooled(self) -> np.ndarray:
        """Return the pool flattened over (b, m): shape (B * M, n_explain, n_features)."""
        return self.values.reshape(-1, self.n_explain, self.n_features)

    def subset(self, resamples=None, model_classes=None) -> "AttributionResult":
        """Restrict the pool to some resamples and model classes (used by the ablations)."""
        b_idx = np.arange(self.n_resamples) if resamples is None else np.asarray(resamples, dtype=int)
        m_idx = np.arange(self.n_model_classes) if model_classes is None else np.asarray(model_classes, dtype=int)
        vals = self.values[b_idx][:, m_idx]
        return AttributionResult(
            values=vals,
            expected=self.expected[b_idx][:, m_idx],
            outputs=self.outputs[b_idx][:, m_idx],
            reference_output=self.outputs[b_idx[0], m_idx[0]].copy(),
            model_class_names=[self.model_class_names[m] for m in m_idx],
            resample_indices=[self.resample_indices[b] for b in b_idx] if self.resample_indices else [],
            seeds=None if self.seeds is None else self.seeds[b_idx][:, m_idx],
            classes=self.classes,
            explained_class=self.explained_class,
        )


def _fit_one(model, X, y, idx, seed):
    est = clone(model)
    params = est.get_params()
    for key in params:
        if key == "random_state" or key.endswith("__random_state"):
            try:
                est.set_params(**{key: int(seed)})
            except Exception:
                pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(X[idx], y[idx])
    return est


def _attribute_one(model, X_train, idx, X_explain, task, explainer, background_size, target_class, kernel_nsamples, permutation_max_evals, seed):
    rng = np.random.default_rng(seed)
    pool_rows = X_train[idx]
    if background_size is not None and background_size < pool_rows.shape[0]:
        bg = pool_rows[rng.choice(pool_rows.shape[0], size=background_size, replace=False)]
    else:
        bg = pool_rows
    values, expected = shapley_values(
        model,
        X_explain,
        bg,
        task,
        explainer=explainer,
        target_class=target_class,
        kernel_nsamples=kernel_nsamples,
        permutation_max_evals=permutation_max_evals,
        seed=seed,
    )
    g = output_function(model, task, target_class)
    return values, expected, g(X_explain)


class AttributionEngine:
    """Bootstrap by model class pool of feature level Shapley values.

    Parameters
    ----------
    model_classes : list of estimators or of (name, estimator) pairs, or None
        The first entry is the reference model class. None selects the
        defaults of :func:`default_model_classes` for the detected task.
    n_resamples : int
        Number of resamples B. Resample 0 is the original training data and
        resamples 1 to B - 1 are bootstrap draws with replacement.
    background_size : int or None
        Size of the background sample drawn from the resample; None uses the
        whole resample.
    explainer : {'auto', 'tree', 'linear', 'kernel', 'exact', 'permutation'}
    task : {'auto', 'regression', 'classification'}
        'auto' takes the task from the reference model class when model
        classes are given (a classifier gives classification) and otherwise
        from the target: binary and multiclass targets in the sense of
        sklearn.utils.multiclass.type_of_target are classified and
        continuous targets are regressed, so an integer valued regression
        target needs an explicit task. An explicit task that contradicts the
        estimator type raises a ValueError.
    target_class : int or label
        Class whose probability is explained for classifiers, given either as
        one of the class labels or as an index into the sorted classes
        (``classes_``). Classification targets are encoded to 0, ..., C - 1
        before fitting, so the default 1 is the second class in sorted order,
        that is the positive class of a 0/1 coding.
    n_jobs : int
        Parallel workers over (resample, model class) pairs.
    random_state : int or None

    Attributes
    ----------
    task_ : str
    classes_ : ndarray
        Sorted class labels; set for classification only.
    target_class_ : int or None
        Index into ``classes_`` of the explained class; None for regression.
    """

    def __init__(
        self,
        model_classes=None,
        n_resamples: int = 50,
        background_size: int | None = 100,
        explainer: str = "auto",
        task: str = "auto",
        target_class=1,
        kernel_nsamples: int | str = "auto",
        permutation_max_evals: int | str = "auto",
        n_jobs: int = 1,
        random_state: int | None = None,
        verbose: int = 0,
    ):
        self.model_classes = model_classes
        self.n_resamples = n_resamples
        self.background_size = background_size
        self.explainer = explainer
        self.task = task
        self.target_class = target_class
        self.kernel_nsamples = kernel_nsamples
        self.permutation_max_evals = permutation_max_evals
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------
    def _resolve_task(self, y) -> str:
        est_task = None
        if self.model_classes is not None:
            first = next(iter(self.model_classes), None)
            if first is not None:
                est_task = _estimator_task(first[1] if isinstance(first, tuple) else first)
        if self.task != "auto":
            if est_task is not None and est_task != self.task:
                kind = "classifier" if est_task == "classification" else "regressor"
                raise ValueError(f"task='{self.task}' contradicts the reference model class, which is a {kind}.")
            return self.task
        if est_task is not None:
            return est_task
        y = np.asarray(y)
        if y.dtype.kind in "OUSb":
            return "classification"
        if type_of_target(y) in ("binary", "multiclass"):
            return "classification"
        return "regression"

    def _resolve_model_classes(self, task: str) -> list[tuple[str, BaseEstimator]]:
        if self.model_classes is None:
            return default_model_classes(task, random_state=self.random_state or 0)
        out = []
        for i, item in enumerate(self.model_classes):
            if isinstance(item, tuple):
                out.append((str(item[0]), item[1]))
            else:
                out.append((f"{type(_final_estimator(item)).__name__}_{i}", item))
        return out

    def fit(self, X, y):
        """Fit the pool of models: every model class on every resample.

        Classification targets are encoded to the integer codes of their
        sorted labels, stored in ``classes_``, and the explained class is
        resolved once into ``target_class_``.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.task_ = self._resolve_task(y)
        if self.task_ == "classification":
            self.classes_, y = np.unique(y, return_inverse=True)
            y = np.asarray(y).ravel()
            self.target_class_ = _class_column(self.classes_, self.target_class)
        else:
            self.target_class_ = None
        self.model_classes_ = self._resolve_model_classes(self.task_)
        B, M = int(self.n_resamples), len(self.model_classes_)
        if B < 1 or M < 1:
            raise ValueError("n_resamples and the number of model classes must be at least one.")
        rng = make_rng(self.random_state)
        n = X.shape[0]
        self.resample_indices_ = [np.arange(n)]
        for _ in range(B - 1):
            self.resample_indices_.append(rng.choice(n, size=n, replace=True))
        self.seeds_ = np.asarray(spawn_seeds(rng, B * M)).reshape(B, M)
        tasks = [(b, m) for b in range(B) for m in range(M)]
        fitted = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_fit_one)(self.model_classes_[m][1], X, y, self.resample_indices_[b], self.seeds_[b, m]) for b, m in tasks
        )
        self.models_ = [[None] * M for _ in range(B)]
        for (b, m), est in zip(tasks, fitted):
            self.models_[b][m] = est
        self.X_train_ = X
        self.n_features_in_ = X.shape[1]
        return self

    @property
    def reference_model_(self):
        return self.models_[0][0]

    def explain(self, X_explain) -> AttributionResult:
        """Compute feature level Shapley values of every pool member for X_explain."""
        X_explain = np.asarray(X_explain, dtype=float)
        B, M = len(self.models_), len(self.model_classes_)
        tasks = [(b, m) for b in range(B) for m in range(M)]
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_attribute_one)(
                self.models_[b][m],
                self.X_train_,
                self.resample_indices_[b],
                X_explain,
                self.task_,
                self.explainer,
                self.background_size,
                self.target_class_,
                self.kernel_nsamples,
                self.permutation_max_evals,
                int(self.seeds_[b, m]) + 1,
            )
            for b, m in tasks
        )
        n, d = X_explain.shape
        values = np.zeros((B, M, n, d))
        expected = np.zeros((B, M))
        outputs = np.zeros((B, M, n))
        for (b, m), (v, e, o) in zip(tasks, results):
            values[b, m] = v
            expected[b, m] = e
            outputs[b, m] = o
        classification = self.task_ == "classification"
        return AttributionResult(
            values=values,
            expected=expected,
            outputs=outputs,
            reference_output=outputs[0, 0].copy(),
            model_class_names=[name for name, _ in self.model_classes_],
            resample_indices=self.resample_indices_,
            seeds=self.seeds_,
            classes=self.classes_ if classification else None,
            explained_class=self.classes_[self.target_class_].item() if classification else None,
        )

    def fit_explain(self, X, y, X_explain) -> AttributionResult:
        return self.fit(X, y).explain(X_explain)


def aggregate_to_concepts(values: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Equation 1: concept level attribution from feature level attribution.

    phi_k = sum_j u_jk phi_j, applied along the last axis of ``values``. Works
    for pools of shape (..., n_features) and returns (..., n_concepts).
    """
    values = np.asarray(values, dtype=float)
    U = np.asarray(U, dtype=float)
    if values.shape[-1] != U.shape[0]:
        raise ValueError("Last axis of values must equal the number of features in U.")
    return values @ U


def redistribute_to_features(concept_values: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Defuzzified feature level vector from concept centroids.

    Each concept value is redistributed to the features in proportion to their
    membership in the concept: psi_j = sum_k c_k u_jk / sum_j' u_j'k. With an
    identity membership matrix this returns the concept values unchanged, so
    the operation reduces to ordinary SHAP for singleton crisp concepts.
    """
    U = np.asarray(U, dtype=float)
    mass = U.sum(axis=0)
    mass = np.where(mass <= 0, 1.0, mass)
    W = U / mass[None, :]
    return np.asarray(concept_values, dtype=float) @ W.T
