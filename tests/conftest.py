import numpy as np
import pytest

from pfca.evaluation.synthetic import make_synthetic


@pytest.fixture(scope="session")
def small_problem():
    """Additive problem with three blocks of correlated features and a fresh test sample."""
    p = make_synthetic("additive", n_samples=300, n_features=9, rho=0.8, n_factors=3, random_state=0, nonlinear=False)
    X_test, _ = p.sample(30, random_state=1)
    return p, X_test


class LinearModel:
    """Deterministic model with a known linear function, used for exact checks."""

    def __init__(self, beta):
        self.beta = np.asarray(beta, dtype=float)

    def fit(self, X, y):
        return self

    def get_params(self, deep=True):
        return {"beta": self.beta}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.beta


@pytest.fixture
def linear_model_cls():
    return LinearModel
