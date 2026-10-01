"""Synthetic data generating processes with known concept structure (Phase A).

Three families are implemented. In the additive family the response is a sum
of functions of latent factors, each factor being measured by a block of
correlated observed features. In the interaction family a product term between
the first two factors is added. In the redundancy family exact and near
duplicates of features are appended, which produces the credit splitting
failure of feature level attribution.

Every latent factor z_k is standard normal and every observed feature of block
k is x_j = sqrt(rho) z_k + sqrt(1 - rho) e_j, so that the within block
correlation equals rho and every feature is standard normal. Given the block,
the factor is normal with mean c_k * sum_j x_j and variance tau_k^2, which
gives the true regression function and the true concept attributions in
closed form (Gauss Hermite quadrature is used for nonlinear links).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from pfca.utils import crisp_membership, make_rng

LINKS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "linear": lambda z: z,
    "sin": lambda z: np.sin(1.5 * z),
    "tanh": lambda z: np.tanh(1.5 * z),
    "square": lambda z: z**2 - 1.0,
    "cubic": lambda z: 0.3 * z**3,
}
_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite_e.hermegauss(40)


@dataclass
class SyntheticProblem:
    """A synthetic problem with known ground truth."""

    family: str
    X: np.ndarray
    y: np.ndarray
    blocks: list[np.ndarray]
    rho: float
    coefficients: np.ndarray
    links: list[str]
    interaction: float
    noise_sd: float
    duplicates: list[tuple[int, int, float]] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)

    @property
    def n_factors(self) -> int:
        return len(self.blocks)

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def true_membership(self) -> np.ndarray:
        labels = np.zeros(self.n_features, dtype=int)
        for k, b in enumerate(self.blocks):
            labels[b] = k
        return crisp_membership(labels, self.n_factors)

    @property
    def block_labels(self) -> np.ndarray:
        return np.argmax(self.true_membership, axis=1)

    # ------------------------------------------------------------------
    def _posterior(self, X: np.ndarray, k: int) -> tuple[np.ndarray, float]:
        """Posterior mean and variance of factor k given its block of original (non duplicate) features."""
        idx = [j for j in self.blocks[k] if j < self._n_original]
        nk = len(idx)
        rho = self.rho
        c = np.sqrt(rho) / (1.0 + (nk - 1) * rho)
        mu = c * X[:, idx].sum(axis=1)
        tau2 = 1.0 - nk * rho / (1.0 + (nk - 1) * rho)
        return mu, max(tau2, 0.0)

    @property
    def _n_original(self) -> int:
        return self.n_features - len(self.duplicates)

    def _expected_link(self, k: int, X: np.ndarray) -> np.ndarray:
        """E[g_k(z_k) | x] by Gauss Hermite quadrature."""
        mu, tau2 = self._posterior(X, k)
        g = LINKS[self.links[k]]
        if self.links[k] == "linear":
            return mu
        tau = np.sqrt(tau2)
        z = mu[:, None] + tau * _GH_NODES[None, :]
        return (g(z) * _GH_WEIGHTS[None, :]).sum(axis=1) / np.sqrt(2 * np.pi)

    def _link_mean(self, k: int) -> float:
        g = LINKS[self.links[k]]
        return float((g(_GH_NODES) * _GH_WEIGHTS).sum() / np.sqrt(2 * np.pi))

    def true_concept_attributions(self, X: np.ndarray) -> np.ndarray:
        """True attribution of every factor for every row of X, shape (n, n_factors).

        Because the regression function is additive across independent blocks,
        the Shapley value of block k with the marginal baseline is the centered
        block contribution. For the interaction term gamma z_1 z_2 the Shapley
        value splits the centered product equally between the two factors.
        """
        X = np.asarray(X, dtype=float)
        A = np.zeros((X.shape[0], self.n_factors))
        for k in range(self.n_factors):
            A[:, k] = self.coefficients[k] * (self._expected_link(k, X) - self._link_mean(k))
        if self.interaction != 0.0 and self.n_factors >= 2:
            m1, _ = self._posterior(X, 0)
            m2, _ = self._posterior(X, 1)
            prod = self.interaction * m1 * m2
            A[:, 0] += 0.5 * prod
            A[:, 1] += 0.5 * prod
        return A

    def true_intercept(self) -> float:
        """E[f(x)], the population mean of the regression function."""
        return float(sum(self.coefficients[k] * self._link_mean(k) for k in range(self.n_factors)))

    def true_function(self, X: np.ndarray) -> np.ndarray:
        """E[y | x], the function a consistent model estimates."""
        X = np.asarray(X, dtype=float)
        f = np.zeros(X.shape[0])
        for k in range(self.n_factors):
            f += self.coefficients[k] * self._expected_link(k, X)
        if self.interaction != 0.0 and self.n_factors >= 2:
            m1, _ = self._posterior(X, 0)
            m2, _ = self._posterior(X, 1)
            f += self.interaction * m1 * m2
        return f

    def true_global_importance(self, X: np.ndarray) -> np.ndarray:
        return np.abs(self.true_concept_attributions(X)).mean(axis=0)

    def sample(self, n: int, random_state=None) -> tuple[np.ndarray, np.ndarray]:
        """Draw a fresh sample from the same process (for held out evaluation)."""
        gen = make_synthetic(
            family=self.family,
            n_samples=n,
            n_features=self._n_original,
            rho=self.rho,
            n_factors=self.n_factors,
            random_state=random_state,
            coefficients=self.coefficients,
            links=self.links,
            interaction=self.interaction,
            noise_sd=self.noise_sd,
            duplicates=self.duplicates,
        )
        return gen.X, gen.y


def default_blocks(n_features: int, n_factors: int) -> list[np.ndarray]:
    sizes = np.full(n_factors, n_features // n_factors)
    sizes[: n_features % n_factors] += 1
    blocks, start = [], 0
    for s in sizes:
        blocks.append(np.arange(start, start + s))
        start += s
    return blocks


def make_synthetic(
    family: str = "additive",
    n_samples: int = 500,
    n_features: int = 30,
    rho: float = 0.6,
    n_factors: int | None = None,
    random_state=None,
    nonlinear: bool = True,
    coefficients: np.ndarray | None = None,
    links: list[str] | None = None,
    interaction: float | None = None,
    noise_sd: float | None = None,
    signal_to_noise: float = 4.0,
    n_duplicates: int | None = None,
    duplicate_noise: float = 0.1,
    duplicates: list[tuple[int, int, float]] | None = None,
) -> SyntheticProblem:
    """Generate a synthetic problem of the requested family.

    Parameters
    ----------
    family : {'additive', 'interaction', 'redundancy'}
    n_samples, n_features, rho : design factors of Phase A; rho must lie in [0, 1], the range
        in which the generating equations and the posterior formulas are valid
    n_factors : number of latent factors; default round(sqrt(n_features)) with a minimum of two
    nonlinear : use a mix of link functions (otherwise all linear)
    coefficients, links : one entry per factor when given, every link being a key of LINKS
    signal_to_noise : ratio of the variance of the signal to the noise variance
    n_duplicates : redundancy family, number of appended duplicate features (default n_factors)
    duplicate_noise : standard deviation of the perturbation of near duplicates (exact copies use 0)
    duplicates : explicit list of (source feature, position, noise sd) used when resampling

    A value outside these ranges raises ValueError.
    """
    if family not in ("additive", "interaction", "redundancy"):
        raise ValueError("family must be 'additive', 'interaction' or 'redundancy'")
    if not 0.0 <= float(rho) <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    rng = make_rng(random_state)
    F = int(n_factors) if n_factors is not None else max(2, int(round(np.sqrt(n_features))))
    F = min(F, n_features)
    blocks = default_blocks(n_features, F)
    if coefficients is None:
        coefficients = np.linspace(2.0, 0.5, F)
        coefficients = coefficients * rng.choice([-1.0, 1.0], size=F)
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape != (F,):
        raise ValueError(f"coefficients must have one entry per factor, expected {F} and got {coefficients.size}")
    if links is None:
        if nonlinear:
            cycle = ["linear", "sin", "tanh", "square", "cubic"]
            links = [cycle[k % len(cycle)] for k in range(F)]
        else:
            links = ["linear"] * F
    links = list(links)
    if len(links) != F:
        raise ValueError(f"links must have one entry per factor, expected {F} and got {len(links)}")
    unknown = [name for name in links if name not in LINKS]
    if unknown:
        raise ValueError(f"unknown link function {unknown}; the available links are {sorted(LINKS)}")
    if interaction is None:
        interaction = 1.5 if family == "interaction" else 0.0
    Z = rng.normal(size=(n_samples, F))
    X = np.zeros((n_samples, n_features))
    for k, b in enumerate(blocks):
        E = rng.normal(size=(n_samples, b.size))
        X[:, b] = np.sqrt(rho) * Z[:, [k]] + np.sqrt(1.0 - rho) * E
    signal = np.zeros(n_samples)
    for k in range(F):
        signal += coefficients[k] * LINKS[links[k]](Z[:, k])
    if interaction != 0.0 and F >= 2:
        signal += interaction * Z[:, 0] * Z[:, 1]
    if noise_sd is None:
        noise_sd = float(np.std(signal) / np.sqrt(signal_to_noise)) if np.std(signal) > 0 else 1.0
    y = signal + rng.normal(size=n_samples) * noise_sd
    dup_list: list[tuple[int, int, float]] = []
    if family == "redundancy":
        if duplicates is not None:
            dup_list = [tuple(t) for t in duplicates]
        else:
            nd = int(n_duplicates) if n_duplicates is not None else F
            sources = [int(blocks[k % F][0]) for k in range(nd)]
            for i, src in enumerate(sources):
                sd = 0.0 if i % 2 == 0 else float(duplicate_noise)
                dup_list.append((src, n_features + i, sd))
        extra = np.zeros((n_samples, len(dup_list)))
        for i, (src, _, sd) in enumerate(dup_list):
            extra[:, i] = X[:, src] + (rng.normal(size=n_samples) * sd if sd > 0 else 0.0)
        X = np.hstack([X, extra])
        blocks = [b.copy() for b in blocks]
        for src, pos, _ in dup_list:
            k = next(k for k, b in enumerate(blocks) if src in b)
            blocks[k] = np.append(blocks[k], pos)
    names = [f"x{j}" for j in range(n_features)] + [f"dup{i}_of_x{src}" for i, (src, _, _) in enumerate(dup_list)]
    return SyntheticProblem(family, X, y, blocks, float(rho), coefficients, links, float(interaction), float(noise_sd), dup_list, names)


def phase_a_grid(
    families=("additive", "interaction", "redundancy"),
    n_samples=(200, 500, 2000),
    n_features=(10, 30, 100),
    rhos=(0.3, 0.6, 0.9),
    n_seeds: int = 50,
) -> list[dict]:
    """Full factorial design of Phase A."""
    cells = []
    for fam in families:
        for n in n_samples:
            for d in n_features:
                for rho in rhos:
                    for seed in range(n_seeds):
                        cells.append({"family": fam, "n_samples": int(n), "n_features": int(d), "rho": float(rho), "seed": int(seed)})
    return cells
