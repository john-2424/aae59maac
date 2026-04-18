"""SDP upper bound on lambda_2 for fixed-support edge reweighting.

Formulation (Boyd & Ghosh, "Growing Well-Connected Graphs", 2006):

    maximize   s
    subject to L(w) + (1/n) 1 1^T  -  s * (I - (1/n) 1 1^T)  succeq  0
               0 <= w_e <= w_max
               sum_e w_e <= budget

Here ``L(w) = sum_e w_e (e_i - e_j)(e_i - e_j)^T`` is the weighted Laplacian
restricted to the support's edge set. The optimal ``s*`` equals the largest
achievable ``lambda_2(L(w))`` subject to the constraints, which gives us the
best-case algebraic connectivity any reweighting policy could possibly attain.

Two solvers:
  - ``fdla_upper_bound``: cvxpy + SCS (preferred; install ``cvxpy`` and ``scs``).
  - ``fdla_upper_bound_subgradient``: pure-numpy spectral subgradient ascent on
    lambda_2, used as a fallback if cvxpy isn't installed.
"""

from __future__ import annotations

import numpy as np

from ..envs.common import edge_index_from_support, weights_vector_to_matrix
from ..graphs.laplacian import fiedler_value, laplacian


def _edge_pair_matrix(n: int, edge_index: np.ndarray) -> np.ndarray:
    """Stack of rank-1 Laplacians, shape (m, n, n)."""
    m = edge_index.shape[0]
    out = np.zeros((m, n, n), dtype=np.float64)
    for k, (i, j) in enumerate(edge_index):
        ei = np.zeros(n); ei[i] = 1.0
        ej = np.zeros(n); ej[j] = 1.0
        d = ei - ej
        out[k] = np.outer(d, d)
    return out


def fdla_upper_bound(
    support: np.ndarray, w_max: float, budget: float, solver: str = "SCS"
) -> tuple[np.ndarray, float]:
    """Solve the SDP; returns (W_opt, lambda2_opt)."""
    try:
        import cvxpy as cp
    except ImportError as e:
        raise ImportError(
            "cvxpy is required for fdla_upper_bound; install with `pip install cvxpy scs` "
            "or use fdla_upper_bound_subgradient() instead"
        ) from e

    A = (support > 0).astype(np.float64)
    np.fill_diagonal(A, 0.0)
    n = A.shape[0]
    edge_index = edge_index_from_support(A)
    m = edge_index.shape[0]
    if m == 0:
        raise ValueError("support has no edges")

    E = _edge_pair_matrix(n, edge_index)  # (m, n, n)
    w = cp.Variable(m, nonneg=True)
    s = cp.Variable()
    L = sum(w[k] * cp.Constant(E[k]) for k in range(m))
    J = np.ones((n, n)) / n
    I = np.eye(n)
    constraints = [
        w <= w_max,
        cp.sum(w) <= budget,
        L + cp.Constant(J) - s * cp.Constant(I - J) >> 0,
    ]
    prob = cp.Problem(cp.Maximize(s), constraints)
    prob.solve(solver=solver, verbose=False)
    if w.value is None:
        raise RuntimeError(f"SDP failed to solve: status={prob.status}")

    w_opt = np.asarray(w.value, dtype=np.float64)
    W_opt = weights_vector_to_matrix(w_opt, edge_index, n)
    lam2 = fiedler_value(laplacian(W_opt))
    return W_opt, float(lam2)


def fdla_upper_bound_subgradient(
    support: np.ndarray,
    w_max: float,
    budget: float,
    n_iters: int = 2000,
    step0: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Pure-numpy fallback using the spectral subgradient of ``lambda_2``.

    At the current ``w``, a valid supergradient for lambda_2 is
    ``g_e = (v_i - v_j)^2`` where ``v`` is the Fiedler vector. We ascend on w
    then project onto the feasible box ``[0, w_max]^m`` intersected with the
    simplex-like constraint ``sum w <= budget``. Not as tight as the SDP but
    close on small instances; used only if cvxpy is unavailable.
    """
    from scipy.linalg import eigh

    A = (support > 0).astype(np.float64)
    np.fill_diagonal(A, 0.0)
    n = A.shape[0]
    edge_index = edge_index_from_support(A)
    m = edge_index.shape[0]

    rng = np.random.default_rng(seed)
    w = np.clip(rng.uniform(0.0, w_max, size=m), 0.0, w_max)
    # start at the budget upper bound if over-budget
    if w.sum() > budget:
        w *= budget / w.sum()

    best_lam2 = -np.inf
    best_w = w.copy()
    for t in range(n_iters):
        W = weights_vector_to_matrix(w, edge_index, n)
        L = laplacian(W)
        vals, vecs = eigh(0.5 * (L + L.T))
        lam2 = float(vals[1])
        v = vecs[:, 1]
        g = np.array([(v[i] - v[j]) ** 2 for (i, j) in edge_index], dtype=np.float64)
        step = step0 / (1.0 + 0.01 * t)
        w = w + step * g
        # project onto [0, w_max]
        w = np.clip(w, 0.0, w_max)
        # budget projection (shrink uniformly if over)
        if w.sum() > budget:
            w *= budget / w.sum()
        if lam2 > best_lam2:
            best_lam2 = lam2
            best_w = w.copy()

    W_opt = weights_vector_to_matrix(best_w, edge_index, n)
    return W_opt, float(best_lam2)
