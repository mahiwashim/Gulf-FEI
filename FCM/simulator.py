"""FCM scenario simulator — Kosko activation propagation with clamping.

Given a signed adjacency matrix A and an activation vector a(0), iterate:

        a(t+1) = f( a(t) + A^T · a(t) )

where f is a squashing function (sigmoid or tanh). Concepts marked as
"drivers" stay clamped to their initial value at every step (what-if
scenario analysis). Returns the full trajectory plus the equilibrium.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def _sigmoid(x: np.ndarray, lam: float = 1.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-lam * x))


class FCMSimulator:
    def __init__(self, concepts: List[str], adjacency: List[List[float]]):
        if not concepts:
            raise ValueError("concepts must be a non-empty list")
        self.concepts: List[str] = list(concepts)
        self.n: int = len(self.concepts)
        self.idx: Dict[str, int] = {c: i for i, c in enumerate(self.concepts)}
        self.A: np.ndarray = np.asarray(adjacency, dtype=float)
        if self.A.shape != (self.n, self.n):
            raise ValueError(
                f"adjacency shape {self.A.shape} does not match {self.n} concepts"
            )

    # ------------------------------------------------------------------
    def simulate(
        self,
        activation: Dict[str, float],
        steps: int = 30,
        squash: str = "sigmoid",
        lam: float = 1.0,
        clamp_drivers: bool = True,
        tol: float = 1e-4,
    ) -> Dict:
        """Propagate activation through the signed adjacency matrix.

        Args:
            activation: {concept_name: initial_value in [-1, 1]}.
            steps:      maximum iterations.
            squash:     "sigmoid" (default, outputs [0,1]) or "tanh" ([-1,1]).
            lam:        steepness of the sigmoid squashing.
            clamp_drivers: keep initially-set concepts at their initial value
                        every step (what-if analysis).
            tol:        convergence tolerance (max |Δa|).

        Returns: dict with trajectory, final state, convergence flag, and
            a per-concept "influence" (change from baseline).
        """
        if squash not in ("sigmoid", "tanh", "none"):
            raise ValueError(f"unknown squash={squash!r}")

        # Baseline initial state — user-provided concepts override, rest = 0
        a = np.zeros(self.n, dtype=float)
        driver_idx: List[int] = []
        for concept, value in activation.items():
            if concept in self.idx:
                i = self.idx[concept]
                a[i] = float(value)
                driver_idx.append(i)

        initial = a.copy()

        history: List[np.ndarray] = [a.copy()]
        converged = False

        for _ in range(steps):
            # Kosko propagation: each concept accumulates weighted inputs
            propagated = a + self.A.T @ a

            if squash == "sigmoid":
                new_a = _sigmoid(propagated, lam=lam)
                # Sigmoid outputs [0, 1]; rescale so baseline (0) maps to 0
                new_a = 2.0 * new_a - 1.0
            elif squash == "tanh":
                new_a = np.tanh(propagated * lam)
            else:
                new_a = propagated

            if clamp_drivers:
                for i in driver_idx:
                    new_a[i] = initial[i]

            diff = float(np.max(np.abs(new_a - a)))
            a = new_a
            history.append(a.copy())

            if diff < tol:
                converged = True
                break

        influence = a - initial
        ranked = sorted(
            ((c, float(a[i]), float(influence[i])) for i, c in enumerate(self.concepts)),
            key=lambda x: abs(x[2]),
            reverse=True,
        )

        return {
            "converged":   converged,
            "iterations":  len(history) - 1,
            "drivers":     list(activation.keys()),
            "squash":      squash,
            "lam":         lam,
            "final":       {c: float(a[i]) for i, c in enumerate(self.concepts)},
            "influence":   {c: float(influence[i]) for i, c in enumerate(self.concepts)},
            "top_effects": [
                {"concept": c, "final": final, "influence": inf}
                for c, final, inf in ranked[:15]
            ],
            "history": [
                {c: float(h[i]) for i, c in enumerate(self.concepts)}
                for h in history
            ],
        }
