from __future__ import annotations

import importlib
import os
from typing import Dict, List

from .schemas import FCMMap


class PyFCMAdapter:
    """
    Adapter designed around PyFCM's adjacency-matrix workflow.

    The upstream repo is script-oriented and commonly expects adjacency matrices
    exported from tools like Mental Modeler. This adapter therefore accepts a
    generated matrix and either:
      1) calls a locally installed/vendored PyFCM module if present, or
      2) falls back to a compact in-app propagation engine for development.
    """

    def __init__(self) -> None:
        self.backend_name = 'fallback'
        self.backend = self._try_import_backend()

    def _try_import_backend(self):
        module_candidates = [
            os.getenv('PYFCM_IMPORT'),
            'PyFCM',
            'pyfcm',
        ]
        for mod_name in module_candidates:
            if not mod_name:
                continue
            try:
                mod = importlib.import_module(mod_name)
                self.backend_name = mod_name
                return mod
            except Exception:
                continue
        return None

    def run_scenario(self, fcm_map: FCMMap, activation: Dict[str, float], steps: int = 5) -> Dict[str, float]:
        if self.backend is not None and hasattr(self.backend, 'run_scenario'):
            return self.backend.run_scenario(fcm_map.adjacency_matrix, activation=activation, steps=steps)
        return self._fallback_simulation(fcm_map, activation=activation, steps=steps)

    def _fallback_simulation(self, fcm_map: FCMMap, activation: Dict[str, float], steps: int) -> Dict[str, float]:
        concepts = fcm_map.concepts
        idx = {c: i for i, c in enumerate(concepts)}
        state: List[float] = [0.0 for _ in concepts]
        for concept, value in activation.items():
            if concept in idx:
                state[idx[concept]] = max(-1.0, min(1.0, float(value)))

        for _ in range(steps):
            new_state = state[:]
            for target_i in range(len(concepts)):
                influence = 0.0
                for source_i in range(len(concepts)):
                    influence += state[source_i] * fcm_map.adjacency_matrix[source_i][target_i]
                new_state[target_i] = max(-1.0, min(1.0, influence + state[target_i]))
            state = new_state

        return {concepts[i]: round(state[i], 4) for i in range(len(concepts))}
