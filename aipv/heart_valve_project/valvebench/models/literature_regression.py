"""A literature-sourced closed-form regression baseline (Idea 1's "Hoashi/
Zhang closed form, no learning" spec) -- with no real coefficients bundled.

This environment has no verified access to Hoashi et al. or Zhang et al.'s
actual published pulmonary-annulus-growth formula, so shipping default
coefficients here would mean fabricating numbers and presenting them as a
real citation -- academic misconduct if it ended up in a science fair
submission. `LiteratureRegressionBaseline` therefore *requires* the caller
to supply real coefficients transcribed from the primary source; it raises
immediately if none are given, and is deliberately NOT registered in
valvebench.core.registry.MODEL_REGISTRY (the registry's `model_cls()`
no-arg construction pattern can't satisfy those required arguments, and a
"literature" baseline that silently falls back to made-up numbers would be
worse than one that doesn't exist).

To use for real: find the published closed-form formula (e.g. in Hoashi et
al. or Zhang et al.), and construct this with the real intercept,
per-feature coefficients, feature order, and citation string.
"""
from typing import Sequence

import numpy as np


class LiteratureRegressionBaseline:
    def __init__(
        self,
        intercept: float,
        coefficients: Sequence[float],
        feature_names: Sequence[str],
        citation: str,
    ):
        if not citation or "REPLACE" in citation.upper():
            raise ValueError(
                "LiteratureRegressionBaseline requires a real citation string "
                "identifying the published source of these coefficients."
            )
        self.intercept = float(intercept)
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.feature_names = list(feature_names)
        self.citation = citation
        if len(self.coefficients) != len(self.feature_names):
            raise ValueError("coefficients and feature_names must have the same length")

    def fit(self, X, y=None):
        """No learning happens here -- this is a fixed published formula.
        fit() only validates that X's feature dimensionality matches, so
        this baseline is interchangeable with the learned baselines in
        valvebench.pipeline.run_pipeline.
        """
        X = np.asarray(X)
        if X.shape[1] != len(self.coefficients):
            raise ValueError(
                f"X has {X.shape[1]} features but this formula was defined over "
                f"{len(self.coefficients)} ({self.feature_names})"
            )
        return self

    def predict(self, X):
        X = np.asarray(X)
        pred = X.dot(self.coefficients) + self.intercept
        # Two targets (Valve_t0, Valve_t12) are expected by the benchmark
        # contract; a single closed-form scalar formula predicts one value,
        # broadcast to both columns unless the caller supplies per-target
        # coefficients (coefficients.ndim == 2).
        if pred.ndim == 1:
            pred = np.column_stack([pred, pred])
        return pred
