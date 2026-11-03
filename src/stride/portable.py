"""Library-version-independent predictors for the deployment artefact.

Pickled scikit-learn estimators only load reliably under the exact scikit-learn
version that created them. For run_episode.py we therefore export the fitted
parameters (scaler, PCA, logistic regression) into these NumPy-only classes.
The maths is identical to the fitted estimators (tested in tests/test_pipeline.py).
"""
from __future__ import annotations

import numpy as np


class LinearSoftmax:
    """Multinomial / binary logistic-regression predictor from coef_, intercept_, classes_."""

    def __init__(self, coef, intercept, classes):
        self.coef_ = np.asarray(coef, dtype=np.float64)
        self.intercept_ = np.asarray(intercept, dtype=np.float64)
        self.classes_ = np.array(list(classes), dtype=object)

    @classmethod
    def from_sklearn(cls, lr):
        return cls(lr.coef_, lr.intercept_, lr.classes_)

    def predict_proba(self, X):
        s = np.asarray(X, dtype=np.float64) @ self.coef_.T + self.intercept_
        if s.shape[1] == 1:                       # binary: sklearn stores one row for classes_[1]
            p1 = 1.0 / (1.0 + np.exp(-s[:, 0]))
            return np.c_[1 - p1, p1]
        s = s - s.max(1, keepdims=True)
        e = np.exp(s)
        return e / e.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class PortablePCA:
    def __init__(self, mean, components):
        self.mean_ = np.asarray(mean, dtype=np.float64)
        self.components_ = np.asarray(components, dtype=np.float64)

    def transform(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean_) @ self.components_.T


class PortableRouter:
    """Standardiser + logistic regression on the static profile (same as RoomRouter)."""

    def __init__(self, router):
        sc, lr = router.model.steps[0][1], router.model.steps[-1][1]
        self.mean_, self.scale_ = np.asarray(sc.mean_, float), np.asarray(sc.scale_, float)
        self.lr = LinearSoftmax.from_sklearn(lr)

    def _f(self, W):
        from .router import static_features
        return (static_features(W) - self.mean_) / self.scale_

    def predict_proba(self, W):
        return self.lr.predict_proba(self._f(W))

    def predict(self, W):
        return self.lr.predict(self._f(W))


def make_portable_sapt(model):
    if getattr(model, "lr", None) is not None and not isinstance(model.lr, LinearSoftmax):
        model.lr = LinearSoftmax.from_sklearn(model.lr)
    return model


def make_portable_features(hf):
    if hasattr(hf.pca, "components_") and not isinstance(hf.pca, PortablePCA):
        hf.pca = PortablePCA(hf.pca.mean_, hf.pca.components_)
    return hf
