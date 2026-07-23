"""Cross-modal contrastive mapping for image-text embedding alignment."""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
from sklearn.model_selection import train_test_split

from isotrieve.mapping.base import (
    Mapping,
    ValidationReport,
    _check_finite,
    l2_normalize,
)
from isotrieve.mapping.registry import register_mapping
from isotrieve.quality.metrics import pairwise_cosine_stats, topk_retention


def _infonce_loss(
    Z_src: np.ndarray,
    Z_tgt: np.ndarray,
    temperature: float,
) -> float:
    """Compute InfoNCE contrastive loss.

    Given aligned pairs (Z_src[i], Z_tgt[i]), this measures how well
    matched pairs score vs. in-batch negatives.

    Parameters
    ----------
    Z_src, Z_tgt:
        L2-normalized embeddings, shape ``(n, d)``.
    temperature:
        Temperature scaling factor. Lower = sharper distribution.

    Returns
    -------
    Scalar loss value (lower = better alignment).
    """
    n = Z_src.shape[0]
    if n < 2:
        return 0.0

    # Similarity matrix: (n, n)
    logits = Z_src @ Z_tgt.T / temperature
    # Labels: diagonal (each src matches its paired tgt)
    labels = np.arange(n, dtype=np.int64)

    # Numerically stable softmax cross-entropy
    logits_max = logits.max(axis=1, keepdims=True)
    logits_shifted = logits - logits_max
    exp_logits = np.exp(logits_shifted)
    log_sum_exp = np.log(exp_logits.sum(axis=1) + 1e-12)
    # Cross-entropy for each row
    loss_per_row = -logits_shifted[np.arange(n), labels] + log_sum_exp
    return float(loss_per_row.mean())


def _contrastive_validation(
    mapping: Mapping,
    X_src: np.ndarray,
    X_tgt: np.ndarray,
    candidate_pool_tgt: np.ndarray | None = None,
    *,
    n_train: int,
    seed: int,
) -> ValidationReport:
    """Validate a cross-modal mapping with retrieval-native metrics."""
    mapped = mapping.transform(X_src)
    cos = pairwise_cosine_stats(mapped, X_tgt)

    if candidate_pool_tgt is not None and len(candidate_pool_tgt) > len(X_tgt):
        # Full retrieval evaluation: find nearest in candidate pool
        t1 = topk_retention(mapped, candidate_pool_tgt, k=1)
        t10 = topk_retention(mapped, candidate_pool_tgt, k=min(10, len(candidate_pool_tgt)))
    else:
        # Pairwise evaluation (each row matched to its pair)
        t1 = topk_retention(mapped, X_tgt, k=1)
        t10 = topk_retention(mapped, X_tgt, k=min(10, len(X_tgt)))

    return ValidationReport(
        holdout_cosine_mean=cos["mean"],
        holdout_cosine_median=cos["median"],
        holdout_cosine_p5=cos["p5"],
        top1_retention=t1,
        top10_retention=t10,
        n_train=n_train,
        n_holdout=len(X_src) - n_train,
        seed=seed,
    )


@register_mapping
class ContrastiveMapping(Mapping):
    """Cross-modal mapping using contrastive (InfoNCE) fitting.

    Unlike RidgeMapping which minimizes pointwise reconstruction error,
    ContrastiveMapping optimizes for retrieval quality by pulling matched
    pairs together and pushing in-batch negatives apart.

    This is the correct choice when mapping between embedding spaces
    trained with contrastive loss (CLIP, SigLIP, etc.) — i.e., cross-modal
    mapping where source and target embeddings describe different modalities
    (images, text) but share semantic content.

    Parameters
    ----------
    temperature:
        InfoNCE temperature. Lower = sharper, higher = smoother.
        Default 0.07 works well for CLIP-scale embeddings.
    bias:
        If True, fit an affine map with bias term.
    seed:
        RNG seed for holdout split.
    normalize_output:
        If True (default), L2-normalize every transformed row.
    holdout_fraction:
        Fraction of data held out for validation.
    rank:
        Optional SVD rank compression of the learned matrix.
    """

    mapping_type = "contrastive"

    def __init__(
        self,
        temperature: float = 0.07,
        *,
        bias: bool = True,
        seed: int = 0,
        normalize_output: bool = True,
        holdout_fraction: float = 0.1,
        rank: int | None = None,
    ) -> None:
        super().__init__()
        self._temperature = temperature
        self._bias = bias
        self._seed = seed
        self._normalize_output = normalize_output
        self._holdout_fraction = holdout_fraction
        self._rank = rank
        self._chosen_alpha: float | None = None

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        candidate_pool_Y: np.ndarray | None = None,
    ) -> ContrastiveMapping:
        """Fit cross-modal mapping using contrastive alignment.

        Parameters
        ----------
        X:
            Source embeddings ``(K, d_src)`` — e.g., image embeddings from Model A.
        Y:
            Target embeddings ``(K, d_tgt)`` — e.g., text embeddings from Model B.
            Must be aligned: X[i] and Y[i] represent the same semantic content.
        candidate_pool_Y:
            Optional larger pool of target embeddings for retrieval validation.
            If provided, recall@k is computed against this pool instead of just
            the paired targets.
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must be 2-D arrays of shape (K, d)")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Sample counts must match: X has {X.shape[0]}, Y has {Y.shape[0]}"
            )
        if X.shape[0] < 2:
            raise ValueError("Need at least 2 samples for contrastive fitting")

        _check_finite("X", X)
        _check_finite("Y", Y)

        self._d_src = int(X.shape[1])
        self._d_tgt = int(Y.shape[1])

        X_train, X_hold, Y_train, Y_hold = train_test_split(
            X, Y,
            test_size=self._holdout_fraction,
            random_state=self._seed,
        )
        if len(X_hold) < 2:
            X_hold, Y_hold = X_train[-2:], Y_train[-2:]
            X_train, Y_train = X_train[:-2], Y_train[:-2]

        # Learn W via alternating optimization of InfoNCE + ridge hybrid:
        # 1. Start with ridge as initialization (fast, good starting point)
        from sklearn.linear_model import Ridge

        X_fit = np.hstack([X_train, np.ones((len(X_train), 1))]) if self._bias else X_train
        ridge = Ridge(alpha=1.0, fit_intercept=False)
        ridge.fit(X_fit, Y_train)
        coef = np.asarray(ridge.coef_, dtype=np.float64)
        if coef.ndim == 1:
            W = coef.reshape(-1, 1)
        else:
            W = coef.T

        # 2. Fine-tune with gradient-free optimization (Nelder-Mead on InfoNCE)
        from scipy.optimize import minimize

        def _objective(w_flat: np.ndarray) -> float:
            W_cand = w_flat.reshape(W.shape)
            Z_src = l2_normalize(X_train @ W_cand) if not self._bias else l2_normalize(
                np.hstack([X_train, np.ones((len(X_train), 1))]) @ W_cand
            )
            Z_tgt = l2_normalize(Y_train)
            return _infonce_loss(Z_src, Z_tgt, self._temperature)

        # Only fine-tune a small number of steps (ridge init is usually good)
        result = minimize(
            _objective,
            W.flatten(),
            method="Nelder-Mead",
            options={"maxiter": 200, "xatol": 1e-6, "fatol": 1e-8},
        )
        W = result.x.reshape(W.shape)

        # Optional rank compression
        if self._rank is not None and self._rank < min(W.shape):
            U, s, Vt = np.linalg.svd(W, full_matrices=False)
            W = (U[:, : self._rank] * s[: self._rank]) @ Vt[: self._rank]

        self._W = W

        # Inverse map: ridge Y -> X (standard ridge, since we need pointwise for inverse)
        Y_fit = np.hstack([Y_train, np.ones((len(Y_train), 1))]) if self._bias else Y_train
        inv_ridge = Ridge(alpha=1.0, fit_intercept=False)
        inv_ridge.fit(Y_fit, X_train)
        inv_coef = np.asarray(inv_ridge.coef_, dtype=np.float64)
        if inv_coef.ndim == 1:
            self._W_inv = inv_coef.reshape(-1, 1)
        else:
            self._W_inv = inv_coef.T

        self._fitted = True
        self._validation_report = _contrastive_validation(
            self,
            X_hold,
            Y_hold,
            candidate_pool_Y,
            n_train=len(X_train),
            seed=self._seed,
        )
        return self

    def transform(self, V: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return self._apply_mapping(
            V,
            self._W,
            self._d_src,
            direction="forward",
            bias=self._bias,
            normalize=self._normalize_output,
        )

    def inverse_transform(self, V: np.ndarray) -> np.ndarray:
        self._require_fitted()
        if self._W_inv is None:
            raise RuntimeError("Inverse mapping not available")
        return self._apply_mapping(
            V,
            self._W_inv,
            self._d_tgt,
            direction="inverse",
            bias=self._bias,
            normalize=self._normalize_output,
        )
