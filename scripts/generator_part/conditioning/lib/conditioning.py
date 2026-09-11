"""Closed-form analysis of the class conditioning space.

The Class-Encoder (Models/class_encoder.py) chains 2 Dense Layers
with no activation, so the map from the one-hoy class vector to
the (100, 768) context is affine:

    context(x) = (x W1 + b1) W2 = x E + u, with E = W1 W2, u = b1 W2

The context of class c is therefore E[c] + u and the unconditional context
used for classfier-free guidance (an all-zeros input) is exactly u.
Every bit of class information reaching the UNet travels through E,
and though nothing else.

Checked numerically on cifar10/epoch31: the closed form sits 3.96-6
from a float64 reference while the model's own float32 output sits 4.1e-4
from it, woth no systematic bias (mean 1.3e-7 against a noise std of 2.6e-5).
The closed form is thus more accurate tha the model itself,
and it is used as the reference
"""

import os

import numpy as np
import tensorflow.keras as tfk
import itertools
import math

from Models.class_encoder import ClassEncoder

SEQ_LEN, EMB_DIM = 100, 768

def checkpoint_path(dataset, exp, epoch):
    return f"Checkpoints/DDPM/Exp_{exp}/{dataset}/MyEmbedding/epoch{epoch}.hdf5"

def build_encoder(dataset, exp, epoch, num_classes):
    """Load a ClassEncoder, failing loudly when the checkpoint is missing.
    ClassEncoder catches load errors and falls back to random weights,
    which would quietly turn every number downstream to noise.
    Hence the check.
    """
    path = checkpoint_path(dataset, exp, epoch)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No encoder checkpoint found at {path}")
    return ClassEncoder(max_length=num_classes, weight_path=path), path

def closed_form(encoder):
    """Return (E, u) in float64, read straight off the Dense layers.

    Raises:
        ValueError: if the architecture is not 2 linear Dense layers
            the closed form assumes.
    """
    dense = [l for l in encoder.layers if isinstance(l, tfk.layers.Dense)]

    if len(dense) != 2:
        raise ValueError(f"Expected 2 Dense layers, found {len(dense)}.")

    for layer in dense:
        if layer.activation.__name__  != "linear":
            raise ValueError(f"Layer {layer.name} has a non-linear activation "
                             f"({layer.activation.__name__}).")

    W1, b1 = (w.astype(np.float64) for w in dense[0].get_weights())
    (W2, ) = (w.astype(np.float64) for w in dense[1].get_weights())

    return W1 @ W2, b1 @ W2

def verify_linearity(encoder, E, u, num_classes):
    """Check the closed form against the model on one-hot and all-zero inputs.

    The criterion is deliberately relative: an absolute tolerance is meaningless
    without knowing the magnitude of the context.
    Rounding noise is accepted, a systematic offset is not - hence the separate
    check on the mean.
    """

    x = np.eye(num_classes, dtype=np.float32)
    predicted = encoder.predict(x, verbose=0).astype(np.float64)
    closed = (x @ E + u).reshape(num_classes, SEQ_LEN, EMB_DIM)

    zeros = np.zeros((1, num_classes), dtype=np.float32)
    uncond = encoder.predict(zeros, verbose=0)[0].astype(np.float64)

    diff = predicted - closed
    scale = float(np.abs(closed).max())
    onehot_err = float(np.abs(diff).max())
    uncond_err = float(np.abs(uncond - u.reshape(SEQ_LEN, EMB_DIM)).max())
    bias, noise = float(diff.mean()), float(diff.std())
    epsilon = float(np.finfo(np.float32).eps)

    return {
        "scale": scale,
        "onehot_abs": onehot_err,
        "onehot_rel": onehot_err / scale,
        "uncond_abs": uncond_err,
        "bias": bias,
        "noise_std": noise,
        "passed": (onehot_err / scale < 1e-3
                   and (abs(bias) < 0.05 * noise or abs(bias) / scale < 5 * epsilon)),
    }

def geometry(E, u):
    """Norms and angles of the conditioning space.

    Returns:
        dict with:
            norm_u: magnitude of the context shared by every class
            norm_E: per-class magnitude of the class-specific direction
            ratio: norm_E / norm_u, how much the class perturbs that shared context
            cos_Eu: alignment of each class direction with the shared context
            cos_classes: pairwise cosine similarity between class directions.
    """

    norm_u = float(np.linalg.norm(u))
    norm_E = np.linalg.norm(E, axis=1)
    unit = E / norm_E[:, None]

    return {
        "norm_u": norm_u,
        "norm_E": norm_E,
        "ratio": norm_E / norm_u,
        "cos_Eu": (E @ u) / (norm_E * norm_u),
        "cos_classes": unit @ unit.T,
    }

def decompose(E):
    """Split class directions into a shared component and per-class residuals.

    E[c] = mean + R[c]. Cosine similarities between raw class directions are
    dominated by the shared component (~0.96 on cifar10),
    which masks the structure rather than showing it: centring exposes what
    actually tells the classes apart.

    Returns:
        dict with:
            mean: the component shared by every class direction
            residuals: E[c] - mean, one row per class
            shared_energy, residual_energy: fractions of the total energy
                                            (they sum to 1, since residuals
                                            sum to zero by construction)
            residual_ratio: per-class ||R[c]|| / ||E[c]||
            cos_residuals: pairwise cosine similarity between residuals
            singular_values, explained_variance_ratio: spectrum of the residuals
            effective_rank: exp of the entropy of that spectrum (Roy&Vetterli)
                            Equals the true rank when directions are uses evenly,
                            falls to 1 as the energy concentrates.
                            At most num_classes - 1.
    """

    mean = E.mean(axis=0)
    residuals = E - mean

    total_sq = float((E ** 2).sum())
    shared_sq = float(len(E) * (mean ** 2).sum())
    residual_sq = float((residuals ** 2).sum())

    norm_R = np.linalg.norm(residuals, axis = 1)
    unit = residuals / norm_R[:, None]

    s = np.linalg.svd(residuals, compute_uv = False)
    ratio = s ** 2 / (s ** 2).sum()
    nonzero = ratio[ratio > 1e-12]

    return {
        "mean": mean,
        "residuals": residuals,
        "shared_energy": shared_sq / total_sq,
        "residual_energy": residual_sq / total_sq,
        "residual_ratio": norm_R / np.linalg.norm(E, axis = 1),
        "cos_residuals": unit @ unit.T,
        "singular_values": s,
        "explained_variance_ratio": ratio,
        "effective_rank": float(np.exp(-(nonzero * np.log(nonzero)).sum())),
    }

def group_cosines(cos_matrix, group_ids):
    """Mean cosine within and between groups, diagonal excluded"""
    same = group_ids[:, None] == group_ids[None, :]
    off = ~np.eye(len(cos_matrix), dtype = bool)
    return {
        "within": float(cos_matrix[same & off].mean()),
        "between": float(cos_matrix[~same].mean()),
    }


def group_separation(cos_matrix, group_ids):
    """How much closer classes are to their own group than to the others."""
    c = group_cosines(cos_matrix, group_ids)
    return c["within"] - c["between"]

def permutation_test(cos_matrix, group_ids,
                     n_permutations=10000, seed=1234,
                     exact_limit=100000):
    """Permutation test on a grouping of the classes.

    The statistic is the mean within-group cosine minus the mean between-group
    cosine. Under the null the grouping carries no information, so labels are
    permuted while group sizes stay fixed.

    With two groups and few classes every distinct partition is enumerated and
    the p-value is exact (C(10, 4) = 210 for the cifar10 split). Otherwise the
    labels are shuffled `n_permutations` times, and the p-value is a Monte Carlo
    estimate bounded below by 1 / (n_permutations + 1).
    """
    group_ids = np.asarray(group_ids)
    observed = group_separation(cos_matrix, group_ids)
    n = len(group_ids)
    unique = np.unique(group_ids)

    k = int((group_ids == unique[0]).sum()) if len(unique) == 2 else None
    exact = k is not None and math.comb(n, k) <= exact_limit

    if exact:
        null = []
        for combo in itertools.combinations(range(n), k):
            ids = np.full(n, unique[1])
            ids[list(combo)] = unique[0]
            null.append(group_separation(cos_matrix, ids))
        p_value = float((np.array(null) >= observed).mean())
    else:
        rng = np.random.default_rng(seed)
        null = [group_separation(cos_matrix, rng.permutation(group_ids))
                for _ in range(n_permutations)]
        p_value = float((1 + (np.array(null) >= observed).sum()) / (1 + len(null)))

    return {
        "observed": observed,
        "null": np.array(null),
        "n_partitions": len(null),
        "exact": exact,
        "p_value": p_value,
    }

def hierarchical_variance(residuals, group_ids):
    """Split residual energy into between-group and within-group parts.

    Residuals already have zero global mean, so the decomposition is exact:

        sum_c ||R[c]||^2 = sum_g n_g ||m_g||^2 + sum_c ||R[c]-m_g(c)||^1

    A large between-group share means the conditioning space mostly encodes
    the coarse category; a large within-group share means it also separates
    the individual classes inside it. Chance level is not zero: with G groups
    over n classes een an uninformative grouping captures about (G-1)/(n-1) of
    the energy, so that reference is returned alongside.
    """

    group_ids = np.asarray(group_ids)
    total = float((residuals ** 2).sum())
    between = within = 0.0

    for g in np.unique(group_ids):
        members = residuals[group_ids == g]
        centroid = members.mean(axis = 0)
        between += len(members) * float((centroid ** 2).sum())
        within += float(((members - centroid) ** 2).sum())

    n, n_groups = len(residuals), len(np.unique(group_ids))

    return {
        "between_energy": between / total,
        "within_energy": within / total,
        "expected_random": (n_groups - 1) / (n - 1),
    }

def subsample_effective_rank(E, group_ids, n_classes=10, n_draws=500, seed=1234):
    """Effective rank of random class subsets, one class per group.

    Puts datasets with different numbers of classes on equal terms: the same
    pipeline (centre, then measure how concentrated the spectrum is) is applied
    to subsets of equal size, drawn from distinct groups so that the subset is
    as heterogeneous as the cifar10 label set.

    Each subset is re-centred on its own mean, exactly as the full analysis
    does, so the result is directly comparable to the effective rank reported
    for a dataset with that many classes.
    """
    rng = np.random.default_rng(seed)
    group_ids = np.asarray(group_ids)
    groups = np.unique(group_ids)

    if len(groups) < n_classes:
        raise ValueError(f"Only {len(groups)} groups available, need "
                         f"{n_classes} to draw one class from each.")

    ranks = []
    for _ in range(n_draws):
        chosen = rng.choice(groups, size=n_classes, replace=False)
        idx = [int(rng.choice(np.flatnonzero(group_ids == g))) for g in chosen]
        ranks.append(decompose(E[idx])["effective_rank"])

    ranks = np.array(ranks)
    return {"effective_rank": ranks, "normalised": ranks / (n_classes - 1)}

def mantel_test(matrix_a, matrix_b, n_permutations=10000, seed=1234):
    """Correlation between two similarity matrices over the same classes.

    Off-diagonal entries only. A plain correlation p-value would be wrong here:
    the entries are not independent, since every class appears in n-1 cells.
    The null therefore shuffles the class order of one matrix, preserving its
    internal structure while destroying the correspondence between the two.

    One-sided: the hypothesis under test is that the conditioning space matches
    visual similarity, which predicts a positive correlation.
    """
    n = len(matrix_a)
    off = ~np.eye(n, dtype=bool)
    a = np.asarray(matrix_a)[off]
    b = np.asarray(matrix_b)

    observed = float(np.corrcoef(a, b[off])[0, 1])

    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(a, b[np.ix_(p, p)][off])[0, 1]
                     for p in (rng.permutation(n) for _ in range(n_permutations))])

    return {
        "correlation": observed,
        "null": null,
        "p_value": float((1 + (null >= observed).sum()) / (1 + len(null))),
    }
















