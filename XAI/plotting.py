"""Figures for the xAI analyses.

Uses the Agg backend: these run on a headless cluster node and write PNGs.
"""
import sys
import matplotlib
# Headless by default: these run on a cluster node with no display. Inside a
# notebook the interactive backend is already set and must be left alone.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import math

def cosine_heatmap(cos_matrix, labels, group_ids=None, group_names=None,
                   title="", out_path=None):
    """Heatmap of a class-by-class cosine matrix, optionally grouped.

    Classes are reordered so members of a group sit next to each other,
    turning semantic structure into visible blocks along the diagonal.
    The diagonal itself is masked: it is 1 by construction and would flatten
    the color scale onto nothing. Past 20 classes the per-class ticks stop
    being readable, so group names are drawn at the centre of each block instead.
    """
    labels = np.asarray(labels)
    order = (np.argsort(group_ids, kind='stable') if group_ids is not None else np.arange(len(labels)))

    m = np.array(cos_matrix, dtype=float)[np.ix_(order, order)]
    np.fill_diagonal(m, np.nan)
    limit = np.nanmax(np.abs(m))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(m, cmap="RdBu_r", vmin=-limit, vmax=limit)

    if group_ids is not None:
        sorted_ids = np.asarray(group_ids)[order]
        for edge in np.flatnonzero(np.diff(sorted_ids)) + 0.5:
            ax.axhline(edge, color="black", lw=0.8)
            ax.axvline(edge, color="black", lw=0.8)

    if len(labels) <= 20:
        ticks, names = range(len(labels)), labels[order]
    elif group_ids is not None and group_names is not None:
        sorted_ids = np.asarray(group_ids)[order]
        unique = np.unique(sorted_ids)
        ticks = [np.mean(np.flatnonzero(sorted_ids == g)) for g in unique]
        names = [group_names[g] for g in unique]
    else:
        ticks, names = [], []

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticklabels(names, fontsize=7)

    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, label="cosine similarity")
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return fig

def mantel_scatter(cos_reference, cos_conditioning,
                   labels, correlation=None, group_ids=None,
                   annotate=0, title="", out_path=None):
    """One point per class pair: visual similarity against conditioning similarity.

    Only the upper triangle is plotted. The matrices are symmetic, so drawing both
    halves would duplicate every point and misrepresent the density.

    The regression line is a visual guide only; the statistic that matters is the
    Mantel correlation, which is reported in the title.
    """
    labels = np.asarray(labels)
    rows, cols = np.triu_indices(len(labels), k=1)
    x = np.asarray(cos_reference)[rows, cols]
    y = np.asarray(cos_conditioning)[rows, cols]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    if group_ids is not None:
        ids = np.asarray(group_ids)
        same = ids[rows] == ids[cols]
        ax.scatter(x[~same], y[~same], s=16, alpha=0.35, edgecolor="none",
                   color="#9ecae1", label="different group", zorder=2)
        ax.scatter(x[same], y[same], s=22, alpha=0.85, edgecolor="none",
                   color="#d62728", label="same group", zorder=3)
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    else:
        ax.scatter(x, y, s=18, alpha=0.5 if len(x) > 200 else 0.85,
                   edgecolor="none", color="#1f77b4", zorder=2)

    slope, intercept = np.polyfit(x, y, 1)
    edges = np.array([x.min(), x.max()])
    ax.plot(edges, slope * edges + intercept, color="crimson", lw=1.2, zorder=4)

    # Label the strongest pairs, skipping any that would collide with one
    # already placed, and flipping the offset near the right edge so the text
    # stays inside the axes.
    placed = []
    midpoint = (x.min() + x.max()) / 2
    for k in np.argsort(y)[::-1]:
        if len(placed) >= annotate:
            break
        if any(abs(x[k] - x[j]) < 0.10 and abs(y[k] - y[j]) < 0.12 for j in placed):
            continue
        right = x[k] > midpoint
        pair = f"{labels[rows[k]][:11]}-{labels[cols[k]][:11]}"
        ax.annotate(pair, (x[k], y[k]),
                    fontsize=7, xytext=(-5 if right else 5, 3),
                    textcoords="offset points",
                    ha="right" if right else "left", zorder=5)
        placed.append(k)

    ax.margins(0.10)

    ax.set_xlabel("visual similarity of the real images")
    ax.set_ylabel("similarity in the conditioning space")
    ax.set_title(f"{title}  (r = {correlation:+.3f})" if correlation is not None
                 else title)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return fig

def energy_spectrum(shared_energy, singular_values, effective_rank,
                    title="", out_path=None):
    """Two panels: how much of the conditioning is shared, and how the rest spreads.

    Left, the split between the component common to every class and the residual that
    carries class information. Right, the spectrum of that residual, with the effective
    rank marked: it says how many directions are really in use, against the n-1 that
    centring leaves available.
    """
    fig, (bar, spec) = plt.subplots(1, 2, figsize=(10,4),
                                    gridspec_kw={"width_ratios": [1,1.4]})

    residual = 1.0 - shared_energy
    bar.barh([0], [shared_energy], height=0.5, color="#bdbdbd")
    bar.barh([0], [residual], left=[shared_energy], height=0.5, color="#d62728")
    bar.text(shared_energy / 2, 0, f"{shared_energy:.1%}\nshared by all classes",
             ha="center", va="center", fontsize=10)
    bar.text(1.03, 0, f"{residual:.1%}\nclass-specific", ha="left", va="center",
             fontsize=10, color="#d62728")
    bar.set_xlim(0, 1.35)
    bar.set_ylim(-0.6, 0.6)
    bar.set_yticks([])
    bar.set_xlabel("fraction of the conditioning energy")


    ratio = singular_values ** 2 / (singular_values ** 2).sum()
    spec.bar(range(1, len(ratio) + 1), ratio, color="#1f77b4")
    spec.axvline(effective_rank, color="crimson", lw=1.4,
                 label=f"effective rank = {effective_rank:.2f}")
    spec.set_xlabel(f"component (at most {len(ratio) - 1} are non-zero)")
    spec.set_ylabel("share of residual energy")
    spec.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return fig

def null_distributions(tests, title="", out_path=None):
    """Permutation nulls with the observed statistic marked on each.

    Shows what a p-value only asserts: where the measured value falls relative
    to what chance produces. When the observed value lands past the entire null,
    the gap is the result.

    Args:
        tests: dicts with keys name, null, observed, p_value.
    """
    fig, axes = plt.subplots(1, len(tests), figsize=(5 * len(tests), 3.6),
                             squeeze=False)

    for ax, test in zip(axes[0], tests):
        null, observed = np.asarray(test["null"]), test["observed"]
        ax.hist(null, bins=40, color="#bdbdbd", edgecolor="none")
        ax.axvline(observed, color="crimson", lw=1.8)

        lo, hi = min(null.min(), observed), max(null.max(), observed)
        right = observed > (lo + hi) / 2
        ax.text(observed, 0.95, f"observed {observed:+.3f}" + (" " if right else ""),
                transform=ax.get_xaxis_transform(),
                ha="right" if right else "left", va="top",
                fontsize=8, color="crimson")

        ax.set_title(f"{test['name']}  (p = {test['p_value']:.4f})", fontsize=10)
        ax.set_xlabel("statistic under the null")
        ax.set_ylabel("permutations")

    fig.suptitle(title)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return fig

def attention_overlay(image, maps, titles, title="", out_path=None):
    """Attention maps laid over the generated image.

    Upsampling is nearest-neighbour on purpose: these maps live on a 16x16
    latent grid or coarser, and smooth interpolation would suggest a spatial
    precision the data does not have.
    """
    panels = len(maps) + 1
    cols = min(panels, 4)
    rows = math.ceil(panels / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows),
                             squeeze=False)
    flat = axes.ravel()

    flat[0].imshow(image)
    flat[0].set_title("generated", fontsize=9)

    grey = image.mean(axis=-1)
    for ax, grid, caption in zip(flat[1:], maps, titles):
        factor = image.shape[0] // grid.shape[0]
        ax.imshow(grey, cmap="gray")
        ax.imshow(np.kron(grid, np.ones((factor, factor))),
                  cmap="inferno", alpha=0.6)
        ax.set_title(caption, fontsize=8)

    for ax in flat:
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in flat[panels:]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return fig

def maps_over_batch(images, maps, titles, title="", out_path=None, cols=4):
    """Each image with its attention map overlaid underneath it."""
    count = len(images)
    blocks = math.ceil(count / cols)

    fig, axes = plt.subplots(2 * blocks, cols,
                             figsize=(2.6 * cols, 5.4 * blocks), squeeze=False)

    for i in range(count):
        block, col = divmod(i, cols)
        axes[2 * block][col].imshow(images[i])
        axes[2 * block][col].set_title(titles[i], fontsize=8)

        factor = images[i].shape[0] // maps[i].shape[0]
        axes[2 * block + 1][col].imshow(images[i].mean(axis=-1), cmap="gray")
        axes[2 * block + 1][col].imshow(
            np.kron(maps[i], np.ones((factor, factor))),
            cmap="inferno", alpha=0.55)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    for i in range(count, blocks * cols):
        block, col = divmod(i, cols)
        axes[2 * block][col].axis("off")
        axes[2 * block + 1][col].axis("off")

    fig.suptitle(title)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return fig

def image_grid(rows, row_labels, col_labels, title="", out_path=None, max_cols=12):
    """One row per configuration, one column per class.

    Meant for comparing generations that differ in a single controlled way, so
    the same column always holds the same class and the same initial noise.
    Beyond max_cols classes an evenly spaced sample is shown: the comparison is
    between rows, and a hundred columns only make every panel too small to see.
    """
    total = len(rows[0])
    if total > max_cols:
        keep = np.linspace(0, total - 1, max_cols).round().astype(int)
        rows = [[images[i] for i in keep] for images in rows]
        col_labels = [col_labels[i] for i in keep]
        title = f"{title} — {max_cols} of {total} classes"
    n_rows, n_cols = len(rows), len(rows[0])
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(1.5 * n_cols, 1.7 * n_rows), squeeze=False)

    for r, images in enumerate(rows):
        for c, image in enumerate(images):
            axes[r][c].imshow(image)
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])
            if r == 0:
                axes[r][c].set_title(col_labels[c], fontsize=7)
        axes[r][0].set_ylabel(row_labels[r], fontsize=7, rotation=0,
                              ha="right", va="center")

    fig.suptitle(title)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return fig

def conditioning_curves(counts, accuracy, error, recovered, chance=None,
                        ceiling=None, title="", out_path=None):
    """Accuracy and pixel recovery against the number of conditioned steps.

    Two axes on purpose: the quantities are not comparable, and the point of the
    figure is that they disagree. RMSE recovery rises steeply and flattens,
    accuracy does the opposite — the early steps change the picture without
    changing what it depicts.
    """
    fig, left = plt.subplots(figsize=(7, 4.5))

    left.errorbar(counts, accuracy, yerr=error, marker="o", color="#d62728",
                  capsize=3, label="accuracy (class survives)")
    if chance is not None:
        left.axhline(chance, color="0.6", ls=":", lw=1)
        left.annotate("chance", (counts[0], chance), fontsize=7, color="0.4",
                      xytext=(2, 3), textcoords="offset points")
    if ceiling is not None:
        left.axhline(ceiling, color="0.6", ls="--", lw=1)
        left.annotate("judge on real images", (counts[0], ceiling), fontsize=7,
                      color="0.4", xytext=(2, 3), textcoords="offset points")

    left.set_xlabel("denoising steps with conditioning active")
    left.set_ylabel("accuracy", color="#d62728")
    left.tick_params(axis="y", labelcolor="#d62728")
    left.set_ylim(0, 1)

    right = left.twinx()
    right.plot(counts, recovered, marker="s", color="#1f77b4", ls="--",
               label="pixel recovery (RMSE)")
    right.set_ylabel("fraction of the pixel difference recovered",
                     color="#1f77b4")
    right.tick_params(axis="y", labelcolor="#1f77b4")
    right.set_ylim(0, 1)

    lines = left.get_lines()[:1] + right.get_lines()[:1]

    accuracy_handle = left.errorbar(counts, accuracy, yerr=error, marker="o",
                                    color="#d62728", capsize=3,
                                    label="accuracy (class survives)")
    recovery_handle, = right.plot(counts, recovered, marker="s",
                                  color="#1f77b4", ls="--",
                                  label="pixel recovery (RMSE)")

    left.legend(handles=[accuracy_handle, recovery_handle], fontsize=8,
                loc="upper left")

    left.set_title(title)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return fig

def rank_vs_quality(ranks, quality, labels, works_above=0.5,
                    title="", out_path=None):
    """Effective rank of the conditioning space against generation quality.

    Quality is the fraction of the attainable margin actually reached:
    (accuracy - chance) / (judge on real - chance). Above 1 means the generated
    images are classified more accurately than the real ones.

    Points are coloured by the *outcome*, never by the rank: colouring by the
    predictor would assume what the figure is meant to show. The shaded band is
    the observed gap — from the highest rank among the failures to the lowest
    among the successes — so it is a measurement, not a chosen threshold.
    """
    ranks = np.asarray(ranks, dtype=float)
    quality = np.asarray(quality, dtype=float)
    works = quality > works_above

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if works.any() and (~works).any():
        ax.axvspan(ranks[~works].max(), ranks[works].min(), color="0.92", zorder=0)

    ax.axhline(1.0, color="0.6", ls="--", lw=1)
    ax.axhline(0.0, color="0.6", ls=":", lw=1)
    ax.annotate("as good as the judge on real images", (ranks.min(), 1.0),
                fontsize=8, color="0.4", xytext=(0, 5), textcoords="offset points")

    ax.scatter(ranks[works], quality[works], s=70, color="#2ca02c", zorder=3,
               label="generation works")
    ax.scatter(ranks[~works], quality[~works], s=70, color="#d62728", zorder=3,
               label="generation fails")

    for rank, value, name in zip(ranks, quality, labels):
        ax.annotate(name, (rank, value), fontsize=8, xytext=(7, -3),
                    textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 10, 20])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(1.0, 25)
    ax.set_xlabel("effective rank of the conditioning residuals")
    ax.set_ylabel("fraction of the attainable margin")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return fig