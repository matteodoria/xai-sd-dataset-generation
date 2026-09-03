"""Generate figures for the conditioning-space analysis.

Reads the .npz written by xai_conditioning.py: no GPU and no model involved.

Usage:
    python -m scripts.xai_figures --dataset cifar10 --exp xAI --enc_epoch 31
"""

import argparse
import os

import numpy as np

from XAI import common, plotting


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", type=str, metavar="NAME",
                        help="Dataset whose analysis should be plotted.")
    parser.add_argument("--exp", required=True, type=str,
                        help="Experiment identifier (e.g., 'xAI').")
    parser.add_argument("--enc_epoch", required=True, type=int,
                        help="Epoch of the analysed encoder checkpoint.")
    return parser.parse_args()


def main():
    args = parse_args()

    npz_path = os.path.join(
        common.results_dir(args.exp, args.dataset, "conditioning"),
        f"epoch{args.enc_epoch}.npz")
    data = np.load(npz_path)
    labels = [str(lab) for lab in data["labels"]]

    groups = common.SEMANTIC_GROUPS.get(args.dataset)
    ids = common.group_ids(labels, groups) if groups else None
    names = list(groups) if groups else None

    fig_dir = common.results_dir(args.exp, args.dataset, "figures")

    out = os.path.join(fig_dir, "cosine_residuals.png")
    plotting.cosine_heatmap(
        data["cos_residuals"], labels, ids, names,
        title=f"{args.dataset}: cosine between class residuals",
        out_path=out)
    print(f"Saved: {out}")

    if "cos_visual" in data:
        out = os.path.join(fig_dir, "mantel_scatter.png")
        plotting.mantel_scatter(
            data["cos_visual"], data["cos_residuals"], labels,
            correlation=float(data["mantel_correlation"]),
            group_ids=ids, annotate=4 if len(labels) <= 20 else 0,
            title=f"{args.dataset}: conditioning vs visual similarity",
            out_path=out)
        print(f"Saved: {out}")

    out = os.path.join(fig_dir, "energy_spectrum.png")
    plotting.energy_spectrum(
        float(data["shared_energy"]),
        data["singular_values"],
        float(data["effective_rank"]),
        title=f"{args.dataset}: conditioning energy and residual spectrum",
        out_path=out)
    print(f"Saved: {out}")

    tests = []
    if "perm_observed" in data:
        tests.append({"name": "semantic grouping", "null": data["perm_null"],
                      "observed": float(data["perm_observed"]),
                      "p_value": float(data["perm_p_value"])})
    if "mantel_null" in data:
        tests.append({"name": "visual similarity (Mantel)",
                      "null": data["mantel_null"],
                      "observed": float(data["mantel_correlation"]),
                      "p_value": float(data["mantel_p_value"])})

    if tests:
        out = os.path.join(fig_dir, "null_distributions.png")
        plotting.null_distributions(
            tests, title=f"{args.dataset}: observed statistics against chance",
            out_path=out)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()