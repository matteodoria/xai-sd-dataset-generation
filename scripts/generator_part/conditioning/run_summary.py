"""Cross-dataset summary: does the conditioning space predict generation quality?

Collects, for each dataset, the effective rank of the conditioning residuals
(computed from the encoder weights alone, no GPU) and the accuracy the generator
achieves under full conditioning. The question is whether the first predicts the
second — i.e. whether hours of generation can be spared by a computation that takes
seconds.

Usage:
    python -m scripts.xai_summary --exp xAI
"""

import argparse
import os

import numpy as np

from scripts.generator_part.conditioning.lib import plotting, common

# dataset -> (conditioning artefact, accuracy artefact)
ARTEFACTS = {
    "cifar10": ("epoch31.npz", "accuracy_steps31_window5.npz"),
    "cifar100": ("epoch41.npz", "accuracy_steps28_window7.npz"),
    "pathmnist": ("epoch28.npz", "accuracy_steps48_window12.npz"),
    "dermamnist": ("epoch5.npz", "accuracy_steps27_window7.npz"),
    "bloodmnist": ("epoch35.npz", "accuracy_steps44_window11.npz"),
    "retinamnist": ("epoch31.npz", "accuracy_steps30_window7.npz"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    names, ranks, quality = [], [], []
    print(f"{'dataset':>12} {'rank':>7} {'all':>8} {'chance':>8} "
          f"{'ceiling':>8} {'quality':>8}")

    for dataset, (conditioning, accuracy) in ARTEFACTS.items():
        space = np.load(os.path.join(
            common.results_dir(args.exp, dataset, "conditioning"), conditioning))
        scores = np.load(os.path.join(
            common.results_dir(args.exp, dataset, "windows"), accuracy))

        # Full conditioning is the configuration with the most conditioned steps.
        full = float(scores["accuracy"][np.argmax(scores["counts"])])
        chance = 1 / len(space["labels"])
        ceiling = float(scores["real_accuracy"])
        # Fraction of the attainable margin: 1.0 means as good as the judge is on
        # real images, and above 1.0 means better.
        fraction = (full - chance) / (ceiling - chance)

        names.append(dataset)
        ranks.append(float(space["effective_rank"]))
        quality.append(fraction)

        print(f"{dataset:>12} {ranks[-1]:>7.2f} {full:>8.1%} {chance:>8.1%} "
              f"{ceiling:>8.1%} {fraction:>8.3f}")

    out = os.path.join(common.results_dir(args.exp, "summary", "figures"),
                       "rank_vs_quality.png")
    plotting.rank_vs_quality(
        ranks, quality, names,
        title="Does the conditioning space predict generation quality?",
        out_path=out)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()