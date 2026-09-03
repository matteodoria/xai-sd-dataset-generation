"""When does the class conditioning actually change the outcome?

Generates the same classes from the same initial noise several times,
conditioning only inside a window of denoising steps. ||d eps|| tells where the
instantaneous signal is largest; this tells where it changes the result, which
need not be the same place — a small nudge at the first step is propagated
through every step that follows, while a large one at the last step only alters
details before the process ends.

Each window is scored by how much of the fully-conditioned image it recovers,
against the distance between the two extremes (always conditioned, never
conditioned) as the unit.

Usage:
    python -m scripts.xai_windows --dataset cifar10 --exp xAI --enc_epoch 31 --dif_epoch 10
"""

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from Models.stable_diffusion import MyStableDiffusion
from XAI import common, instrumentation, plotting
from XAI import scoring


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--exp", required=True, type=str)
    parser.add_argument("--enc_epoch", required=True, type=int)
    parser.add_argument("--dif_epoch", required=True, type=int)
    parser.add_argument("--steps", default=20, type=int)
    parser.add_argument("--window", default=5, type=int)
    parser.add_argument("--ugs", default=1.0, type=float)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--per_class", default=10, type=int,
                        help="Images generated per class. Ten per class gives a "
                             "100-image sample per configuration, enough for an "
                             "accuracy with a few points of standard error.")
    parser.add_argument("--chunk", default=100, type=int,
                        help="Images generated per forward batch. Lower it if "
                             "the GPU runs out of memory.")
    return parser.parse_args()


def per_image_rmse(a, b):
    """Root mean squared pixel difference, one value per image."""
    difference = a.astype(np.float64) - b.astype(np.float64)
    return np.sqrt((difference.reshape(len(a), -1) ** 2).mean(axis=1))



def main():
    args = parse_args()
    labels = common.get_labels(args.dataset)
    num_classes = len(labels)
    resolution = 32 if args.dataset.startswith("cifar") else 28

    # keras_cv builds the schedule as range(1, 1000, 1000 // num_steps), which does
    # not yield exactly num_steps values: 31 gives 32, 44 gives 45. The windows must
    # be built on the real count, or the last steps fall outside every prefix.
    effective_steps = len(range(1, 1000, 1000 // args.steps))
    if effective_steps != args.steps:
        print(f"note: {args.steps} steps requested, {effective_steps} actually run")

    ddpm = MyStableDiffusion(
        res=resolution, num_classes=num_classes,
        original_diffusion=True, original_text_encoder=False,
        enc_weight_path=f"Checkpoints/DDPM/Exp_{args.exp}/{args.dataset}/MyEmbedding/epoch{args.enc_epoch}.hdf5",
        diff_weight_path=f"Checkpoints/DDPM/Exp_{args.exp}/{args.dataset}/DiffusionFt/epoch{args.dif_epoch}.hdf5")

    # Tiled, not repeated: the order becomes c0..c9, c0..c9, ... so every chunk
    # holds all classes in equal parts.
    context = tf.tile(ddpm.text_encoder(tf.eye(num_classes)),
                      [args.per_class, 1, 1])
    class_index = np.tile(np.arange(num_classes), args.per_class)
    batch = num_classes * args.per_class

    # Position 0 is the noisiest step; None conditions everywhere, the empty
    # set nowhere. Same seed throughout, so every run starts from the same noise
    # and the only difference is where conditioning was allowed to act.
    configurations = [("all", None)]
    for stop in range(args.window, effective_steps + 1, args.window):
        configurations.append((f"0-{stop - 1}", set(range(stop))))
    configurations.append(("none", set()))

    generated, timesteps = {}, None
    for name, positions in configurations:
        print(f"... conditioning on steps: {name} ...")
        generated[name], timesteps = instrumentation.generate_in_chunks(
            ddpm, context, args.chunk, args.seed,
            num_steps=args.steps, ugs=args.ugs, condition_steps=positions)

    # The judge is applied here, where all the generated images still exist. Only
    # one image per class is saved to disk, so scoring later is not possible.
    judge = scoring.load_judge(args.dataset, resolution, num_classes)
    predictions = {name: scoring.classify(judge, generated[name],
                                          resolution).argmax(axis=1)
                   for name, _ in configurations}

    full, none = generated["all"], generated["none"]
    baseline = per_image_rmse(full, none)

    print(f"\n------------ CONDITIONING WINDOWS ------------"
          f"\n- {args.steps} steps, windows of {args.window}, ugs {args.ugs}"
          f"\n- distance between the two extremes: {baseline.mean():.2f} RMSE\n")
    print(f"{'window':>10} {'timesteps':>16} {'vs all':>9} {'vs none':>9} "
          f"{'recovered':>11}")

    recovery = {}
    for name, positions in configurations:
        distance = per_image_rmse(generated[name], full)
        recovery[name] = 1.0 - distance / baseline
        span = ("-" if positions is None or not positions
                else f"{timesteps[min(positions)]}-{timesteps[max(positions)]}")
        print(f"{name:>10} {span:>16} {distance.mean():>9.2f} "
              f"{per_image_rmse(generated[name], none).mean():>9.2f} "
              f"{recovery[name].mean():>10.1%}")

    first = np.arange(num_classes)

    out_dir = common.results_dir(args.exp, args.dataset, "windows")
    np.savez_compressed(
        os.path.join(out_dir, f"steps{args.steps}_window{args.window}.npz"),
        labels=np.array(labels), timesteps=np.array(timesteps),
        names=np.array([n for n, _ in configurations]),
        images=np.stack([generated[n][first] for n, _ in configurations]),
        recovery=np.stack([recovery[n] for n, _ in configurations]),
        class_index=class_index,
        per_class=args.per_class,
        predictions=np.stack([predictions[n] for n, _ in configurations]),
    )

    out = os.path.join(common.results_dir(args.exp, args.dataset, "figures"),
                       "conditioning_prefixes.png")
    plotting.image_grid(
        [generated[name][first] for name, _ in configurations],
        [name for name, _ in configurations], labels,
        title=f"{args.dataset}: conditioning restricted to a window of steps",
        out_path=out)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()