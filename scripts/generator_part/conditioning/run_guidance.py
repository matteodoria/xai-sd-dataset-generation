"""How strongly, and when, the class conditioning acts during denoising.

eps(c) - eps(0) is the class signal by definition: the only quantity through
which the class enters generation, and exactly what classifier-free guidance
multiplies by UGS. Measuring it needs no attention weights and carries none of
their caveats.

Usage:
    python -m scripts.generator_part.conditioning.run_guidance --dataset cifar10 --exp xAI --enc_epoch 31 --dif_epoch 10
"""

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from Models.stable_diffusion import MyStableDiffusion
from scripts.generator_part.conditioning.lib import instrumentation, common


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--exp", required=True, type=str)
    parser.add_argument("--enc_epoch", required=True, type=int)
    parser.add_argument("--dif_epoch", required=True, type=int)
    parser.add_argument("--steps", default=20, type=int)
    parser.add_argument("--ugs", default=1.0, type=float)
    parser.add_argument("--seed", default=1234, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    labels = common.get_labels(args.dataset)
    num_classes = len(labels)
    resolution = 32 if args.dataset.startswith("cifar") else 28

    ddpm = MyStableDiffusion(
        res=resolution, num_classes=num_classes,
        original_diffusion=True, original_text_encoder=False,
        enc_weight_path=f"Models/Checkpoints/DDPM/Exp_{args.exp}/{args.dataset}/MyEmbedding/epoch{args.enc_epoch}.hdf5",
        diff_weight_path=f"Models/Checkpoints/DDPM/Exp_{args.exp}/{args.dataset}/DiffusionFt/epoch{args.dif_epoch}.hdf5")

    one_hot = tf.eye(num_classes)
    context = ddpm.text_encoder(one_hot)

    images, _, trace = instrumentation.generate(
        ddpm, context, batch_size=num_classes, num_steps=args.steps,
        ugs=args.ugs, seed=args.seed)

    timesteps = np.array([t["timestep"] for t in trace])
    relative = np.stack([t["guidance_norm"] / t["eps_norm"] for t in trace])
    maps = np.stack([t["guidance_map"] for t in trace])

    absolute = np.stack([t["guidance_norm"] for t in trace])
    eps = np.stack([t["eps_norm"] for t in trace])

    print(f"\n------------ CLASS SIGNAL ALONG DENOISING ------------"
          f"\n- dataset: {args.dataset}, {num_classes} classes"
          f"\n- {args.steps} steps, ugs {args.ugs}\n")
    print(f"{'timestep':>9} {'||d eps||':>12} {'||eps(0)||':>12} "
          f"{'ratio':>10} {'spread':>10}")
    for i, timestep in enumerate(timesteps):
        print(f"{timestep:>9} {absolute[i].mean():>12.4f} {eps[i].mean():>12.4f} "
              f"{relative[i].mean():>10.4f} {relative[i].std():>10.4f}")

    print(f"\n||d eps||   first/last: {absolute[0].mean() / absolute[-1].mean():.2f}x")
    print(f"||eps(0)||  first/last: {eps[0].mean() / eps[-1].mean():.2f}x")
    print(f"ratio       first/last: {relative[0].mean() / relative[-1].mean():.2f}x")

    strongest = int(np.argmax(absolute.mean(axis=1)))
    print(f"largest absolute class signal at timestep {timesteps[strongest]} "
          f"(step {strongest + 1} of {len(timesteps)})")

    out_dir = common.results_dir(args.exp, args.dataset, "guidance")
    out = os.path.join(out_dir, f"steps{args.steps}_ugs{args.ugs}.npz")
    np.savez_compressed(out, labels=np.array(labels), timesteps=timesteps,
                        relative=relative, maps=maps.astype(np.float32),
                        images=images, absolute=absolute, eps=eps,)
    print(f"\nSaved: {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()