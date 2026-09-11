"""Sanity checks: is what we measured really the class conditioning?

Three runs, identical in everything except the context handed to the UNet:

    original   the trained ClassEncoder
    shifted    every class given the next class's embedding
    random     a ClassEncoder whose weights were never trained

The shifted run is the sharper test. Accuracy against the *requested* label must
fall to chance while accuracy against the class whose embedding was actually
supplied stays high: the conditioning is not merely doing something, it is doing
what it claims. A pipeline artefact would survive randomisation but not this.

With ugs = 1.0 the guidance reduces to eps(c), so replacing the conditional
context is enough — the unconditional one does not reach the output.

Usage:
    python -m scripts.xai_sanity --dataset cifar10 --exp xAI --enc_epoch 31 --dif_epoch 10
"""

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from Models.class_encoder import ClassEncoder
from Models.stable_diffusion import MyStableDiffusion
from XAI.Generator.Conditioning import instrumentation, scoring, common


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--exp", required=True, type=str)
    parser.add_argument("--enc_epoch", required=True, type=int)
    parser.add_argument("--dif_epoch", required=True, type=int)
    parser.add_argument("--steps", default=20, type=int)
    parser.add_argument("--ugs", default=1.0, type=float)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--per_class", default=50, type=int)
    parser.add_argument("--chunk", default=100, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    labels = common.get_labels(args.dataset)
    num_classes = len(labels)
    resolution = common.get_dataset(args.dataset).get_resolution()

    ddpm = MyStableDiffusion(
        res=resolution, num_classes=num_classes,
        original_diffusion=True, original_text_encoder=False,
        enc_weight_path=f"Checkpoints/DDPM/Exp_{args.exp}/{args.dataset}/MyEmbedding/epoch{args.enc_epoch}.hdf5",
        diff_weight_path=f"Checkpoints/DDPM/Exp_{args.exp}/{args.dataset}/DiffusionFt/epoch{args.dif_epoch}.hdf5")

    one_hot = tf.eye(num_classes)
    trained = ddpm.text_encoder(one_hot)

    # ClassEncoder swallows load errors and falls back to random weights. That
    # normally hides bugs — the conditioning analysis guards against it — but
    # here it is precisely the ablation we want, so the failure below is
    # deliberate.
    print("\n[expected] the next weight-loading error is intentional: "
          "the 'random' context needs an untrained encoder")
    untrained = ClassEncoder(max_length=num_classes,
                             weight_path="__no_such_checkpoint__")(one_hot)

    # Each class receives the next class's embedding.
    shift = (np.arange(num_classes) + 1) % num_classes

    contexts = {
        "original": trained,
        "shifted": tf.gather(trained, shift),
        "random": untrained,
    }

    class_index = np.tile(np.arange(num_classes), args.per_class)
    supplied = shift[class_index]

    accuracy = {}
    for name, context in contexts.items():
        print(f"\n... generating with the {name} context ...")
        images, _ = instrumentation.generate_in_chunks(
            ddpm, tf.tile(context, [args.per_class, 1, 1]),
            args.chunk, args.seed, num_steps=args.steps, ugs=args.ugs)

        judge = scoring.load_judge(args.dataset, resolution, num_classes)
        predicted = scoring.classify(judge, images, resolution).argmax(axis=1)
        accuracy[name] = predicted

    print(f"\n------------ SANITY CHECKS ------------")
    print(f"{'context':>10} {'vs requested label':>20} {'vs supplied embedding':>23}")
    for name, predicted in accuracy.items():
        against_supplied = ((predicted == supplied).mean()
                            if name == "shifted" else float("nan"))
        print(f"{name:>10} {(predicted == class_index).mean():>19.1%} "
              f"{against_supplied:>22.1%}")

    print(f"\nchance level: {1 / num_classes:.1%}")


if __name__ == "__main__":
    main()