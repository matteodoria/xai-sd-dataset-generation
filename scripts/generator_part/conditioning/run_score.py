"""Does the image still belong to its class, after restricting conditioning?

Judges the generated images with a ResNet20 trained on real CIFAR-10 only, so
its verdict is independent of the generator. RMSE told us how much the pixels
change; this tells us whether the class survives, which is the question that
matters and the one the pixel distance answers badly.

Usage:
    python -m scripts.generator_part.conditioning.run_score --dataset cifar10 --exp xAI --npz steps20_window5.npz
"""

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from scripts.generator_part.conditioning.lib import plotting, common

from Models.ImageClassifiers import resnet

CLASSIFIER = "Models/Checkpoints/Classifiers/resnet20/{dataset}/real/best_model.h5"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--exp", required=True, type=str)
    parser.add_argument("--npz", required=True, type=str,
                        help="File under Results/Exp_<exp>/<dataset>/windows/")
    return parser.parse_args()


def classify(model, images, resolution):
    """Resize to the classifier's resolution and rescale, as in training.

    The training pipeline applies Rescaling(1./255) to images loaded at 32x32
    by image_dataset_from_directory, which resizes bilinearly. Anything else
    here would produce low accuracies that look like a finding and are a bug.
    """
    resized = tf.image.resize(tf.cast(images, tf.float32), (resolution, resolution))
    return model.predict(resized / 255.0, verbose=0)


def main():
    args = parse_args()
    labels = common.get_labels(args.dataset)
    resolution = common.get_dataset(args.dataset).get_resolution()

    path = os.path.join(common.results_dir(args.exp, args.dataset, "windows"),
                        args.npz)
    data = np.load(path)
    names = [str(n) for n in data["names"]]
    images = data["images"]
    truth = data["class_index"]

    # The .h5 holds a custom class, so load_model cannot rebuild it. The
    # architecture is recreated with the hyper-parameters used in training
    # (load_classifier passes initial_filters=64) and the weights loaded into it.
    model = resnet.ResNet20(input_shape=(resolution, resolution, 3),
                            num_classes=len(labels), initial_filters=64, seed=1234)
    model.load_weights(CLASSIFIER.format(dataset=args.dataset))
    print(f"classifier: {model.count_params():,} parameters, "
          f"input {model.input_shape}")

    ## Sanity check: the judge must reproduce its own accuracy on real data.
    (_, _, (x_test, y_test)) = common.get_dataset(args.dataset).load_data_real(
        return_raw=True)
    predicted = classify(model, x_test[:2000], resolution).argmax(axis=1)
    real_accuracy = float((predicted == np.asarray(y_test[:2000]).ravel()).mean())
    print(f"accuracy on 2000 real test images: {real_accuracy:.1%} "
          f"(expected around 85%)\n")

    print(f"{'window':>10} {'accuracy':>10}")
    for i, name in enumerate(names):
        correct = data["predictions"][i] == truth
        print(f"{name:>10} {correct.mean():>10.1%}")

    def conditioned_count(name):
        if name == "all":
            return int(data["timesteps"].shape[0])
        if name == "none":
            return 0
        start, stop = name.split("-")
        return int(stop) - int(start) + 1

    # "all" is dropped only when a prefix already covers the same number of
    # steps, which happens when total steps is a multiple of the window.
    curve = [n for n in names if n != "all"]
    if conditioned_count("all") not in {conditioned_count(n) for n in curve}:
        curve.append("all")
    order = np.argsort([conditioned_count(n) for n in curve])
    curve = [curve[i] for i in order]

    counts, accuracy, error, recovered = [], [], [], []
    for name in curve:
        i = names.index(name)
        correct = data["predictions"][i] == truth
        counts.append(conditioned_count(name))
        accuracy.append(correct.mean())
        error.append(np.sqrt(correct.mean() * (1 - correct.mean()) / len(correct)))
        recovered.append(float(data["recovery"][i].mean()))

    np.savez(os.path.join(common.results_dir(args.exp, args.dataset, "windows"),
                          f"accuracy_{args.npz}"),
             names=np.array(curve), counts=np.array(counts),
             accuracy=np.array(accuracy), error=np.array(error),
             recovered=np.array(recovered), real_accuracy=real_accuracy)

    out = os.path.join(common.results_dir(args.exp, args.dataset, "figures"),
                       "conditioning_curves.png")
    plotting.conditioning_curves(
        counts, accuracy, error, recovered,
        chance=1 / len(labels), ceiling=real_accuracy,
        title=f"{args.dataset}: what each conditioned step contributes",
        out_path=out)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()