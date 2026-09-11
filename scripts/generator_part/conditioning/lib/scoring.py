"""Judging generated images with a classifier trained on real data only."""

import tensorflow as tf

from Models.ImageClassifiers import resnet

JUDGE = "Models/Checkpoints/Classifiers/resnet20/{dataset}/real/best_model.h5"


def load_judge(dataset, resolution, num_classes):
    """Rebuild the ResNet20 and load its trained weights.

    The .h5 holds a custom class, so load_model cannot reconstruct it; the
    architecture is recreated with the hyper-parameters used in training
    (initial_filters=64, giving 4,291,530 parameters) and the weights loaded in.
    """
    model = resnet.ResNet20(input_shape=(resolution, resolution, 3),
                            num_classes=num_classes, initial_filters=64,
                            seed=1234)
    model.load_weights(JUDGE.format(dataset=dataset))
    return model


def classify(model, images, resolution):
    """Resize and rescale exactly as the training pipeline did.

    Training applied Rescaling(1./255) to images loaded at the target resolution
    by image_dataset_from_directory, which resizes bilinearly. Any other
    preprocessing yields low accuracies that look like a finding and are a bug.
    """
    resized = tf.image.resize(tf.cast(images, tf.float32), (resolution, resolution))
    return model.predict(resized / 255.0, verbose=0)