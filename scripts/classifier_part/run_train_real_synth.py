import argparse

from pathlib import Path

import numpy as np
import tensorflow as tf

from Models.ImageClassifiers.resnet import ResNet20


SEED = 1234
N_CLASSES = 10
N_TRAIN_PER_CLASS = 4000

BATCH_SIZE = 256
EPOCHS = 50

SYNTHETIC_PATH = Path(
    "Data/Synthetic/Exp_xAI/cifar10/40.0kEnc31Dif10Is20Ugs1.0"
)

SAVE_DIR = Path("Models/Checkpoints/Classifiers/resnet20")

REAL_MODEL_PATH = SAVE_DIR / "resnet20_real_matched.h5"
SYNTHETIC_MODEL_PATH = SAVE_DIR / "resnet20_synthetic.h5"


def load_datasets(synthetic_path, batch_size):
    (x_train_full, y_train_full), (x_test, y_test) = (
        tf.keras.datasets.cifar10.load_data()
    )

    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train_full = y_train_full.reshape(-1)
    y_test = y_test.reshape(-1)

    rng = np.random.default_rng(SEED)

    train_indices = []
    val_indices = []

    for class_id in range(N_CLASSES):
        class_indices = np.where(y_train_full == class_id)[0]
        class_indices = rng.permutation(class_indices)

        train_indices.extend(
            class_indices[:N_TRAIN_PER_CLASS]
        )
        val_indices.extend(
            class_indices[N_TRAIN_PER_CLASS:]
        )

    train_indices = np.asarray(train_indices)
    val_indices = np.asarray(val_indices)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    x_train = x_train_full[train_indices]
    y_train = y_train_full[train_indices]

    x_val = x_train_full[val_indices]
    y_val = y_train_full[val_indices]

    synthetic_ds = tf.keras.utils.image_dataset_from_directory(
        synthetic_path,
        labels="inferred",
        label_mode="int",
        image_size=(32, 32),
        batch_size=batch_size,
        shuffle=True,
        seed=SEED,
    )

    synthetic_ds = synthetic_ds.map(
        lambda images, labels: (
            tf.cast(images, tf.float32) / 255.0,
            labels,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    return (
        (x_train, y_train),
        (x_val, y_val),
        (x_test, y_test),
        synthetic_ds,
    )


def count_images_per_class(dataset):
    counts = np.zeros(N_CLASSES, dtype=int)

    for _, labels in dataset:
        counts += np.bincount(
            labels.numpy(),
            minlength=N_CLASSES,
        )

    return counts


def create_resnet20():
    model = ResNet20(
        input_shape=(32, 32, 3),
        num_classes=N_CLASSES,
        initial_filters=16,
        seed=SEED,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model


def make_callbacks():
    return [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.1,
            patience=10,
            min_lr=1e-5,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=25,
            restore_best_weights=True,
            verbose=1,
        ),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Train the two ResNet20 classifiers explained by notebooks 05-06."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the existing checkpoints",
    )
    args = parser.parse_args()

    existing = [
        path for path in (REAL_MODEL_PATH, SYNTHETIC_MODEL_PATH) if path.exists()
    ]
    if existing and not args.overwrite:
        raise SystemExit(
            "Refusing to overwrite the checkpoints notebooks 05-06 were computed from:\n"
            + "\n".join(f"- {path}" for path in existing)
            + "\nPass --overwrite to retrain anyway, then rerun notebooks 05-06 "
            "with FORCE_RECOMPUTE = True: the saved results would no longer match."
        )

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    real_train, real_val, real_test, synthetic_ds = load_datasets(
        str(SYNTHETIC_PATH),
        BATCH_SIZE,
    )

    x_train, y_train = real_train
    x_val, y_val = real_val
    x_test, y_test = real_test

    real_counts = np.bincount(
        y_train,
        minlength=N_CLASSES,
    )

    val_counts = np.bincount(
        y_val,
        minlength=N_CLASSES,
    )

    synthetic_counts = count_images_per_class(
        synthetic_ds
    )

    print("Real training images per class:", real_counts)
    print("Real validation images per class:", val_counts)
    print("Synthetic training images per class:", synthetic_counts)

    print("Total real training:", len(x_train))
    print("Total real validation:", len(x_val))
    print("Total synthetic training:", synthetic_counts.sum())
    print("Total real test:", len(x_test))

    assert len(x_train) == 40000
    assert len(x_val) == 10000
    assert len(x_test) == 10000

    assert np.all(real_counts == 4000)
    assert np.all(val_counts == 1000)
    assert np.all(synthetic_counts == 4000)

    model_real = create_resnet20()

    initial_weights = [
        weight.copy()
        for weight in model_real.get_weights()
    ]

    print("\nTRAINING ON 40,000 REAL CIFAR-10 IMAGES")

    model_real.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        shuffle=True,
        validation_data=(x_val, y_val),
        callbacks=make_callbacks(),
    )

    real_results = model_real.evaluate(
        x_test,
        y_test,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    model_real.save(
        str(REAL_MODEL_PATH)
    )

    del model_real
    tf.keras.backend.clear_session()

    print("\nTRAINING ON 40,000 SYNTHETIC CIFAR-10 IMAGES")

    model_synthetic = create_resnet20()

    model_synthetic.set_weights(
        initial_weights
    )

    model_synthetic.fit(
        synthetic_ds,
        epochs=EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=make_callbacks(),
    )

    synthetic_results = model_synthetic.evaluate(
        x_test,
        y_test,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    model_synthetic.save(
        str(SYNTHETIC_MODEL_PATH)
    )

    print(
        f"\nAccuracy real-trained: "
        f"{real_results[1]:.4f}"
    )

    print(
        f"Accuracy synthetic-trained: "
        f"{synthetic_results[1]:.4f}"
    )


if __name__ == "__main__":
    main()