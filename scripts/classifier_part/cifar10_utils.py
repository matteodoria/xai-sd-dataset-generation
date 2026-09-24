"""Upload CIFAR-10 """

import os
import pickle
import tarfile
from pathlib import Path

import numpy as np
import tensorflow as tf


PROJECT_DIR = Path(__file__).resolve().parents[2]
LOCAL_CIFAR10_ARCHIVE = PROJECT_DIR / "Data/cifar-10-python.tar.gz"


def _load_test_batch_from_file(batch_file):
    with batch_file.open("rb") as file:
        batch = pickle.load(file, encoding="latin1")
    images = batch["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(batch["labels"], dtype=np.int64)
    return images, labels


def _load_test_batch_from_archive(archive_path):
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            member = archive.getmember("cifar-10-batches-py/test_batch")
            with archive.extractfile(member) as file:
                batch = pickle.load(file, encoding="latin1")
    except (tarfile.TarError, EOFError, KeyError, pickle.PickleError, OSError) as error:
        raise RuntimeError(
            f"Local CIFAR-10 not valid or incomplete: {archive_path}"
        ) from error

    images = batch["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(batch["labels"], dtype=np.int64)
    return images, labels


def load_cifar10_test_data():
    """Upload the test set CIFAR-10 from local cache, without automated download."""
    extracted_candidates = (
        PROJECT_DIR / "Data/cifar-10-batches-py/test_batch",
        Path.home() / ".keras/datasets/cifar-10-batches-py/test_batch",
    )
    for batch_file in extracted_candidates:
        if batch_file.exists():
            x_test, y_test = _load_test_batch_from_file(batch_file)
            return x_test.astype("float32") / 255.0, y_test

    archive_candidates = (
        LOCAL_CIFAR10_ARCHIVE,
        Path.home() / ".keras/datasets/cifar-10-python.tar.gz",
    )
    for archive_path in archive_candidates:
        if archive_path.exists():
            x_test, y_test = _load_test_batch_from_archive(archive_path)
            return x_test.astype("float32") / 255.0, y_test

    if os.environ.get("ALLOW_CIFAR10_DOWNLOAD") == "1":
        (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        return x_test.astype("float32") / 255.0, y_test.reshape(-1)

    raise RuntimeError(
        "CIFAR-10 not found locally. The automated download is disabled "
        "to avoid TimeoutError. Place a complete archive in "
        f"{LOCAL_CIFAR10_ARCHIVE}, or run once with "
        "ALLOW_CIFAR10_DOWNLOAD=1 if you have internet access."
    )
