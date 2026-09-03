"""Shared helpers for the xAI analysis"""

import os
import numpy as np

from Data.target_datasets import Cifar10, Cifar100, MedMnist

RESULTS_ROOT = "XAI_Results"


def get_dataset(dataset):
    """Instantiate the loader for a target dataset."""
    match dataset:
        case "cifar10": return Cifar10.CIFAR10()
        case "cifar100": return Cifar100.CIFAR100()
        case "pathmnist": return MedMnist.PathMNIST()
        case "dermamnist": return MedMnist.DermaMNIST()
        case "bloodmnist": return MedMnist.BloodMNIST()
        case "retinamnist": return MedMnist.RetinaMNIST()
        case _: raise ValueError(f"Unknown dataset: {dataset}")


def get_labels(dataset):
    """Class names of a target dataset, in label order."""
    return get_dataset(dataset).get_labels()


def class_means(dataset):
    """Mean training image of each class, flattened.

    A deliberately crude description of what each class looks like, computed
    from the real data with no model involved. Its value is not precision but
    independence: unlike a hand-picked taxonomy, nobody chose it to fit the
    result. Accumulation is in float64 without materialising a float copy of
    the whole training set, which for cifar100 would be over a gigabyte.
    """
    loader = get_dataset(dataset)
    (x_train, y_train), _, _ = loader.load_data_real(return_raw=True)

    x, y = np.asarray(x_train), np.asarray(y_train).ravel()
    return np.stack([x[y == c].mean(axis=0, dtype=np.float64).ravel()
                     for c in range(len(loader.get_labels()))])


def results_dir(exp, dataset, kind):
    """Create and return XAI_Results/Exp_<exp>/<dataset>/<kind>/."""
    path = os.path.join(RESULTS_ROOT, f"Exp_{exp}", dataset, kind)
    os.makedirs(path, exist_ok=True)
    return path

# Canonical coarse grouping of CIFAR-10, used as an independent reference when
# testing whether the conditioning space recovers semantic structure. It is a
# property of the dataset, not a grouping chosen after seeing our results.
SEMANTIC_GROUPS = {
    "cifar10": {
        "vehicle": ["airplane", "automobile", "ship", "truck"],
        "animal": ["bird", "cat", "deer", "dog", "frog", "horse"],
    },
    "cifar100": {
        "aquatic_mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
        "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
        "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
        "food_containers": ["bottle", "bowl", "can", "cup", "plate"],
        "fruit_and_vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
        "household_electrical": ["clock", "keyboard", "lamp", "telephone", "television"],
        "household_furniture": ["bed", "chair", "couch", "table", "wardrobe"],
        "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
        "large_carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
        "large_manmade_outdoor": ["bridge", "castle", "house", "road", "skyscraper"],
        "large_natural_outdoor": ["cloud", "forest", "mountain", "plain", "sea"],
        "large_omnivores_herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
        "medium_mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
        "non_insect_invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
        "people": ["baby", "boy", "girl", "man", "woman"],
        "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
        "small_mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
        "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
        "vehicles_1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
        "vehicles_2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
    },
    # BloodMNIST has no official coarse labels. This is the primary
    # morphological division in haematology: granulocytes show cytoplasmic
    # granules and a lobed nucleus, the others do not — a distinction that is
    # visual as much as biological, which is what this space encodes.
    # Caveat: unlike the CIFAR groupings, this one was chosen after seeing the
    # data, so its p-value deserves more caution.
    "bloodmnist": {
        "granulocytes": ["neutrophils", "eosinophils", "basophils",
                         "immature granulocytes"],
        "other": ["lymphocytes", "monocytes", "erythroblasts", "platelets"],
    },
}

def group_ids(labels, groups):
    """Map each label to the index of its semantic group.

    Raises:
        ValueError: if a label belongs to no group.
                    Without this check a single misspelled name would silently
                    produce a wrong grouping, and a perfectly respectable-looking
                    p-value computed on nonsense.
    """
    lookup = {lab: i for i, members in enumerate(groups.values()) for lab in members}
    missing = [lab for lab in labels if lab not in lookup]

    if missing:
        raise ValueError(f"Labels assigned to no group: {missing}")

    return np.array([lookup[lab] for lab in labels])




