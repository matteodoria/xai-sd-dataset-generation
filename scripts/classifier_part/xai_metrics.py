"""Common metrics for Grad-CAM and Integrated Gradients."""

import numpy as np
import tensorflow as tf

TOP_FRACTION = 0.20
PERTURBATION_STEPS = 10
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)

def saliency_to_2d(saliency):
    """Converts a 2D or 3D saliency map to a normalized positive importance map."""
    saliency = np.asarray(saliency, dtype=np.float32).squeeze()
    if saliency.ndim == 3:
        saliency = np.sum(np.abs(saliency), axis=-1)
    else:
        saliency = np.maximum(saliency, 0.0)

    saliency = saliency - np.min(saliency)
    maximum = np.max(saliency)
    if maximum > 0:
        saliency = saliency / maximum
    return saliency


def resize_saliency(saliency, size=(32, 32)):
    saliency = saliency_to_2d(saliency)
    resized = tf.image.resize(
        saliency[..., None],
        size,
        method="bilinear",
    ).numpy()[..., 0]
    return saliency_to_2d(resized)


def pearson_correlation(first_saliency, second_saliency):
    first = resize_saliency(first_saliency).reshape(-1)
    second = resize_saliency(second_saliency).reshape(-1)
    if np.std(first) == 0.0 or np.std(second) == 0.0:
        return np.nan
    return float(np.corrcoef(first, second)[0, 1])


def top_fraction_mask(saliency, top_fraction=TOP_FRACTION):
    saliency = resize_saliency(saliency)
    flat = saliency.reshape(-1)
    top_count = max(1, int(round(top_fraction * len(flat))))
    top_indices = np.argpartition(flat, -top_count)[-top_count:]
    mask = np.zeros(len(flat), dtype=bool)
    mask[top_indices] = True
    return mask.reshape(saliency.shape)


def top_fraction_iou(first_saliency, second_saliency, top_fraction=TOP_FRACTION):
    first_mask = top_fraction_mask(first_saliency, top_fraction)
    second_mask = top_fraction_mask(second_saliency, top_fraction)
    intersection = np.logical_and(first_mask, second_mask).sum()
    union = np.logical_or(first_mask, second_mask).sum()
    if union == 0:
        return np.nan
    return float(intersection / union)



def deletion_insertion_auc(
    model,
    image,
    saliency,
    target_class,
    steps=PERTURBATION_STEPS,
    baseline=None,
):
    """Compute deletion AUC and insertion AUC using the same saliency map."""
    image = np.asarray(image, dtype=np.float32)
    saliency = resize_saliency(saliency, size=image.shape[:2])
    order = np.argsort(saliency.reshape(-1))[::-1]
    pixel_count = saliency.size
    fractions = np.linspace(0.0, 1.0, steps + 1)

    if baseline is None:
        baseline = np.broadcast_to(CIFAR10_MEAN, image.shape).copy()

    deletion_images = []
    insertion_images = []
    for fraction in fractions:
        selected_count = int(round(fraction * pixel_count))
        mask = np.zeros(pixel_count, dtype=np.float32)
        mask[order[:selected_count]] = 1.0
        mask = mask.reshape(image.shape[0], image.shape[1], 1)

        deletion_images.append(image * (1.0 - mask) + baseline * mask)
        insertion_images.append(baseline * (1.0 - mask) + image * mask)

    batch = np.concatenate(
        [np.asarray(deletion_images), np.asarray(insertion_images)],
        axis=0,
    )
    scores = model(batch, training=False).numpy()[:, target_class]
    split_index = len(fractions)
    deletion_scores = scores[:split_index]
    insertion_scores = scores[split_index:]

    deletion_auc = np.trapz(deletion_scores, fractions)
    insertion_auc = np.trapz(insertion_scores, fractions)
    return float(deletion_auc), float(insertion_auc)


def saliency_metrics_against_reference(first_saliency, second_saliency):
    return {
        "pearson_correlation": pearson_correlation(first_saliency, second_saliency),
        "top20_iou": top_fraction_iou(first_saliency, second_saliency),
    }

