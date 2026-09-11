"""Aggregation of the captured cross-attention weights.

Raw weights have shape (batch, heads, positions, slots) per layer,
per denoising step, for each of the conditional and unconditional
passes. This module reduces them to quantities that can actually be
looked at.

One property drives the design: the softmax is taken over the context slots,
so every spatial position holds a distribution summing to 1.
Averaging attention over slots therefore yields a constant map and says nothing.
What varies in space is how that distribution *differs* between the conditional
and the unconditional pass, which is what conditioning_map measures.
"""

import math
import numpy as np

def as_grid(flat):
    """Reshape a (..., h*w) array to (..., h, w), assuming a square latent."""
    positions = flat.shape[-1]
    side = int(round(math.sqrt(positions)))
    if side * side != positions:
        raise ValueError(f"{positions} positions do not form a square grid.")
    return flat.reshape(flat.shape[:-1] + (side, side))

def token_weights(tensors):
    """Attention received by each context slot, one row per layer.

    Averaged over heads and spatial positions. Because the softmax normalises
    over slots, each row is a probability distribution: it tells which of the
    slots emitted by the ClassEncoder the UNet actually reads.
    Nothing in training forced all 100 to carry information, they exist because
    the UNet expects that context length.

    Returns:
         (n_layers, n_slots)
    """
    return np.stack([np.asarray(t).mean(axis=(0,1,2)) for t in tensors])

def conditioning_map(conditional, unconditional):
    """Where the class changes how the latent reads the context.

    For every spatial position, the total variation distance between the conditional
    and the unconditional attention distribution: 0 where the class leaves the
    reading untouched, 1 where it redirects it entirely.
    Layers are grouped by gris size, since the UNet attends at several resolutions.

    Returns:
        dict mapping grid side to a (side, side) map averaged over the layers
        operating at the resolution.
    """
    maps = {}
    for cond, uncond in zip(conditional, unconditional):
        c = np.asarray(cond).mean(axis=1)[0]
        u = np.asarray(uncond).mean(axis=1)[0]
        distance = 0.5 * np.abs(c - u).sum(axis=-1)
        grid = as_grid(distance)
        maps.setdefault(grid.shape[0],[]).append(grid)

    return {
        side: np.mean(grids, axis=0) for side, grids in maps.items()
    }

def head_selectivity(conditional, unconditional):
    """Spatial selectivity of every (layer, head) pair, with no averaging.
    Aggregated maps can hide structure: if one head attends to the top-left
    and another to the bottom-rights, their mean is flat while both are
    selective. This keeps every head separate.

    Two quantities per head, both as coefficient of variation across positions
    (0 for a perfectly uniform map, growing with concentration):
        conditional_cv: on the conditional-vs-unconditional total variation,
            i.e. how unevenly the class changes the reading context.
        slot_cv: on the attention paid to that head's most attended slot.
            This is the DAAM-style map, and unlike the distribution per position
            is not normalised, the softmax runs over slots, not over space, so it
            is free to vary spatially.
    If localisation exists anywhere, it is here.

    Return:
        list of dicts, one per (layer, head).
    """
    rows = []

    for layer, (cond, uncond) in enumerate(zip(conditional, unconditional)):
        c = np.asarray(cond)[0]     # (head, position, slots)
        u = np.asarray(uncond)[0]

        for head in range(c.shape[0]):
            tv = 0.5 * np.abs(c[head] - u[head]).sum(axis=-1)
            slot = int(c[head].mean(axis=0).argmax())
            slot_map = c[head][:, slot]

            rows.append({
                "layer": layer,
                "head": head,
                "side": int(round(math.sqrt(c.shape[1]))),
                "slot": slot,
                "conditioning_cv": float(tv.std() / tv.mean()),
                "slot_cv": float(slot_map.std() / slot_map.mean()),
                "slot_map": as_grid(slot_map),
            })

    return rows

def batch_slot_maps(tensor, head, slot):
    """Map of one (head, slot) for every image in the batch.

    Returns:
        (batch, side, side)
    """
    return as_grid(np.asarray(tensor)[:, head, :, slot])


def pairwise_correlation(maps):
    """Mean correlation between all pairs of maps.

    Near 1 means the map barely changes across images: the head responds to
    position in the grid rather than to content. Near 0 means it follows what
    is actually in the image.
    """
    flat = np.asarray(maps).reshape(len(maps), -1)
    correlation = np.corrcoef(flat)
    rows, cols = np.triu_indices(len(maps), k=1)
    return float(correlation[rows, cols].mean())

def image_detail(image, side):
    """Local standard deviation of an image on a side x side grid.

    A crude proxy for where content is: in CIFAR the subject is textured and
    the background largely flat. Used to ask whether an attention map follows
    the subject or merely varies.
    """
    factor = image.shape[0] // side
    blocks = np.asarray(image, dtype=np.float64).mean(axis=-1)
    blocks = blocks.reshape(side, factor, side, factor)
    return blocks.std(axis=(1, 3))

def image_luminance(image, side):
    """Mean brightness of the image on a side x side grid.

    Companion to image_detail: dark regions and detailed regions overlap in
    these images (a dark subject on a light background produces both), so an
    attention map correlating with one may simply be tracking the other.
    """
    factor = image.shape[0] // side
    blocks = np.asarray(image, dtype=np.float64).mean(axis=-1)
    return blocks.reshape(side, factor, side, factor).mean(axis=(1, 3))