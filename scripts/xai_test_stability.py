"""Do the selective attention heads follow content, or something simpler?

Generates one image per class and, at the last denoising step (when the latent
finally holds the image), asks three questions of every attention head:

  1. does its map change across images, or is it a fixed positional pattern?
  2. does it track image detail, or image brightness? The two overlap in these
     pictures — a dark subject on a light background produces both — so a map
     correlating with one may simply be following the other.
  3. does the conditional attention behave differently from the unconditional
     one? If not, whatever we see is a property of the UNet's cross-attention
     rather than of the class conditioning.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from Models.stable_diffusion import MyStableDiffusion
from XAI import attention, common, instrumentation, plotting

DATASET, EXP, ENC_EPOCH, DIF_EPOCH = "cifar10", "xAI", 31, 10
NUM_CLASSES, RES, SEED, STEPS, UGS, BATCH = 10, 32, 1234, 20, 1.0, 8


def correlations(tensors, rows, images, reference):
    """Correlation of each head's map with an image property, averaged over images.

    Args:
        tensors: the per-layer attention of one denoising step.
        rows: output of head_selectivity, giving the (layer, head, slot) to read.
        reference: image_detail or image_luminance.

    Returns:
        one value per row, in the same order.
    """
    values = []
    for row in rows:
        maps = attention.batch_slot_maps(
            tensors[row["layer"]], row["head"], row["slot"])
        side = maps.shape[1]
        values.append(np.mean([
            np.corrcoef(maps[i].ravel(), reference(images[i], side).ravel())[0, 1]
            for i in range(len(images))]))
    return np.array(values)


def main():
    ddpm = MyStableDiffusion(
        res=RES, num_classes=NUM_CLASSES,
        original_diffusion=True, original_text_encoder=False,
        enc_weight_path=f"Checkpoints/DDPM/Exp_{EXP}/{DATASET}/MyEmbedding/epoch{ENC_EPOCH}.hdf5",
        diff_weight_path=f"Checkpoints/DDPM/Exp_{EXP}/{DATASET}/DiffusionFt/epoch{DIF_EPOCH}.hdf5")

    labels = tf.eye(NUM_CLASSES)[:BATCH]
    context = ddpm.text_encoder(labels)

    # Only the first and last step are kept: with a batch of 8 and 20 steps,
    # holding every step's attention would take close to 2 GB.
    with instrumentation.AttentionRecorder() as rec:
        images, steps, _ = instrumentation.generate(
            ddpm, context, batch_size=BATCH, num_steps=STEPS, ugs=UGS,
            seed=SEED, recorder=rec, record_steps={0, STEPS - 1})

    step = steps[-1]
    tensors = step["conditional"]
    print(f"\n{images.shape[0]} images, timestep {step['timestep']} "
          f"(last step: the latent now holds the image)")

    ## 1. Which heads are selective, and do their maps survive a change of image?
    rows = attention.head_selectivity(tensors, step["unconditional"])
    top = sorted(rows, key=lambda r: r["slot_cv"], reverse=True)[:10]
    best = top[0]
    best_index = next(i for i, r in enumerate(rows) if r is best)

    print(f"\n{'layer':>5} {'head':>5} {'slot':>5} {'grid':>8} {'CV':>6} "
          f"{'corr across images':>20}")
    for row in top:
        maps = attention.batch_slot_maps(
            tensors[row["layer"]], row["head"], row["slot"])
        print(f"{row['layer']:>5} {row['head']:>5} {row['slot']:>5} "
              f"{row['side']:>3}x{row['side']:<4} {row['slot_cv']:>6.2f} "
              f"{attention.pairwise_correlation(maps):>20.3f}")

    across = np.array([attention.pairwise_correlation(
        attention.batch_slot_maps(tensors[r["layer"]], r["head"], r["slot"]))
        for r in rows])

    print(f"\nall {len(rows)} heads: median correlation across images "
          f"{np.median(across):.3f}, min {across.min():.3f}, max {across.max():.3f}")

    ## 2. Detail or brightness? Conditional or unconditional?
    detail_cond = correlations(tensors, rows, images, attention.image_detail)
    detail_uncond = correlations(step["unconditional"], rows, images,
                                 attention.image_detail)
    lumin_cond = correlations(tensors, rows, images, attention.image_luminance)
    lumin_uncond = correlations(step["unconditional"], rows, images,
                                attention.image_luminance)

    print(f"\ncorrelation of the attention maps with image properties, "
          f"over all {len(rows)} heads")
    print(f"{'':>16} {'median':>9} {'mean':>9} {'min':>9} {'max':>9}")
    for name, values in (("detail cond", detail_cond),
                         ("detail uncond", detail_uncond),
                         ("luminance cond", lumin_cond),
                         ("luminance uncond", lumin_uncond)):
        print(f"{name:>16} {np.median(values):>9.3f} {values.mean():>9.3f} "
              f"{values.min():>9.3f} {values.max():>9.3f}")

    print(f"\nmost selective head (L{best['layer']} H{best['head']} "
          f"slot {best['slot']}, CV {best['slot_cv']:.2f}):")
    print(f"  detail    {detail_cond[best_index]:+.3f} conditional, "
          f"{detail_uncond[best_index]:+.3f} unconditional")
    print(f"  luminance {lumin_cond[best_index]:+.3f} conditional, "
          f"{lumin_uncond[best_index]:+.3f} unconditional")

    ## 3. Look at it
    maps = attention.batch_slot_maps(tensors[best["layer"]], best["head"],
                                     best["slot"])
    out = os.path.join(common.results_dir(EXP, DATASET, "figures"),
                       "attention_across_classes.png")
    plotting.maps_over_batch(
        images, maps, common.get_labels(DATASET)[:BATCH],
        title=f"L{best['layer']} H{best['head']} slot {best['slot']} "
              f"(CV {best['slot_cv']:.2f}) at timestep {step['timestep']}",
        out_path=out)
    print(f"\nSaved: {out}")

    ## 4. Store the evidence, so notebook 02 can read it without a GPU.
    saved = {
        "timestep": int(step["timestep"]),
        "images": images,
        "labels": np.array(common.get_labels(DATASET)[:BATCH]),
        "across": across,
        "detail_cond": detail_cond,
        "detail_uncond": detail_uncond,
        "lumin_cond": lumin_cond,
        "lumin_uncond": lumin_uncond,
        "best_index": best_index,
        "best_maps": maps,
    }
    for field in ("layer", "head", "side", "slot", "slot_cv", "conditioning_cv"):
        saved[field] = np.array([r[field] for r in rows])

    out = os.path.join(common.results_dir(EXP, DATASET, "attention"), "stability.npz")
    np.savez(out, **saved)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()