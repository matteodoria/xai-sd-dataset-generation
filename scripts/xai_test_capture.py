"""Check that the attention capture sees what it should, on one forward pass."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

from Models.stable_diffusion import MyStableDiffusion

DATASET, EXP, ENC_EPOCH, DIF_EPOCH = "cifar10", "xAI", 31, 10
NUM_CLASSES, RES = 10, 32

ddpm = MyStableDiffusion(
    res=RES, num_classes=NUM_CLASSES,
    original_diffusion=True, original_text_encoder=False,
    enc_weight_path=f"Checkpoints/DDPM/Exp_{EXP}/{DATASET}/MyEmbedding/epoch{ENC_EPOCH}.hdf5",
    diff_weight_path=f"Checkpoints/DDPM/Exp_{EXP}/{DATASET}/DiffusionFt/epoch{DIF_EPOCH}.hdf5")

import numpy as np

SEED, STEPS, UGS = 1234, 20, 1.0

label = tf.eye(NUM_CLASSES)[3:4]          # cat
context = ddpm.text_encoder(label)

print(f"\n... reference generation (original generate_image) ...")
reference = ddpm.gen(label, num_images=1, num_inference_steps=STEPS,
                     ugs=UGS, seed=SEED)

print(f"\n... instrumented generation (eager loop) ...")
with instrumentation.AttentionRecorder() as rec:
    images, steps, _ = instrumentation.generate(
        ddpm, context, batch_size=1, num_steps=STEPS, ugs=UGS, seed=SEED,
        recorder=rec)

difference = np.abs(reference.astype(int) - images.astype(int))
print(f"\nimages: reference {reference.shape}, instrumented {images.shape}")
print(f"max pixel difference : {difference.max()} (out of 255)")
print(f"mean difference      : {difference.mean():.4f}")
print(f"pixels differing     : {(difference > 0).sum()} / {difference.size}")

print(f"\ncaptured {len(steps)} steps")
first = steps[0]
print(f"first step: timestep {first['timestep']}, "
      f"{len(first['conditional'])} conditional + "
      f"{len(first['unconditional'])} unconditional tensors")

from XAI.Generator.Conditioning import attention, plotting, common

saved = {}
for tag, name, step in (("first", "first (noisiest)", steps[0]),
                        ("last",  "last (cleanest)",  steps[-1])):
    print(f"\n--- {name} step, timestep {step['timestep']} ---")

    weights = attention.token_weights(step["conditional"])
    average = weights.mean(axis=0)
    ranking = np.argsort(average)[::-1]
    print(f"  slot distribution: top-10 slots hold "
          f"{average[ranking[:10]].sum():.1%} of the attention")
    print(f"  most attended slots: {ranking[:8].tolist()}")

    maps = attention.conditioning_map(step['conditional'], step['unconditional'])
    for side, grid in sorted(maps.items()):
        print(f"  {side:2d}x{side:<2d} conditioning map: "
              f"min {grid.min():.3f}, max {grid.max():.3f}, mean {grid.mean():.3f}")

    rows = attention.head_selectivity(step["conditional"], step["unconditional"])
    cond_cv = np.array([r["conditioning_cv"] for r in rows])
    slot_cv = np.array([r["slot_cv"] for r in rows])

    print(f"  {len(rows)} (layer, head) pairs, no averaging")
    print(f"    conditioning CV: median {np.median(cond_cv):.3f}, "
          f"90th pct {np.percentile(cond_cv, 90):.3f}, max {cond_cv.max():.3f}")
    print(f"    slot-map     CV: median {np.median(slot_cv):.3f}, "
          f"90th pct {np.percentile(slot_cv, 90):.3f}, max {slot_cv.max():.3f}")

    best = max(rows, key=lambda r: r["slot_cv"])
    print(f"    most selective: layer {best['layer']} head {best['head']} "
          f"at {best['side']}x{best['side']}, slot {best['slot']}, "
          f"CV {best['slot_cv']:.3f}")

    # Stored so notebook 02 can recompute these numbers without a GPU.
    saved[f"{tag}_timestep"] = int(step["timestep"])
    saved[f"{tag}_token_weights"] = weights
    saved[f"{tag}_conditioning_cv"] = cond_cv
    saved[f"{tag}_slot_cv"] = slot_cv
    for field in ("layer", "head", "side", "slot"):
        saved[f"{tag}_{field}"] = np.array([r[field] for r in rows])
    for side, grid in maps.items():
        saved[f"{tag}_condmap_{side}"] = grid

step = steps[0]
rows = attention.head_selectivity(step["conditional"], step["unconditional"])
top = sorted(rows, key=lambda r: r["slot_cv"], reverse=True)[:7]

out = os.path.join(common.results_dir(EXP, DATASET, "figures"), "attention_heads.png")
plotting.attention_overlay(
    images[0],
    [r["slot_map"] for r in top],
    [f"L{r['layer']} H{r['head']} slot {r['slot']}\n{r['side']}x{r['side']}  "
     f"CV {r['slot_cv']:.2f}" for r in top],
    title=f"{DATASET} 'cat', timestep {step['timestep']}: most selective heads",
    out_path=out)
print(f"\nSaved: {out}")

out = os.path.join(common.results_dir(EXP, DATASET, "attention"), "uniformity.npz")
np.savez(out, **saved)
print(f"Saved: {out}")