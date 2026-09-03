# Artefacts

What `XAI_Results/` contains, who writes each file and who reads it back.

The directory is git-ignored: `.gitattributes` would send every `.npz` to Git LFS, and
the whole tree is ~446 MB of quantities that the scripts regenerate. It is delivered
separately from the repository. Nothing here is a source: every file can be rebuilt
with the commands in [results.md](results.md#how-to-reproduce).

## Layout

```
XAI_Results/Exp_<exp>/<dataset>/<kind>/<name>.npz
```

`<name>` encodes the hyper-parameters of the run — `epoch31`, `steps31_window5`,
`steps31_ugs1.752` — so runs with different settings sit side by side instead of
overwriting each other. The values used for the final runs are tabulated in
[results.md](results.md#how-to-reproduce).

## Coverage

| dataset | conditioning | windows | guidance | attention | figures |
|---|---|---|---|---|---|
| cifar10 | `epoch31` | `steps31_window5` | `steps31_ugs1.752` | ✓ | 9 |
| cifar100 | `epoch41` | `steps28_window7` | — | — | 6 |
| pathmnist | `epoch28` | `steps48_window12` | — | — | 2 |
| dermamnist | `epoch5` | `steps27_window7` | — | — | 2 |
| bloodmnist | `epoch35` | `steps44_window11` | — | — | 6 |
| retinamnist | `epoch31` | `steps30_window7` | — | — | 2 |
| summary | — | — | — | — | 1 |

**The asymmetry is deliberate, not a gap.** `conditioning` costs seconds and no GPU, so
it was run everywhere. `windows` costs about an hour per dataset and was still run
everywhere, because Result 3 is the main claim and had to hold beyond one dataset.
`guidance` and `attention` were run on CIFAR-10 only: the first is a measurement of the
same phenomenon `windows` covers, kept as a cross-check rather than repeated six times;
the second supports a negative result, and a negative result established on one dataset
does not become more negative on six.

---

## `conditioning/epoch<N>.npz`

Written by `python -m scripts.xai_conditioning`. Read by
`notebooks/01_conditioning.ipynb`, `scripts/xai_figures` and `scripts/xai_summary`.
The geometry of the class-conditioning space, computed from the `ClassEncoder` weights
alone — no GPU, no generation. `E` is flattened: 100 context slots × 768 dimensions.

| keys | what they hold |
|---|---|
| `labels`, `checkpoint` | class names in label order; the checkpoint path the run read |
| `E` (C, 76800), `mean`, `u`, `norm_u`, `norm_E`, `ratio`, `cos_Eu` | the embeddings, their mean, the shared component and each class's alignment with it |
| `residuals` (C, 76800), `shared_energy`, `residual_energy`, `residual_ratio`, `cos_residuals` | what is left once the shared component is removed — the part that actually distinguishes classes |
| `singular_values`, `explained_variance_ratio`, `effective_rank` | the spectrum of the residuals; `effective_rank` is the scalar Result 4 correlates with generation quality |
| `perm_observed`, `perm_null` (210), `perm_p_value`, `between_energy`, `within_energy` | permutation test of the semantic grouping |
| `mantel_correlation`, `mantel_p_value`, `mantel_null` (10000), `cos_visual` | Mantel test of conditioning geometry against visual similarity of the real data |

## `guidance/steps<S>_ugs<U>.npz`

Written by `python -m scripts.xai_guidance`. Read by `notebooks/03_temporal.ipynb`.
The class signal `eps(c) − eps(0)` along denoising, one row per step, one column per
class. Note there is one more step than `--steps` asks for: 31 gives 32.

| key | shape | what it holds |
|---|---|---|
| `timesteps` | (S+1,) | the diffusion timestep of each row, descending |
| `absolute` | (S+1, C) | ‖eps(c) − eps(0)‖, the magnitude of the class signal |
| `eps` | (S+1, C) | ‖eps(c)‖, the total predicted noise |
| `relative` | (S+1, C) | `absolute / eps` — the share of the prediction the class accounts for |
| `maps` | (S+1, C, 16, 16) | the same signal resolved spatially, at latent resolution |
| `images` | (C, 128, 128, 3) | the generation the trace was taken from |

## `windows/steps<S>_window<W>.npz`

Written by `python -m scripts.xai_windows`. The intervention behind Result 3: the same
classes from the same initial noise, conditioned only inside a window of steps.

| key | shape | what it holds |
|---|---|---|
| `names` | (8,) | the configurations: `all`, then prefixes `0-4`, `0-9`, …, then `none` |
| `images` | (8, N, 128, 128, 3) | the generations, one block per configuration |
| `recovery` | (8, N) | per image, how much of the fully-conditioned result it recovers: 1 = identical to `all`, 0 = as far from it as `none` is |
| `class_index` | (N,) | the requested class of each image |
| `labels`, `timesteps`, `per_class` | | class names, the timestep of each step, images per class |

`all` means conditioning applied at every step, `none` at no step. They are the two
extremes that define the unit of `recovery`, not windows in their own right.

## `windows/accuracy_steps<S>_window<W>.npz`

Written by `python -m scripts.xai_score`, which judges the images above with a ResNet20
trained on real data only. Read by `notebooks/03_temporal.ipynb`.

| key | shape | what it holds |
|---|---|---|
| `names` | (8,) | same configurations — **re-sorted**, see the warning below |
| `counts` | (8,) | how many steps were conditioned in that configuration |
| `accuracy` | (8,) | share of images the judge assigns to the requested class |
| `error` | (8,) | standard error of that share, √(p(1−p)/n) |
| `recovered` | (8,) | the mean of `recovery` for that configuration |
| `real_accuracy` | scalar | the same judge on the real test set — the ceiling to read `accuracy` against |

> **⚠️ The two `windows` files order their configurations differently.**
> `steps<S>_window<W>.npz` stores them as generated — `all`, `0-4`, …, `none`.
> `accuracy_…npz` stores them sorted by `counts` — `none`, `0-4`, …, `all`.
> The names are the same and both arrays have length 8, so joining the two files **by
> index** silently pairs the fully-conditioned run with the unconditioned one and
> inverts every conclusion. Join by `names`.

## `attention/uniformity.npz` and `attention/stability.npz`

Written by `python -m scripts.xai_test_capture` and `xai_test_stability`. Read by
`notebooks/02_attention.ipynb`. The evidence for the negative result.

`uniformity.npz` holds, prefixed `first_` and `last_` for the two ends of denoising:
`timestep`; `token_weights` (layers × 100 slots); `condmap_16`, `condmap_8`, `condmap_4`,
`condmap_2`, the conditional-vs-unconditional total variation at each attention
resolution; and per (layer, head) `layer`, `head`, `side`, `slot`, `conditioning_cv`,
`slot_cv`.

`stability.npz` holds, for the last step over a batch of 8: `images`, `labels`,
`timestep`; per (layer, head) `layer`, `head`, `side`, `slot`, `slot_cv`,
`conditioning_cv`, `across` (map correlation across images), and `detail_cond`,
`detail_uncond`, `lumin_cond`, `lumin_uncond`; plus `best_index` and `best_maps`, the
maps of the most selective head over the eight images.

## `figures/`

PNGs written by the scripts as they run — working output, regenerated on every run. The
selected, final figures are tracked in the repository under
[`docs/figures/`](figures/); those are the ones the documents refer to.

## Sizes

Three files account for 350 of the 446 MB, all of them stored generations:

| file | size |
|---|---|
| `cifar10/windows/steps31_window5.npz` | 147 MB |
| `cifar100/conditioning/epoch41.npz` | 113 MB |
| `bloodmnist/windows/steps44_window11.npz` | 90 MB |

If the archive has to be made smaller, the `images` array inside the `windows` files is
where the weight is; every derived quantity survives without it, but the generations
cannot then be re-scored or re-inspected.