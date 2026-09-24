# Artefacts

What `Results/` contains, who writes each file and who reads it back.

The directory is git-ignored: `.gitattributes` would send every `.npz` to Git LFS, and
the whole tree is ~688 MB of quantities that the code regenerates. It is not in the
repository: it is delivered inside `xAI-Project.zip`, next to the code (see the
[README](../../README.md#the-delivery-archive)). Nothing here is a source: every file can be rebuilt
with the commands in [results.md](results.md#how-to-reproduce), or, for
`cifar10/classifier/`, by running notebooks 05-06 with `FORCE_RECOMPUTE = True`.

## Layout

```
Results/Exp_<exp>/<dataset>/<kind>/<name>.npz
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

`cifar10` also holds `classifier/`, the artefacts of notebooks 05-06 — see
[the classifier section](#classifier-casea-and-caseb) below.

**The asymmetry is deliberate, not a gap.** `conditioning` costs seconds and no GPU, so
it was run everywhere. `windows` costs about an hour per dataset and was still run
everywhere, because Result 3 is the main claim and had to hold beyond one dataset.
`guidance` and `attention` were run on CIFAR-10 only: the first is a measurement of the
same phenomenon `windows` covers, kept as a cross-check rather than repeated six times;
the second supports a negative result, and a negative result established on one dataset
does not become more negative on six.

---

## `conditioning/epoch<N>.npz`

Written by `python -m scripts.generator_part.conditioning.run_conditioning`. Read by
`Notebooks/01_generator_conditioning.ipynb`, `scripts/generator_part/conditioning/run_figures` and `scripts/generator_part/conditioning/run_summary`.
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

Written by `python -m scripts.generator_part.conditioning.run_guidance`. Read by `Notebooks/03_generator_temporal.ipynb`.
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

Written by `python -m scripts.generator_part.conditioning.run_windows`. The intervention behind Result 3: the same
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

Written by `python -m scripts.generator_part.conditioning.run_score`, which judges the images above with a ResNet20
trained on real data only. Read by `Notebooks/03_generator_temporal.ipynb`.

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

Written by `python -m scripts.generator_part.conditioning.run_attention_capture` and `run_attention_stability`. Read by
`Notebooks/02_generator_attention.ipynb`. The evidence for the negative result.

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
[`Results/docs/figures/`](figures/); those are the ones the documents refer to.

## Sizes

Of the generator's ~446 MB, three files account for 350, all of them stored
generations (the classifier's ~240 MB are itemised in its own section):

| file | size |
|---|---|
| `cifar10/windows/steps31_window5.npz` | 147 MB |
| `cifar100/conditioning/epoch41.npz` | 113 MB |
| `bloodmnist/windows/steps44_window11.npz` | 90 MB |

If the archive has to be made smaller, the `images` array inside the `windows` files is
where the weight is; every derived quantity survives without it, but the generations
cannot then be re-scored or re-inspected.

## `CAS/test_accuracies.json`

Written and re-read by `python -m scripts.tools.run_classifier_training`, which
merges into it rather than overwriting: one file per experiment, accumulated
across runs. **Not in the delivered archive**: it is written only when
classifiers are retrained.

It is the one artefact outside the `<dataset>/<kind>/` scheme above — it is
cross-dataset by construction, nested as
`<classifier>/<dataset>/<"real" or "x<N>">/{top_1_accuracy, top_5_accuracy}`.
The classifier weights it scores live under
`Models/Checkpoints/Classifiers/<classifier>/<dataset>/<"real" or "x<N>">/`.

## Classifier: caseA and caseB

Written and re-read by `Notebooks/05_classifier_caseA.ipynb` and
`06_classifier_caseB.ipynb`, under `cifar10/classifier/`. They explain two ResNet20
classifiers on the real CIFAR-10 test set: one trained on real images
(`resnet20_real_matched.h5`), one on synthetic images (`resnet20_synthetic.h5`), both
in `Models/Checkpoints/Classifiers/resnet20/`.

This part does not follow the `<kind>/<name>.npz` scheme above: it is mostly CSV, one
row per image or per summary cell, plus one PNG per explained image. Each notebook
computes on GPU once and reloads afterwards: a heavy cell skips its computation when
its output file already exists, unless `FORCE_RECOMPUTE = True` in the configuration
cell.

### `caseA/` — images both classifiers get right (159 MB)

1000 test images, 100 per class, drawn at random among those both classifiers classify
correctly. The indices are stored once, in `gradcam/`, and Integrated Gradients and
LIME reuse them, so the three methods are compared on the same images. LIME runs on
200 of them, 20 per class.

| file | what it holds |
|---|---|
| `gradcam/caseA_test_indices.npy` (1000,) | test-set indices of the selected images — **the set every caseA analysis reads** |
| `gradcam/caseA_class_counts.csv` | images selected per class |
| `gradcam/gradcam_per_image_metrics.csv` | per image: both predictions and confidences; agreement of the two Grad-CAM maps (Pearson, top-20% IoU); deletion and insertion AUC of each model |
| `gradcam/perturbation_per_image.csv` | per image × model × perturbation (`mean`, `blur`) × condition × fraction of 4×4 patches replaced: logit and probability before and after, and whether the prediction changed |
| `gradcam/perturbation_summary_curves.csv` | the same, averaged overall and per class |
| `gradcam/perturbation_paired_effects.csv` | paired differences between conditions, with 95% bootstrap intervals |
| `gradcam/gradcam_summary_metrics.csv`, `all_summary_metrics.csv` | mean and std of the metrics, overall and per class; the second stacks every summary in one table |
| `gradcam/probability_drop_curves.png`, `prediction_flip_curves.png` | the perturbation curves |
| `integrated_gradients/integrated_gradients_raw.npz` | `explanations_real`, `explanations_synthetic` (1000, 32, 32, 1); `selected_images`, `selected_labels`, `selected_indices` |
| `integrated_gradients/integrated_gradients_metrics.csv`, `…_summary_metrics.csv` | the Grad-CAM metrics, computed on the IG maps |
| `lime/lime_caseA_per_image.csv`, `lime_caseA_summary.csv` | per image, the Jaccard overlap of the two models' top-5 superpixels; its mean per class |

One figure per explained image, in a folder per class `<class_id>_<class_name>/`:
`gradcam/visuals/…/gradcam_index_<i>.png`,
`integrated_gradients/…/integrated_gradients_index_<i>.png`,
`lime/visuals/…/lime_index_<i>.png`, where `<i>` is the test-set index.

In the perturbation files, `condition` says which patches are replaced first:
`self_top`, those the model's own Grad-CAM ranks highest; `other_top`, those the
*other* model's Grad-CAM ranks highest; `self_bottom`, its own lowest; `random`, a
random order. **`random` is repeated 5 times**, so it has five times the rows of the
others: average per condition, never over the whole file.

> **The LIME figures are the ones the metrics were computed on.** LIME draws random
> perturbations, so explaining an image again gives different superpixel masks.
> The notebook displays the saved PNGs rather than recomputing them; any new figure
> has to be produced together with its metrics, not afterwards.

### `caseB/` — images only the real-trained classifier gets right (81 MB)

Test images the real-trained model classifies correctly and the synthetic-trained one
does not: up to 100 per class, drawn at random. **The classes are not balanced** —
bird, cat, deer, dog, horse and ship reach 100, the others have fewer errors to draw
from (airplane 92, automobile 81, frog 35, truck 93), for 901 images in all. The
indices sit at the root, in `caseB_test_indices.npy`, and all three analyses below read
them; LIME runs on 200 of them, 20 per class.

| file | what it holds |
|---|---|
| `caseB_test_indices.npy` (901,) | test-set indices of the selected images — **the set every caseB analysis reads** |
| `cross_model_ig/caseB_cross_model_ig_maps.npz` | `ig_real`, `ig_synthetic` (901, 32, 32), the true-class IG maps of both models; `selected_indices` |
| `cross_model_ig/CaseB_Summary_ByClass.csv` | per true class: how often caseB occurs, true-class confidence of both models, agreement of the two maps (cosine, top-20% IoU), missing-evidence mass, and the real model's confidence drop when the top 10% of the missing evidence is masked (`mask10_…`) |
| `cross_model_ig/CaseB_Selection_Counts.csv` | per class: test images, caseB errors available and their rate, errors selected — the counts of `synthetic_model/caseB_selection_counts.csv`, plus the test-set totals |
| `cross_model_ig/CaseB_Confusion_AllEligible.csv` | true class × class the synthetic-trained model predicts, over **all** 1381 caseB errors of the test set, not only the 901 selected |
| `synthetic_model/caseB_selection_counts.csv` | per class, errors available and errors selected |
| `synthetic_model/caseB_synthetic_only_per_image_metrics.csv` | per image, on the synthetic-trained model only: probability of the true and of the predicted class; agreement of true- vs wrong-class maps and of Grad-CAM vs IG (Pearson, top-20% IoU); deletion and insertion AUC of each map |
| `synthetic_model/caseB_summary_by_true_class.csv`, `…_by_true_predicted_pair.csv` | mean and std of the same, per true class and per (true, predicted) pair |
| `synthetic_model/caseB_masking_per_image.csv` | per image × map (`gradcam_true`, `gradcam_wrong`, `ig_true`, `ig_wrong`) × fraction masked: true- and wrong-class probability, whether the prediction flipped, and whether it flipped to the true class |
| `synthetic_model/caseB_masking_summary.csv`, `caseB_masking_curves.png` | the same, averaged per map and fraction |
| `lime/lime_caseB_per_image.csv`, `lime_caseB_summary.csv` | per image, the Jaccard overlap of the top-5 superpixels for the true and the wrong class; its mean per class |
| `lime/lime_caseB_masking_per_image.csv`, `…_masking_summary.csv`, `…_masking_curves.png` | wrong-class probability drop, and recovery of the true class, as the superpixels supporting the error are masked |

One figure per explained image, in a folder per class `<class_id>_<class_name>/`:
`synthetic_model/visuals/…/caseB_index_<i>_true_<class>_pred_<class>.png` and
`lime/visuals/…/lime_index_<i>.png`. The synthetic-model Grad-CAM and IG maps are
stored **only** in these figures (`SAVE_RAW = False`): analysing them again means
recomputing them on GPU.

> **The two IG analyses of caseB are not on the same scale.** `cross_model_ig/` uses
> Xplique's `IntegratedGradients` with 50 steps and a black baseline;
> `synthetic_model/` uses its own implementation with 64 steps and the CIFAR-10 mean as
> baseline. Compare their maps by shape and ranking, never by magnitude.