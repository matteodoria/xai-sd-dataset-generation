# script/tools/fig_attention.py
"""One-off figure for the slides: cross-attention does not localise.

Reads the stored artefacts (no GPU, no generation) and draws three panels,
one per line of evidence:
  A  aggregated conditioning map, last step      -> flat
  B  most selective head, last step              -> speckle, not a subject
  C  detail-following with vs without the class  -> the same thing
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec

DATASET, EXP, IMG = "cifar10", "xAI", 5      # IMG: which of the 8 stability images
BASE = f"Results/Exp_{EXP}/{DATASET}/attention"
OUT = f"Results/docs/figures/{DATASET}_attention.png"

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})

uni = np.load(f"{BASE}/uniformity.npz", allow_pickle=True)
sta = np.load(f"{BASE}/stability.npz", allow_pickle=True)

agg = uni["last_condmap_16"].astype(np.float64)
best = int(sta["best_index"])
head = sta["best_maps"][IMG].astype(np.float64)
label = str(sta["labels"][IMG])
lay, hd = int(sta["layer"][best]), int(sta["head"][best])

# Both maps relative to their own mean, on ONE shared scale centred at 1.
# Without this, matplotlib would rescale the flat map to its own min/max and
# make it look as structured as the selective one.
agg_r, head_r = agg / agg.mean(), head / head.mean()
vmax = float(np.ceil(max(head_r.max(), agg_r.max())))
norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=vmax)

fig = plt.figure(figsize=(14, 4.8), layout="constrained")
gs = GridSpec(1, 3, width_ratios=[1, 1, 1.5], figure=fig)

img_axes = []
for k, (m, title, sub) in enumerate([
    (agg_r, "1 · aggregated over 16 layers x 8 heads",
     f"max/mean = {agg.max() / agg.mean():.2f}   ->  flat"),
    (head_r, f"2 · sharpest head (layer {lay}, head {hd}) - '{label}'",
     f"CV = {float(sta['slot_cv'][best]):.2f}   ->  speckle, r = -0.428 with detail"),
]):
    ax = fig.add_subplot(gs[k])
    im = ax.imshow(m, cmap="RdBu_r", norm=norm, interpolation="nearest")
    ax.set_title(f"{title}\n{sub}", pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    img_axes.append(ax)

fig.colorbar(im, ax=img_axes, orientation="horizontal", aspect=45,
             fraction=0.07, pad=0.02, label="attention change / its own mean")

x, y = sta["detail_uncond"], sta["detail_cond"]
r = np.corrcoef(x, y)[0, 1]
ax = fig.add_subplot(gs[2])
lim = [min(x.min(), y.min()) - 0.05, max(x.max(), y.max()) + 0.05]
ax.plot(lim, lim, color="0.6", lw=1, ls="--", zorder=1)
ax.scatter(x, y, s=26, color="#3b6ea5", alpha=0.75, edgecolor="none", zorder=2)
ax.scatter(x[best], y[best], s=90, facecolor="none", edgecolor="crimson",
           lw=2, zorder=3, label="the sharpest head")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_box_aspect(1)
ax.set_xlabel("detail-following  WITHOUT conditioning")
ax.set_ylabel("detail-following  WITH conditioning")
ax.set_title(f"3 · same behaviour with and without the class\n"
             f"r = {r:.2f} over all 128 (layer, head) pairs")
ax.legend(loc="upper left", frameon=False, fontsize=10)

fig.savefig(OUT, dpi=200, facecolor="white")
print(f"written {OUT}")

for end in ("first", "last"):
    for side in (16, 8, 4, 2):
        m = uni[f"{end}_condmap_{side}"]
        print(f"  {end:5s} {side:2d}x{side:<2d}  max/mean = {m.max() / m.mean():.3f}")

print(f"  sharpest head  CV = {float(sta['slot_cv'][best]):.3f}  layer {lay} head {hd}")
print(f"  detail  r = {r:.3f}     luminance  r = "
      f"{np.corrcoef(sta['lumin_uncond'], sta['lumin_cond'])[0, 1]:.3f}")