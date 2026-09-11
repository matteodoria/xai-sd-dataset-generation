# script/tools/peek_attention.py
import numpy as np

for name in ("uniformity", "stability"):
    path = f"XAI_Results/Exp_xAI/cifar10/attention/{name}.npz"
    print(f"\n===== {name}.npz =====")
    with np.load(path, allow_pickle=True) as z:
        for k in z.files:
            a = z[k]
            print(f"  {k:28s} shape={str(a.shape):20s} dtype={a.dtype}")
            if a.ndim == 0 or a.size <= 12:
                print(f"      -> {a}")
            elif a.dtype.kind in "fiu":
                print(f"      -> min={a.min():.4g} max={a.max():.4g} mean={a.mean():.4g}")