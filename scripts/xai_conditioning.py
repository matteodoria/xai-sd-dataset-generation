"""Analyse the class conditioning space of a fine-tuned ClassEncoder.

Each run re-validates the closed form it relies on before computing anything.
Then reports the geometry of the space and stores for it for the notebooks.

Usage:
    python -m scripts.xai_conditioning --dataset cifar10 --exp xAI --enc_epoch 31
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import numpy as np

from XAI.Generator.Conditioning import conditioning, common


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", type=str, metavar="NAME",
                        choices=["cifar10", "cifar100", "pathmnist", "dermamnist",
                                 "bloodmnist", "retinamnist"],
                        help="Dataset whose ClassEncoder is analysed.")
    parser.add_argument("--exp", required=True, type=str,
                        help="Experiment identifier (e.g., 'xAI').")
    parser.add_argument("--enc_epoch", required=True, type=int,
                        help="Epoch of the encoder checkpoint to load.")
    return parser.parse_args()

def main():
    args = parse_args()
    labels = common.get_labels(args.dataset)
    num_classes = len(labels)

    encoder, path = conditioning.build_encoder(
        args.dataset,
        args.exp,
        args.enc_epoch,
        num_classes
    )

    E, u = conditioning.closed_form(encoder)

    print(f"\n------------ CONDITIONING SPACE ------------"
          f"\n- dataset: {args.dataset} ({num_classes} classes)"
          f"\n- checkpoint: {path}")

    ## 1. Gate: the closed form must reproduce the model
    check = conditioning.verify_linearity(
        encoder, E, u, num_classes
    )

    print(f"\n[gate] context scale          : {check['scale']:.4f}")
    print(f"[gate] one-hot deviation        : {check['onehot_abs']:.3e} "
          f"({check['onehot_rel']:.2e} relative)")
    print(f"[gate] unconditonal deviation   : {check['uncond_abs']:.3e}")
    print(f"[gate] bias {check['bias']:+.3e} against noise std {check['noise_std']:.3e}")

    if not check["passed"]:
        raise SystemExit("\nFAILED: the ClassEncoder is not the affine map this "
                         "analysis assumes. Stop and re-check the architecture.")
    print("[gate] PASSED")

    ## 2. Geometry of the space
    geo = conditioning.geometry(E, u)
    print(f"\n||u|| (shared context) = {geo['norm_u']:.3f}\n")
    print(f"{'class':<12} {'||E[c]||':>9} {'ratio':>8} {'cos(E,u)':>10}")
    for c, name in enumerate(labels):
        print(f"{name:<12} {geo['norm_E'][c]:>9.3f} {geo['ratio'][c]:>8.3f} "
              f"{geo['cos_Eu'][c]:>10.3f}")

    off = ~np.eye(num_classes, dtype=bool)
    print(f"\nratio ||E[c]||/||u||   : mean {geo['ratio'].mean():.3f}, "
          f"min {geo['ratio'].min():.3f}, max {geo['ratio'].max():.3f}")
    print(f"cosine between classes : mean {geo['cos_classes'][off].mean():+.3f}, "
          f"min {geo['cos_classes'][off].min():+.3f}, "
          f"max {geo['cos_classes'][off].max():+.3f}")

    ## 3. Remove the shared component and look at what tells the classes apart
    dec = conditioning.decompose(E)
    print(f"\nshared component : {dec['shared_energy']:.1%} of the energy")
    print(f"residuals        : {dec['residual_energy']:.1%} "
          f"(||R[c]||/||E[c]|| mean {dec['residual_ratio'].mean():.3f})")
    print(f"effective rank of residuals: {dec['effective_rank']:.2f} "
          f"(at most {num_classes - 1}, as residuals sum to zero)")
    print(f"mean cosine between residuals: {dec['cos_residuals'][off].mean():+.3f}")

    print("\nclosest class after centring:")
    cos_r = dec["cos_residuals"].copy()
    np.fill_diagonal(cos_r, -np.inf)
    for c in range(num_classes):
        j = int(np.argmax(cos_r[c]))
        print(f"  {labels[c]:<12} -> {labels[j]:<12} ({cos_r[c, j]:+.3f})")

    ## 4. Is that structure more than chance?
    extra = {}
    groups = common.SEMANTIC_GROUPS.get(args.dataset)
    if groups is not None:
        ids = common.group_ids(labels, groups)
        perm = conditioning.permutation_test(dec["cos_residuals"], ids)
        extra = {"perm_observed": perm["observed"], "perm_null": perm["null"],
                 "perm_p_value": perm["p_value"]}

        kind = "exact" if perm["exact"] else "Monte Carlo"
        print(f"\npermutation test on {len(groups)} semantic groups ({kind})")
        print(f"  within minus between cosine : {perm['observed']:+.3f}")
        print(f"  null over {perm['n_partitions']} draws: "
              f"mean {perm['null'].mean():+.3f}, max {perm['null'].max():+.3f}")
        print(f"  p-value                     : {perm['p_value']:.4f}")

        var = conditioning.hierarchical_variance(dec["residuals"], ids)
        cos = conditioning.group_cosines(dec["cos_residuals"], ids)
        extra.update({"between_energy": var["between_energy"],
                      "within_energy": var["within_energy"]})

        print(f"\nresidual energy split by the grouping")
        print(f"  between groups : {var['between_energy']:.1%} "
              f"(chance level {var['expected_random']:.1%})")
        print(f"  within groups  : {var['within_energy']:.1%}")
        print(f"  mean cosine    : {cos['within']:+.3f} within, "
              f"{cos['between']:+.3f} between")

        if len(groups) >= 10:
            sub = conditioning.subsample_effective_rank(E, ids)
            extra["subsample_effective_rank"] = sub["effective_rank"]
            print(f"\ncontrol: 10 classes from 10 distinct groups, "
                  f"{len(sub['effective_rank'])} draws")
            print(f"  effective rank : {sub['effective_rank'].mean():.2f} "
                  f"+/- {sub['effective_rank'].std():.2f} "
                  f"(range {sub['effective_rank'].min():.2f}"
                  f"-{sub['effective_rank'].max():.2f})")
            print(f"  normalised     : {sub['normalised'].mean():.1%}")

    ## 5. Does the space match how the real images actually look?
    visual = conditioning.decompose(common.class_means(args.dataset))
    mantel = conditioning.mantel_test(dec["cos_residuals"], visual["cos_residuals"])
    extra.update({"mantel_correlation": mantel["correlation"],
                  "mantel_p_value": mantel["p_value"],
                  "mantel_null": mantel["null"],
                  "cos_visual": visual["cos_residuals"]})

    print(f"\nMantel test against mean-image similarity of the real data")
    print(f"  correlation : {mantel['correlation']:+.3f}")
    print(f"  null        : mean {mantel['null'].mean():+.3f}, "
          f"max {mantel['null'].max():+.3f}")
    print(f"  p-value     : {mantel['p_value']:.4f}")

    ## 6. Persist for the notebooks
    out_dir = common.results_dir(args.exp, args.dataset, "conditioning")
    out_path = os.path.join(out_dir, f"epoch{args.enc_epoch}.npz")
    np.savez_compressed(out_path, labels=np.array(labels), checkpoint=path,
                        E=E, u=u, **geo, **dec, **extra)
    print(f"\nSaved: {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")

if __name__ == "__main__":
    main()
