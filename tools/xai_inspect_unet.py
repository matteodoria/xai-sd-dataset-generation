"""Map the cross-attention blocks of the diffusion UNet.

Reconnaissance for the attention axis: how many attention blocks there are,
at which spatial resolution they operate, how many heads and what context length they expect.
Written defensively, if keras_cv0.6.1 names things differently than assumed,
it reports the available attributes instead of raising.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from collections import Counter

from Models.stable_diffusion import MyStableDiffusion

DATASET, EXP, ENC_EPOCH, DIF_EPOCH = 'cifar10', 'xAI', 31, 10
NUM_CLASSES = 10
RES = 32

enc_path = f"Checkpoints/DDPM/Exp_{EXP}/{DATASET}/MyEmbedding/epoch{ENC_EPOCH}.hdf5"
dif_path = f"Checkpoints/DDPM/Exp_{EXP}/{DATASET}/DiffusionFt/epoch{DIF_EPOCH}.hdf5"

for path in [enc_path, dif_path]:
    assert os.path.isfile(path), f"File {path} does not exist"

ddpm = MyStableDiffusion(res=RES, num_classes=NUM_CLASSES,
                         original_diffusion = True,
                         original_text_encoder = False,
                         enc_weight_path = enc_path,
                         diff_weight_path = dif_path
                         )

unet = ddpm.diffusion_model

print(f"\n------------ UNET ------------")
print(f"requested resolution : {ddpm.input_res}")
print(f"model resolution     : {ddpm.img_height}x{ddpm.img_width}")
print(f"latent resolution    : {ddpm.img_height // 8}x{ddpm.img_width // 8}")
print(f"top-level layers     : {len(unet.layers)}")

print("\nlayer types:")
for name, count in Counter(type(l).__name__ for l in unet.layers).most_common():
    print(f"  {name:<24} {count}")

def attributes(obj):
    """Public attribute names, to see what the API actually offers."""
    return [k for k in vars(obj) if not k.startswith("_")]

print(f"\n{'spatial transformer':<26} {'output shape':>20} {'heads':>6} "
      f"{'head dim':>9} {'context dim':>12}")

for layer in unet.layers:
    if type(layer).__name__ != "SpatialTransformer":
        continue

    block = getattr(layer, "transformer_block", None)
    if block is None:
        print(f"{layer.name}: no 'transformer_block'; has {attributes(layer)}")
        continue

    attn = getattr(block, "attn2", None)
    if attn is None:
        print(f"{layer.name}: no 'attn2'; block has {attributes(block)}")
        continue

    heads = getattr(attn, "num_heads", "?")
    head_dim = getattr(attn, "head_size", "?")
    to_k = getattr(attn, "to_k", None)
    context_dim = to_k.kernel.shape[0] if to_k is not None else "?"

    print(f"{layer.name:<26} {str(layer.output_shape):>20} {heads:>6} "
          f"{head_dim:>9} {context_dim:>12}")
    if to_k is None:
        print(f"    attn2 has {attributes(attn)}")
