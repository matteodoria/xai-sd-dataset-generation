import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow.keras as tfk
from Models.class_encoder import ClassEncoder

DATASET, EXP, EPOCH, NUM_CLASSES = "cifar10", "xAI", 31, 10
path = f"Models/Checkpoints/DDPM/Exp_{EXP}/{DATASET}/MyEmbedding/epoch{EPOCH}.hdf5"
assert os.path.isfile(path), f"Checkpoint non trovato: {path}"

enc = ClassEncoder(max_length=NUM_CLASSES, weight_path=path)
dense = [l for l in enc.layers if isinstance(l, tfk.layers.Dense)]
W1, b1 = dense[0].get_weights()
(W2,) = dense[1].get_weights()

x = np.eye(NUM_CLASSES, dtype=np.float32)

pred32 = enc.predict(x, verbose=0)                              # modello, float32
closed32 = ((x @ W1 + b1) @ W2).reshape(NUM_CLASSES, 100, 768)  # forma chiusa, float32

# Riferimento: stessa formula in float64, errore di arrotondamento trascurabile.
ref64 = ((x.astype(np.float64) @ W1.astype(np.float64) + b1.astype(np.float64))
         @ W2.astype(np.float64)).reshape(NUM_CLASSES, 100, 768)

scale = np.abs(ref64).max()
err_model = np.abs(pred32 - ref64).max()
err_closed = np.abs(closed32 - ref64).max()

print(f"\nscala (max |context|)            = {scale:.4f}")
print(f"errore modello TF    vs float64  = {err_model:.3e}   ({err_model / scale:.2e} rel.)")
print(f"errore forma chiusa  vs float64  = {err_closed:.3e}   ({err_closed / scale:.2e} rel.)")

# Errore teorico atteso accumulando 500 somme in float32.
eps = np.finfo(np.float32).eps
print(f"errore teorico ~sqrt(500)*eps*scala = {np.sqrt(500) * eps * scale:.3e}")

diff = (pred32 - closed32).astype(np.float64)
print(f"\ndifferenza modello - forma chiusa")
print(f"  max   = {np.abs(diff).max():.3e}")
print(f"  media = {diff.mean():+.3e}    <- un errore sistematico la spingerebbe lontano da 0")
print(f"  std   = {diff.std():.3e}")

corr = np.corrcoef(pred32.ravel(), closed32.ravel())[0, 1]
print(f"\ncorrelazione modello vs forma chiusa = {corr:.12f}")