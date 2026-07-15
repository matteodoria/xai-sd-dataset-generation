#!/usr/bin/env bash
#
# setup_cuda.sh
# -----------------------------------------------------------------------------
# Configura l'ambiente CUDA per far girare generate_dataset.py sul cluster.
#
# Risolve tre problemi noti su questo cluster (RHEL 9 + vGPU + stack CUDA pip):
#
#   1. libcudnn cerca "libcuda.so" (senza suffisso versione) ma sul sistema
#      esiste solo "libcuda.so.1". Creiamo un symlink pulito verso la libreria
#      driver di sistema in /lib64.
#
#   2. TensorFlow/XLA cerca "ptxas" per compilare i kernel PTX, ma non e'
#      incluso nello stack pip. Lo prendiamo dal modulo Spack cuda/11.8.0.
#
#   3. Le librerie CUDA (cudnn, cublas, cudart, ...) sono installate via pip
#      dentro site-packages/nvidia/*/lib, ma quelle cartelle non sono nel
#      percorso di ricerca del linker: senza, TensorFlow non vede la GPU
#      ("Could not find cuda drivers on your machine, GPU will not be used").
#
# Le tre correzioni vengono rese permanenti tramite l'activate hook
# dell'ambiente conda: si attivano da sole ad ogni "conda activate sd_dataset".
#
# PREREQUISITO: l'ambiente conda 'sd_dataset' deve gia' esistere ed essere
#               ATTIVO quando lanci questo script:
#                   conda activate sd_dataset
#                   bash setup_cuda.sh
#
# Lo script e' idempotente: puoi rilanciarlo senza fare danni.
# -----------------------------------------------------------------------------

set -euo pipefail

# --- Parametri configurabili -------------------------------------------------
CUDA_MODULE="cuda/11.8.0"      # modulo Spack da cui prendere ptxas
CUDALIBS_DIR="$HOME/cudalibs"  # dove mettiamo i symlink
SYSTEM_LIBCUDA="/lib64/libcuda.so.1"   # driver di sistema

# --- Utility di stampa -------------------------------------------------------
info()  { echo "[INFO]  $*"; }
ok()    { echo "[ OK ]  $*"; }
warn()  { echo "[WARN]  $*" >&2; }
fail()  { echo "[FAIL]  $*" >&2; exit 1; }

echo "==============================================="
echo "  Setup ambiente CUDA per generate_dataset.py"
echo "==============================================="

# --- 0. Verifica che l'env conda giusto sia attivo ---------------------------
if [ -z "${CONDA_PREFIX:-}" ]; then
    fail "Nessun ambiente conda attivo. Esegui prima: conda activate sd_dataset"
fi
if [ "$(basename "$CONDA_PREFIX")" != "sd_dataset" ]; then
    warn "L'env attivo e' '$(basename "$CONDA_PREFIX")', non 'sd_dataset'."
    warn "Se e' voluto ignora; altrimenti: conda activate sd_dataset"
fi
info "Env conda attivo: $CONDA_PREFIX"

# --- 1. Fix libcuda.so -------------------------------------------------------
echo ""
info "[1/4] Configuro libcuda.so ..."

if [ ! -e "$SYSTEM_LIBCUDA" ]; then
    fail "Non trovo $SYSTEM_LIBCUDA. Il driver NVIDIA e' installato su questo nodo?
       Controlla con: nvidia-smi   e   ldconfig -p | grep libcuda"
fi
REAL_LIBCUDA="$(readlink -f "$SYSTEM_LIBCUDA")"
if [ ! -e "$REAL_LIBCUDA" ]; then
    fail "$SYSTEM_LIBCUDA e' un symlink rotto (punta a $REAL_LIBCUDA, inesistente).
       La catena del driver e' danneggiata: contatta chi gestisce il cluster."
fi
info "libcuda di sistema: $SYSTEM_LIBCUDA -> $REAL_LIBCUDA"

mkdir -p "$CUDALIBS_DIR"
ln -sf "$SYSTEM_LIBCUDA" "$CUDALIBS_DIR/libcuda.so"
ok "Symlink: $CUDALIBS_DIR/libcuda.so -> $SYSTEM_LIBCUDA"

# --- 2. Fix ptxas ------------------------------------------------------------
echo ""
info "[2/4] Configuro ptxas (dal modulo $CUDA_MODULE) ..."

if ! type module >/dev/null 2>&1; then
    for mod_init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
        # shellcheck disable=SC1090
        [ -f "$mod_init" ] && source "$mod_init" && break
    done
fi
if ! type module >/dev/null 2>&1; then
    fail "Comando 'module' non disponibile. Caricalo manualmente, poi rilancia:
       module load $CUDA_MODULE"
fi

if ! module load "$CUDA_MODULE" 2>/dev/null; then
    fail "Impossibile caricare il modulo $CUDA_MODULE.
       Verifica il nome con: module avail 2>&1 | grep -i cuda
       poi modifica CUDA_MODULE in cima a questo script."
fi

PTXAS_PATH="$(command -v ptxas || true)"
module unload "$CUDA_MODULE" 2>/dev/null || true

if [ -z "$PTXAS_PATH" ]; then
    fail "Modulo $CUDA_MODULE caricato ma 'ptxas' non trovato nel PATH."
fi
info "ptxas trovato in: $PTXAS_PATH"

if ! "$PTXAS_PATH" --version >/dev/null 2>&1; then
    fail "ptxas non e' autocontenuto: non gira senza il modulo caricato.
       Alternativa: pip install nvidia-cuda-nvcc-cu11"
fi

ln -sf "$PTXAS_PATH" "$CUDALIBS_DIR/ptxas"
ok "Symlink: $CUDALIBS_DIR/ptxas -> $PTXAS_PATH"

# --- 3. Librerie CUDA installate via pip -------------------------------------
echo ""
info "[3/4] Individuo le librerie CUDA installate via pip ..."

NVIDIA_LIBS="$(python -c "
import os, site, glob
paths = []
for sp in site.getsitepackages():
    paths += glob.glob(os.path.join(sp, 'nvidia', '*', 'lib'))
print(':'.join(sorted(paths)))
" 2>/dev/null || true)"

if [ -z "$NVIDIA_LIBS" ]; then
    fail "Nessuna libreria CUDA pip trovata in site-packages/nvidia/*/lib.
       L'ambiente e' stato creato da environment.yml? Verifica con:
       pip list | grep nvidia"
fi
info "Trovate $(echo "$NVIDIA_LIBS" | tr ':' '\n' | wc -l) cartelle di librerie CUDA (pip)"

# --- 4. Persistenza tramite activate hook ------------------------------------
echo ""
info "[4/4] Rendo le correzioni permanenti nell'activate hook ..."

HOOK_DIR="$CONDA_PREFIX/etc/conda/activate.d"
HOOK_FILE="$HOOK_DIR/zz_cudalib_fix.sh"
mkdir -p "$HOOK_DIR"

# Scriviamo l'hook da zero (idempotente: niente righe duplicate).
# I path delle librerie pip vengono ricalcolati ad ogni attivazione, cosi'
# l'hook resta valido anche se l'ambiente viene spostato o aggiornato.
cat > "$HOOK_FILE" <<'EOF'
# Correzioni CUDA per l'ambiente sd_dataset (generato da setup_cuda.sh).
# Non modificare a mano: rilancia setup_cuda.sh per rigenerarlo.

# 1+2. libcuda.so e ptxas (symlink in ~/cudalibs)
export LD_LIBRARY_PATH=$HOME/cudalibs:$LD_LIBRARY_PATH
export PATH=$HOME/cudalibs:$PATH

# 3. Librerie CUDA installate via pip (nvidia-*-cu11)
_NVIDIA_LIBS="$(python -c "
import os, site, glob
paths = []
for sp in site.getsitepackages():
    paths += glob.glob(os.path.join(sp, 'nvidia', '*', 'lib'))
print(':'.join(sorted(paths)))
" 2>/dev/null)"
if [ -n "$_NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH="$_NVIDIA_LIBS:$LD_LIBRARY_PATH"
fi
unset _NVIDIA_LIBS
EOF
ok "Hook scritto: $HOOK_FILE"

# Applichiamo le stesse variabili alla shell corrente per il self-test
export LD_LIBRARY_PATH="$NVIDIA_LIBS:$CUDALIBS_DIR:${LD_LIBRARY_PATH:-}"
export PATH="$CUDALIBS_DIR:$PATH"

# --- Self-test ---------------------------------------------------------------
echo ""
info "Eseguo self-test ..."

# a) libcuda.so caricabile via dlopen
if python -c "import ctypes; ctypes.CDLL('libcuda.so')" 2>/dev/null; then
    ok "libcuda.so si carica correttamente"
else
    fail "libcuda.so NON si carica."
fi

# b) ptxas invocabile dal PATH
if ptxas --version >/dev/null 2>&1; then
    PTXAS_VER="$(ptxas --version | grep -oi 'release [0-9.]*' | head -1)"
    ok "ptxas raggiungibile dal PATH ($PTXAS_VER)"
else
    fail "ptxas NON raggiungibile dal PATH."
fi

# c) IL TEST CHE CONTA: TensorFlow vede la GPU?
info "Verifico che TensorFlow rilevi la GPU (puo' richiedere ~30s) ..."
GPU_COUNT="$(python -c "
import tensorflow as tf
print(len(tf.config.list_physical_devices('GPU')))
" 2>/dev/null | tail -1)"

if [ "${GPU_COUNT:-0}" -ge 1 ]; then
    ok "TensorFlow rileva $GPU_COUNT GPU"
else
    fail "TensorFlow NON rileva alcuna GPU.
       Controlla che 'nvidia-smi' mostri una GPU su questo nodo.
       Se nvidia-smi funziona ma TF no, incolla l'output completo di:
         python -c 'import tensorflow as tf; tf.config.list_physical_devices()'"
fi

echo ""
echo "==============================================="
ok "Setup CUDA completato con successo."
echo ""
echo "  Le correzioni sono permanenti: si attivano"
echo "  automaticamente ad ogni 'conda activate sd_dataset'."
echo ""
echo "  Prossimo passo: scarica i pesi con download_weights.sh"
echo "==============================================="
