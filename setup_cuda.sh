#!/usr/bin/env bash
#
# setup_cuda.sh
# -----------------------------------------------------------------------------
# Configura l'ambiente CUDA per far girare generate_dataset.py sul cluster.
#
# Risolve due problemi noti su questo cluster (RHEL 9 + vGPU + stack CUDA pip):
#   1. libcudnn cerca "libcuda.so" (senza suffisso versione) ma sul sistema
#      esiste solo "libcuda.so.1". Creiamo un symlink pulito verso la libreria
#      di sistema integra in /lib64.
#   2. TensorFlow/XLA cerca "ptxas" per compilare i kernel PTX, ma non e'
#      incluso nello stack pip. Lo prendiamo dal modulo Spack cuda/11.8.0.
#
# Entrambe le correzioni vengono rese permanenti tramite l'activate hook
# dell'ambiente conda, cosi' si attivano da sole ad ogni "conda activate".
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
SYSTEM_LIBCUDA="/lib64/libcuda.so.1"   # driver di sistema (integro su questo cluster)

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
info "[1/3] Configuro libcuda.so ..."

# Verifica che la libreria di sistema esista e risolva a un file reale
if [ ! -e "$SYSTEM_LIBCUDA" ]; then
    fail "Non trovo $SYSTEM_LIBCUDA. Il driver NVIDIA e' installato su questo nodo?
       Controlla con: ldconfig -p | grep libcuda"
fi
REAL_LIBCUDA="$(readlink -f "$SYSTEM_LIBCUDA")"
if [ ! -e "$REAL_LIBCUDA" ]; then
    fail "$SYSTEM_LIBCUDA e' un symlink rotto (punta a $REAL_LIBCUDA, inesistente).
       La catena del driver e' danneggiata: contatta chi gestisce il cluster."
fi
info "libcuda di sistema: $SYSTEM_LIBCUDA -> $REAL_LIBCUDA"

mkdir -p "$CUDALIBS_DIR"
ln -sf "$SYSTEM_LIBCUDA" "$CUDALIBS_DIR/libcuda.so"
ok "Creato symlink: $CUDALIBS_DIR/libcuda.so -> $SYSTEM_LIBCUDA"

# --- 2. Fix ptxas ------------------------------------------------------------
echo ""
info "[2/3] Configuro ptxas (dal modulo $CUDA_MODULE) ..."

# 'module' e' una shell function definita dagli script di init del cluster.
# In uno script non interattivo potrebbe non essere caricata: proviamo a
# sourcare il profilo dei moduli se 'module' non e' disponibile.
if ! type module >/dev/null 2>&1; then
    for mod_init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash \
                    /share/apps/spack/latest_x86_64/*/lmod/*/init/bash; do
        # shellcheck disable=SC1090
        [ -f "$mod_init" ] && source "$mod_init" && break
    done
fi
if ! type module >/dev/null 2>&1; then
    fail "Comando 'module' non disponibile. Caricalo manualmente, poi rilancia:
       module load $CUDA_MODULE"
fi

# Carichiamo il modulo SOLO per scoprire dove sta ptxas, poi lo scarichiamo
# per non far intromettere le sue librerie nell'ambiente pip.
if ! module load "$CUDA_MODULE" 2>/dev/null; then
    fail "Impossibile caricare il modulo $CUDA_MODULE.
       Verifica il nome con: module avail 2>&1 | grep -i cuda
       poi modifica CUDA_MODULE in cima a questo script."
fi

PTXAS_PATH="$(command -v ptxas || true)"
module unload "$CUDA_MODULE" 2>/dev/null || true

if [ -z "$PTXAS_PATH" ]; then
    fail "Modulo $CUDA_MODULE caricato ma 'ptxas' non trovato nel PATH.
       Il modulo potrebbe non includere il compilatore."
fi
info "ptxas trovato in: $PTXAS_PATH"

# Verifica che ptxas giri ANCHE a modulo scaricato (dev'essere autocontenuto)
if ! "$PTXAS_PATH" --version >/dev/null 2>&1; then
    fail "ptxas non e' autocontenuto: non gira senza il modulo caricato.
       Serve un approccio diverso (es. pip install nvidia-cuda-nvcc-cu11)."
fi

ln -sf "$PTXAS_PATH" "$CUDALIBS_DIR/ptxas"
ok "Creato symlink: $CUDALIBS_DIR/ptxas -> $PTXAS_PATH"

# --- 3. Persistenza tramite activate hook ------------------------------------
echo ""
info "[3/3] Rendo le correzioni permanenti nell'activate hook ..."

HOOK_DIR="$CONDA_PREFIX/etc/conda/activate.d"
HOOK_FILE="$HOOK_DIR/zz_cudalib_fix.sh"
mkdir -p "$HOOK_DIR"

# Scriviamo l'hook da zero (idempotente: sovrascrive, niente righe doppie)
cat > "$HOOK_FILE" <<'EOF'
# Correzioni CUDA per sd_dataset (vedi setup_cuda.sh)
export LD_LIBRARY_PATH=$HOME/cudalibs:$LD_LIBRARY_PATH
export PATH=$HOME/cudalibs:$PATH
EOF
ok "Scritto hook: $HOOK_FILE"

# Applichiamo le variabili anche alla shell corrente per il self-test
export LD_LIBRARY_PATH="$CUDALIBS_DIR:${LD_LIBRARY_PATH:-}"
export PATH="$CUDALIBS_DIR:$PATH"

# --- 4. Self-test ------------------------------------------------------------
echo ""
info "Eseguo self-test ..."

# 4a. libcuda.so caricabile via dlopen?
if python -c "import ctypes; ctypes.CDLL('libcuda.so')" 2>/dev/null; then
    ok "libcuda.so si carica correttamente"
else
    fail "libcuda.so NON si carica. Il fix non ha funzionato come previsto."
fi

# 4b. ptxas invocabile dal PATH?
if ptxas --version >/dev/null 2>&1; then
    PTXAS_VER="$(ptxas --version | grep -oi 'release [0-9.]*' | head -1)"
    ok "ptxas raggiungibile dal PATH ($PTXAS_VER)"
else
    fail "ptxas NON raggiungibile dal PATH. Il fix non ha funzionato."
fi

echo ""
echo "==============================================="
ok "Setup CUDA completato con successo."
echo ""
echo "  Le correzioni sono ora permanenti: si attivano"
echo "  automaticamente ad ogni 'conda activate sd_dataset'."
echo ""
echo "  Prossimo passo: scarica i pesi con download_weights.sh"
echo "==============================================="
