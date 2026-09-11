#!/usr/bin/env bash
#
# download_weights.sh
# -----------------------------------------------------------------------------
# Scarica i pesi del modello di diffusione fine-tunato (DiffusionFt) per i
# dataset richiesti, a partire dai link OneDrive/SharePoint contenuti nei
# file Models/Checkpoints/DDPM/Exp_xAI/<dataset>/DiffusionFt/link.txt
#
# NOTA: gli embedding (MyEmbedding) e i dataset MedMNIST (Data/MNIST) NON
#       vanno scaricati con questo script: arrivano automaticamente col
#       'git clone' perche' sono tracciati con Git LFS. Assicurati di aver
#       installato git-lfs PRIMA di clonare (vedi README).
#
# USO:
#   bash download_weights.sh                      # scarica TUTTI i dataset
#   bash download_weights.sh cifar10 bloodmnist   # solo quelli indicati
#
# ATTENZIONE SPAZIO: ogni peso DiffusionFt pesa ~3.4 GB.
#   Scaricarli tutti e sei richiede ~20 GB liberi.
# -----------------------------------------------------------------------------

set -uo pipefail

# --- Config ------------------------------------------------------------------
# Radice dei checkpoint, relativa alla posizione dello script (root del repo)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$SCRIPT_DIR/Models/Checkpoints/DDPM/Exp_xAI"

ALL_DATASETS=(cifar10 cifar100 dermamnist retinamnist bloodmnist pathmnist)

# --- Utility -----------------------------------------------------------------
info()  { echo "[INFO]  $*"; }
ok()    { echo "[ OK ]  $*"; }
warn()  { echo "[WARN]  $*" >&2; }
err()   { echo "[FAIL]  $*" >&2; }

# --- Selezione dataset -------------------------------------------------------
if [ "$#" -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=("${ALL_DATASETS[@]}")
    info "Nessun dataset specificato: verranno scaricati TUTTI (${#DATASETS[@]})."
    info "Servono ~20 GB liberi. Ctrl+C entro 5s per annullare..."
    sleep 5
fi

echo "==============================================="
echo "  Download pesi DiffusionFt"
echo "  Dataset richiesti: ${DATASETS[*]}"
echo "==============================================="

FAILED=()

for ds in "${DATASETS[@]}"; do
    echo ""
    echo "############### $ds ###############"

    DS_DIR="$EXP_DIR/$ds/DiffusionFt"
    LINK_FILE="$DS_DIR/link.txt"

    # Validazioni
    if [ ! -d "$DS_DIR" ]; then
        err "Cartella non trovata: $DS_DIR (dataset '$ds' inesistente?)"
        FAILED+=("$ds"); continue
    fi
    if [ ! -f "$LINK_FILE" ]; then
        err "link.txt mancante in $DS_DIR"
        FAILED+=("$ds"); continue
    fi

    # Costruisci l'URL di download diretto: prendi l'URL, togli tutto
    # dopo il primo '?' e appendi '?download=1'
    RAW="$(tr -d '[:space:]' < "$LINK_FILE")"
    URL="${RAW%%\?*}?download=1"

    # Ricava il nome file (con l'epoca giusta) dal redirect di SharePoint
    info "Interrogo il link per ricavare il nome file ..."
    FNAME="$(wget --spider --server-response "$URL" 2>&1 \
             | grep -i 'Location:' | grep -o 'epoch[0-9]*\.hdf5' | head -1)"

    if [ -z "$FNAME" ]; then
        err "Impossibile ricavare il nome file dal link per '$ds'."
        err "Il link potrebbe essere scaduto o richiedere autenticazione."
        FAILED+=("$ds"); continue
    fi

    DEST="$DS_DIR/$FNAME"

    # Se gia' presente e valido, salta
    if [ -f "$DEST" ] && file "$DEST" | grep -q "Hierarchical"; then
        ok "$ds: $FNAME gia' presente e valido, salto."
        continue
    fi

    info "Scarico $FNAME (~3.4 GB) ..."
    if ! wget --no-verbose --show-progress -O "$DEST" "$URL"; then
        err "Download fallito per '$ds'."
        rm -f "$DEST"
        FAILED+=("$ds"); continue
    fi

    # Verifica che sia un HDF5 vero (non una pagina HTML di errore)
    if file "$DEST" | grep -q "Hierarchical"; then
        SIZE="$(du -h "$DEST" | cut -f1)"
        ok "$ds: $FNAME scaricato e verificato ($SIZE)"
    else
        err "$ds: il file scaricato non e' un HDF5 valido (link scaduto?)."
        rm -f "$DEST"
        FAILED+=("$ds"); continue
    fi

    df -h "$SCRIPT_DIR" | tail -1
done

# --- Riepilogo ---------------------------------------------------------------
echo ""
echo "==============================================="
if [ "${#FAILED[@]}" -eq 0 ]; then
    ok "Tutti i download richiesti sono andati a buon fine."
else
    warn "Alcuni dataset non sono stati scaricati: ${FAILED[*]}"
    warn "Controlla i link (potrebbero essere scaduti) e riprova."
fi
echo "==============================================="
