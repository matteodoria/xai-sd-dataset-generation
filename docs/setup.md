# xAI-Project — Generazione di dataset sintetici

Pipeline per generare dataset sintetici tramite Stable Diffusion con embedding
di classe fine-tunati, per esperimenti di explainability (xAI) su dataset
naturali (CIFAR-10/100) e biomedici (MedMNIST: blood, derma, path, retina).

Questo repository contiene tutto il necessario per partire: codice, ambiente,
pesi pre-addestrati e script di setup. Segui i passi **nell'ordine indicato**.

---

## ⚠️ Prima di iniziare — leggi questi due avvisi

1. **NON scaricare il repository come archivio ZIP.** Lo ZIP di GitHub **non**
   include i file gestiti con Git LFS (pesi ed embedding): otterresti dei
   segnaposto di testo da poche centinaia di byte al posto dei file veri, e il
   codice fallirebbe. **Usa esclusivamente `git clone`** come descritto sotto.

2. **Installa Git LFS PRIMA di clonare.** Se cloni senza Git LFS attivo,
   ottieni comunque i segnaposto. L'ordine corretto è: installa git-lfs →
   `git lfs install` → poi `git clone`.

---

## 0. Prerequisiti

Si assume di lavorare **sul cluster** (stesso ambiente su cui è stato preparato
il progetto), con a disposizione:

- `conda` / `miniforge` installato nella propria home
- il sistema di `module` (per il toolkit CUDA)
- `git`

Verifica rapida:

```bash
conda --version
git --version
module avail 2>&1 | grep -i "cuda/11.8"   # deve comparire cuda/11.8.0
```

---

## 1. Configura l'accesso SSH a GitHub (repo privato)

Il repository è **privato**: per clonarlo devi (a) essere stato aggiunto come
collaboratore (chiedi a Matteo), e (b) avere una chiave SSH collegata al tuo
account GitHub. Se hai già una chiave SSH funzionante, salta al punto 2.

**Verifica se sei già a posto:**

```bash
ssh -T git@github.com
```

Se risponde `Hi <username>!` sei pronto, vai al punto 2. Se dà
`Permission denied`, configura la chiave:

```bash
# 1. Genera una chiave (premi Invio ad ogni domanda per i default)
ssh-keygen -t ed25519 -C "tua.email@example.com"

# 2. Mostra la chiave PUBBLICA e copiala tutta
cat ~/.ssh/id_ed25519.pub
```

Poi su **github.com** → Settings → SSH and GPG keys → **New SSH key**,
incolla la chiave pubblica e salva. Infine ri-verifica:

```bash
ssh -T git@github.com    # ora deve rispondere "Hi <username>!"
```

> Nota: incolla solo il file `.pub` (chiave **pubblica**). La chiave privata
> (`id_ed25519`, senza `.pub`) non va mai condivisa.

---

## 2. Installa Git LFS e clona il repository

```bash
# Installa git-lfs nell'ambiente conda 'base' (se non presente sul sistema)
conda install -n base -c conda-forge git-lfs
conda activate base
git lfs install

# Clona il repo (via SSH). Gli embedding (~920 MB) e i dataset MedMNIST
# (~264 MB) vengono scaricati automaticamente da Git LFS durante il clone.
cd ~/Desktop      # o dove preferisci tenere il progetto
git clone git@github.com:matteodoria/xAI-Project.git
cd xAI-Project
```

**Verifica che i pesi siano veri e non segnaposto:**

```bash
file Checkpoints/DDPM/Exp_xAI/cifar100/MyEmbedding/epoch41.hdf5
# Deve dire: "Hierarchical Data Format (version 5) data"
# Se dice "ASCII text", Git LFS non era attivo: reinstallalo e ri-clona.
```

---

## 3. Crea l'ambiente Python

```bash
conda env create -f environment.yml
conda activate sd_dataset
```

Questo crea l'ambiente `sd_dataset` con TensorFlow 2.13, lo stack CUDA 11.8
(via pip) e tutte le dipendenze, incluse le versioni esatte testate.

---

## 4. Configura CUDA (una volta sola)

Sul cluster, lo stack CUDA installato via pip ha bisogno di due aggiustamenti
(collegamento alla libreria driver `libcuda.so` e al compilatore `ptxas`).
Lo script li applica e li rende permanenti per l'ambiente `sd_dataset`.

```bash
# assicurati che l'ambiente sia attivo
conda activate sd_dataset
bash setup_cuda.sh
```

Al termine deve stampare `Setup CUDA completato con successo`. Le correzioni
si attiveranno automaticamente ad ogni `conda activate sd_dataset`.

---

## 5. Scarica i pesi del modello di diffusione (DiffusionFt)

Gli embedding sono già arrivati col clone (LFS). Mancano solo i pesi del
modello di diffusione fine-tunato, che sono grandi (~3.4 GB ciascuno) e vanno
scaricati a parte dai link forniti.

```bash
# Scarica solo i dataset che ti servono (consigliato):
bash download_weights.sh cifar10 bloodmnist

# Oppure tutti e sei (richiede ~20 GB liberi):
bash download_weights.sh
```

> **Spazio disco**: ogni peso pesa ~3.4 GB. Controlla lo spazio con
> `df -h ~` prima di scaricarne molti.

---

## 6. Genera un dataset sintetico

```bash
conda activate sd_dataset
python -m scripts.generate_dataset --dataset cifar10 --img_total 400 \
    --enc_epoch 25 --dif_epoch 5 --inf_steps 20 --ugs 1.0 --exp xAI
```

Dataset disponibili: `cifar10`, `cifar100`, `bloodmnist`, `dermamnist`,
`pathmnist`, `retinamnist`. Le immagini generate finiscono in
`Data/Synthetic/Exp_xAI/<dataset>/`.

---

## Struttura del repository

```
xAI-Project/
├── environment.yml            # ambiente conda (sd_dataset)
├── setup_cuda.sh              # fix CUDA (libcuda.so + ptxas)
├── download_weights.sh        # scarica i pesi DiffusionFt
├── generate_dataset.py        # script principale di generazione
├── classifier_training.py     # training del classificatore
├── xai_*.py                   # analisi di explainability
├── XAI/                       # moduli per l'analisi (vedi README_xAI.md)
├── Models/                    # definizioni dei modelli
├── Data/
│   ├── MNIST/                 # dataset MedMNIST (.npz, via LFS)
│   └── target_datasets/       # loader dei dataset (codice)
└── Checkpoints/DDPM/Exp_xAI/<dataset>/
    ├── MyEmbedding/*.hdf5      # embedding pre-addestrati (via LFS)
    └── DiffusionFt/link.txt    # link per scaricare i pesi diffusione
```

La parte di explainability — cosa abbiamo chiesto al modello, i risultati e come
riprodurli — è documentata in [results.md](results.md).
I dataset CIFAR-10/100 vengono scaricati automaticamente da Keras al primo
uso; non serve prepararli.

---

## Troubleshooting

**`file signature not found` / `ASCII text` al posto di un `.hdf5` o `.npz`**
Git LFS non era attivo al momento del clone: il file è un segnaposto di testo.
Installa git-lfs (`git lfs install`), poi `git lfs pull` per scaricare i file
veri, oppure ri-clona da capo con LFS attivo.

**`Could not load library libcudnn_...: libcuda.so: cannot open shared object file`**
Manca la configurazione CUDA. Assicurati di aver eseguito `bash setup_cuda.sh`
con l'ambiente `sd_dataset` attivo, e di aver riattivato l'ambiente dopo
(`conda deactivate && conda activate sd_dataset`).

**`Couldn't invoke ptxas` / `Relying on driver to perform ptx compilation`**
`ptxas` non è nel PATH. Anche questo lo risolve `setup_cuda.sh`. Se persiste,
verifica che il modulo `cuda/11.8.0` sia disponibile (`module avail | grep cuda`).

**`ran out of memory trying to allocate ...GiB`**
La GPU è al limite di memoria. Non è un errore fatale: la generazione prosegue
con un algoritmo più parco. Se invece crasha davvero, prova a ridurre il
batch/numero di immagini, o imposta `TF_FORCE_GPU_ALLOW_GROWTH=true` prima del
comando.

**Il download di un DiffusionFt scarica un file piccolo / non valido**
Il link SharePoint potrebbe essere scaduto. Verifica i link nei file
`Checkpoints/DDPM/Exp_xAI/<dataset>/DiffusionFt/link.txt` e chiedi a Matteo
link aggiornati se necessario.
