# Setup

How to get the project running: repository access, Python environment, CUDA and
pre-trained weights. Follow the steps **in the order given**. What the project
does, and how it is organised, is in the [README](../README.md).

---

## ⚠️ Before you start — two warnings

1. **Do not download the repository as a ZIP archive.** GitHub's ZIP does **not**
   include files managed by Git LFS (weights and embeddings): you would get
   text placeholders of a few hundred bytes in place of the real files, and the
   code would fail. **Use `git clone` only**, as described below.

2. **Install Git LFS BEFORE cloning.** Cloning without Git LFS active gives you
   the placeholders anyway. The right order is: install git-lfs →
   `git lfs install` → then `git clone`.

---

## 0. Prerequisites

These instructions assume you are working **on the cluster** (the same
environment the project was prepared on), with:

- `conda` / `miniforge` installed in your home directory
- the `module` system (for the CUDA toolkit)
- `git`

Quick check:

```bash
conda --version
git --version
module avail 2>&1 | grep -i "cuda/11.8"   # cuda/11.8.0 must appear
```

---

## 1. Set up SSH access to GitHub (the repository is private)

The repository is **private**: to clone it you need (a) to have been added as a
collaborator — ask Matteo — and (b) an SSH key linked to your GitHub account. If
you already have a working SSH key, skip to step 2.

**Check whether you are already set:**

```bash
ssh -T git@github.com
```

If it answers `Hi <username>!` you are ready, go to step 2. If it says
`Permission denied`, set up the key:

```bash
# 1. Generate a key (press Enter at every prompt for the defaults)
ssh-keygen -t ed25519 -C "your.email@example.com"

# 2. Print the PUBLIC key and copy all of it
cat ~/.ssh/id_ed25519.pub
```

Then on **github.com** → Settings → SSH and GPG keys → **New SSH key**, paste the
public key and save. Finally check again:

```bash
ssh -T git@github.com    # must now answer "Hi <username>!"
```

> Note: paste only the `.pub` file (the **public** key). The private key
> (`id_ed25519`, without `.pub`) must never be shared.

---

## 2. Install Git LFS and clone the repository

```bash
# Install git-lfs in the 'base' conda environment (if not already on the system)
conda install -n base -c conda-forge git-lfs
conda activate base
git lfs install

# Clone the repository (over SSH). The embeddings (~920 MB) and the MedMNIST
# datasets (~264 MB) are fetched automatically by Git LFS during the clone.
cd ~/Desktop      # or wherever you want to keep the project
git clone git@github.com:matteodoria/xAI-Project.git
cd xAI-Project
```

**Check that the weights are real files and not placeholders:**

```bash
file Checkpoints/DDPM/Exp_xAI/cifar100/MyEmbedding/epoch41.hdf5
# Must say: "Hierarchical Data Format (version 5) data"
# If it says "ASCII text", Git LFS was not active: reinstall it and clone again.
```

---

## 3. Create the Python environment

```bash
conda env create -f environment.yml
conda activate sd_dataset
```

This creates the `sd_dataset` environment with TensorFlow 2.13, the CUDA 11.8
stack (via pip) and every dependency, pinned to the versions that were tested.

---

## 4. Configure CUDA (once)

On the cluster, the pip-installed CUDA stack needs two adjustments: linking the
driver library `libcuda.so` and the `ptxas` compiler. The script applies them and
makes them permanent for the `sd_dataset` environment.

```bash
# make sure the environment is active
conda activate sd_dataset
bash setup_cuda.sh
```

It must end by printing `Setup CUDA completato con successo`. The fixes are then
applied automatically on every `conda activate sd_dataset`.

---

## 5. Download the diffusion model weights (DiffusionFt)

The embeddings arrived with the clone, through LFS. What is missing are the
weights of the fine-tuned diffusion model: they are large (~3.4 GB each) and are
downloaded separately from the links provided.

```bash
# Download only the datasets you need (recommended):
bash download_weights.sh cifar10 bloodmnist

# Or all six (needs ~20 GB free):
bash download_weights.sh
```

> **Disk space**: each weight file is ~3.4 GB. Check with `df -h ~` before
> downloading several of them.

---

## 6. Generate a synthetic dataset

```bash
conda activate sd_dataset
python -m scripts.generate_dataset --dataset cifar10 --img_total 400 \
    --enc_epoch 25 --dif_epoch 5 --inf_steps 20 --ugs 1.0 --exp xAI
```

Available datasets: `cifar10`, `cifar100`, `bloodmnist`, `dermamnist`,
`pathmnist`, `retinamnist`. The generated images end up in
`Data/Synthetic/Exp_xAI/<dataset>/`.

---

## Where to look next

CIFAR-10/100 are downloaded automatically by Keras on first use; there is
nothing to prepare for them.

The repository layout is described in the [README](../README.md); the
explainability work — the questions asked of the model, the results, and how to
reproduce them — is in [results.md](results.md).

---

## Troubleshooting

**`file signature not found` / `ASCII text` where a `.hdf5` or `.npz` should be**
Git LFS was not active when you cloned: the file is a text placeholder. Install
git-lfs (`git lfs install`), then `git lfs pull` to fetch the real files, or
clone again from scratch with LFS active.

**`Could not load library libcudnn_...: libcuda.so: cannot open shared object file`**
The CUDA configuration is missing. Make sure you ran `bash setup_cuda.sh` with
the `sd_dataset` environment active, and that you re-activated the environment
afterwards (`conda deactivate && conda activate sd_dataset`).

**`Couldn't invoke ptxas` / `Relying on driver to perform ptx compilation`**
`ptxas` is not on the PATH. `setup_cuda.sh` fixes this too. If it persists, check
that the `cuda/11.8.0` module is available (`module avail | grep cuda`).

**`ran out of memory trying to allocate ...GiB`**
The GPU is at its memory limit. This is not fatal: generation carries on with a
more frugal algorithm. If it really crashes, reduce the batch or the number of
images, or set `TF_FORCE_GPU_ALLOW_GROWTH=true` before the command.

**A DiffusionFt download produces a small or invalid file**
The SharePoint link may have expired. Check the links in
`Checkpoints/DDPM/Exp_xAI/<dataset>/DiffusionFt/link.txt` and ask Matteo for
updated ones if needed.