# xAI-Project

Explainability analysis of the class-conditional Stable Diffusion generator
introduced in *Stable Diffusion Dataset Generation for Downstream Classification
Tasks* (Lomurno, D'Oria, Matteucci — ESANN 2024). The paper shows **that** the
class-conditional adaptation works; this project asks **why**.

Course project for *Interpretability and Explainability in Machine Learning*,
PhD, Politecnico di Milano.

## Documentation

| | |
|---|---|
| [docs/results.md](docs/results.md) | the questions, the results, how to reproduce them — **start here** |
| [docs/setup.md](docs/setup.md) | installation: repository access, Git LFS, conda, CUDA, weights |

## Layout

```
├── XAI/            analysis library — the logic lives here
├── scripts/        entry points, one per experiment
├── tools/          one-off diagnostics (UNet reconnaissance, GPU, linearity)
├── notebooks/      Results 1 and 3, on stored artefacts, no GPU needed
├── docs/           documentation and final figures
├── Models/         model definitions            (from the paper's repository)
├── Data/           dataset loaders and data     (from the paper's repository)
├── Models/Checkpoints/    model weights                (from the paper's repository)
└── XAI_Results/    generated artefacts, git-ignored
```

Plus `environment.yml`, `setup_cuda.sh` and `download_weights.sh` in the root.
Directories in CamelCase come from the paper's repository and are left as they
are, so the diff against it stays readable; everything added here is lowercase.

## Running

Entry points import the project's packages and reach `Models/Checkpoints/` and `Data/`
by relative path, so they run **as modules, from the repository root**:

```bash
conda activate sd_dataset
python -m scripts.xai_conditioning --dataset cifar10 --exp xAI --enc_epoch 31
```

`python scripts/xai_conditioning.py` fails on the imports. The full command list,
with the hyper-parameters of every final run, is in
[docs/results.md](docs/results.md#how-to-reproduce).

## What the repository does not contain

Git-ignored, because large and reproducible:

|                                                         | how to get it back |
|---------------------------------------------------------|---|
| `Models/Checkpoints/DDPM/**/DiffusionFt/*.hdf5` (~3.4 GB each) | `./download_weights.sh` — links are in the tracked `link.txt` |
| `Models/Checkpoints/Classifiers/**/*.h5` (~50 MB each)         | `python -m scripts.classifier_training` |
| `Data/Synthetic/`                                       | `python -m scripts.generate_dataset` |
| `XAI_Results/`                                          | the scripts/xai_* entry points — contents documented in [docs/artefacts.md](docs/artefacts.md) |

The pre-trained class embeddings and the MedMNIST datasets **are** in the
repository, through Git LFS. Cloning without `git-lfs` installed leaves text
pointers in place of the files, and the failure that follows does not look like
a missing-file error — see [docs/setup.md](docs/setup.md).