# xAI-Project

Explainability analysis of the class-conditional Stable Diffusion generator
introduced in *Stable Diffusion Dataset Generation for Downstream Classification
Tasks* (Lomurno, D'Oria, Matteucci — ESANN 2024). The paper shows **that** the
class-conditional adaptation works; this project asks **why**, from both ends
of the pipeline:

- **the generator** — how the class condition shapes the image along
  denoising (notebooks 01-04);
- **the classifier** — how a ResNet20 trained on synthetic CIFAR-10 decides,
  compared with the same network trained on real images, explained with
  Grad-CAM, Integrated Gradients and LIME (notebooks 05-06).

Course project for *Interpretability and Explainability in Machine Learning*,
PhD, Politecnico di Milano.

## Documentation

| | |
|---|---|
| [Results/docs/results.md](Results/docs/results.md) | the questions, the results, how to reproduce them — **start here** |
| [Results/docs/setup.md](Results/docs/setup.md) | installation: repository access, Git LFS, conda, CUDA, weights |

## Layout

```
├── Data/                  dataset loaders and data     (from the paper's repository)
├── Models/                model definitions            (from the paper's repository)
│   └── Checkpoints/       model weights                (from the paper's repository)
├── Notebooks/             one per analysis, on stored artefacts, no GPU needed
├── scripts/
│   ├── generator_part/
│   │   ├── conditioning/  notebooks 01-03: the logic in lib/, one run_*.py per experiment
│   │   └── ugs_analysis/  notebook 04: generation of the guidance-scale sweep
│   ├── classifier_part/   notebooks 05-06, the classifier side
│   └── tools/             shared entry points and one-off diagnostics
└── Results/
    ├── docs/              documentation and final figures
    └── Exp_<exp>/         generated artefacts, git-ignored
```

Plus `environment.yml`, `setup_cuda.sh` and `download_weights.sh` in the root.
`Data/`, `Models/` and `Checkpoints/` keep the CamelCase of the paper's
repository, so the diff against it stays readable; `Results/` is capitalised to
sit beside them, everything else added here is lowercase.

## Running

Entry points import the project's packages and reach `Models/Checkpoints/` and `Data/`
by relative path, so they run **as modules, from the repository root**:

```bash
conda activate sd_dataset
python -m scripts.generator_part.conditioning.run_conditioning --dataset cifar10 --exp xAI --enc_epoch 31
```

`python scripts/generator_part/conditioning/run_conditioning.py` fails on the
imports. The full command list, with the hyper-parameters of every final run, is
in [Results/docs/results.md](Results/docs/results.md#how-to-reproduce).

## What the repository does not contain

Git-ignored, because large and reproducible:

|                                                         | how to get it back |
|---------------------------------------------------------|---|
| `Models/Checkpoints/DDPM/**/DiffusionFt/*.hdf5` (~3.4 GB each) | `./download_weights.sh` — links are in the tracked `link.txt` |
| `Models/Checkpoints/Classifiers/<classifier>/<dataset>/**/*.h5` (~50 MB each) | `python -m scripts.tools.run_classifier_training` |
| `Models/Checkpoints/Classifiers/resnet20/resnet20_{real_matched,synthetic}.h5` (3.5 MB each) | shipped with the data archive: retraining gives different models from those notebooks 05-06 were computed from. The recipe is `python -m scripts.classifier_part.run_train_real_synth`, which needs `--overwrite` to replace them and the 40,000 synthetic images in `Data/Synthetic/Exp_xAI/cifar10/40.0kEnc31Dif10Is20Ugs1.0/` |
| `Data/Synthetic/`                                       | `python -m scripts.tools.run_generate_dataset` |
| `Results/Exp_*/`                                        | the `run_*.py` entry points, and notebooks 05-06 for `cifar10/classifier/` — contents documented in [Results/docs/artefacts.md](Results/docs/artefacts.md) |

The pre-trained class embeddings and the MedMNIST datasets **are** in the
repository, through Git LFS. Cloning without `git-lfs` installed leaves text
pointers in place of the files, and the failure that follows does not look like
a missing-file error — see [Results/docs/setup.md](Results/docs/setup.md).