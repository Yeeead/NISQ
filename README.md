# QINR Cross-Resolution Backdoor Experiments

This project trains and evaluates cross-resolution backdoor attacks on MNIST.
For QINR, the victim classifier and coordinate-continuous trigger generator are
trained together with alternating victim/generator updates, then evaluated for
base-resolution attack strength, defense behavior, and resolution transfer.

## Structure

- `models/classifiers/resnet.py`: local ResNet-style MNIST victim model.
- `models/quantum/`: QINR generators built from quantum data re-uploading.
- `models/generators/`: input-aware and classical INR generator builders.
- `training/poison.py`: coordinate grid creation, QINR delta generation, additive injection, and clamp.
- `methods/triggers.py`: BadNets, Blended, and WaNet trigger generation.
- `methods/poisoning.py`: poison-mask sampling and mixed-batch poisoning.
- `training/train_clean.py`: optional clean victim training.
- `training/train_qinr.py`: alternating QINR generator and poisoned victim training.
- `training/train_baselines.py`: shared training loop for classic backdoor baselines.
- `evaluation/backdoor.py`: unified clean accuracy, ASR, perturbation stats, and class overview visualization.
- `evaluation/cross_resolution.py`: QINR-only scenario evaluation plus cross-method resolution-transfer comparison.
- `utils/visualization.py`: class-organized clean/poisoned/diff eval overview images.
- `configs/qinr_mnist.yaml`: default experiment config.
- `configs/qinr_debug.yaml`: small smoke-test config.

## QINR

Coordinates are generated in `[-1, 1]` by default. For each qubit `i`, a
frequency `b_i` is sampled from the configured distribution, and the encoding
layer applies `RX(b_i * x)` and `RY(b_i * y)`.
Set `model.qinr_freq_mode: constant` or `Fixmode` to use the fixed
`model.qinr_baseline_freq` coefficient on every qubit as a no-random-frequency
variant. `Randommode` is accepted as an explicit alias for the default random
per-qubit frequency mode.
Set `model.qinr_freq_mode: gaussian_reparam` to train one Gaussian frequency
distribution per data-encoding layer. The generator resamples one shared
coefficient for each layer on every forward pass with the reparameterization
trick; `model.qinr_freq_std_init` controls the initial standard deviation.

Each parameter layer applies `RZ -> RY -> RZ` on every qubit followed by a ring
CNOT entanglement layer. The first qubit's finite-shot Pauli-Z measurement is
used directly as the per-coordinate perturbation value.

## Run

Install dependencies first:

```bash
pip install -r requirements.txt
```

Run the combined experiment:

```bash
python3 exp_run.py --config configs/qinr_mnist.yaml
```

`exp_run.py` runs two groups in order:

1. the original cross-resolution backdoor attack experiment
2. the QINR finite-shot and frequency-sampling ablation

Experiment flow:

1. `scripts/run_all_backdoor_methods.py` trains the selected backdoor methods
   from a copied config. The QINR method uses `Randommode` frequencies sampled
   at initialization and kept fixed.
2. Each trained method is evaluated for clean accuracy and ASR.
3. If enabled, cross-resolution protocols evaluate transfer across source
   resolutions, interpolation modes, antialias settings, and preprocessing
   order.
4. If enabled, base-resolution mitigation evaluates STRIP and Neural Cleanse.
5. `scripts/run_qinr_ablation.py` then runs the QINR ablation group.

The ablation group trains four QINR backdoor variants with the same base
config:

- `finite_shots_freq_sample`
- `finite_shots_no_sample`
- `infinite_shots_freq_sample`
- `infinite_shots_no_sample`

`finite_shots` means `model.qinr_shots` is taken from the config, so the
generator includes finite-shot measurement noise during perturbation
generation.
`infinite_shots` means `model.qinr_shots` is set to `None`, so the generator
uses analytic Pauli-Z expectation values without finite-shot measurement noise.
`freq_sample` means the frequency coefficients are sampled at initialization
and then kept fixed. `no_sample` means the generator uses the fixed
`model.qinr_baseline_freq` coefficient and does not sample frequency
coefficients.

Each ablation variant is a separate sub-experiment: the script deep-copies the
base config, clears checkpoint paths, builds a fresh QINR generator and a fresh
poisoned victim model, retrains both with alternating updates, then evaluates
only at base resolution for clean accuracy, ASR, and ASR after STRIP. Ablation
checkpoints are not reused across variants.

Use a different config or output directory:

```bash
python3 exp_run.py --config configs/qinr_debug.yaml --output-dir runs/qinr_debug/combined
```

The output directory contains:

- `cross_resolution_backdoor/`: original cross-resolution backdoor results
- `qinr_ablation/`: ablation checkpoints, base-resolution STRIP metrics, and
  `qinr_ablation_summary.json`
- `exp_run_summary.json`: paths and summaries for both groups
