# SAI - Synthetic AI Image Detector

A production-reliability layer for detecting AI-generated images, aimed at
journalists, fact-checkers, and national-security analysts who cannot afford
to be fooled by deepfakes but also cannot afford false accusations.

> Most open-source detectors publish a single score and call it a day. SAI
> reports a calibrated probability, an uncertainty estimate, and a verdict
> that can refuse to commit when the evidence is weak. The eval harness
> measures cross-generator generalization - the failure mode that breaks
> detectors the day a new generator ships.

## Why this exists

Synthetic image generation has crossed the photorealistic threshold.
Stable Diffusion XL, Midjourney v6, DALL-E 3, and FLUX produce images that
pass casual human inspection. For newsrooms, intelligence analysts, and
election-integrity teams, a wrong call has asymmetric cost: flagging a real
photo as fake destroys credibility, while missing a fake enables
disinformation.

The public landscape of detectors falls into two camps:

1. **Academic research code** - strong on novel signals, weak on
   calibration, uncertainty, and deployment. Trained on one generator,
   evaluated on the same generator.
2. **Closed platforms** - opaque, no way to audit confidence, no way to
   measure generalization to new generators.

SAI fills the gap between these: a multi-signal detector with the
production-reliability primitives that real deployments need.

## What makes SAI different

| Primitive | What it does | Why it matters |
|-----------|--------------|----------------|
| Multi-signal ensemble | Frequency, reconstruction, noise residual | No single signal generalizes; ensemble hedges |
| Temperature calibration | Maps raw scores to true probabilities | A 0.9 score should mean 90% chance of AI, not 60% |
| Epistemic uncertainty | Signal disagreement | Catches inputs the detector has never seen |
| Aleatoric uncertainty | Weight entropy | Catches inputs where no signal is confident |
| Refusal verdict | Returns "inconclusive" when uncertainty is high | Prevents forced wrong calls on hard inputs |
| Cross-generator eval | Train calibration on generators A,B; test on C | Measures the new-generator failure mode |

## Signals

### 1. Frequency-domain (`FrequencySignal`)

Diffusion and GAN generators leave periodic artifacts in the DCT spectrum
that real-camera captures do not. This signal measures:

- **High-frequency energy ratio**: AI images often have anomalous spectral tails.
- **Cross-channel spectral correlation**: generators synthesize channels
  jointly, producing a cross-channel signature distinct from the
  per-channel Bayer-pattern demosaicing of a real camera.

### 2. Reconstruction error (`ReconstructionSignal`)

Synthetic images live on the generator's manifold and reconstruct cleanly
under a generic natural-image denoiser. Real photographs contain sensor
noise and demosaicing artifacts that survive as residual. This signal
measures:

- **Residual energy**: low residual energy suggests a smooth synthetic manifold.
- **Residual spectral concentration**: AI residuals concentrate energy in
  low frequencies (a generator failure mode); real residuals spread
  across high frequencies (sensor noise).

Uses a wavelet denoiser (no torch dependency) as the reconstructor.

### 3. Noise residual (`NoiseResidualSignal`)

Real cameras leave a Photo Response Non-Uniformity (PRNU) noise fingerprint
from sensor demosaicing and lens distortion. Synthetic images lack this
fingerprint. This signal measures:

- **Channel variance asymmetry**: real cameras have channel-dependent noise
  (Bayer pattern); AI generators typically do not.
- **Spatial consistency of noise**: real PRNU has stable spatial structure;
  AI noise is more uniform.

## Calibration and uncertainty

### Temperature scaling

Post-hoc calibration that maps raw ensemble scores to well-calibrated
probabilities. Fit a single temperature `T` on a held-out labeled set:

```
calibrated_score = sigmoid(logit(raw_score) / T)
```

This is Platt scaling applied to the ensemble output. It does not change
ranking (AUROC is invariant), only the probability interpretation.

### Uncertainty quantification

Two forms of uncertainty are reported:

- **Epistemic** (model uncertainty): weighted variance of signal scores.
  High when signals disagree - typically on inputs from a generator the
  detector has never seen.
- **Aleatoric** (data uncertainty): entropy of the signal weight
  distribution. High when no single signal is confident.

When `total_uncertainty >= refuse_threshold`, the verdict is
`inconclusive` rather than `ai` or `real`. This is the refusal mechanism
that prevents forced wrong calls.

## Evaluation harness

The harness is the production-reliability core. It measures:

- **AUROC**: standard discrimination metric.
- **Accuracy at threshold 0.5**.
- **Refusal-aware accuracy**: accuracy over non-refused inputs only.
  A high refusal-aware accuracy with a non-trivial refusal rate is the
  goal: the detector is right when it commits, and honest when it is unsure.
- **Expected Calibration Error (ECE)**: reliability of probability scores.
- **Per-generator AUROC**: discrimination broken down by generator.
- **Cross-generator generalization**: fit calibration on generators A, B;
  evaluate on held-out generator C. This exposes the new-generator failure
  mode that breaks most open-source detectors.

## Install

```bash
pip install -e .
```

Requires Python 3.9+ with numpy, scipy, scikit-learn, scikit-image,
PyWavelets, Pillow, pydantic, typer, rich, and matplotlib. Optional
torch backend for neural signals (`pip install -e ".[torch]"`). Dev
dependencies: `pip install -e ".[dev]"`.

## CLI

```bash
# Detect a single image
sai detect path/to/image.png

# Detect with JSON output (for pipelines)
sai detect path/to/image.png --json

# Evaluate on a directory of real vs AI images
sai eval-dir real/ ai/ --generator sd-xl

# Fit temperature calibration on a labeled set
sai calibrate real/ ai/ --out calibration.json
```

## Library

```python
from sai.pipeline import DetectorPipeline
from sai.io import load_image

pipeline = DetectorPipeline()
result = pipeline.detect(load_image("image.png"))

print(result.verdict)              # "ai", "real", or "inconclusive"
print(result.calibrated_score)     # well-calibrated probability
print(result.total_uncertainty)    # 0..1, high = refuse
```

Cross-generator evaluation:

```python
from sai.eval import cross_generator_eval
from sai.pipeline import DetectorPipeline

samples = [...]  # list of {image, label, generator}
pipeline = DetectorPipeline()
report = cross_generator_eval(
    pipeline, samples,
    train_generators=["sd-xl", "real-camera"],
    held_out_generator="midjourney-v6",
)
print(report.auroc, report.ece, report.refusal_rate)
```

## Repository structure

```
SAI/
  sai/
    signals/           # Detection signals (frequency, reconstruction, noise)
    calibration.py     # Temperature scaling + uncertainty quantification
    pipeline.py        # Ensemble pipeline
    eval.py            # Evaluation harness + cross-generator generalization
    cli.py             # sai detect / sai eval / sai calibrate
    io.py              # Image loading
  tests/               # Synthetic fixtures + signal sanity + calibration checks
  docs/                # Architecture and method notes
  pyproject.toml
  LICENSE
```

## Limitations and honest scope

- The signals are model-agnostic and do not require training data. They
  trade raw discrimination power for generalization: they will not beat a
  fine-tuned detector on the generator it was trained on, but they will
  not collapse on a new generator either.
- The calibration fit needs a labeled set. For production use, fit on a
  representative sample of your deployment distribution.
- The refusal threshold is a policy knob. For newsroom use, set it high
  (refuse often). For triage use, set it low (commit often, audit later).
- This is not a forensic provenance tool. It tells you whether an image is
  AI-generated, not which generator produced it or whether it was edited.

## National-security framing

Synthetic image detection is a textbook national-security capability:
election disinformation, fabricated evidence in conflict zones, impersonation
of public figures, and forged documentation all rely on the inability of
analysts to reliably distinguish real from synthetic. SAI's contribution is
not a better classifier - it is the reliability layer that makes a
classifier safe to deploy in high-cost decisions:

- **Calibrated probabilities** mean an analyst can reason about risk
  rather than reading a binary verdict.
- **Uncertainty quantification** means the detector can refuse instead of
  being forced into a wrong call.
- **Cross-generator evaluation** means the deployment team can measure
  how the detector will perform on next month's generator, not just
  today's.

## License

MIT - see [LICENSE](LICENSE).
