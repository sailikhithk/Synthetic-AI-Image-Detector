# Synthetic AI Image Detector (SAI) - Architecture

## Design principles

1. **No training data required for the base detector.** Signals are
   model-agnostic and based on image-formation physics. This trades raw
   discrimination for cross-generator generalization.
2. **Calibration is a separate layer.** The base detector produces raw
   scores; temperature scaling maps them to calibrated probabilities.
   This lets you re-calibrate for your deployment distribution without
   retraining signals.
3. **Uncertainty is first-class.** Every detection returns epistemic and
   aleatoric uncertainty. The verdict can be "inconclusive".
4. **The eval harness is the product.** A detector without a
   generalization measurement is a demo. The harness measures the
   failure modes that matter in production.

## Signal contract

Every signal implements:

```python
class Signal(ABC):
    name: str
    def analyze(self, image: np.ndarray) -> SignalResult
```

`SignalResult` contains:
- `score` in [0, 1]: probability the image is AI-generated.
- `weight` in [0, 1]: how much the ensemble should trust this signal.
- `features`: dict of diagnostic features for logging and interpretability.

Signals are stateless across calls. Config lives on the instance.

## Ensemble

The pipeline computes a weighted average of signal scores:

```
raw_score = sum(weight_i * score_i) / sum(weight_i)
```

Weights are static defaults (0.7, 0.65, 0.6 for frequency, reconstruction,
noise). A learned weighting scheme is a future extension.

## Calibration

Temperature scaling fits a single scalar `T` on a held-out labeled set by
minimizing NLL of calibrated probabilities. The fit uses gradient descent
on `T` with the gradient:

```
dNLL/dT = mean((y - p) * logits / T^2)
```

where `p = sigmoid(logits / T)` and `logits = logit(raw_score)`.

## Uncertainty

- **Epistemic**: `weighted_variance(scores) * 4.0`, clipped to [0, 1].
  High when signals disagree.
- **Aleatoric**: normalized entropy of the weight distribution.
  High when no signal dominates.
- **Total**: `0.6 * epistemic + 0.4 * aleatoric`, clipped to [0, 1].
- **Verdict**: `inconclusive` if total >= refuse_threshold, else
  `ai` if calibrated_score >= 0.5, else `real`.

## Cross-generator generalization

The key experiment:

1. Fit calibration on generators A and B (e.g. SDXL + real-camera).
2. Evaluate on held-out generator C (e.g. Midjourney v6).
3. Report AUROC, ECE, refusal rate on C.

A detector that scores well on C without seeing C during calibration is
the goal. A detector that collapses on C is the failure mode this project
exists to expose.

## Future work

- Learned signal weighting via logistic regression on a labeled set.
- Neural signals (CLIP-based embedding anomaly detection) as optional
  torch-backed signals.
- Per-signal calibration (one temperature per signal).
- Reliability diagrams as plot output from the eval harness.
- Streaming inference for video frames.
