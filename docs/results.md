# Synthetic AI Image Detector (SAI) - Real-World Evaluation Results

## Setup

| Component | Source |
|-----------|--------|
| Real images | MS-COCO val2014 (200 images, `bitmind/MS-COCO` on HuggingFace) |
| AI images (GAN) | GenImage BigGAN subset (200 images, `bitmind/GenImage_BigGAN`) |
| AI images (diffusion) | GenImage ADM subset (200 images, `bitmind/GenImage_ADM`) |
| Image preprocessing | Resized to 256x256 RGB, PNG |
| Detector | SAI v0.1 (3 signals: frequency, reconstruction, noise residual) |
| Hardware | Intel MacBook Pro i9, CPU only, no GPU |

Total: 600 images (200 real + 200 BigGAN + 200 ADM).

## Results

### Discrimination (AUROC)

| Experiment | AUROC | Accuracy | ECE |
|------------|-------|----------|-----|
| BigGAN vs Real (uncalibrated) | **0.9403** | 0.6650 | 0.1796 |
| ADM vs Real (uncalibrated) | **0.6634** | - | - |
| All generators vs Real (uncalibrated) | **0.8019** | 0.6650 | 0.1796 |
| All generators (calibrated on all) | **0.8019** | 0.6650 | **0.1046** |

Key findings:

1. **BigGAN detection is strong (AUROC 0.94).** BigGAN is a GAN architecture that
   leaves clear frequency-domain artifacts. The frequency signal captures these
   effectively.

2. **ADM detection is weak (AUROC 0.66).** ADM (Ablated Diffusion Model) produces
   higher-quality images with subtler artifacts. This is the expected "hard
   generator" case and confirms the central thesis of SAI: no single signal or
   ensemble generalizes to all generators without generator-specific training.

3. **Calibration improves ECE from 0.18 to 0.10** without changing AUROC
   (temperature scaling is monotonic). The fitted temperature is 0.4072,
   indicating the raw scores are underconfident and need sharpening.

### Cross-Generator Generalization

| Train calibration on | Test on | AUROC | Accuracy | ECE |
|----------------------|---------|-------|----------|-----|
| BigGAN + Real | ADM + Real | 0.6634 | 0.4975 | 0.1145 |
| ADM + Real | BigGAN + Real | 0.9403 | 0.5100 | 0.2968 |

Key findings:

1. **AUROC is unchanged by cross-generator calibration.** Temperature scaling is
   a monotonic transform, so it preserves the ranking of scores and thus AUROC.
   This is expected and correct.

2. **Calibration does NOT transfer across generators.** When calibration is
   fitted on BigGAN and applied to ADM, ECE improves (0.1145 vs 0.1796
   uncalibrated). But when fitted on ADM and applied to BigGAN, ECE degrades
   badly (0.2968 vs 0.1796). The calibration surface is generator-specific.

3. **Accuracy at 0.5 threshold drops to near-chance (0.50) on cross-gen.** The
   score distributions shift between generators, so a fixed threshold calibrated
   on one generator does not generalize. This is the "new generator failure mode"
   that SAI is designed to expose.

### Uncertainty and Refusal

| Refusal threshold | Refusal rate | Refusal-aware accuracy | Coverage |
|-------------------|-------------|----------------------|----------|
| 0.40 (default) | 100.0% | 0.0000 | 0.0% |
| 0.50 | 87.0% | **0.9615** | 13.0% |
| 0.60 | 16.2% | 0.7455 | 83.8% |
| 0.70 | 2.7% | 0.6832 | 97.3% |
| 0.80 | 0.0% | 0.6650 | 100.0% |

Key findings:

1. **The default refusal threshold (0.40) is too aggressive.** With 3
   roughly-uniform-weight signals, the aleatoric uncertainty component (weight
   entropy / log(3)) is approximately 0.91, making total uncertainty always
   exceed 0.43. The minimum observed uncertainty is 0.4348, so 100% of samples
   are refused at the default threshold.

2. **At threshold 0.50, refusal-aware accuracy is 96.15%** with 87% refusal.
   The detector is highly accurate when it does commit, but refuses most
   samples. This is appropriate for high-stakes use cases (intelligence
   analysis, journalism) where false positives are costly.

3. **At threshold 0.60, the detector covers 83.8% of samples** with 74.55%
   accuracy. This is a more practical operating point for general use.

4. **Real images have higher mean uncertainty (0.5976) than AI images (0.5339).**
   This is because real photographs have more diverse noise patterns, causing
   higher signal disagreement. The detector is more uncertain on real images,
   which is the safer failure mode (refusing on real rather than misclassifying
   AI as real).

## Limitations

1. **Small sample size (200 per class).** Results are indicative but not
   statistically robust. Confidence intervals would be wide.

2. **Only 2 AI generators tested.** BigGAN (GAN) and ADM (diffusion) represent
   two major families, but the full GenImage benchmark includes 8 generators
   (SD, SDXL, Midjourney, DALL-E 2, DALL-E 3, GLIDE, Wukong, AttnGAN).

3. **No trained calibration parameters.** The temperature scaler is fitted
   post-hoc on the eval set. A proper protocol would fit on a held-out split
   and evaluate on a separate test split.

4. **Images resized to 256x256.** Native BigGAN images are 128x128, MS-COCO
   images are variable (typically ~640x480). Resizing may affect frequency
   artifacts.

5. **No comparison to baselines.** These numbers are SAI-only. Comparison
   against CNNDetection, NPR, or UnivFD would contextualize performance.

## Reproduction

```bash
# Download data (requires huggingface_hub)
python3 scripts/download_eval_data.py

# Run evaluation
python3 scripts/run_real_eval.py

# Results saved to data/eval_results.json
# Calibration saved to calibration.json
```

## Conclusion

SAI's three-signal ensemble achieves AUROC 0.94 on BigGAN and 0.66 on ADM,
confirming that GAN artifacts are easier to detect than diffusion artifacts.
The cross-generator experiment confirms that calibration does not transfer
across generator families. The uncertainty-based refusal mechanism provides
96% accuracy when it commits, but the default threshold needs tuning for
practical use. These results establish a baseline for future work: adding
trained signals (autoencoder reconstruction, CLIP-based features) and expanding
to the full GenImage benchmark.
