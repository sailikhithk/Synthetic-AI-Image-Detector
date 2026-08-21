# LinkedIn Projects Section - SAI Entry

## For the "Projects" section on your LinkedIn profile

**Title:** Synthetic AI Image Detector (SAI)

**Description:**

Open-source multi-signal detector for AI-generated images, built for journalists, fact-checkers, and national-security analysts who need calibrated confidence rather than a binary verdict.

Most open-source detectors publish a single score. SAI reports a calibrated probability, an uncertainty estimate, and a verdict that can refuse to commit when evidence is weak.

Three model-agnostic signals (no training data required):
- Frequency-domain (DCT cross-channel correlation) - catches GAN spectral artifacts
- Reconstruction error (wavelet denoiser residual) - synthetic images reconstruct cleanly, real photos leave sensor noise
- Noise residual (PRNU-style channel asymmetry) - real cameras leave Bayer-pattern fingerprints

Production-reliability primitives:
- Temperature scaling for post-hoc probability calibration
- Epistemic uncertainty (signal disagreement) + aleatoric uncertainty (weight entropy)
- Refusal verdict when total uncertainty exceeds threshold
- Cross-generator evaluation harness: train calibration on generator A, test on generator B

Evaluated on the GenImage benchmark (600 images, CPU-only):
- BigGAN vs Real: AUROC 0.9403
- ADM (diffusion) vs Real: AUROC 0.6634
- All generators (calibrated): AUROC 0.8019, ECE 0.1046
- Cross-generator: calibration does NOT transfer across generator families (ECE degrades from 0.18 to 0.30)
- Refusal mechanism: 96.15% accuracy at 87% refusal rate

Key finding: GAN artifacts are detectable at AUROC 0.94, but diffusion artifacts are harder at AUROC 0.66. No single signal generalizes to all generators. The cross-generator experiment confirms that calibration is generator-specific - the failure mode that breaks detectors the day a new generator ships.

Tech stack: Python, NumPy, SciPy, scikit-image, scikit-learn, PyWavelets, Pydantic, Typer

GitHub: https://github.com/sailikhithk/Synthetic-AI-Image-Detector

---

## For a LinkedIn post (if you want to announce it)

Built and open-sourced a multi-signal detector for AI-generated images.

The problem: most open-source deepfake detectors publish a single score with no calibration, no uncertainty, and no way to measure how they'll perform on next month's generator. That's not safe to deploy in high-stakes decisions (journalism, intelligence analysis, election integrity).

The approach: three model-agnostic signals (frequency-domain DCT, wavelet reconstruction error, PRNU noise residual) combined in an ensemble with:
- Temperature scaling for calibrated probabilities
- Epistemic + aleatoric uncertainty quantification
- Refusal verdicts (the detector can say "inconclusive" instead of guessing wrong)
- Cross-generator evaluation harness

Results on GenImage benchmark (600 images, CPU-only, no GPU):
- BigGAN detection: AUROC 0.94
- ADM (diffusion) detection: AUROC 0.66
- Refusal mechanism: 96% accuracy when it commits

The honest finding: GAN artifacts are easy. Diffusion artifacts are hard. And calibration does NOT transfer across generator families - which is exactly the failure mode that breaks detectors in production.

Open source, MIT licensed: https://github.com/sailikhithk/Synthetic-AI-Image-Detector

#AI #MachineLearning #DeepfakeDetection #ImageForensics #OpenSource #NationalSecurity #ComputerVision
