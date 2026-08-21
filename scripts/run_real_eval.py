#!/usr/bin/env python3
"""Run SAI evaluation on real benchmark data (GenImage BigGAN + ADM vs MS-COCO real).

Fixes vs v1:
- Per-generator AUROC computed as (generator + real) not (generator only)
- Cross-gen held-out set includes real images
- Refusal threshold sweep to find operating point
- Reports raw_score AUROC (threshold-independent) as primary metric

Usage:
    python3 scripts/run_real_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from sai.pipeline import DetectorPipeline
from sai.eval import auroc, expected_calibration_error

DATA = Path(__file__).resolve().parent.parent / "data" / "eval"
N_PER_GEN = 200


def load_images(directory: Path, n: int = N_PER_GEN):
    """Load up to n images from a directory as uint8 HxWxC arrays."""
    imgs = []
    for p in sorted(directory.glob("*.png"))[:n]:
        imgs.append(np.array(Image.open(p).convert("RGB")))
    return imgs


def score_all(pipeline: DetectorPipeline, real_imgs, biggan_imgs, adm_imgs):
    """Run detector on all images, return dict of scores and labels."""
    rows = []
    for img in real_imgs:
        r = pipeline.detect(img)
        rows.append({"raw": r.raw_score, "cal": r.calibrated_score, "label": 0,
                      "generator": "real-camera", "verdict": r.verdict,
                      "total_unc": r.total_uncertainty})
    for img in biggan_imgs:
        r = pipeline.detect(img)
        rows.append({"raw": r.raw_score, "cal": r.calibrated_score, "label": 1,
                      "generator": "biggan", "verdict": r.verdict,
                      "total_unc": r.total_uncertainty})
    for img in adm_imgs:
        r = pipeline.detect(img)
        rows.append({"raw": r.raw_score, "cal": r.calibrated_score, "label": 1,
                      "generator": "adm", "verdict": r.verdict,
                      "total_unc": r.total_uncertainty})
    return rows


def metrics_at_threshold(scores, labels, threshold=0.5):
    """Accuracy, refusal rate at a given score threshold."""
    preds = (np.array(scores) >= threshold).astype(int)
    return float((preds == np.array(labels)).mean())


def refusal_aware_accuracy(scores, labels, uncertainties, refuse_thresh, score_thresh=0.5):
    """Accuracy over non-refused samples."""
    scores = np.array(scores)
    labels = np.array(labels)
    uncertainties = np.array(uncertainties)
    refused = uncertainties >= refuse_thresh
    if refused.all():
        return 0.0, 1.0
    preds = (scores[~refused] >= score_thresh).astype(int)
    raa = float((preds == labels[~refused]).mean())
    rate = float(refused.mean())
    return raa, rate


def per_generator_auroc(rows, score_key="raw"):
    """AUROC for each AI generator vs real."""
    real_rows = [r for r in rows if r["generator"] == "real-camera"]
    real_scores = np.array([r[score_key] for r in real_rows])
    real_labels = np.zeros(len(real_rows))
    result = {}
    for gen in ["biggan", "adm"]:
        gen_rows = [r for r in rows if r["generator"] == gen]
        gen_scores = np.array([r[score_key] for r in gen_rows])
        gen_labels = np.ones(len(gen_rows))
        all_scores = np.concatenate([real_scores, gen_scores])
        all_labels = np.concatenate([real_labels, gen_labels])
        result[gen] = auroc(all_scores, all_labels)
    return result


def main():
    print("Loading images (200 real + 200 BigGAN + 200 ADM)...")
    real_imgs = load_images(DATA / "real")
    biggan_imgs = load_images(DATA / "biggan")
    adm_imgs = load_images(DATA / "adm")
    print(f"  Loaded {len(real_imgs)} real, {len(biggan_imgs)} BigGAN, {len(adm_imgs)} ADM")

    # --- Run detector on all images (uncalibrated) ---
    print("\nRunning detector (uncalibrated)...")
    pipeline = DetectorPipeline()
    rows = score_all(pipeline, real_imgs, biggan_imgs, adm_imgs)

    raw_scores = np.array([r["raw"] for r in rows])
    cal_scores = np.array([r["cal"] for r in rows])
    labels = np.array([r["label"] for r in rows])
    uncs = np.array([r["total_unc"] for r in rows])

    # --- Experiment 1: Per-generator AUROC (raw scores, threshold-independent) ---
    print("\n" + "=" * 70)
    print("  Experiment 1: Per-generator AUROC (raw scores)")
    print("=" * 70)
    per_gen_raw = per_generator_auroc(rows, "raw")
    overall_auroc = auroc(raw_scores, labels)
    print(f"  Overall AUROC (all vs real):    {overall_auroc:.4f}")
    for g, a in sorted(per_gen_raw.items()):
        print(f"  {g:20s} vs real:  {a:.4f}")
    accuracy_05 = metrics_at_threshold(raw_scores, labels, 0.5)
    print(f"  Accuracy @ 0.5 threshold:       {accuracy_05:.4f}")
    ece_raw = expected_calibration_error(raw_scores, labels)
    print(f"  ECE (raw scores):               {ece_raw:.4f}")

    # --- Experiment 2: Uncertainty analysis ---
    print("\n" + "=" * 70)
    print("  Experiment 2: Uncertainty analysis")
    print("=" * 70)
    print(f"  Mean total uncertainty:         {uncs.mean():.4f}")
    print(f"  Median total uncertainty:       {np.median(uncs):.4f}")
    print(f"  Min / Max uncertainty:          {uncs.min():.4f} / {uncs.max():.4f}")
    # Uncertainty by class
    ai_unc = uncs[labels == 1]
    real_unc = uncs[labels == 0]
    print(f"  Mean uncertainty (AI images):   {ai_unc.mean():.4f}")
    print(f"  Mean uncertainty (real images): {real_unc.mean():.4f}")

    # Refusal threshold sweep
    print(f"\n  Refusal threshold sweep:")
    print(f"  {'Threshold':>10s}  {'Refusal%':>8s}  {'RAA':>6s}  {'Coverage':>8s}")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]:
        raa, rrate = refusal_aware_accuracy(raw_scores, labels, uncs, thresh)
        print(f"  {thresh:>10.2f}  {rrate*100:>7.1f}%  {raa:>6.4f}  {(1-rrate)*100:>7.1f}%")

    # --- Experiment 3: Cross-generator generalization ---
    print("\n" + "=" * 70)
    print("  Experiment 3: Cross-generator generalization")
    print("=" * 70)

    # Train calibration on BigGAN+real, test on ADM+real
    train_rows = [r for r in rows if r["generator"] in ("biggan", "real-camera")]
    train_raw = [r["raw"] for r in train_rows]
    train_labels = [r["label"] for r in train_rows]
    pipeline_cross1 = DetectorPipeline()
    T1 = pipeline_cross1.fit_calibration(train_raw, train_labels)
    print(f"  Train on BigGAN+real: fitted T = {T1:.4f}")

    # Re-score ADM+real with calibrated pipeline
    adm_real_rows = [r for r in rows if r["generator"] in ("adm", "real-camera")]
    cross1_scores = []
    cross1_labels = []
    for r in adm_real_rows:
        # Re-detect with calibrated pipeline to get calibrated scores
        # But we already have raw scores; just re-calibrate
        cal = pipeline_cross1.scaler.transform(r["raw"])
        cross1_scores.append(cal)
        cross1_labels.append(r["label"])
    cross1_auroc = auroc(np.array(cross1_scores), np.array(cross1_labels))
    cross1_acc = metrics_at_threshold(cross1_scores, cross1_labels, 0.5)
    cross1_ece = expected_calibration_error(np.array(cross1_scores), np.array(cross1_labels))
    print(f"  Test on ADM+real: AUROC = {cross1_auroc:.4f}, Acc = {cross1_acc:.4f}, ECE = {cross1_ece:.4f}")

    # Train calibration on ADM+real, test on BigGAN+real
    train_rows2 = [r for r in rows if r["generator"] in ("adm", "real-camera")]
    train_raw2 = [r["raw"] for r in train_rows2]
    train_labels2 = [r["label"] for r in train_rows2]
    pipeline_cross2 = DetectorPipeline()
    T2 = pipeline_cross2.fit_calibration(train_raw2, train_labels2)
    print(f"  Train on ADM+real: fitted T = {T2:.4f}")

    biggan_real_rows = [r for r in rows if r["generator"] in ("biggan", "real-camera")]
    cross2_scores = []
    cross2_labels = []
    for r in biggan_real_rows:
        cal = pipeline_cross2.scaler.transform(r["raw"])
        cross2_scores.append(cal)
        cross2_labels.append(r["label"])
    cross2_auroc = auroc(np.array(cross2_scores), np.array(cross2_labels))
    cross2_acc = metrics_at_threshold(cross2_scores, cross2_labels, 0.5)
    cross2_ece = expected_calibration_error(np.array(cross2_scores), np.array(cross2_labels))
    print(f"  Test on BigGAN+real: AUROC = {cross2_auroc:.4f}, Acc = {cross2_acc:.4f}, ECE = {cross2_ece:.4f}")

    # --- Experiment 4: Calibration on all data ---
    print("\n" + "=" * 70)
    print("  Experiment 4: Calibration fitted on all data")
    print("=" * 70)
    pipeline_cal = DetectorPipeline()
    T_all = pipeline_cal.fit_calibration(raw_scores.tolist(), labels.tolist())
    print(f"  Fitted temperature: {T_all:.4f}")
    cal_all = np.array([pipeline_cal.scaler.transform(r["raw"]) for r in rows])
    cal_auroc = auroc(cal_all, labels)
    cal_acc = metrics_at_threshold(cal_all, labels, 0.5)
    cal_ece = expected_calibration_error(cal_all, labels)
    print(f"  AUROC (calibrated):  {cal_auroc:.4f}")
    print(f"  Accuracy:            {cal_acc:.4f}")
    print(f"  ECE:                 {cal_ece:.4f}")

    # Save calibration
    cal_path = Path(__file__).resolve().parent.parent / "calibration.json"
    cal_path.write_text(json.dumps({"temperature": T_all}, indent=2))
    print(f"\n  Saved calibration to {cal_path}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Experiment':<50s} {'AUROC':>7s} {'Acc':>6s} {'ECE':>6s}")
    print(f"  {'-'*50} {'-'*7} {'-'*6} {'-'*6}")
    print(f"  {'BigGAN vs Real (raw, uncalibrated)':<50s} {per_gen_raw['biggan']:>7.4f} {accuracy_05:>6.4f} {ece_raw:>6.4f}")
    print(f"  {'ADM vs Real (raw, uncalibrated)':<50s} {per_gen_raw['adm']:>7.4f} {'':>6s} {'':>6s}")
    print(f"  {'All generators (raw, uncalibrated)':<50s} {overall_auroc:>7.4f} {accuracy_05:>6.4f} {ece_raw:>6.4f}")
    print(f"  {'Cross-gen: train BigGAN, test ADM (calibrated)':<50s} {cross1_auroc:>7.4f} {cross1_acc:>6.4f} {cross1_ece:>6.4f}")
    print(f"  {'Cross-gen: train ADM, test BigGAN (calibrated)':<50s} {cross2_auroc:>7.4f} {cross2_acc:>6.4f} {cross2_ece:>6.4f}")
    print(f"  {'All generators (calibrated on all)':<50s} {cal_auroc:>7.4f} {cal_acc:>6.4f} {cal_ece:>6.4f}")
    print()

    # Save results
    results = {
        "n_per_generator": N_PER_GEN,
        "datasets": {
            "real": "MS-COCO val2014 (bitmind/MS-COCO on HuggingFace)",
            "biggan": "GenImage BigGAN (bitmind/GenImage_BigGAN on HuggingFace)",
            "adm": "GenImage ADM (bitmind/GenImage_ADM on HuggingFace)",
        },
        "image_size": "256x256 RGB (resized from native)",
        "experiments": {
            "biggan_vs_real_auroc": per_gen_raw["biggan"],
            "adm_vs_real_auroc": per_gen_raw["adm"],
            "all_generators_auroc": overall_auroc,
            "all_generators_accuracy": accuracy_05,
            "all_generators_ece": ece_raw,
            "cross_gen_train_biggan_test_adm": {
                "auroc": cross1_auroc, "accuracy": cross1_acc, "ece": cross1_ece,
                "temperature": T1,
            },
            "cross_gen_train_adm_test_biggan": {
                "auroc": cross2_auroc, "accuracy": cross2_acc, "ece": cross2_ece,
                "temperature": T2,
            },
            "calibrated_on_all": {
                "auroc": cal_auroc, "accuracy": cal_acc, "ece": cal_ece,
                "temperature": T_all,
            },
        },
        "uncertainty_stats": {
            "mean": float(uncs.mean()),
            "median": float(np.median(uncs)),
            "mean_ai": float(ai_unc.mean()),
            "mean_real": float(real_unc.mean()),
        },
    }
    results_path = Path(__file__).resolve().parent.parent / "data" / "eval_results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"  Saved results to {results_path}")


if __name__ == "__main__":
    main()
