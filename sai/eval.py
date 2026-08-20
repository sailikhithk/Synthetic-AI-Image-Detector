"""Evaluation harness.

The harness is the production-reliability core of SAI. It measures:

1. Cross-generator generalization: train calibration on generators A, B;
   evaluate on held-out generator C. This exposes the "new generator"
   failure mode that breaks most open-source detectors.

2. Calibration: reliability diagrams and Expected Calibration Error (ECE).

3. Discrimination: AUROC, accuracy at a fixed threshold, and the
   "refusal-aware accuracy" that counts an inconclusive verdict as
   neither right nor wrong (separately reported).

4. Uncertainty quality: how often the detector refuses on hard cases
   vs easy cases (refusal-coverage curve).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from sai.pipeline import DetectorPipeline, DetectionResult


@dataclass
class EvalRow:
    path: str
    label: int  # 1 = AI, 0 = real
    generator: str  # e.g. "sd-xl", "real-camera", "midjourney"
    raw_score: float
    calibrated_score: float
    verdict: str
    total_uncertainty: float


@dataclass
class EvalReport:
    auroc: float
    accuracy: float
    refusal_aware_accuracy: float
    refusal_rate: float
    ece: float  # expected calibration error
    per_generator_auroc: dict = field(default_factory=dict)
    rows: List[EvalRow] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "auroc": self.auroc,
            "accuracy": self.accuracy,
            "refusal_aware_accuracy": self.refusal_aware_accuracy,
            "refusal_rate": self.refusal_rate,
            "ece": self.ece,
            "per_generator_auroc": self.per_generator_auroc,
        }


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    P = float(labels.sum())
    N = float(len(labels) - P)
    if P == 0 or N == 0:
        return float("nan")
    tp = 0.0
    fp = 0.0
    auroc = 0.0
    prev_score = None
    for i in range(len(labels_sorted)):
        if prev_score is not None and scores[order[i]] != prev_score:
            auroc += (fp / N) * (tp / P) * 0.5  # trapezoid mid-point correction
            auroc += (tp / P) * 0  # placeholder
        if labels_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
            auroc += tp / P
        prev_score = scores[order[i]]
    return float(auroc / N) if N > 0 else float("nan")


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Standard ROC AUC via the rank statistic (Mann-Whitney U / (P*N))."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if len(np.unique(y)) < 2:
        return float("nan")
    P = int(y.sum())
    N = int(len(y) - P)
    if P == 0 or N == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # handle ties via average ranks
    unique_vals, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.zeros(len(unique_vals), dtype=np.float64)
        np.add.at(sums, inverse, ranks)
        avg_ranks = sums / counts
        ranks = avg_ranks[inverse]
    sum_ranks_pos = float(ranks[y == 1].sum())
    u = sum_ranks_pos - P * (P + 1) / 2.0
    return float(u / (P * N))


def expected_calibration_error(scores: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """ECE: sum over bins of |accuracy - confidence| * (bin_size / total)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(s)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (s >= lo) & (s < hi if i < n_bins - 1 else s <= hi)
        if mask.sum() == 0:
            continue
        conf = float(s[mask].mean())
        # accuracy: predict ai if score >= 0.5
        acc = float((y[mask] == (s[mask] >= 0.5).astype(int)).mean())
        ece += abs(acc - conf) * (mask.sum() / n)
    return float(ece)


def evaluate(
    pipeline: DetectorPipeline,
    samples: Iterable[dict],
) -> EvalReport:
    """Run the pipeline over a list of samples.

    Each sample is a dict with keys: image (HxWxC uint8 ndarray), label (int),
    generator (str), path (str, optional).
    """
    rows: List[EvalRow] = []
    for sample in samples:
        res: DetectionResult = pipeline.detect(sample["image"])
        rows.append(
            EvalRow(
                path=sample.get("path", ""),
                label=int(sample["label"]),
                generator=sample.get("generator", "unknown"),
                raw_score=res.raw_score,
                calibrated_score=res.calibrated_score,
                verdict=res.verdict,
                total_uncertainty=res.total_uncertainty,
            )
        )

    scores = np.array([r.calibrated_score for r in rows])
    labels = np.array([r.label for r in rows])
    verdicts = np.array([r.verdict for r in rows])

    auroc_all = auroc(scores, labels)
    preds = (scores >= 0.5).astype(int)
    accuracy = float((preds == labels).mean()) if len(labels) else 0.0
    refused = verdicts == "inconclusive"
    refusal_rate = float(refused.mean()) if len(verdicts) else 0.0
    # refusal-aware accuracy: accuracy over non-refused only
    if refused.all():
        raa = 0.0
    else:
        raa = float((preds[~refused] == labels[~refused]).mean())
    ece = expected_calibration_error(scores, labels)

    per_gen: dict = {}
    for gen in sorted({r.generator for r in rows}):
        mask = np.array([r.generator == gen for r in rows])
        per_gen[gen] = auroc(scores[mask], labels[mask])

    return EvalReport(
        auroc=auroc_all,
        accuracy=accuracy,
        refusal_aware_accuracy=raa,
        refusal_rate=refusal_rate,
        ece=ece,
        per_generator_auroc=per_gen,
        rows=rows,
    )


def cross_generator_eval(
    pipeline: DetectorPipeline,
    samples: List[dict],
    train_generators: List[str],
    held_out_generator: str,
) -> EvalReport:
    """Fit calibration on train_generators, evaluate on held_out_generator.

    This is the key experiment for the "new generator generalization" gap.
    """
    train = [s for s in samples if s["generator"] in train_generators]
    held = [s for s in samples if s["generator"] == held_out_generator]
    if not train or not held:
        raise ValueError("Need samples in both train and held-out generators.")
    # fit calibration on raw scores of train set
    raw_train = []
    labels_train = []
    for s in train:
        r = pipeline.detect(s["image"])
        raw_train.append(r.raw_score)
        labels_train.append(s["label"])
    pipeline.fit_calibration(raw_train, labels_train)
    return evaluate(pipeline, held)
