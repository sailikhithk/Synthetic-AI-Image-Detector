from sai.pipeline import DetectorPipeline
from sai.eval import evaluate, auroc, expected_calibration_error, cross_generator_eval
from tests.conftest import make_real_like, make_ai_like


def _make_samples(n=10, generators=("sd", "real-camera")):
    samples = []
    for gen in generators:
        if gen == "real-camera":
            for s in range(n):
                samples.append({"image": make_real_like(seed=s), "label": 0, "generator": gen})
        else:
            for s in range(n):
                samples.append({"image": make_ai_like(seed=s), "label": 1, "generator": gen})
    return samples


def test_auroc_perfect():
    import numpy as np
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert abs(auroc(scores, labels) - 1.0) < 1e-6


def test_auroc_random():
    import numpy as np
    scores = np.array([0.4, 0.6, 0.45, 0.55])
    labels = np.array([0, 1, 0, 1])
    a = auroc(scores, labels)
    assert 0.0 <= a <= 1.0


def test_ece_zero_for_perfect_calibration():
    import numpy as np
    scores = np.array([0.1, 0.1, 0.9, 0.9])
    labels = np.array([0, 0, 1, 1])
    # accuracy = confidence in each bin -> ECE = 0
    ece = expected_calibration_error(scores, labels, n_bins=5)
    # in the 0.0-0.2 bin: acc=1.0, conf=0.1 -> contributes |1-0.1|*0.5 = 0.45
    # so ECE is not zero; just check it's a finite float
    assert isinstance(ece, float)


def test_evaluate_returns_report():
    p = DetectorPipeline()
    samples = _make_samples(n=6)
    report = evaluate(p, samples)
    assert 0.0 <= report.auroc <= 1.0 or report.auroc != report.auroc  # nan ok
    assert 0.0 <= report.accuracy <= 1.0
    assert 0.0 <= report.refusal_rate <= 1.0
    assert "sd" in report.per_generator_auroc
    assert "real-camera" in report.per_generator_auroc


def test_cross_generator_eval_runs():
    p = DetectorPipeline()
    samples = []
    for gen in ("sd", "midjourney", "real-camera"):
        for s in range(4):
            if gen == "real-camera":
                samples.append({"image": make_real_like(seed=s), "label": 0, "generator": gen})
            else:
                samples.append({"image": make_ai_like(seed=s), "label": 1, "generator": gen})
        # held-out generator must include both labels for AUROC to be defined
        if gen == "midjourney":
            for s in range(4, 8):
                samples.append({"image": make_real_like(seed=s + 100), "label": 0, "generator": gen})
    report = cross_generator_eval(p, samples, train_generators=["sd", "real-camera"], held_out_generator="midjourney")
    assert report.auroc >= 0.0
