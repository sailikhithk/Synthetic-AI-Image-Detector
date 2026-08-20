import numpy as np

from sai.pipeline import DetectorPipeline
from sai.calibration import TemperatureScaler
from tests.conftest import make_real_like, make_ai_like


def test_pipeline_detect_returns_valid_result():
    p = DetectorPipeline()
    img = make_real_like(seed=10)
    res = p.detect(img)
    assert 0.0 <= res.raw_score <= 1.0
    assert 0.0 <= res.calibrated_score <= 1.0
    assert res.verdict in {"ai", "real", "inconclusive"}
    assert 0.0 <= res.total_uncertainty <= 1.0
    assert len(res.signal_results) == 3


def test_pipeline_separates_real_vs_ai():
    p = DetectorPipeline()
    real_scores = [p.detect(make_real_like(seed=s)).raw_score for s in range(5)]
    ai_scores = [p.detect(make_ai_like(seed=s)).raw_score for s in range(5)]
    mean_real = float(np.mean(real_scores))
    mean_ai = float(np.mean(ai_scores))
    assert mean_ai > mean_real, f"AI {mean_ai} should > real {mean_real}"


def test_temperature_scaler_monotonic():
    ts = TemperatureScaler(temperature=2.0)
    # higher raw scores should produce higher calibrated scores
    a = ts.transform(0.2)
    b = ts.transform(0.5)
    c = ts.transform(0.8)
    assert a < b < c


def test_temperature_fit_reduces_nll():
    rng = np.random.default_rng(0)
    raw = np.clip(rng.normal(0.7, 0.15, 200), 0.01, 0.99)
    labels = (raw > 0.5).astype(int)
    ts = TemperatureScaler()
    T = ts.fit(raw.tolist(), labels.tolist())
    assert T > 0
    # calibrated should be closer to labels than raw
    cal = np.array([ts.transform(r) for r in raw])
    nll_raw = -np.mean(labels * np.log(np.clip(raw, 1e-6, 1)) + (1 - labels) * np.log(np.clip(1 - raw, 1e-6, 1)))
    nll_cal = -np.mean(labels * np.log(np.clip(cal, 1e-6, 1)) + (1 - labels) * np.log(np.clip(1 - cal, 1e-6, 1)))
    assert nll_cal <= nll_raw + 1e-6


def test_uncertainty_refuses_on_disagreement():
    from sai.calibration import uncertainty
    import numpy as np
    # signals strongly disagree
    scores = np.array([0.95, 0.05, 0.95])
    weights = np.array([1.0, 1.0, 1.0])
    res = uncertainty(scores, weights, calibrated_score=0.65, refuse_threshold=0.3)
    assert res.epistemic_uncertainty > 0.3
    assert res.verdict == "inconclusive"
