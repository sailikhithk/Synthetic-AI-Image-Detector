from sai.signals import FrequencySignal, ReconstructionSignal, NoiseResidualSignal
from tests.conftest import make_real_like, make_ai_like


def test_frequency_signal_separates():
    s = FrequencySignal()
    real = s.analyze(make_real_like(seed=1))
    ai = s.analyze(make_ai_like(seed=1))
    # AI image should score higher (more AI-like) than real
    assert ai.score > real.score, f"AI {ai.score} should > real {real.score}"
    assert 0.0 <= ai.score <= 1.0
    assert 0.0 <= real.score <= 1.0


def test_reconstruction_signal_separates():
    s = ReconstructionSignal()
    real = s.analyze(make_real_like(seed=2))
    ai = s.analyze(make_ai_like(seed=2))
    assert ai.score > real.score, f"AI {ai.score} should > real {real.score}"


def test_noise_signal_separates():
    s = NoiseResidualSignal()
    real = s.analyze(make_real_like(seed=3))
    ai = s.analyze(make_ai_like(seed=3))
    assert ai.score > real.score, f"AI {ai.score} should > real {real.score}"


def test_signal_results_have_features():
    s = FrequencySignal()
    r = s.analyze(make_real_like(seed=4))
    assert "hf_ratio" in r.features
    assert "cross_corr" in r.features
