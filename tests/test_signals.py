from sai.signals import (
    FrequencySignal,
    ReconstructionSignal,
    NoiseResidualSignal,
    MetadataSignal,
    SemanticSignal,
    CrossImageConsistencySignal,
)
from tests.conftest import make_real_like, make_ai_like, make_ai_batch, make_real_batch


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


# --- New signal tests ---


def test_noise_signal_detects_jpeg_blockiness():
    """JPEG-compressed images should have blockiness detected and weight reduced."""
    s = NoiseResidualSignal()
    ai = s.analyze(make_ai_like(seed=5))
    assert "jpeg_blockiness" in ai.features
    assert "is_jpeg_compressed" in ai.features
    assert "weight_adjusted" in ai.features


def test_semantic_signal_separates():
    """SemanticSignal should score AI images higher than real images."""
    s = SemanticSignal()
    real = s.analyze(make_real_like(seed=6))
    ai = s.analyze(make_ai_like(seed=6))
    assert ai.score > real.score, f"AI {ai.score} should > real {real.score}"
    assert "palette_ratio" in real.features
    assert "channel_corr" in real.features
    assert "edge_density" in real.features


def test_semantic_signal_features_present():
    s = SemanticSignal()
    r = s.analyze(make_real_like(seed=7))
    assert "palette_ratio" in r.features
    assert "edge_density" in r.features
    assert "smooth_ratio" in r.features
    assert "channel_corr" in r.features
    assert "gradient_smoothness" in r.features


def test_metadata_signal_no_filepath_neutral():
    """Without a file path, MetadataSignal returns neutral with low weight."""
    s = MetadataSignal()
    import numpy as np

    img = make_real_like(seed=8)
    r = s.analyze(img)
    assert r.score == 0.5
    assert r.weight < 0.2


def test_metadata_signal_detects_no_camera_exif(tmp_path):
    """A synthetic image saved as JPEG should have no camera EXIF."""
    import numpy as np
    from PIL import Image

    img = make_ai_like(seed=9)
    path = tmp_path / "synthetic.jpg"
    Image.fromarray(img).save(path, "JPEG")

    s = MetadataSignal()
    r = s.analyze_file(str(path))
    # No camera EXIF -> should lean toward AI
    assert r.score > 0.5, f"Synthetic image should score > 0.5, got {r.score}"
    assert r.features["has_camera_exif"] is False


def test_metadata_signal_detects_progressive_jpeg(tmp_path):
    """Progressive JPEG without camera EXIF should score high for AI."""
    import numpy as np
    from PIL import Image

    img = make_ai_like(seed=10)
    path = tmp_path / "progressive.jpg"
    Image.fromarray(img).save(path, "JPEG", progressive=True)

    s = MetadataSignal()
    r = s.analyze_file(str(path))
    assert r.features["is_progressive"] is True
    assert r.score > 0.6


def test_cross_image_single_image_neutral():
    """Single image analysis should return neutral with low weight."""
    s = CrossImageConsistencySignal()
    import numpy as np

    r = s.analyze(make_real_like(seed=11))
    assert r.score == 0.5
    assert r.weight < 0.1


def test_cross_image_batch_ai_uniform():
    """AI batch (uniform images) should score higher than real batch."""
    s = CrossImageConsistencySignal()
    ai_batch = make_ai_batch(n=5, seed=20)
    real_batch = make_real_batch(n=5, seed=20)

    ai_results = s.analyze_batch(ai_batch)
    real_results = s.analyze_batch(real_batch)

    ai_score = ai_results[0].score
    real_score = real_results[0].score
    assert ai_score > real_score, f"AI batch {ai_score} should > real batch {real_score}"
    assert "batch_size" in ai_results[0].features
    assert "batch_color_std" in ai_results[0].features
    assert "batch_hist_corr" in ai_results[0].features


def test_cross_image_batch_size_in_features():
    s = CrossImageConsistencySignal()
    batch = make_ai_batch(n=3, seed=30)
    results = s.analyze_batch(batch)
    for r in results:
        assert r.features["batch_size"] == 3
