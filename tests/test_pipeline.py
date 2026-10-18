"""数据生成 / 特征 / 训练评估 / 预测 测试。"""
from pathlib import Path

import numpy as np
import pytest

from app.dataset import CLASSES, generate_dataset, generate_image
from app.features import FEATURE_NAMES, extract
from app import model


@pytest.fixture(scope="module")
def images(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("images")
    generate_dataset(out, per_class=30, seed=7)
    return out


def test_generated_images_exist(images: Path) -> None:
    for kind in CLASSES:
        files = list((images / kind).glob("*.png"))
        assert len(files) == 30
        assert files[0].stat().st_size > 0


def test_generate_image_differs_between_classes() -> None:
    healthy = np.asarray(generate_image("healthy")).astype("int16")
    rust = np.asarray(generate_image("leaf_rust")).astype("int16")
    assert np.abs(healthy - rust).mean() > 1.0


def test_feature_vector_shape(images: Path) -> None:
    vec = extract(next((images / "healthy").glob("*.png")))
    assert vec.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(vec).all()


def test_train_reaches_expected_accuracy(images: Path) -> None:
    result = model.train(images)
    assert result.accuracy > 0.9
    assert len(result.confusion) == len(CLASSES)
    assert result.n_train + result.n_test == 90


def test_predict_returns_label_and_confidence(images: Path) -> None:
    model.train(images)
    out = model.predict(next((images / "powdery_mildew").glob("*.png")))
    assert out["label"] in CLASSES
    assert 0.0 <= out["confidence"] <= 1.0
    assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-3
