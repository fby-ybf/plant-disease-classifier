"""分类模型：随机森林 + 训练 / 评估 / 保存。"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from .dataset import CLASSES
from .features import extract_many

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "clf.pkl"


@dataclass
class TrainResult:
    accuracy: float
    report: str
    confusion: list[list[int]]
    n_train: int
    n_test: int


def build_dataset(images_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    paths, labels = [], []
    for idx, kind in enumerate(CLASSES):
        for p in sorted((images_dir / kind).glob("*.png")):
            paths.append(p)
            labels.append(idx)
    X = extract_many(paths)
    y = np.array(labels)
    return X, y, [str(p) for p in paths]


def train(images_dir: Path, test_size: float = 0.25, seed: int = 42) -> TrainResult:
    X, y, _ = build_dataset(images_dir)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=seed)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    result = TrainResult(
        accuracy=float(accuracy_score(y_test, pred)),
        report=classification_report(y_test, pred, target_names=CLASSES, digits=3),
        confusion=confusion_matrix(y_test, pred).tolist(),
        n_train=len(y_train),
        n_test=len(y_test),
    )
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_bytes(pickle.dumps({"clf": clf, "classes": CLASSES, "features": None}))
    return result


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("模型不存在，请先运行 python -m app.cli train")
    return pickle.loads(MODEL_PATH.read_bytes())


def predict(path: Path) -> dict:
    from .features import extract

    payload = load_model()
    x = extract(path).reshape(1, -1)
    proba = payload["clf"].predict_proba(x)[0]
    idx = int(np.argmax(proba))
    return {
        "label": payload["classes"][idx],
        "confidence": round(float(proba[idx]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(payload["classes"], proba)},
    }
