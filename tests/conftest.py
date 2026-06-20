from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import joblib
import pytest


ROOT = Path(__file__).resolve().parents[1]
COMBINED_DIR = ROOT / "combined"
MODEL_DIR = ROOT / "Youtube comment classification" / "models"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def comment_processor():
    return load_module("combined_comment_processor", COMBINED_DIR / "comment_processor.py")


@pytest.fixture(scope="session")
def model_and_vectorizer():
    return (
        joblib.load(MODEL_DIR / "best_model.pkl"),
        joblib.load(MODEL_DIR / "vectorizer.pkl"),
    )


@pytest.fixture(scope="session")
def combined_module():
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "setup_combined_app.py")],
        cwd=ROOT,
        check=True,
    )
    sys.path.insert(0, str(COMBINED_DIR))
    try:
        return load_module("combined_app", COMBINED_DIR / "app.py")
    finally:
        sys.path.remove(str(COMBINED_DIR))


@pytest.fixture()
def client(combined_module):
    combined_module.app.config.update(TESTING=True)
    return combined_module.app.test_client()
