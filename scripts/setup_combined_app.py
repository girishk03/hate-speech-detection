from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import joblib
import nltk


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "Youtube comment classification" / "models"
TARGET_DIR = ROOT / "combined" / "models"
ARTIFACTS = ("best_model.pkl", "vectorizer.pkl")
NLTK_RESOURCES = {
    "vader_lexicon": "sentiment/vader_lexicon.zip",
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifacts() -> None:
    model = joblib.load(TARGET_DIR / "best_model.pkl")
    vectorizer = joblib.load(TARGET_DIR / "vectorizer.pkl")

    if not callable(getattr(model, "predict", None)):
        raise TypeError("best_model.pkl does not expose predict()")
    if not callable(getattr(vectorizer, "transform", None)):
        raise TypeError("vectorizer.pkl does not expose transform()")


def ensure_nltk_resources() -> None:
    for package, resource_path in NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            if not nltk.download(package, quiet=True):
                raise RuntimeError(f"Unable to download required NLTK resource: {package}")


def setup_combined_app() -> list[Path]:
    missing = [path for name in ARTIFACTS if not (path := SOURCE_DIR / name).is_file()]
    if missing:
        missing_names = ", ".join(str(path.relative_to(ROOT)) for path in missing)
        raise FileNotFoundError(f"Missing source model artifacts: {missing_names}")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ARTIFACTS:
        source = SOURCE_DIR / name
        destination = TARGET_DIR / name
        if not destination.exists() or file_hash(source) != file_hash(destination):
            shutil.copy2(source, destination)
            copied.append(destination)

    validate_artifacts()
    ensure_nltk_resources()
    return copied


if __name__ == "__main__":
    copied_files = setup_combined_app()
    action = "Copied" if copied_files else "Verified"
    print(f"{action} combined app artifacts in {TARGET_DIR.relative_to(ROOT)}")
