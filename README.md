# Hate Speech Detection and Moderation Assistant

[![CI](https://github.com/girishk03/hate-speech-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/girishk03/hate-speech-detection/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3-black?logo=flask)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Project Overview

This repository explores assisted text moderation through two Flask applications:

- a YouTube comment classifier built with TF-IDF features and a LinearSVC model; and
- a Socket.IO chatroom that detects strongly negative messages and offers more polite wording.

The trained classifier predicts the repository's three dataset labels—`positive`, `negative`, and `neutral`. These labels are useful moderation signals, but they are not a complete or authoritative definition of hate speech.

## Problem Statement

Online communities need scalable ways to prioritize potentially harmful content without treating an automated prediction as a final moderation decision. This project demonstrates how classical NLP, lightweight web services, and human-facing interventions can support that workflow while keeping the system's limitations visible.

## Use Cases

- Review sentiment-style classifications across public YouTube comments.
- Surface comments that may deserve closer moderator attention.
- Demonstrate real-time message intervention in a Socket.IO chatroom.
- Compare a trained text classifier with a separate rules-and-sentiment moderation approach.

## Live Demo

[Open the combined Flask application](https://hate-speech-detection-zqjy.onrender.com)

The link is the deployment URL recorded by this repository. Availability is not guaranteed; free-tier services may sleep, restart, or take time to respond.

## Screenshots

### YouTube Comment Classifier

![YouTube analysis home](docs/screenshots/01-youtube-home.png)

![Example YouTube analysis results](docs/screenshots/02-youtube-results.png)

![Classified YouTube comments](docs/screenshots/04-comment-list.png)

The values visible in screenshots describe that captured run only; they are not repository-wide performance metrics.

### AI Polite Chatroom

![AI polite chatroom](docs/screenshots/07-chatroom-home.png)

![Polite wording suggestion](docs/screenshots/09-polite-conversion.png)

![Converted chatroom message](docs/screenshots/08-chatroom-result.png)

## Architecture

The YouTube classifier and chatroom share a Flask interface but use different moderation paths. The classifier uses the saved ML pipeline; the chatroom uses VADER sentiment and phrase substitutions.

```mermaid
flowchart LR
    User["User"] --> UI["Flask UI"]
    UI --> Input["YouTube comments"]
    Input --> Prep["Preprocessing"]
    Prep --> TFIDF["TF-IDF vectorizer"]
    TFIDF --> SVC["LinearSVC model"]
    SVC --> Prediction["Prediction"]
    Prediction --> Results["YouTube results"]
    UI --> Message["Chat message"]
    Message --> VADER["VADER sentiment and phrase rules"]
    VADER --> Moderation["Chatroom moderation"]
```

## Repository Structure

```text
hate-speech-detection/
├── .github/workflows/ci.yml          # GitHub Actions import smoke checks
├── Chatroom/                         # Standalone Socket.IO polite-chat application
├── Youtube comment classification/  # Training, inference, dataset, and standalone UI
│   ├── Datasets/                     # CSV data used by training scripts
│   └── models/                       # Saved vectorizer and classifier artifacts
├── combined/                         # Combined deployable Flask application
├── data/                             # Additional repository data assets
├── docs/screenshots/                 # README screenshots
├── LICENSE                           # MIT license
└── README.md
```

The directory with spaces is retained for compatibility with existing scripts and CI. Quote the path in shell commands.

## Dataset

The training script reads `Youtube comment classification/Datasets/ytfinal.csv`.

| Property | Verified value | Notes |
|---|---:|---|
| Raw rows | 32,120 | Count in `ytfinal.csv` |
| Usable rows | 32,119 | After dropping missing or empty processed text |
| Classes | `positive`, `negative`, `neutral` | Sentiment-style labels, not explicit hate-speech categories |
| Training samples | 25,695 | Stratified 80% split |
| Test samples | 6,424 | Stratified 20% split |
| Split seed | 42 | Set in `model.py` |

The repository also contains `ytdata.csv` and an augmentation script using WordNet substitutions. However, it does not document the original external dataset source, license, or a reproducible provenance chain from `ytdata.csv` to `ytfinal.csv`. Verify those details before redistributing the data or using it in production.

## Preprocessing

`model.py` applies the following transformations before training and inference:

1. Convert text to lowercase.
2. Remove URLs, user mentions, and hashtag markers.
3. Remove non-alphabetic characters.
4. Collapse repeated whitespace.
5. Drop empty processed rows.
6. Transform text with TF-IDF word unigrams and bigrams, English stop words, and a 5,000-feature limit.

## Model Training

The training script compares Logistic Regression, Multinomial Naive Bayes, and LinearSVC on the same stratified split. It selects the model with the highest test accuracy, then stores the selected classifier and fitted vectorizer as Joblib artifacts.

```bash
cd "Youtube comment classification"
python model.py
```

Training overwrites files under `models/`. Review data provenance and evaluation output before publishing newly generated artifacts.

## Model Performance

The checked-in data and training configuration reproduce LinearSVC as the best evaluated model. The figures below are rounded from a local rerun of the code path in `model.py`; they describe this specific held-out split, not expected production performance.

## Evaluation Metrics

| Metric | Value | Notes |
|---|---:|---|
| Accuracy | 78.50% | 5,043 correct predictions across 6,424 test samples |
| Macro F1 | 0.7832 | Gives equal weight to each class |
| Weighted F1 | 0.7845 | Weights each class by its test support |
| Test samples | 6,424 | Stratified 20% holdout with `random_state=42` |

These metrics evaluate `positive`, `negative`, and `neutral` classification. They do not directly measure hate-speech recall, demographic fairness, calibration, or moderation safety.

## How the Chatroom Moderator Works

The standalone and combined chatroom modules do not use the LinearSVC classifier. They:

1. receive messages over Flask-SocketIO;
2. score message sentiment with NLTK VADER;
3. identify sufficiently negative messages using a configured threshold or phrase rules;
4. replace selected terms with a randomly chosen polite alternative; and
5. return or broadcast the original message with the suggested wording.

This mechanism is a demonstration of conversational intervention, not a semantic rewrite system. It can miss harmful language and can change tone without preserving intent.

## Quick Start

```bash
git clone https://github.com/girishk03/hate-speech-detection.git
cd hate-speech-detection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r combined/requirements.txt
python combined/app.py
```

Open `http://127.0.0.1:5000`. The combined app exposes the YouTube interface at `/youtube` and chatroom at `/chatroom`.

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

## Setup Guide

Use Python 3.11, matching the existing GitHub Actions workflow and deployment runtime. The repository has module-specific dependency files rather than one root requirements file:

- `combined/requirements.txt` for the integrated application;
- `Youtube comment classification/requirements.txt` for the standalone classifier; and
- `Chatroom/requirements.txt` for the standalone chatroom.

The YouTube feature downloads public comments and therefore requires network access. No API key is required by the current downloader implementation.

## Run YouTube Comment Classifier

```bash
source .venv/bin/activate
pip install -r "Youtube comment classification/requirements.txt"
cd "Youtube comment classification"
python app.py
```

Open `http://127.0.0.1:5001`.

## Run AI Polite Chatroom

```bash
source .venv/bin/activate
pip install -r Chatroom/requirements.txt
cd Chatroom
python app.py
```

Open `http://127.0.0.1:5000`.

## Testing

The repository does not currently contain a pytest test suite. `Youtube comment classification/tmp_test_comments.py` is an exploratory script, not an automated test module, and should not be presented as test coverage.

The existing CI performs application import smoke checks. Run equivalent checks locally after installing the relevant requirements:

```bash
(cd "Youtube comment classification" && python -c "import app; print('YouTube app import OK')")
(cd Chatroom && python -c "import app; print('Chatroom app import OK')")
```

Recommended future tests include `tests/test_preprocessing.py`, `tests/test_model_inference.py`, `tests/test_chatroom_moderation.py`, and `tests/test_routes.py`.

## CI/CD

`.github/workflows/ci.yml` runs on pushes and pull requests targeting `main` with Python 3.11. It installs each standalone module's dependencies and checks that both Flask applications import successfully.

This is smoke validation only: the workflow does not run pytest, retrain the model, validate metric thresholds, or deploy the application. The Render configuration in `combined/` provides the deployment entry point, but deployment automation is not defined in this repository.

## Limitations

- Sarcasm, irony, oblique abuse, and reclaimed language may be misclassified.
- The English-oriented cleaner and vectorizer may fail on code-mixed or multilingual text.
- Sentiment labels are only indirect moderation signals and can produce false positives or false negatives.
- LinearSVC decision scores are not calibrated probabilities; confidence-like UI values should not be interpreted as probabilities without calibration.
- Keyword and phrase substitution can miss context or produce awkward rewrites.
- Dataset origin and demographic representation are not documented, limiting bias analysis.
- The system is not suitable as the final moderation authority without human review and an appeals process.

## Ethics and Responsible Use

This project is intended to assist moderation and learning, not automate censorship. A real deployment should keep trained reviewers in the decision loop, provide explanations and appeal routes, monitor false-positive rates across communities and language varieties, and document retention and privacy practices.

Training data can encode cultural, demographic, and annotation bias. Evaluate the model on representative data before use, and do not infer a person's character or intent from one classification.

## Future Improvements

- Document and version the dataset source, license, and transformation lineage.
- Replace sentiment-style labels with a clearly defined moderation taxonomy.
- Add pytest coverage for preprocessing, inference, routes, and Socket.IO events.
- Calibrate classifier scores and report per-class confusion matrices.
- Evaluate code-mixed and multilingual language performance.
- Add threshold configuration, moderator feedback, and audit logging.
- Rename the spaced module directory in a coordinated compatibility change.
- Make dependency installation deterministic and remove permissive CI fallbacks.

## License

This project is available under the [MIT License](LICENSE).
