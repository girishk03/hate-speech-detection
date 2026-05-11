# Hate Speech Detection Suite
[![CI](https://github.com/girishk03/hate-speech-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/girishk03/hate-speech-detection/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask-black.svg)](https://flask.palletsprojects.com/)
[![Model](https://img.shields.io/badge/Model-TF--IDF%20%2B%20LinearSVC-success.svg)](#modeling-approach)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

Production-style NLP project for detecting toxic/hate speech across two real-world channels:

- **YouTube Comment Intelligence**: batch + interactive comment analysis
- **Chatroom Moderation**: near real-time toxicity/hate flagging

---

## Table of Contents
- [Highlights](#highlights)
- [System Architecture](#system-architecture)
- [Repository Layout](#repository-layout)
- [Modeling Approach](#modeling-approach)
- [Results](#results)
- [Quickstart](#quickstart)
- [Run Applications](#run-applications)
- [API/Socket Events (Overview)](#apisocket-events-overview)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## Highlights

- End-to-end hate/toxicity detection workflow
- TF-IDF + LinearSVC pipeline with persisted artifacts
- Flask + Flask-SocketIO apps
- Structured data + model folders for reproducibility
- GitHub Actions CI integrated

---

## System Architecture

```mermaid
flowchart LR
    A[Raw Text / YouTube Comments] --> B[Preprocessing]
    B --> C[TF-IDF Vectorization]
    C --> D[Classifier: LinearSVC]
    D --> E[Prediction + Confidence Signals]
    E --> F1[YouTube Web App]
    E --> F2[Chatroom Moderation App]
