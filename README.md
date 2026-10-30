<div align="center">

# 🔗 LinkForge (EntityLink AI)

**High-Performance ML-Powered Business Entity Resolution**

[![CI](https://github.com/harinish45/LinkForge/actions/workflows/ci.yml/badge.svg)](https://github.com/harinish45/LinkForge/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)

> *High-performance, scalable entity resolution and link analysis system — Amazon ML Challenge 2026*

[📋 TODO](TODO.md) · [📓 Notebooks](notebooks/) · [📚 Docs](docs/)

</div>

---

## 🏗️ Architecture

```mermaid
graph LR
    subgraph Input["📥 Input"]
        Raw["Raw Entity\nRecords (CSV/JSON)"]
        URLs["URLs /\nDomain Names"]
    end

    subgraph Pipeline["⚙️ ML Pipeline"]
        Extract["🔤 Feature\nExtractor"]
        Embed["🧠 Embedding\nModel (SBERT)"]
        Dedup["🔀 Deduplication\n(Blocking + Matching)"]
        Score["📊 Similarity\nScorer"]
        Cluster["🗂️ Entity\nClustering"]
    end

    subgraph API["🌐 FastAPI"]
        Single["POST /resolve\n(single entity)"]
        Batch["POST /resolve/batch\n(bulk CSV)"]
        Health["GET /health"]
    end

    subgraph Output["📤 Output"]
        Linked["✅ Linked\nEntity Groups"]
        Canonical["📋 Canonical\nRecord"]
        Conf["📊 Confidence\nScore"]
    end

    Raw --> Extract
    URLs --> Extract
    Extract --> Embed
    Embed --> Dedup
    Dedup --> Score
    Score --> Cluster
    Cluster --> Linked
    Cluster --> Canonical
    Cluster --> Conf
    API --> Pipeline
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/harinish45/LinkForge.git
cd LinkForge
pip install -r requirements.txt
python -m src.api.main   # → http://localhost:8000
```

---

## 📁 Structure

```
LinkForge/
├── 🐍  src/
│   ├── models/        # ML models
│   ├── api/           # FastAPI endpoints
│   └── processors/    # Link analysis pipeline
├── 📓  notebooks/     # Research notebooks
├── 🧪  tests/         # Test suite
├── ⚙️  configs/       # Configuration
└── 📚  docs/          # Documentation
```

---

<div align="center">
<a href="LICENSE">MIT License</a> · Built by harinish45
</div>