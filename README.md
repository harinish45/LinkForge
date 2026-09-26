# LinkForge 🔗

> High-Performance Business Entity Resolution System for Amazon ML Challenge 2026

LinkForge is an entity resolution pipeline engineered to link and deduplicate business records arriving from multiple independent, heterogeneous, and noisy data sources (Source 1, Source 2, and Source 3) across multiple countries (US, India, France).

---

## 📌 Problem Overview

In commercial platforms, business entity records arrive from disjoint data streams without common foreign keys or unique identifiers. LinkForge resolves which records across sources refer to the exact same real-world business entity.

- **Reference Source:** Source 1 (`S1-*`)
- **Target Sources:** Source 2 (`S2-*`) and Source 3 (`S3-*`)
- **Key Challenges:**
  - Abbreviations & legal suffix variants (`Corp` vs `Corporation`, `Pvt Ltd`)
  - Noisy address descriptions, landmark references, and component reordering
  - Transliteration discrepancies and missing postal codes
  - Cross-border multi-lingual entity matching (US, India, and France)

---

## 📁 Repository Structure

```text
LinkForge/
├── .gitignore                          # Excludes large TSV datasets & runtime caches
├── README.md                           # Main project documentation
└── student_resource/                   # Challenge resource bundle
    ├── Documentation_template.md       # Solution writeup & methodology template
    ├── README.md                       # Official challenge specification
    ├── utils/
    │   └── validate_submission.py      # Format & constraint verification script
    └── dataset/                        # Datasets (place TSV files here locally)
        ├── train/
        └── test/
```

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup

```bash
# Clone the repository
git clone https://github.com/harinish45/LinkForge.git
cd LinkForge

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Dataset Setup

Place the competition TSV dataset inside `student_resource/dataset/`:
- `student_resource/dataset/train/`
- `student_resource/dataset/test/`

> **Note:** The raw `.tsv` datasets exceed GitHub's file size limit and are kept locally via `.gitignore`.

### 3. Submission Validation

Before submitting any predictions, validate formatting using the provided utility:

```bash
python student_resource/utils/validate_submission.py --submission path/to/submission.tsv --test-source1 student_resource/dataset/test/test_source1.tsv
```

---

## 📄 License & Attribution

Developed for the Amazon ML Challenge 2026.
