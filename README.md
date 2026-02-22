# 🧬 Genomic Variant Pathogenicity Predictor

An AI-powered full-stack bioinformatics platform that predicts whether a genomic variant is **benign** or **pathogenic** using biologically-informed machine learning.

> **Model accuracy: 94.3% (Random Forest) | 92.8% (Logistic Regression)**

---

## 📋 Overview

This tool simulates real-world clinical variant interpretation workflows (similar to ClinVar-style logic), combining:

- **Biological feature engineering** from molecular biology principles
- **Random Forest + Logistic Regression** ensemble ML
- **Explainable AI** with feature importance and human-readable interpretation
- **Protein impact visualization** with highlighted amino acid changes
- **Full-stack deployment** via Flask API + standalone HTML frontend

---

## 🗂️ Project Structure

```
genomic-predictor/
├── data/
│   └── synthetic_variants.csv      # Generated training dataset (3,000 samples)
├── ml_model/
│   ├── model_rf.pkl                # Trained Random Forest (primary)
│   ├── model_lr.pkl                # Logistic Regression (baseline)
│   ├── scaler.pkl                  # Feature scaler for LR
│   └── model_meta.json             # Accuracy, feature importances, metadata
├── backend/
│   └── app.py                      # Flask REST API
├── frontend/
│   └── index.html                  # Single-file interactive web app
├── utils/
│   └── bio_utils.py                # Genetic code, translation, feature engineering
├── train_model.py                  # ML training pipeline
└── README.md
```

---

## 🧪 Biological Logic

### Feature Engineering

| Feature | Description | Biological Relevance |
|---|---|---|
| `aa_substitution_score` | Physicochemical distance between ref/alt AAs | Protein structure disruption |
| `variant_type_*` | One-hot: synonymous/missense/nonsense/frameshift | Functional consequence category |
| `stop_gained` | Binary: stop codon introduced | NMD or truncated protein |
| `gc_content` | GC% of sequence | mRNA stability, splice signals |
| `position_normalized` | Position / seq_length | Start region = more critical |

### Pathogenicity Rules (Labeling Logic)

```
Frameshift         → P(pathogenic) = 0.75–1.00  (destroys reading frame)
Nonsense/Stop gain → P(pathogenic) = 0.70–0.95  (premature termination)
Radical missense   → P(pathogenic) = 0.45–0.85  (physicochemical change)
Conservative miss. → P(pathogenic) = 0.15–0.55  (limited structural impact)
Synonymous         → P(pathogenic) = 0.02–0.20  (no AA change)
```

---

## 🚀 Quick Start

### 1. Option A — Frontend Only (No Server)

Just open `frontend/index.html` in your browser. The frontend includes a full client-side ML inference engine that mirrors the trained model.

### 2. Option B — Full Stack (Backend + Frontend)

**Install dependencies:**
```bash
pip install scikit-learn pandas numpy flask flask-cors joblib
```

**Train the model:**
```bash
cd genomic-predictor
python train_model.py
```

**Start the API:**
```bash
python backend/app.py
# → Running on http://localhost:5050
```

**Open frontend:**
```bash
open frontend/index.html
```

---

## 🔌 API Reference

### `GET /health`
```json
{"status": "ok", "model_version": "1.0.0"}
```

### `GET /model/info`
Returns model accuracy and feature importances.

### `POST /predict`
Predict from DNA sequence + mutation.

**Request:**
```json
{
  "sequence": "ATGAAAGCAATTTTCGTACTGAAAGGTTTTGTT...",
  "position": 10,
  "mutated_nucleotide": "A"
}
```

**Response:**
```json
{
  "prediction": 1,
  "classification": "Pathogenic",
  "probability_pathogenic": 0.873,
  "probability_benign": 0.127,
  "lr_probability_pathogenic": 0.801,
  "variant_summary": {
    "reference_codon": "GCA",
    "alternate_codon": "TAA",
    "reference_amino_acid": "A",
    "alternate_amino_acid": "*",
    "variant_type": "nonsense",
    "stop_gained": true,
    "gc_content": 0.397,
    "aa_substitution_score": 1.0
  },
  "feature_importances": {...},
  "protein_sequences": {
    "reference": "MKAIFVLKGFVGFLEIAККDNTKA",
    "mutated": "MKAIFVLK*",
    "mutation_aa_position": 3
  },
  "interpretation": "This variant is predicted PATHOGENIC with high confidence..."
}
```

### `POST /predict/codon`
Predict from direct codon input.

**Request:**
```json
{
  "ref_codon": "GAG",
  "alt_codon": "GTG",
  "gc_content": 0.52,
  "position": 50,
  "seq_length": 300
}
```

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Random Forest | **94.3%** | 0.96/0.92 | 0.94/0.95 | 0.95/0.94 |
| Logistic Regression | 92.8% | 0.92/0.93 | 0.95/0.90 | 0.94/0.92 |

*5-fold CV: 93.5% ± 1.0%*

### Feature Importance (Random Forest)

```
aa_substitution_score    ████████████████████████  56.7%
variant_type_synonymous  ███████                   16.4%
variant_type_frameshift  ███                        6.8%
position_normalized      ██                         5.0%
gc_content               ██                         4.7%
stop_gained              █                          3.9%
variant_type_missense    █                          3.6%
variant_type_nonsense    █                          2.9%
```

---

## 🎓 Skills Demonstrated

- **Machine Learning**: Random Forest, Logistic Regression, cross-validation, feature importance
- **Bioinformatics**: Genetic code, codon translation, amino acid properties, variant classification
- **Feature Engineering**: Physicochemical scoring, one-hot encoding, normalization
- **Full-Stack Development**: REST API (Flask) + interactive HTML/JS frontend
- **Explainable AI**: Feature attribution + natural language interpretation

---

## 🔬 Biological Accuracy Notes

1. **Genetic code**: Standard NCBI genetic code implemented verbatim
2. **Amino acid properties**: Hydrophobicity (Kyte-Doolittle scale), charge, size, polarity
3. **Variant classification**: HGVS-style categories (synonymous, missense, nonsense, frameshift)
4. **Pathogenicity logic**: Informed by ACMG variant classification guidelines (PM1–PP5 criteria)
5. **NMD**: Nonsense-mediated decay predicted for premature stop codons

---

## ⚠️ Disclaimer

This tool is for educational and portfolio demonstration purposes. Predictions are based on a synthetic training dataset and should **not** be used for clinical decision-making. Always consult validated databases (ClinVar, gnomAD) and clinical geneticists for real variant interpretation.
