---
title: Variant Predictor
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# ⌬ Variant Analyzer.

> **Discover the pathogenic potential of genomic variants with premium, clinical-grade AI.**

![Project Status](https://img.shields.io/badge/Status-Active-7C3AED?style=for-the-badge)
![Model Accuracy](https://img.shields.io/badge/Random_Forest-94.3%25_Accuracy-10B981?style=for-the-badge)
![Tech Stack](https://img.shields.io/badge/Deployed-Hugging_Face-EF4444?style=for-the-badge)

Variant Analyzer isn't just a bioinformatics tool—it's a **sleek, fully-responsive AI copilot** for processing and understanding DNA mutations. 

Crafted with a minimalist, high-contrast aesthetic, it combines deep molecular biology feature extraction with powerful Random Forest ensembles to give you instant, explainable predictions on genomic variant pathogenicity.

---

## ✨ Why Variant Analyzer?

### ⚡ Lightning-Fast Clinical Insight
No clunky workflows. Input a reference DNA sequence, pick your target nucleotide, or enter a direct codon mutation. Get instantaneous, clinical-grade predictions right in your browser.

### 🧬 Deep Biological Feature Arrays
We don't just look at letters. The engine extracts profound biological descriptors in real-time:
* **Physicochemical Amino Acid Substitution Scoring**
* **Reading Frame Preservation (Nonsense/Frameshift detection)**
* **Local GC Content Metrics**
* **Positional Normalization**

### 📊 Explainable AI (XAI) Built-In
Trust the algorithm. Every prediction comes with a dynamic **Feature Importance Matrix** and clear, natural language synthesis explaining *why* a variant was marked Benign or Pathogenic. See the exact percentage weight of the amino acid disruption.

### 💎 "Vero-Class" UI/UX
Bioinformatics doesn't have to be ugly. Experience a completely redesigned, ultra-modern interface:
* **Buttery Smooth Animations**: Hardware-accelerated 60fps entrance sequences and live probability bars.
* **Flawless Formatting**: Crisp *Satoshi* & *JetBrains Mono* typography wrapped in a stark white and deep purple (`#7C3AED`) palette.
* **100% Mobile Responsive**: Analyze variants flawlessly on your desktop or your phone. 

---

## 🚀 Live Demo

**Experience the app instantly, zero setup required:**
👉 **[Launch Variant Analyzer on Hugging Face Spaces](https://huggingface.co/spaces/govind1112/Variant-Analyzer)**

---

## 🛠 For Developers

Want to run the engine yourself or interface with the Flask API? 

<details>
<summary><b>Click to expand local deployment instructions</b></summary>

### 1. Zero-Install Client Side
The entire front-end application contains mirrored client-side JS inference. Just double click `frontend/index.html` on your desktop—no servers required!

### 2. Full-Stack Docker Deployment (Recommended)
This requires Docker installed on your machine.
```bash
# Clone the repository
git clone https://github.com/govindharajan-m/Variant-Predictor.git
cd Variant-Predictor

# Build the container (this handles all dependencies and trains the ML models)
docker build -t variant-analyzer .

# Run the container
docker run -p 7860:7860 variant-analyzer
```
Then open `http://localhost:7860` in your browser.

### 3. Native Python Deployment
```bash
# Install requirements
pip install -r requirements.txt

# Generate the ML models
python train_model.py

# Start the Flask API
python backend/app.py
```

### The API `POST /predict`
Send JSON directly to the engine:
```json
{
  "sequence": "ATGAAAGCAATTTTCGTACTGAAAG...",
  "position": 10,
  "mutated_nucleotide": "A"
}
```
*Returns full pathogenic probabilities, translated sequence diffs, and feature importances.*

</details>

---

*Disclaimer: This tool is a high-fidelity demonstration of machine learning in bioinformatics, trained on synthetic data simulating ClinVar parameters. It is not intended for true clinical diagnostic use.*


## Biological Accuracy Notes

1. **Genetic code**: Standard NCBI genetic code implemented verbatim
2. **Amino acid properties**: Hydrophobicity (Kyte-Doolittle scale), charge, size, polarity
3. **Variant classification**: HGVS-style categories (synonymous, missense, nonsense, frameshift)
4. **Pathogenicity logic**: Informed by ACMG variant classification guidelines (PM1–PP5 criteria)
5. **NMD**: Nonsense-mediated decay predicted for premature stop codons

---

## ⚠️ Disclaimer

This tool is for educational and portfolio demonstration purposes. Predictions are based on a synthetic training dataset and should **not** be used for clinical decision-making. Always consult validated databases (ClinVar, gnomAD) and clinical geneticists for real variant interpretation.
