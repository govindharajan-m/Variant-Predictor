"""
train_model.py - ML pipeline for Genomic Variant Pathogenicity Predictor.

Generates a synthetic but biologically-informed training dataset,
trains Random Forest + Logistic Regression classifiers, evaluates them,
and saves the best model for inference.

Biological labeling logic:
  - Frameshift / nonsense → highly pathogenic
  - Stop codon gained → higher pathogenic probability
  - Radical missense (high aa_score) → moderately pathogenic
  - Synonymous → mostly benign
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

from bio_utils import (
    GENETIC_CODE, translate_codon, gc_content,
    amino_acid_substitution_score, classify_variant,
    build_feature_vector, FEATURE_NAMES
)


# ─────────────────────────────────────────────
# 1. Synthetic Dataset Generation
# ─────────────────────────────────────────────

def generate_synthetic_dataset(n_samples: int = 3000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic variant dataset with biologically-informed labels.
    Simulates realistic distributions of variant types and pathogenicity.
    """
    rng = np.random.default_rng(seed)
    all_codons = [c for c in GENETIC_CODE if c not in ('TAA', 'TAG', 'TGA')]
    stop_codons = ['TAA', 'TAG', 'TGA']

    records = []

    for i in range(n_samples):
        # Pick a random reference codon (non-stop)
        ref_codon = all_codons[rng.integers(0, len(all_codons))]
        ref_aa = translate_codon(ref_codon)

        # Randomly pick variant scenario with realistic distribution
        scenario = rng.choice(
            ['synonymous', 'missense_conservative', 'missense_radical',
             'nonsense', 'frameshift'],
            p=[0.30, 0.25, 0.25, 0.10, 0.10]
        )

        if scenario == 'synonymous':
            # Find synonymous codon
            synonyms = [c for c, a in GENETIC_CODE.items()
                        if a == ref_aa and c != ref_codon and c not in stop_codons]
            alt_codon = synonyms[rng.integers(0, len(synonyms))] if synonyms else ref_codon
            variant_type = 'synonymous'

        elif scenario == 'missense_conservative':
            # Find non-synonymous, non-stop codon with low aa_score
            candidates = [c for c, a in GENETIC_CODE.items()
                          if a != ref_aa and a != '*']
            scored = [(c, amino_acid_substitution_score(ref_aa, GENETIC_CODE[c]))
                      for c in candidates]
            conservative = [(c, s) for c, s in scored if s < 0.35]
            if conservative:
                alt_codon = conservative[rng.integers(0, len(conservative))][0]
            else:
                alt_codon = candidates[rng.integers(0, len(candidates))]
            variant_type = 'missense'

        elif scenario == 'missense_radical':
            candidates = [c for c, a in GENETIC_CODE.items()
                          if a != ref_aa and a != '*']
            scored = [(c, amino_acid_substitution_score(ref_aa, GENETIC_CODE[c]))
                      for c in candidates]
            radical = [(c, s) for c, s in scored if s >= 0.35]
            if radical:
                alt_codon = radical[rng.integers(0, len(radical))][0]
            else:
                alt_codon = candidates[rng.integers(0, len(candidates))]
            variant_type = 'missense'

        elif scenario == 'nonsense':
            alt_codon = stop_codons[rng.integers(0, 3)]
            variant_type = 'nonsense'

        else:  # frameshift
            # Simulate frameshift: modify codon length artificially
            alt_codon = ref_codon[:2]  # deletion of 1 bp (not a valid codon = frameshift)
            variant_type = 'frameshift'

        # Compute features
        alt_aa = translate_codon(alt_codon) if len(alt_codon) == 3 else 'fs'
        stop_gained = 1 if (alt_aa == '*' and ref_aa != '*') else 0
        aa_score = (
            amino_acid_substitution_score(ref_aa, alt_aa)
            if len(alt_codon) == 3 else 1.0
        )
        seq_len = rng.integers(100, 2000)
        position = rng.integers(1, seq_len)

        # Generate a dummy sequence to compute GC content
        dummy_seq = ''.join(rng.choice(list('ATGC'), size=seq_len))
        gc = gc_content(dummy_seq)

        pos_normalized = position / seq_len

        # ── Biological labeling logic ──────────────────────────────
        # Base pathogenic probability from variant type
        if variant_type == 'frameshift':
            base_prob = rng.uniform(0.75, 1.0)
        elif variant_type == 'nonsense':
            base_prob = rng.uniform(0.70, 0.95)
        elif variant_type == 'missense' and aa_score >= 0.35:
            base_prob = rng.uniform(0.45, 0.85)
        elif variant_type == 'missense' and aa_score < 0.35:
            base_prob = rng.uniform(0.15, 0.55)
        else:  # synonymous
            base_prob = rng.uniform(0.02, 0.20)

        # Modifiers
        if stop_gained:
            base_prob = min(base_prob + 0.15, 1.0)
        if gc > 0.65 or gc < 0.35:  # extreme GC content can affect splicing
            base_prob = min(base_prob + 0.05, 1.0)
        if pos_normalized < 0.1:  # start region mutations often deleterious
            base_prob = min(base_prob + 0.05, 1.0)

        label = 1 if base_prob >= 0.5 else 0

        records.append({
            'variant_id': f'VAR_{i:05d}',
            'ref_codon': ref_codon,
            'alt_codon': alt_codon,
            'ref_aa': ref_aa,
            'alt_aa': alt_aa,
            'aa_substitution_score': round(aa_score, 4),
            'gc_content': round(gc, 4),
            'position': position,
            'seq_length': seq_len,
            'position_normalized': round(pos_normalized, 4),
            'stop_gained': stop_gained,
            'variant_type': variant_type,
            'variant_type_synonymous': 1 if variant_type == 'synonymous' else 0,
            'variant_type_missense': 1 if variant_type == 'missense' else 0,
            'variant_type_nonsense': 1 if variant_type == 'nonsense' else 0,
            'variant_type_frameshift': 1 if variant_type == 'frameshift' else 0,
            'label': label,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# 2. Train Models
# ─────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame):
    """Train RF and LR models, evaluate, return the better one."""

    X = df[FEATURE_NAMES].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Logistic Regression (baseline) ──────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict(X_test_scaled)
    lr_acc = accuracy_score(y_test, lr_preds)

    print("\n═══════════════════════════════════════")
    print("  LOGISTIC REGRESSION (Baseline)")
    print(f"  Accuracy: {lr_acc:.4f}")
    print("═══════════════════════════════════════")
    print(classification_report(y_test, lr_preds, target_names=['Benign', 'Pathogenic']))

    # ── Random Forest (primary) ──────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1,
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)

    print("\n═══════════════════════════════════════")
    print("  RANDOM FOREST (Primary)")
    print(f"  Accuracy: {rf_acc:.4f}")
    print("═══════════════════════════════════════")
    print(classification_report(y_test, rf_preds, target_names=['Benign', 'Pathogenic']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, rf_preds))

    # Cross-validation
    cv_scores = cross_val_score(rf, X, y, cv=5)
    print(f"\nRF 5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importances
    importances = dict(zip(FEATURE_NAMES, rf.feature_importances_))
    print("\nFeature Importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        bar = '█' * int(imp * 40)
        print(f"  {feat:<35} {imp:.4f} {bar}")

    return rf, lr, scaler, importances, rf_acc, lr_acc


# ─────────────────────────────────────────────
# 3. Save Artifacts
# ─────────────────────────────────────────────

def save_artifacts(rf, lr, scaler, importances, rf_acc, lr_acc):
    os.makedirs('ml_model', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    joblib.dump(rf, 'ml_model/model_rf.pkl')
    joblib.dump(lr, 'ml_model/model_lr.pkl')
    joblib.dump(scaler, 'ml_model/scaler.pkl')

    meta = {
        'feature_names': FEATURE_NAMES,
        'rf_accuracy': round(rf_acc, 4),
        'lr_accuracy': round(lr_acc, 4),
        'feature_importances': {k: round(float(v), 6) for k, v in importances.items()},
        'model_version': '1.0.0',
    }
    with open('ml_model/model_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print("\n✓ Models saved to ml_model/")
    print(f"  - ml_model/model_rf.pkl  (RF accuracy: {rf_acc:.4f})")
    print(f"  - ml_model/model_lr.pkl  (LR accuracy: {lr_acc:.4f})")
    print("  - ml_model/scaler.pkl")
    print("  - ml_model/model_meta.json")


# ─────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("🧬 Genomic Variant Pathogenicity Predictor - ML Training Pipeline")
    print("=" * 60)

    print("\n[1/3] Generating synthetic training dataset (n=3000)...")
    df = generate_synthetic_dataset(n_samples=3000)
    
    # Ensure data directory exists before saving
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/synthetic_variants.csv', index=False)
    print(f"  Dataset saved → data/synthetic_variants.csv")
    print(f"  Class distribution: {dict(df['label'].value_counts())}")
    print(f"  Variant types:\n{df['variant_type'].value_counts().to_string()}")

    print("\n[2/3] Training models...")
    rf, lr, scaler, importances, rf_acc, lr_acc = train_and_evaluate(df)

    print("\n[3/3] Saving model artifacts...")
    save_artifacts(rf, lr, scaler, importances, rf_acc, lr_acc)

    print("\n✅ Training complete!")
