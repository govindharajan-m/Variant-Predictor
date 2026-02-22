"""
backend/app.py - Flask REST API for Genomic Variant Pathogenicity Predictor.

Endpoints:
  POST /predict        - Full prediction from DNA sequence + mutation details
  POST /predict/codon  - Prediction from direct codon input
  GET  /health         - Health check
  GET  /model/info     - Model metadata
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import joblib
from flask import Flask, request, jsonify, send_from_directory

from bio_utils import (
    apply_mutation, translate_sequence, gc_content,
    build_feature_vector, classify_variant, translate_codon,
    amino_acid_substitution_score, FEATURE_NAMES
)

app = Flask(__name__)


# ── Load model artifacts ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'ml_model')

rf_model = joblib.load(os.path.join(MODEL_DIR, 'model_rf.pkl'))
lr_model = joblib.load(os.path.join(MODEL_DIR, 'model_lr.pkl'))
scaler   = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))

with open(os.path.join(MODEL_DIR, 'model_meta.json')) as f:
    model_meta = json.load(f)


# ── Helper: Generate biological interpretation ───────────────────────────────

def generate_interpretation(features: dict, prediction: int, probability: float) -> str:
    """Generate a human-readable biological interpretation of the prediction."""
    variant_type = features['_variant_type']
    stop_gained  = features['stop_gained']
    aa_score     = features['aa_substitution_score']
    ref_aa       = features['_ref_aa']
    alt_aa       = features['_alt_aa']
    ref_codon    = features['_ref_codon']
    alt_codon    = features['_alt_codon']
    gc           = features['gc_content']

    reasons = []
    mitigating = []

    # Variant type impact
    if variant_type == 'frameshift':
        reasons.append(
            "the variant introduces a frameshift, disrupting the entire downstream reading "
            "frame and likely producing a non-functional truncated or aberrant protein"
        )
    elif variant_type == 'nonsense' or stop_gained:
        reasons.append(
            f"a premature stop codon is introduced ({ref_codon}→{alt_codon}), leading to "
            "protein truncation via nonsense-mediated decay (NMD) or production of a "
            "truncated non-functional protein"
        )
    elif variant_type == 'missense':
        aa_label = "radical" if aa_score >= 0.4 else "moderate" if aa_score >= 0.2 else "conservative"
        reasons.append(
            f"the {aa_label} amino acid substitution {ref_aa}→{alt_aa} (score: {aa_score:.2f}) "
            f"alters the physicochemical properties of the protein, potentially affecting "
            f"structure, folding, or binding function"
        )
        if aa_score < 0.2:
            mitigating.append(
                "the amino acid change is chemically conservative, suggesting limited "
                "structural disruption"
            )
    elif variant_type == 'synonymous':
        mitigating.append(
            "the variant is synonymous — the amino acid sequence is unchanged — "
            "though rare splicing or regulatory effects cannot be excluded"
        )

    # GC content modifier
    if gc > 0.65:
        reasons.append(
            f"the high GC content ({gc:.0%}) may affect local chromatin accessibility "
            f"or create cryptic splice sites"
        )
    elif gc < 0.35:
        reasons.append(
            f"the low GC content ({gc:.0%}) in this region may affect mRNA stability"
        )

    # Assemble
    class_label = "PATHOGENIC" if prediction == 1 else "BENIGN"
    conf_desc = (
        "high confidence" if probability > 0.85 or probability < 0.15
        else "moderate confidence" if probability > 0.70 or probability < 0.30
        else "low confidence (variant of uncertain significance)"
    )

    if prediction == 1:
        main = (
            f"This variant is predicted {class_label} with {conf_desc} "
            f"(probability: {probability:.2%}). "
        )
        if reasons:
            main += f"Key pathogenic factors: {'; '.join(reasons)}."
        if mitigating:
            main += f" Note: {'; '.join(mitigating)}."
    else:
        main = (
            f"This variant is predicted {class_label} with {conf_desc} "
            f"(probability of pathogenicity: {probability:.2%}). "
        )
        if mitigating:
            main += f"Supporting evidence: {'; '.join(mitigating)}."
        if reasons:
            main += f" However, consider: {'; '.join(reasons)}."

    main += (
        " This prediction is based on a machine learning model trained on simulated "
        "variant data and should be interpreted alongside experimental evidence and "
        "population databases (e.g., ClinVar, gnomAD) in a clinical context."
    )
    return main


# ── Helper: Run inference ────────────────────────────────────────────────────

def run_inference(features: dict) -> dict:
    """Extract feature vector and run both models."""
    X = np.array([[features[f] for f in FEATURE_NAMES]])
    X_scaled = scaler.transform(X)

    rf_prob = rf_model.predict_proba(X)[0]
    lr_prob = lr_model.predict_proba(X_scaled)[0]

    rf_pred = int(rf_model.predict(X)[0])
    prob_pathogenic = float(rf_prob[1])

    # Feature contributions (RF importances weighted by feature value)
    importances = model_meta['feature_importances']
    feature_importance_display = {
        name: {
            'importance': round(importances[name], 4),
            'value': round(features[name], 4),
        }
        for name in FEATURE_NAMES
    }

    return {
        'prediction': rf_pred,
        'classification': 'Pathogenic' if rf_pred == 1 else 'Benign',
        'probability_pathogenic': round(prob_pathogenic, 4),
        'probability_benign': round(float(rf_prob[0]), 4),
        'lr_probability_pathogenic': round(float(lr_prob[1]), 4),
        'feature_importances': feature_importance_display,
        'model_accuracy': model_meta['rf_accuracy'],
    }


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def index():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_version': model_meta['model_version']})


@app.route('/model/info', methods=['GET'])
def model_info():
    return jsonify(model_meta)


@app.route('/predict', methods=['POST'])
def predict_from_sequence():
    """
    Predict pathogenicity from DNA sequence + mutation.

    Input JSON:
    {
        "sequence": "ATGCGT...",
        "position": 7,          # 1-based
        "mutated_nucleotide": "A"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON payload provided'}), 400

    required = ['sequence', 'position', 'mutated_nucleotide']
    missing = [r for r in required if r not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    sequence = str(data['sequence']).upper().strip()
    position = int(data['position'])
    mut_nuc  = str(data['mutated_nucleotide']).upper().strip()

    if not all(b in 'ATGC' for b in sequence):
        return jsonify({'error': 'Sequence contains invalid nucleotides (use A, T, G, C only)'}), 400
    if mut_nuc not in 'ATGC':
        return jsonify({'error': 'Mutated nucleotide must be A, T, G, or C'}), 400
    if len(sequence) < 3:
        return jsonify({'error': 'Sequence must be at least 3 nucleotides'}), 400

    try:
        ref_codon, alt_codon, mut_sequence = apply_mutation(sequence, position, mut_nuc)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    gc = gc_content(sequence)
    features = build_feature_vector(ref_codon, alt_codon, gc, position, len(sequence))

    result = run_inference(features)
    result['interpretation'] = generate_interpretation(features, result['prediction'],
                                                       result['probability_pathogenic'])

    ref_protein = translate_sequence(sequence)
    mut_protein = translate_sequence(mut_sequence)

    # Find mutation position in protein
    aa_position = (position - 1) // 3

    return jsonify({
        **result,
        'variant_summary': {
            'input_sequence_length': len(sequence),
            'mutation_position': position,
            'reference_nucleotide': sequence[position - 1],
            'mutated_nucleotide': mut_nuc,
            'reference_codon': ref_codon,
            'alternate_codon': alt_codon,
            'reference_amino_acid': features['_ref_aa'],
            'alternate_amino_acid': features['_alt_aa'],
            'variant_type': features['_variant_type'],
            'stop_gained': bool(features['stop_gained']),
            'gc_content': round(gc, 4),
            'aa_substitution_score': round(features['aa_substitution_score'], 4),
        },
        'protein_sequences': {
            'reference': ref_protein,
            'mutated': mut_protein,
            'mutation_aa_position': aa_position,
        },
    })


@app.route('/predict/codon', methods=['POST'])
def predict_from_codon():
    """
    Predict pathogenicity from direct codon input.

    Input JSON:
    {
        "ref_codon": "GAG",
        "alt_codon": "GTG",
        "gc_content": 0.52,        # optional, default 0.5
        "position": 50,            # optional, default 50
        "seq_length": 300          # optional, default 300
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON payload provided'}), 400

    ref_codon = str(data.get('ref_codon', '')).upper().strip()
    alt_codon = str(data.get('alt_codon', '')).upper().strip()

    if len(ref_codon) < 2 or len(alt_codon) < 2:
        return jsonify({'error': 'ref_codon and alt_codon are required'}), 400

    gc = float(data.get('gc_content', 0.50))
    position  = int(data.get('position', 50))
    seq_length = int(data.get('seq_length', 300))

    features = build_feature_vector(ref_codon, alt_codon, gc, position, seq_length)
    result = run_inference(features)
    result['interpretation'] = generate_interpretation(features, result['prediction'],
                                                       result['probability_pathogenic'])

    return jsonify({
        **result,
        'variant_summary': {
            'reference_codon': ref_codon,
            'alternate_codon': alt_codon,
            'reference_amino_acid': features['_ref_aa'],
            'alternate_amino_acid': features['_alt_aa'],
            'variant_type': features['_variant_type'],
            'stop_gained': bool(features['stop_gained']),
            'gc_content': round(gc, 4),
            'aa_substitution_score': round(features['aa_substitution_score'], 4),
        },
    })


if __name__ == '__main__':
    print("🧬 Genomic Variant Pathogenicity Predictor API")
    print("   Running on http://localhost:5050")
    app.run(host='0.0.0.0', port=5050, debug=False)
