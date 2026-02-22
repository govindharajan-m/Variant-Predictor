"""
bio_utils.py - Biological utility functions for genomic variant analysis.
Implements codon translation, amino acid properties, and variant classification
without external bioinformatics dependencies.
"""

import numpy as np
from typing import Optional, Tuple, Dict

# Standard genetic code (codon -> amino acid, single letter)
GENETIC_CODE: Dict[str, str] = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

STOP_CODONS = {'TAA', 'TAG', 'TGA'}

# Amino acid physicochemical properties
# (hydrophobicity, charge, size, polarity) - normalized 0-1
AA_PROPERTIES: Dict[str, Dict[str, float]] = {
    'A': {'hydrophobicity': 0.61, 'charge': 0.0, 'size': 0.12, 'polarity': 0.0},
    'R': {'hydrophobicity': 0.0,  'charge': 1.0, 'size': 0.94, 'polarity': 1.0},
    'N': {'hydrophobicity': 0.19, 'charge': 0.0, 'size': 0.47, 'polarity': 1.0},
    'D': {'hydrophobicity': 0.10, 'charge': -1.0,'size': 0.41, 'polarity': 1.0},
    'C': {'hydrophobicity': 0.68, 'charge': 0.0, 'size': 0.28, 'polarity': 0.3},
    'Q': {'hydrophobicity': 0.0,  'charge': 0.0, 'size': 0.60, 'polarity': 1.0},
    'E': {'hydrophobicity': 0.0,  'charge': -1.0,'size': 0.55, 'polarity': 1.0},
    'G': {'hydrophobicity': 0.45, 'charge': 0.0, 'size': 0.0,  'polarity': 0.0},
    'H': {'hydrophobicity': 0.23, 'charge': 0.5, 'size': 0.66, 'polarity': 0.7},
    'I': {'hydrophobicity': 1.0,  'charge': 0.0, 'size': 0.64, 'polarity': 0.0},
    'L': {'hydrophobicity': 0.92, 'charge': 0.0, 'size': 0.64, 'polarity': 0.0},
    'K': {'hydrophobicity': 0.06, 'charge': 1.0, 'size': 0.76, 'polarity': 1.0},
    'M': {'hydrophobicity': 0.64, 'charge': 0.0, 'size': 0.64, 'polarity': 0.3},
    'F': {'hydrophobicity': 0.97, 'charge': 0.0, 'size': 0.80, 'polarity': 0.0},
    'P': {'hydrophobicity': 0.36, 'charge': 0.0, 'size': 0.40, 'polarity': 0.0},
    'S': {'hydrophobicity': 0.50, 'charge': 0.0, 'size': 0.22, 'polarity': 0.8},
    'T': {'hydrophobicity': 0.50, 'charge': 0.0, 'size': 0.38, 'polarity': 0.7},
    'W': {'hydrophobicity': 0.88, 'charge': 0.0, 'size': 1.0,  'polarity': 0.4},
    'Y': {'hydrophobicity': 0.76, 'charge': 0.0, 'size': 0.87, 'polarity': 0.6},
    'V': {'hydrophobicity': 0.86, 'charge': 0.0, 'size': 0.50, 'polarity': 0.0},
    '*': {'hydrophobicity': 0.0,  'charge': 0.0, 'size': 0.0,  'polarity': 0.0},
}


def translate_codon(codon: str) -> str:
    """Translate a single codon to its amino acid (single-letter code)."""
    codon = codon.upper()
    return GENETIC_CODE.get(codon, '?')


def translate_sequence(dna: str) -> str:
    """Translate a DNA sequence to protein, stopping at first stop codon."""
    dna = dna.upper().replace(' ', '')
    protein = []
    for i in range(0, len(dna) - 2, 3):
        codon = dna[i:i+3]
        if len(codon) < 3:
            break
        aa = translate_codon(codon)
        if aa == '*':
            protein.append('*')
            break
        protein.append(aa)
    return ''.join(protein)


def gc_content(dna: str) -> float:
    """Compute GC content of a DNA sequence."""
    dna = dna.upper()
    if not dna:
        return 0.0
    gc = sum(1 for b in dna if b in 'GC')
    return gc / len(dna)


def amino_acid_substitution_score(ref_aa: str, alt_aa: str) -> float:
    """
    Compute physicochemical distance between two amino acids.
    Returns 0 for synonymous, high values for radical substitutions.
    """
    if ref_aa == alt_aa:
        return 0.0
    if ref_aa == '*' or alt_aa == '*' or ref_aa not in AA_PROPERTIES or alt_aa not in AA_PROPERTIES:
        return 1.0  # Max impact for stop codon changes

    props_ref = AA_PROPERTIES[ref_aa]
    props_alt = AA_PROPERTIES[alt_aa]
    
    diff = np.sqrt(
        (props_ref['hydrophobicity'] - props_alt['hydrophobicity']) ** 2 +
        (props_ref['charge'] - props_alt['charge']) ** 2 +
        (props_ref['size'] - props_alt['size']) ** 2 +
        (props_ref['polarity'] - props_alt['polarity']) ** 2
    ) / 2.0  # normalize to ~0-1 range
    return min(diff, 1.0)


def classify_variant(ref_codon: str, alt_codon: str) -> str:
    """
    Classify variant type based on codon change.
    Returns: 'synonymous', 'missense', 'nonsense', or 'frameshift'
    """
    ref_codon = ref_codon.upper()
    alt_codon = alt_codon.upper()

    # Frameshift: different lengths (indel)
    if len(ref_codon) != len(alt_codon) or len(ref_codon) % 3 != 0:
        return 'frameshift'

    ref_aa = translate_codon(ref_codon[:3])
    alt_aa = translate_codon(alt_codon[:3])

    if alt_aa == '*' and ref_aa != '*':
        return 'nonsense'
    elif ref_aa == alt_aa:
        return 'synonymous'
    else:
        return 'missense'


def apply_mutation(sequence: str, position: int, mutated_nucleotide: str) -> Tuple[str, str, str]:
    """
    Apply a point mutation to a DNA sequence.
    
    Args:
        sequence: Reference DNA sequence
        position: 1-based mutation position
        mutated_nucleotide: The new nucleotide at position
    
    Returns:
        (ref_codon, alt_codon, mutated_sequence)
    """
    sequence = sequence.upper()
    pos = position - 1  # Convert to 0-based

    if pos < 0 or pos >= len(sequence):
        raise ValueError(f"Position {position} out of range for sequence of length {len(sequence)}")

    # Find which codon this position falls in
    codon_index = pos // 3
    codon_start = codon_index * 3
    codon_end = codon_start + 3

    if codon_end > len(sequence):
        raise ValueError("Mutation position falls in incomplete codon at end of sequence")

    ref_codon = sequence[codon_start:codon_end]
    
    # Build mutated sequence
    mut_seq = list(sequence)
    mut_seq[pos] = mutated_nucleotide.upper()
    mutated_sequence = ''.join(mut_seq)
    alt_codon = mutated_sequence[codon_start:codon_end]

    return ref_codon, alt_codon, mutated_sequence


def build_feature_vector(
    ref_codon: str,
    alt_codon: str,
    gc: float,
    position: int,
    seq_length: int,
) -> Dict:
    """
    Build a complete feature dictionary for ML inference.
    """
    ref_codon = ref_codon.upper()
    alt_codon = alt_codon.upper()

    ref_aa = translate_codon(ref_codon)
    alt_aa = translate_codon(alt_codon)
    variant_type = classify_variant(ref_codon, alt_codon)
    stop_gained = 1 if (alt_aa == '*' and ref_aa != '*') else 0
    aa_score = amino_acid_substitution_score(ref_aa, alt_aa)
    pos_normalized = position / max(seq_length, 1)

    # One-hot encode variant type
    vt_synonymous = 1 if variant_type == 'synonymous' else 0
    vt_missense = 1 if variant_type == 'missense' else 0
    vt_nonsense = 1 if variant_type == 'nonsense' else 0
    vt_frameshift = 1 if variant_type == 'frameshift' else 0

    return {
        'gc_content': gc,
        'position_normalized': pos_normalized,
        'stop_gained': stop_gained,
        'aa_substitution_score': aa_score,
        'variant_type_synonymous': vt_synonymous,
        'variant_type_missense': vt_missense,
        'variant_type_nonsense': vt_nonsense,
        'variant_type_frameshift': vt_frameshift,
        # Metadata (not used in ML)
        '_ref_codon': ref_codon,
        '_alt_codon': alt_codon,
        '_ref_aa': ref_aa,
        '_alt_aa': alt_aa,
        '_variant_type': variant_type,
    }


FEATURE_NAMES = [
    'gc_content',
    'position_normalized',
    'stop_gained',
    'aa_substitution_score',
    'variant_type_synonymous',
    'variant_type_missense',
    'variant_type_nonsense',
    'variant_type_frameshift',
]
