"""
Feature extraction for amyloidogenicity prediction.

This module provides comprehensive extraction of biophysical and
contextual features from protein sequences. Features are designed
to capture the determinants of amyloid formation:

- Hydrophobicity patterns (driving force for aggregation)
- Secondary structure propensity (β-sheet formation capability)
- Charge distribution (gatekeeper effects, electrostatic repulsion)
- Sequence composition (amino acid classes, dipeptides)
- Structural motifs (hydrophobic stretches, repeats, low complexity)

The features can be used for:
1. Direct scoring of aggregation propensity
2. Input to machine learning models
3. Interpretable analysis of amyloidogenic sequences
"""

from .extraction import (
    FeatureExtractor,
    SequenceFeatures,
    extract_features,
    get_feature_names,
    calculate_aggregation_score,
    # Amino acid scales
    HYDROPHOBICITY_KD,
    HYDROPHOBICITY_EISENBERG,
    BETA_PROPENSITY_CF,
    ALPHA_PROPENSITY_CF,
    AGGREGATION_PROPENSITY,
    CHARGE_PH7,
)

__all__ = [
    "FeatureExtractor",
    "SequenceFeatures",
    "extract_features",
    "get_feature_names",
    "calculate_aggregation_score",
    "HYDROPHOBICITY_KD",
    "HYDROPHOBICITY_EISENBERG",
    "BETA_PROPENSITY_CF",
    "ALPHA_PROPENSITY_CF",
    "AGGREGATION_PROPENSITY",
    "CHARGE_PH7",
]
