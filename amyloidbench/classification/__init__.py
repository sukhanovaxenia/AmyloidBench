"""
Classification modules for amyloid prediction and structural typing.

This module addresses two classification problems in amyloidogenicity:

**Binary classification: Amyloid vs Non-amyloid**

The fundamental question of whether a protein will form amyloid under
physiological or pathological conditions. This integrates:
- Per-residue APR predictions into protein-level scores
- Consideration of APR length, intensity, and position
- Contextual factors (gatekeeper density, disorder propensity)

**Structural classification: Amyloid polymorphs**

Amyloid fibrils exhibit remarkable structural diversity. The polymorph
module provides prediction of:

*Steric Zipper Classes (Eisenberg 8-class system)*
Based on symmetry of the cross-β interface: strand orientation,
sheet packing, and face orientation.

*Cross-β Geometry*
Parallel in-register, parallel out-of-register, antiparallel, or mixed.

*Higher-Order Folds*
Steric zipper, β-solenoid, β-arcade, Greek key, β-helix, serpentine.

The biological significance of polymorph classification includes:
- Prion strain typing (different PrP^Sc conformers)
- Synucleinopathy distinction (PD vs MSA α-synuclein polymorphs)
- Tauopathy classification (AD vs Pick's tau polymorphs)
- Functional amyloid identification (HET-s, curli)

References:
    Sawaya et al. (2007) - Steric zipper classification
    Eisenberg & Sawaya (2017) - Amyloid Atlas
    Fitzpatrick et al. (2017) - Tau polymorphs
    Schweighauser et al. (2020) - α-synuclein polymorphs
"""

from .polymorph import (
    # Enums
    StericZipperClass,
    CrossBetaGeometry,
    AmyloidFold,
    # Data classes
    PolymorphPrediction,
    # Main classifier
    PolymorphClassifier,
    # Convenience functions
    predict_polymorph,
    find_similar_structures,
    get_known_structures,
    # Database
    KNOWN_AMYLOID_STRUCTURES,
)

__all__ = [
    # Enums
    "StericZipperClass",
    "CrossBetaGeometry", 
    "AmyloidFold",
    # Data classes
    "PolymorphPrediction",
    # Main classifier
    "PolymorphClassifier",
    # Convenience functions
    "predict_polymorph",
    "find_similar_structures",
    "get_known_structures",
    # Database
    "KNOWN_AMYLOID_STRUCTURES",
]
