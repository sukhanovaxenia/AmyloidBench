"""
Classification modules for amyloid prediction and structural typing.

This module addresses two classification problems in amyloidogenicity:

**Binary classification: Amyloid vs Non-amyloid**

The fundamental question of whether a protein will form amyloid under
physiological or pathological conditions. This integrates:
- Per-residue APR predictions into protein-level scores
- Consideration of APR length, intensity, and position
- Contextual factors (gatekeeper density, disorder propensity)

A protein may contain APRs yet remain soluble due to:
1. APRs buried in the native fold
2. Effective gatekeeper protection
3. Stable native state kinetically trapping the protein
4. Chaperone availability in cellular context

**Structural classification: Amyloid polymorphs**

Amyloid fibrils exhibit remarkable structural diversity:

*Cross-β parallel in-register* - The canonical amyloid architecture
where β-strands stack perpendicular to the fibril axis with parallel
orientation. Characteristic of most disease-associated amyloids (Aβ,
α-synuclein, tau). Features steric zipper interfaces.

*Cross-β antiparallel* - β-strands in antiparallel arrangement, found
in some functional amyloids and designed amyloid-forming peptides.

*β-solenoid/β-helix* - Helical β-structure with triangular or quadrilateral
cross-section. Found in HET-s prion domain, curli subunits, and some
bacterial functional amyloids.

*Cross-α* - Emerging class of amyloid-like fibrils formed by α-helical
stacking. PSMα3 from Staphylococcus aureus exemplifies this category.

Structural classification requires:
1. Integration of structure prediction (ESMFold, AlphaFold)
2. Secondary structure content analysis
3. Pattern recognition for repeat architectures
4. Potentially ML models trained on solved amyloid structures

Submodules:
    binary: Protein-level amyloid/non-amyloid classification
    polymorph: Structural type prediction (cross-β, β-solenoid, cross-α)
    structural_pred: Interface to structure prediction tools
"""

# Classification module will be implemented in Phase 5
__all__ = []
