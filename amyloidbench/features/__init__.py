"""
Feature extraction for machine learning-based amyloidogenicity prediction.

This module provides systematic extraction of sequence and structure-derived
features relevant to amyloid formation. The feature space is grounded in
the biophysical and biochemical principles underlying protein aggregation:

**Compositional features**
Amino acid frequencies, dipeptide composition, and grouped amino acid
classes. These capture the overall chemical character of sequences—
amyloidogenic regions are enriched in hydrophobic, β-sheet-prone residues
(V, I, L, F, Y) and depleted in aggregation-inhibiting "gatekeepers"
(P, K, R, E, D).

**Biophysical features**
Hydrophobicity profiles (Kyte-Doolittle, Eisenberg consensus), charge
patterns, and structural propensity scales (Chou-Fasman α/β). These
directly relate to the thermodynamic driving forces of aggregation.

**Structural features**
Secondary structure predictions, intrinsic disorder propensity, and
when available, solvent accessibility. These determine which regions
are structurally predisposed to β-sheet formation and whether they
are exposed for intermolecular contacts.

**Contextual features**
Flanking sequence analysis—gatekeeper residues adjacent to hydrophobic
stretches act as evolutionary safeguards against aberrant aggregation.
Position within the protein (N/C-terminal bias) also influences
aggregation kinetics.

**Evolutionary features**
Conservation scores from multiple sequence alignments. Aggregation-prone
regions in functional amyloids show conservation, while pathological
APRs often involve mutation-induced disruption of protective features.

Feature extraction follows a sliding window approach for per-residue
predictions, with window sizes optimized for each feature type (typically
5-21 residues for local properties, whole-protein for global features).

Submodules:
    biophysical: Hydrophobicity, charge, propensity scales
    structural: Secondary structure, disorder prediction
    evolutionary: Conservation and position-specific scoring
    context: Flanking region and gatekeeper analysis
"""

# Features module will be implemented in Phase 3
__all__ = []
