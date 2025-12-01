"""
Amyloid polymorph classification and structural prediction.

This module provides classification of amyloidogenic sequences into
structural polymorph classes, enabling prediction of the cross-β
architecture that a given sequence is likely to adopt.

Biological Background
---------------------
Amyloid fibrils share the cross-β architecture where β-strands run
perpendicular to the fibril axis, but this core structure admits
remarkable variation:

**Steric Zipper Classes (Eisenberg)**
The steric zipper—the dry interface between β-sheets in the fibril
core—can adopt 8 distinct geometries based on:
1. Parallel vs antiparallel β-strands within each sheet
2. Parallel vs antiparallel packing of the two sheets
3. Face-to-face vs face-to-back sheet orientation

**Cross-β Geometries**
- Parallel in-register: Identical residues stack along fibril axis
  (Aβ, α-synuclein, tau in most disease forms)
- Parallel out-of-register: Shifted stacking creates different contacts
- Antiparallel: Adjacent strands run in opposite directions
  (common in short peptides, some designed amyloids)

**Higher-Order Folds**
- β-solenoid: Single chain winds into helical β-structure (HET-s, curli)
- β-helix: Triangular/square cross-section coils
- Greek key: Complex topology with multiple strand arrangements

**Biological Significance**
Polymorph identity determines:
- Toxicity profiles (oligomer vs fibril, membrane interaction)
- Seeding efficiency (cross-seeding between polymorphs is often impaired)
- Strain properties in prion diseases
- Therapeutic targeting (polymorph-specific antibodies, inhibitors)

References
----------
- Sawaya et al. (2007) - Steric zipper classification
- Eisenberg & Sawaya (2017) - Amyloid Atlas
- Fitzpatrick et al. (2017) - Tau polymorphs in AD vs Pick's
- Schweighauser et al. (2020) - α-synuclein polymorphs in PD vs MSA
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np

from amyloidbench.core.models import ProteinRecord, Region
from amyloidbench.features.extraction import (
    FeatureExtractor,
    HYDROPHOBICITY_KD,
    BETA_PROPENSITY_CF,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Polymorph Classification Enums
# =============================================================================

class StericZipperClass(str, Enum):
    """
    Eisenberg's 8 steric zipper classes based on symmetry.
    
    Classification is based on three binary features:
    1. Strand orientation within sheet (parallel/antiparallel)
    2. Sheet-sheet packing (parallel/antiparallel)
    3. Sheet face orientation (face-to-face/face-to-back)
    
    Classes 1-4: Parallel strands within sheets
    Classes 5-8: Antiparallel strands within sheets
    """
    CLASS_1 = "class_1"  # Parallel strands, parallel sheets, face-to-face
    CLASS_2 = "class_2"  # Parallel strands, parallel sheets, face-to-back
    CLASS_3 = "class_3"  # Parallel strands, antiparallel sheets, face-to-face
    CLASS_4 = "class_4"  # Parallel strands, antiparallel sheets, face-to-back
    CLASS_5 = "class_5"  # Antiparallel strands, parallel sheets, face-to-face
    CLASS_6 = "class_6"  # Antiparallel strands, parallel sheets, face-to-back
    CLASS_7 = "class_7"  # Antiparallel strands, antiparallel sheets, face-to-face
    CLASS_8 = "class_8"  # Antiparallel strands, antiparallel sheets, face-to-back
    UNKNOWN = "unknown"
    
    @property
    def strand_orientation(self) -> str:
        """Parallel or antiparallel strands within each sheet."""
        if self in [self.CLASS_1, self.CLASS_2, self.CLASS_3, self.CLASS_4]:
            return "parallel"
        elif self in [self.CLASS_5, self.CLASS_6, self.CLASS_7, self.CLASS_8]:
            return "antiparallel"
        return "unknown"
    
    @property
    def sheet_packing(self) -> str:
        """Parallel or antiparallel packing of the two sheets."""
        if self in [self.CLASS_1, self.CLASS_2, self.CLASS_5, self.CLASS_6]:
            return "parallel"
        elif self in [self.CLASS_3, self.CLASS_4, self.CLASS_7, self.CLASS_8]:
            return "antiparallel"
        return "unknown"
    
    @property
    def face_orientation(self) -> str:
        """Face-to-face or face-to-back sheet arrangement."""
        if self in [self.CLASS_1, self.CLASS_3, self.CLASS_5, self.CLASS_7]:
            return "face_to_face"
        elif self in [self.CLASS_2, self.CLASS_4, self.CLASS_6, self.CLASS_8]:
            return "face_to_back"
        return "unknown"


class CrossBetaGeometry(str, Enum):
    """Cross-β strand arrangement geometries."""
    PARALLEL_IN_REGISTER = "parallel_in_register"
    PARALLEL_OUT_OF_REGISTER = "parallel_out_of_register"
    ANTIPARALLEL = "antiparallel"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class AmyloidFold(str, Enum):
    """Higher-order amyloid fold types."""
    STERIC_ZIPPER = "steric_zipper"  # Classic cross-β with dry interface
    BETA_SOLENOID = "beta_solenoid"  # Helical winding (HET-s, curli)
    BETA_HELIX = "beta_helix"        # Triangular/square β-helix
    GREEK_KEY = "greek_key"          # Complex topology
    BETA_ARCADE = "beta_arcade"      # Stacked β-arches (Aβ, tau cryo-EM)
    SERPENTINE = "serpentine"        # Extended serpentine fold
    UNKNOWN = "unknown"


# =============================================================================
# Polymorph Classification Data Classes
# =============================================================================

@dataclass
class PolymorphPrediction:
    """
    Complete polymorph classification result.
    
    Attributes:
        sequence: Input sequence
        predicted_fold: Primary fold type prediction
        fold_probabilities: Probability distribution over fold types
        predicted_geometry: Cross-β geometry prediction
        geometry_probabilities: Probability distribution over geometries
        steric_zipper_class: Most likely steric zipper class
        zipper_probabilities: Probability distribution over 8 classes
        structural_features: Extracted features used for classification
        template_matches: Similar known structures (if available)
        confidence: Overall prediction confidence (0-1)
        notes: Additional interpretation notes
    """
    sequence: str
    predicted_fold: AmyloidFold
    fold_probabilities: dict[str, float] = field(default_factory=dict)
    predicted_geometry: CrossBetaGeometry = CrossBetaGeometry.UNKNOWN
    geometry_probabilities: dict[str, float] = field(default_factory=dict)
    steric_zipper_class: StericZipperClass = StericZipperClass.UNKNOWN
    zipper_probabilities: dict[str, float] = field(default_factory=dict)
    structural_features: dict[str, float] = field(default_factory=dict)
    template_matches: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Polymorph Prediction for {len(self.sequence)}-residue sequence",
            f"  Predicted fold: {self.predicted_fold.value} ({self.confidence:.0%} confidence)",
            f"  Cross-β geometry: {self.predicted_geometry.value}",
            f"  Steric zipper: {self.steric_zipper_class.value}",
        ]
        if self.notes:
            lines.append("  Notes:")
            for note in self.notes:
                lines.append(f"    - {note}")
        return "\n".join(lines)


# =============================================================================
# Sequence-Based Polymorph Predictors
# =============================================================================

# Amino acid preferences for parallel vs antiparallel arrangements
# Based on analysis of solved amyloid structures
PARALLEL_PREFERENCE = {
    # Residues favoring parallel in-register (identical residue stacking)
    # Hydrophobic residues pack well in parallel steric zippers
    'V': 0.8, 'I': 0.9, 'L': 0.7, 'F': 0.6, 'Y': 0.5,
    # Polar residues can form ladders in parallel arrangement
    'Q': 0.7, 'N': 0.7, 'S': 0.4, 'T': 0.4,
    # Charged residues disfavor parallel (same-charge stacking)
    'K': -0.3, 'R': -0.2, 'E': -0.4, 'D': -0.4,
    # Aromatic stacking favors parallel
    'W': 0.5,
    # Small residues are neutral
    'A': 0.2, 'G': 0.0,
    # Cysteine can form disulfide ladders
    'C': 0.6,
    # Proline and histidine
    'P': -0.8, 'H': 0.1, 'M': 0.5,
}

# Alternating hydrophobic-polar patterns favor antiparallel
# (allows complementary packing)
ANTIPARALLEL_MOTIFS = [
    r'[VILF][QNST][VILF]',  # Hydrophobic-polar-hydrophobic
    r'[KR][VILF]{2,}[DE]',  # Charged flanks with hydrophobic core
    r'([VILF][QNST]){2,}',  # Alternating pattern
]

# β-solenoid signatures (from HET-s, curli analysis)
BETA_SOLENOID_FEATURES = {
    'repeat_length': (18, 30),  # Typical repeat unit size
    'glycine_positions': True,   # Conserved glycines at turns
    'asparagine_ladders': True,  # Asparagine stacking
}


class PolymorphClassifier:
    """
    Sequence-based amyloid polymorph classifier.
    
    Predicts the likely structural class of an amyloidogenic sequence
    based on sequence features correlated with known structures.
    
    The classifier uses:
    1. Amino acid composition preferences
    2. Sequence patterns (alternating, repeating)
    3. Charge distribution
    4. Hydrophobic moment (amphipathicity)
    
    This is a heuristic approach; high-confidence predictions require
    experimental validation or structure determination.
    
    Usage:
        >>> classifier = PolymorphClassifier()
        >>> result = classifier.predict(sequence)
        >>> print(result.predicted_geometry)
    """
    
    def __init__(self):
        """Initialize the polymorph classifier."""
        self.feature_extractor = FeatureExtractor(window_size=7)
    
    def predict(
        self,
        sequence: str,
        regions: Optional[list[Region]] = None,
    ) -> PolymorphPrediction:
        """
        Predict polymorph class for a sequence.
        
        Args:
            sequence: Protein sequence
            regions: Optional list of APRs to focus analysis on
            
        Returns:
            PolymorphPrediction with classification results
        """
        sequence = sequence.upper()
        
        # Extract features
        features = self.feature_extractor.extract(sequence)
        structural_features = self._extract_structural_features(sequence, features)
        
        # Predict cross-β geometry
        geometry, geometry_probs = self._predict_geometry(sequence, structural_features)
        
        # Predict steric zipper class
        zipper_class, zipper_probs = self._predict_zipper_class(sequence, geometry, structural_features)
        
        # Predict higher-order fold
        fold, fold_probs = self._predict_fold(sequence, structural_features)
        
        # Calculate confidence
        confidence = self._calculate_confidence(geometry_probs, zipper_probs, fold_probs)
        
        # Generate notes
        notes = self._generate_notes(sequence, structural_features, geometry, fold)
        
        return PolymorphPrediction(
            sequence=sequence,
            predicted_fold=fold,
            fold_probabilities=fold_probs,
            predicted_geometry=geometry,
            geometry_probabilities=geometry_probs,
            steric_zipper_class=zipper_class,
            zipper_probabilities=zipper_probs,
            structural_features=structural_features,
            confidence=confidence,
            notes=notes,
        )
    
    def _extract_structural_features(
        self,
        sequence: str,
        features,
    ) -> dict[str, float]:
        """Extract features relevant to polymorph classification."""
        n = len(sequence)
        
        # Parallel preference score
        parallel_score = sum(PARALLEL_PREFERENCE.get(aa, 0) for aa in sequence) / n
        
        # Alternating pattern score
        alternating_score = self._calculate_alternating_score(sequence)
        
        # Charge distribution
        charges = [1 if aa in 'KRH' else (-1 if aa in 'DE' else 0) for aa in sequence]
        charge_asymmetry = self._calculate_charge_asymmetry(charges)
        
        # Repeat detection
        repeat_score, repeat_length = self._detect_repeats(sequence)
        
        # β-turn propensity (influences fold type)
        turn_residues = sum(1 for aa in sequence if aa in 'GNPSD')
        turn_fraction = turn_residues / n
        
        # Glycine periodicity (important for β-solenoid)
        glycine_positions = [i for i, aa in enumerate(sequence) if aa == 'G']
        glycine_periodicity = self._calculate_periodicity(glycine_positions, n)
        
        # Asparagine/glutamine content (ladder formation)
        qn_fraction = sum(1 for aa in sequence if aa in 'QN') / n
        
        # Aromatic content (π-stacking in parallel structures)
        aromatic_fraction = sum(1 for aa in sequence if aa in 'FYW') / n
        
        # Hydrophobic moment
        hydrophobic_moment = features.global_features.get('hydrophobic_moment', 0)
        
        return {
            'parallel_preference': parallel_score,
            'alternating_score': alternating_score,
            'charge_asymmetry': charge_asymmetry,
            'repeat_score': repeat_score,
            'repeat_length': repeat_length,
            'turn_fraction': turn_fraction,
            'glycine_periodicity': glycine_periodicity,
            'qn_fraction': qn_fraction,
            'aromatic_fraction': aromatic_fraction,
            'hydrophobic_moment': hydrophobic_moment,
            'mean_hydrophobicity': features.global_features['mean_hydrophobicity'],
            'mean_beta_propensity': features.global_features['mean_beta_propensity'],
            'frac_hydrophobic': features.global_features['frac_hydrophobic'],
            'length': n,
        }
    
    def _calculate_alternating_score(self, sequence: str) -> float:
        """
        Calculate score for alternating hydrophobic-polar patterns.
        
        Such patterns favor antiparallel arrangements where complementary
        residues can pack across the sheet interface.
        """
        hydrophobic = set('VILFYWM')
        polar = set('QNSTDE')
        
        alternations = 0
        for i in range(len(sequence) - 1):
            if (sequence[i] in hydrophobic and sequence[i+1] in polar) or \
               (sequence[i] in polar and sequence[i+1] in hydrophobic):
                alternations += 1
        
        return alternations / (len(sequence) - 1) if len(sequence) > 1 else 0
    
    def _calculate_charge_asymmetry(self, charges: list[int]) -> float:
        """
        Calculate asymmetry in charge distribution.
        
        High asymmetry (charges clustered at one end) suggests specific
        structural constraints on fibril geometry.
        """
        n = len(charges)
        if n == 0:
            return 0.0
        
        # Calculate center of positive and negative charges
        pos_positions = [i for i, c in enumerate(charges) if c > 0]
        neg_positions = [i for i, c in enumerate(charges) if c < 0]
        
        if not pos_positions or not neg_positions:
            return 0.0
        
        pos_center = np.mean(pos_positions) / n
        neg_center = np.mean(neg_positions) / n
        
        return abs(pos_center - neg_center)
    
    def _detect_repeats(self, sequence: str) -> tuple[float, float]:
        """
        Detect sequence repeats characteristic of β-solenoid folds.
        
        Returns:
            Tuple of (repeat_score, estimated_repeat_length)
        """
        n = len(sequence)
        if n < 10:
            return 0.0, 0.0
        
        best_score = 0.0
        best_length = 0
        
        # Check repeat lengths typical for β-solenoids (15-30 residues)
        for repeat_len in range(10, min(40, n // 2)):
            matches = 0
            comparisons = 0
            
            for i in range(n - repeat_len):
                if sequence[i] == sequence[i + repeat_len]:
                    matches += 1
                comparisons += 1
            
            if comparisons > 0:
                score = matches / comparisons
                if score > best_score:
                    best_score = score
                    best_length = repeat_len
        
        return best_score, best_length
    
    def _calculate_periodicity(self, positions: list[int], length: int) -> float:
        """Calculate periodicity of positions (e.g., glycine spacing)."""
        if len(positions) < 3:
            return 0.0
        
        # Calculate spacing between consecutive positions
        spacings = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        
        if not spacings:
            return 0.0
        
        # Periodicity is inverse of spacing variance (normalized)
        mean_spacing = np.mean(spacings)
        if mean_spacing == 0:
            return 0.0
        
        variance = np.var(spacings)
        periodicity = 1.0 / (1.0 + variance / mean_spacing)
        
        return periodicity
    
    def _predict_geometry(
        self,
        sequence: str,
        features: dict[str, float],
    ) -> tuple[CrossBetaGeometry, dict[str, float]]:
        """
        Predict cross-β strand geometry.
        
        Based on sequence features correlated with known structures.
        """
        # Initialize probabilities
        probs = {
            CrossBetaGeometry.PARALLEL_IN_REGISTER.value: 0.0,
            CrossBetaGeometry.PARALLEL_OUT_OF_REGISTER.value: 0.0,
            CrossBetaGeometry.ANTIPARALLEL.value: 0.0,
            CrossBetaGeometry.MIXED.value: 0.0,
        }
        
        # Parallel in-register favored by:
        # - High parallel preference score
        # - Low alternating pattern score
        # - Aromatic residues (π-stacking)
        # - Q/N residues (side chain ladders)
        parallel_score = (
            0.3 * (features['parallel_preference'] + 1) / 2 +  # Normalize to 0-1
            0.2 * features['aromatic_fraction'] * 3 +
            0.2 * features['qn_fraction'] * 3 +
            0.2 * (1 - features['alternating_score']) +
            0.1 * features['mean_hydrophobicity'] / 4  # Normalize
        )
        probs[CrossBetaGeometry.PARALLEL_IN_REGISTER.value] = min(1.0, max(0.0, parallel_score))
        
        # Antiparallel favored by:
        # - High alternating score
        # - Short sequences
        # - Low parallel preference
        antiparallel_score = (
            0.4 * features['alternating_score'] +
            0.2 * (1 if features['length'] < 15 else 0) +
            0.2 * (1 - (features['parallel_preference'] + 1) / 2) +
            0.2 * features['charge_asymmetry']
        )
        probs[CrossBetaGeometry.ANTIPARALLEL.value] = min(1.0, max(0.0, antiparallel_score))
        
        # Out-of-register is less common
        probs[CrossBetaGeometry.PARALLEL_OUT_OF_REGISTER.value] = 0.1
        
        # Mixed is rare
        probs[CrossBetaGeometry.MIXED.value] = 0.05
        
        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v/total for k, v in probs.items()}
        
        # Get most likely
        best = max(probs.items(), key=lambda x: x[1])
        geometry = CrossBetaGeometry(best[0])
        
        return geometry, probs
    
    def _predict_zipper_class(
        self,
        sequence: str,
        geometry: CrossBetaGeometry,
        features: dict[str, float],
    ) -> tuple[StericZipperClass, dict[str, float]]:
        """
        Predict steric zipper class based on geometry and sequence.
        
        Uses the predicted geometry to constrain possible zipper classes,
        then uses sequence features to distinguish within that subset.
        """
        probs = {cls.value: 0.0 for cls in StericZipperClass if cls != StericZipperClass.UNKNOWN}
        
        if geometry in [CrossBetaGeometry.PARALLEL_IN_REGISTER, 
                        CrossBetaGeometry.PARALLEL_OUT_OF_REGISTER]:
            # Classes 1-4 (parallel strands)
            # Class 1 (face-to-face) favored by symmetric hydrophobic residues
            # Class 2 (face-to-back) favored by asymmetric amphipathic
            
            hydrophobic_symmetry = 1.0 - features['hydrophobic_moment']
            
            probs[StericZipperClass.CLASS_1.value] = 0.3 + 0.2 * hydrophobic_symmetry
            probs[StericZipperClass.CLASS_2.value] = 0.2 + 0.2 * features['hydrophobic_moment']
            probs[StericZipperClass.CLASS_3.value] = 0.15
            probs[StericZipperClass.CLASS_4.value] = 0.15
            
        elif geometry == CrossBetaGeometry.ANTIPARALLEL:
            # Classes 5-8 (antiparallel strands)
            probs[StericZipperClass.CLASS_5.value] = 0.25
            probs[StericZipperClass.CLASS_6.value] = 0.25
            probs[StericZipperClass.CLASS_7.value] = 0.25
            probs[StericZipperClass.CLASS_8.value] = 0.25
        else:
            # Unknown - equal probability across all
            for cls in StericZipperClass:
                if cls != StericZipperClass.UNKNOWN:
                    probs[cls.value] = 0.125
        
        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v/total for k, v in probs.items()}
        
        # Get most likely
        best = max(probs.items(), key=lambda x: x[1])
        zipper_class = StericZipperClass(best[0])
        
        return zipper_class, probs
    
    def _predict_fold(
        self,
        sequence: str,
        features: dict[str, float],
    ) -> tuple[AmyloidFold, dict[str, float]]:
        """
        Predict higher-order amyloid fold type.
        
        Distinguishes between steric zipper, β-solenoid, β-arcade, etc.
        Based on sequence length, repeat patterns, turn residue distribution,
        and glycine periodicity.
        """
        probs = {fold.value: 0.0 for fold in AmyloidFold if fold != AmyloidFold.UNKNOWN}
        
        length = features['length']
        
        # β-solenoid indicators (HET-s, curli, prion repeats)
        # Key features: glycine periodicity, regular repeats, moderate length
        solenoid_score = (
            0.25 * features['repeat_score'] +
            0.35 * features['glycine_periodicity'] +  # Strong weight for glycine periodicity
            0.20 * (1 if 15 <= features['repeat_length'] <= 35 else 0) +
            0.10 * features['turn_fraction'] +
            0.10 * (1 if 30 <= length <= 100 else 0.3)  # Length bonus
        )
        # Boost if high glycine periodicity even with lower repeat score
        if features['glycine_periodicity'] > 0.7 and length > 30:
            solenoid_score += 0.15
        probs[AmyloidFold.BETA_SOLENOID.value] = min(0.85, solenoid_score)
        
        # Steric zipper (most common for short APRs, 5-15 residues)
        zipper_score = (
            0.35 * (1 - features['repeat_score']) +
            0.25 * features['mean_beta_propensity'] +
            0.20 * features['frac_hydrophobic'] +
            0.20 * (1 if length < 20 else (0.5 if length < 30 else 0.2))
        )
        probs[AmyloidFold.STERIC_ZIPPER.value] = min(0.85, zipper_score)
        
        # β-arcade (typical for longer sequences like Aβ, tau core)
        arcade_score = (
            0.30 * (1 if 25 <= length <= 60 else 0.3) +
            0.25 * features['turn_fraction'] +
            0.25 * (1 - features['repeat_score']) +
            0.20 * max(0, features['mean_hydrophobicity'] / 2)
        )
        probs[AmyloidFold.BETA_ARCADE.value] = min(0.75, arcade_score)
        
        # Greek key (complex topology, longer sequences)
        greek_key_score = (
            0.25 * (1 if 30 <= length <= 80 else 0.2) +
            0.25 * features['turn_fraction'] +
            0.25 * (1 - features['glycine_periodicity']) +  # Less regular than solenoid
            0.25 * features['mean_hydrophobicity'] / 3
        )
        probs[AmyloidFold.GREEK_KEY.value] = min(0.6, greek_key_score)
        
        # β-helix (rare, very regular repeats)
        helix_score = 0.1 + 0.2 * features['repeat_score']
        probs[AmyloidFold.BETA_HELIX.value] = min(0.4, helix_score)
        
        # Serpentine (kinked, longer)
        serpentine_score = 0.1 + 0.1 * (1 if length > 60 else 0)
        probs[AmyloidFold.SERPENTINE.value] = min(0.4, serpentine_score)
        
        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v/total for k, v in probs.items()}
        
        # Get most likely
        best = max(probs.items(), key=lambda x: x[1])
        fold = AmyloidFold(best[0])
        
        return fold, probs
    
    def _calculate_confidence(
        self,
        geometry_probs: dict[str, float],
        zipper_probs: dict[str, float],
        fold_probs: dict[str, float],
    ) -> float:
        """
        Calculate overall prediction confidence.
        
        Based on probability margins and agreement between predictions.
        """
        # Get top probabilities
        geo_max = max(geometry_probs.values()) if geometry_probs else 0
        zip_max = max(zipper_probs.values()) if zipper_probs else 0
        fold_max = max(fold_probs.values()) if fold_probs else 0
        
        # Confidence is geometric mean of top probabilities
        confidence = (geo_max * zip_max * fold_max) ** (1/3)
        
        return float(confidence)
    
    def _generate_notes(
        self,
        sequence: str,
        features: dict[str, float],
        geometry: CrossBetaGeometry,
        fold: AmyloidFold,
    ) -> list[str]:
        """Generate interpretive notes about the prediction."""
        notes = []
        
        # Length-based notes
        n = len(sequence)
        if n < 10:
            notes.append("Short peptide - structural predictions less reliable")
        elif n > 50:
            notes.append("Long sequence may adopt complex multi-strand topology")
        
        # Geometry-specific notes
        if geometry == CrossBetaGeometry.PARALLEL_IN_REGISTER:
            if features['qn_fraction'] > 0.15:
                notes.append("High Q/N content suggests polar zipper with asparagine ladders")
            if features['aromatic_fraction'] > 0.15:
                notes.append("Aromatic residues may contribute to parallel stacking via π-π interactions")
        
        if geometry == CrossBetaGeometry.ANTIPARALLEL:
            if features['alternating_score'] > 0.4:
                notes.append("Strong alternating pattern favors antiparallel β-sheet")
        
        # Fold-specific notes
        if fold == AmyloidFold.BETA_SOLENOID:
            notes.append(f"Repeat pattern detected (period ~{features['repeat_length']:.0f} residues)")
            notes.append("Consider similarity to HET-s, curli, or other β-solenoid amyloids")
        
        if fold == AmyloidFold.BETA_ARCADE:
            notes.append("β-arcade topology similar to Aβ and tau cryo-EM structures")
        
        # Confidence notes
        if features['mean_beta_propensity'] < 1.0:
            notes.append("Low β-sheet propensity - may require cofactors for aggregation")
        
        return notes


# =============================================================================
# Known Amyloid Structure Database
# =============================================================================

# Representative solved amyloid structures for template matching
KNOWN_AMYLOID_STRUCTURES = {
    "abeta_2nao": {
        "name": "Aβ(1-42) fibril",
        "pdb_id": "2NAO",
        "sequence": "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA",
        "fold": AmyloidFold.BETA_ARCADE,
        "geometry": CrossBetaGeometry.PARALLEL_IN_REGISTER,
        "zipper_class": StericZipperClass.CLASS_1,
        "disease": "Alzheimer's disease",
        "reference": "Gremer et al. (2017) Science",
    },
    "asyn_2n0a": {
        "name": "α-synuclein fibril",
        "pdb_id": "2N0A",
        "sequence": "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA",
        "fold": AmyloidFold.GREEK_KEY,
        "geometry": CrossBetaGeometry.PARALLEL_IN_REGISTER,
        "zipper_class": StericZipperClass.CLASS_1,
        "disease": "Parkinson's disease",
        "reference": "Tuttle et al. (2016) Nat Struct Mol Biol",
    },
    "tau_5o3l": {
        "name": "Tau paired helical filament",
        "pdb_id": "5O3L",
        "sequence": "VQIINK",  # Core hexapeptide
        "fold": AmyloidFold.BETA_ARCADE,
        "geometry": CrossBetaGeometry.PARALLEL_IN_REGISTER,
        "zipper_class": StericZipperClass.CLASS_1,
        "disease": "Alzheimer's disease",
        "reference": "Fitzpatrick et al. (2017) Nature",
    },
    "hets_2rnm": {
        "name": "HET-s prion domain",
        "pdb_id": "2RNM",
        "sequence": "GIDAASHIQNVGDGAGPNPQ",  # Repeat unit
        "fold": AmyloidFold.BETA_SOLENOID,
        "geometry": CrossBetaGeometry.PARALLEL_IN_REGISTER,
        "zipper_class": StericZipperClass.CLASS_1,
        "disease": None,  # Functional amyloid
        "reference": "Wasmer et al. (2008) Science",
    },
    "gnnqqny_1yjp": {
        "name": "GNNQQNY steric zipper",
        "pdb_id": "1YJP",
        "sequence": "GNNQQNY",
        "fold": AmyloidFold.STERIC_ZIPPER,
        "geometry": CrossBetaGeometry.ANTIPARALLEL,
        "zipper_class": StericZipperClass.CLASS_8,
        "disease": None,  # Yeast prion model
        "reference": "Nelson et al. (2005) Nature",
    },
    "klvffa_2y3j": {
        "name": "KLVFFA steric zipper",
        "pdb_id": "2Y3J",
        "sequence": "KLVFFA",
        "fold": AmyloidFold.STERIC_ZIPPER,
        "geometry": CrossBetaGeometry.PARALLEL_IN_REGISTER,
        "zipper_class": StericZipperClass.CLASS_1,
        "disease": "Alzheimer's disease (core)",
        "reference": "Colletier et al. (2011) PNAS",
    },
}


def find_similar_structures(
    sequence: str,
    min_identity: float = 0.5,
) -> list[dict]:
    """
    Find known amyloid structures similar to the query sequence.
    
    Args:
        sequence: Query sequence
        min_identity: Minimum sequence identity for matches
        
    Returns:
        List of matching structures with similarity scores
    """
    matches = []
    
    for struct_id, struct_info in KNOWN_AMYLOID_STRUCTURES.items():
        template_seq = struct_info["sequence"]
        
        # Simple identity calculation (for short sequences)
        # For longer sequences, would use proper alignment
        identity = calculate_sequence_identity(sequence, template_seq)
        
        if identity >= min_identity:
            matches.append({
                **struct_info,
                "struct_id": struct_id,
                "identity": identity,
            })
    
    # Sort by identity
    matches.sort(key=lambda x: x["identity"], reverse=True)
    
    return matches


def calculate_sequence_identity(seq1: str, seq2: str) -> float:
    """
    Calculate sequence identity between two sequences.
    
    For short sequences, uses ungapped comparison.
    For longer sequences, would ideally use alignment.
    """
    # Ensure uppercase
    seq1 = seq1.upper()
    seq2 = seq2.upper()
    
    # Check if one is contained in the other
    if seq1 in seq2 or seq2 in seq1:
        shorter = min(len(seq1), len(seq2))
        return shorter / max(len(seq1), len(seq2))
    
    # Sliding window comparison for partial matches
    if len(seq1) < len(seq2):
        query, template = seq1, seq2
    else:
        query, template = seq2, seq1
    
    best_identity = 0.0
    for i in range(len(template) - len(query) + 1):
        matches = sum(1 for j, aa in enumerate(query) if template[i+j] == aa)
        identity = matches / len(query)
        best_identity = max(best_identity, identity)
    
    return best_identity


# =============================================================================
# Convenience Functions
# =============================================================================

def predict_polymorph(sequence: str) -> PolymorphPrediction:
    """
    Quick polymorph prediction for a sequence.
    
    Args:
        sequence: Protein sequence
        
    Returns:
        PolymorphPrediction with classification results
    """
    classifier = PolymorphClassifier()
    return classifier.predict(sequence)


def get_known_structures() -> dict:
    """Get dictionary of known amyloid structures."""
    return KNOWN_AMYLOID_STRUCTURES.copy()
