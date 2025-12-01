"""
FoldAmyloid re-implementation for amyloidogenicity prediction.

FoldAmyloid identifies amyloidogenic regions based on the observation that
residues in amyloid fibrils exhibit unusually high packing density due to
the tightly interdigitated "steric zipper" architecture of the cross-β core.

The algorithm uses two amino acid scales derived from globular protein
structures:
1. Expected average number of contacts per residue
2. Expected packing density (related to side chain volume and contacts)

Amyloidogenic segments are identified as regions where the mean expected
packing density exceeds a threshold, indicating propensity to form the
dense intermolecular contacts characteristic of amyloid structures.

Algorithm:
1. Calculate per-residue expected packing density from the scale
2. Apply sliding window averaging (default: 5 residues)
3. Identify regions exceeding the amyloidogenicity threshold (21.4)

The biological rationale is that sequences forming amyloid fibrils must
be capable of achieving the extremely dense packing observed in steric
zipper structures, where side chains from opposing β-sheets interlock
like the teeth of a zipper.

Reference:
    Garbuzynskiy, S.O., Lobanov, M.Y., & Galzitskaya, O.V. (2010).
    FoldAmyloid: a method of prediction of amyloidogenic regions from
    protein sequence. Bioinformatics, 26(3), 326-332.
    DOI: 10.1093/bioinformatics/btp691
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from amyloidbench.core.models import (
    PerResidueScores,
    PredictionResult,
    Region,
)
from amyloidbench.predictors.base import (
    BasePredictor,
    PredictorCapability,
    PredictorConfig,
    PredictorType,
    register_predictor,
)

logger = logging.getLogger(__name__)


# =============================================================================
# FoldAmyloid Amino Acid Scales
# =============================================================================

# Expected number of contacts per residue
# Derived from analysis of globular protein structures
# Higher values indicate residues that typically make more contacts
EXPECTED_CONTACTS = {
    'A': 5.46, 'R': 5.88, 'N': 5.07, 'D': 4.89, 'C': 5.80,
    'Q': 5.23, 'E': 4.87, 'G': 4.67, 'H': 5.64, 'I': 6.53,
    'L': 6.23, 'K': 5.32, 'M': 6.08, 'F': 6.61, 'P': 4.76,
    'S': 4.98, 'T': 5.36, 'W': 6.51, 'Y': 6.22, 'V': 6.21,
}

# Expected packing density scale
# This is the primary scale used by FoldAmyloid
# Calculated as: (mean number of contacts) × (contact scale factor)
# Higher values indicate denser expected packing
PACKING_DENSITY = {
    'A': 20.90, 'R': 21.56, 'N': 19.53, 'D': 19.02, 'C': 22.30,
    'Q': 20.37, 'E': 18.94, 'G': 17.12, 'H': 21.56, 'I': 24.96,
    'L': 23.66, 'K': 20.31, 'M': 23.24, 'F': 25.52, 'P': 18.28,
    'S': 19.31, 'T': 20.79, 'W': 25.51, 'Y': 24.16, 'V': 23.96,
}

# Mean packing density across all amino acids (for normalization)
MEAN_PACKING_DENSITY = sum(PACKING_DENSITY.values()) / len(PACKING_DENSITY)

# Amyloidogenicity threshold from original FoldAmyloid
# Regions with mean packing density above this are predicted amyloidogenic
FOLDAMYLOID_THRESHOLD = 21.4

# Alternative "soft count" scale from the paper
# Based on expected number of contacts within 8Å
SOFT_CONTACTS = {
    'A': 5.89, 'R': 6.42, 'N': 5.47, 'D': 5.33, 'C': 6.31,
    'Q': 5.72, 'E': 5.32, 'G': 5.03, 'H': 6.16, 'I': 7.14,
    'L': 6.81, 'K': 5.84, 'M': 6.66, 'F': 7.24, 'P': 5.18,
    'S': 5.41, 'T': 5.83, 'W': 7.15, 'Y': 6.83, 'V': 6.80,
}


@register_predictor
class FoldAmyloidPredictor(BasePredictor):
    """
    FoldAmyloid predictor based on expected packing density.
    
    This implementation follows the algorithm described in Garbuzynskiy et al.
    (2010). The method identifies amyloidogenic regions by calculating the
    expected packing density for each residue and finding segments that
    exceed the empirically-derived threshold.
    
    The biological basis rests on the observation that amyloid fibrils
    achieve extraordinarily dense packing through the steric zipper motif,
    where side chains from opposing β-sheets interdigitate. Sequences
    capable of forming such structures must have compatible side chain
    volumes and shapes, reflected in high expected packing density.
    
    Performance characteristics (from original paper):
    - Sensitivity: 78.7% on amyloidogenic hexapeptides
    - Specificity: 66.2% on non-amyloidogenic sequences
    - Best performance on sequences with well-defined aggregation cores
    
    Attributes:
        window_size: Sliding window size for averaging (default: 5)
        scale: Which amino acid scale to use ('packing' or 'contacts')
        
    Usage:
        >>> predictor = FoldAmyloidPredictor()
        >>> protein = ProteinRecord(id="test", sequence="KLVFFAE")
        >>> result = predictor.predict(protein)
        >>> print(result.is_amyloidogenic)
        True
    """
    
    name = "FoldAmyloid"
    version = "1.0-reimpl"
    predictor_type = PredictorType.SEQUENCE_HEURISTIC
    capabilities = {
        PredictorCapability.PER_RESIDUE_SCORES,
        PredictorCapability.REGION_DETECTION,
        PredictorCapability.BINARY_CLASSIFICATION,
    }
    
    # Default threshold from original publication
    default_threshold = FOLDAMYLOID_THRESHOLD
    default_window_size = 5
    
    # Score range for packing density scale
    score_min = 17.0  # Glycine (lowest)
    score_max = 26.0  # Phenylalanine (highest)
    
    citation = (
        "Garbuzynskiy, S.O., Lobanov, M.Y., & Galzitskaya, O.V. (2010). "
        "FoldAmyloid: a method of prediction of amyloidogenic regions from "
        "protein sequence. Bioinformatics, 26(3), 326-332."
    )
    url = "http://bioinfo.protres.ru/fold-amyloid/"
    description = (
        "Predicts amyloidogenic regions based on expected packing density. "
        "High-density regions are prone to forming the tightly packed "
        "steric zipper structures characteristic of amyloid fibrils."
    )
    
    def __init__(
        self,
        config: Optional[PredictorConfig] = None,
        scale: str = "packing",
        use_triple_window: bool = False,
    ):
        """
        Initialize FoldAmyloid predictor.
        
        Args:
            config: Predictor configuration
            scale: Amino acid scale to use:
                   - 'packing': Expected packing density (default, recommended)
                   - 'contacts': Expected number of contacts
                   - 'soft': Soft contact count within 8Å
            use_triple_window: If True, use triple-window scheme from paper
                              for improved specificity (windows at -5, 0, +5)
        """
        super().__init__(config)
        
        if scale == "packing":
            self._scale = PACKING_DENSITY
            self._threshold = FOLDAMYLOID_THRESHOLD
        elif scale == "contacts":
            self._scale = EXPECTED_CONTACTS
            self._threshold = 5.8  # Adjusted threshold for contact scale
        elif scale == "soft":
            self._scale = SOFT_CONTACTS
            self._threshold = 6.3  # Adjusted threshold for soft contacts
        else:
            raise ValueError(f"Unknown scale: {scale}. Use 'packing', 'contacts', or 'soft'")
        
        self.scale_name = scale
        self.use_triple_window = use_triple_window
        
        logger.info(f"FoldAmyloid initialized with '{scale}' scale, threshold={self._threshold}")
    
    def _get_residue_score(self, aa: str) -> float:
        """
        Get the scale value for an amino acid.
        
        Args:
            aa: Single-letter amino acid code
            
        Returns:
            Scale value (uses mean for unknown residues)
        """
        return self._scale.get(aa.upper(), MEAN_PACKING_DENSITY)
    
    def _calculate_raw_scores(self, sequence: str) -> np.ndarray:
        """
        Calculate raw per-residue scores from the scale.
        
        Args:
            sequence: Protein sequence
            
        Returns:
            Array of scale values for each residue
        """
        return np.array([self._get_residue_score(aa) for aa in sequence])
    
    def _apply_window_averaging(
        self, 
        scores: np.ndarray, 
        window_size: int
    ) -> np.ndarray:
        """
        Apply sliding window averaging to smooth the score profile.
        
        The window averaging captures the local context that determines
        whether a residue contributes to an aggregation-prone segment.
        Single high-scoring residues are less significant than contiguous
        stretches of amyloidogenic residues.
        
        Args:
            scores: Raw per-residue scores
            window_size: Size of averaging window
            
        Returns:
            Smoothed scores (same length as input)
        """
        if window_size <= 1:
            return scores
        
        # Use convolution for efficient moving average
        kernel = np.ones(window_size) / window_size
        
        # 'same' mode preserves array length, padding with zeros
        # This matches the behavior in the original FoldAmyloid
        smoothed = np.convolve(scores, kernel, mode='same')
        
        return smoothed
    
    def _apply_triple_window(self, sequence: str) -> np.ndarray:
        """
        Apply the triple-window scheme from the original paper.
        
        This scheme evaluates each position using three overlapping windows
        centered at positions (i-5, i, i+5), taking the maximum. This
        captures broader context and can improve specificity for identifying
        extended aggregation-prone regions.
        
        Args:
            sequence: Protein sequence
            
        Returns:
            Triple-window smoothed scores
        """
        raw_scores = self._calculate_raw_scores(sequence)
        window = self.window_size
        n = len(sequence)
        
        # Calculate window averages at three offsets
        scores_center = self._apply_window_averaging(raw_scores, window)
        
        # For shifted windows, we pad and then take max
        final_scores = scores_center.copy()
        
        for offset in [-5, 5]:
            shifted = np.zeros(n)
            for i in range(n):
                # Window centered at i + offset
                start = max(0, i + offset - window // 2)
                end = min(n, i + offset + window // 2 + 1)
                if start < end:
                    shifted[i] = np.mean(raw_scores[start:end])
            
            # Take maximum across windows
            final_scores = np.maximum(final_scores, shifted)
        
        return final_scores
    
    def _predict_impl(
        self,
        sequence: str,
        structure_path=None,  # Not used - FoldAmyloid is sequence-only
    ) -> PredictionResult:
        """
        Run FoldAmyloid prediction.
        
        The algorithm:
        1. Look up packing density for each residue
        2. Apply sliding window averaging
        3. Identify regions exceeding threshold
        4. Merge nearby regions and filter by minimum length
        
        Args:
            sequence: Protein sequence
            structure_path: Ignored (sequence-only method)
            
        Returns:
            PredictionResult with per-residue scores and identified APRs
        """
        # Calculate scores
        if self.use_triple_window:
            scores = self._apply_triple_window(sequence)
        else:
            raw_scores = self._calculate_raw_scores(sequence)
            scores = self._apply_window_averaging(raw_scores, self.window_size)
        
        # Create per-residue scores object
        per_residue = PerResidueScores(
            scores=scores.tolist(),
            sequence=sequence,
            predictor=self.name,
            score_type="raw",
            threshold=self._threshold,
            min_score=self.score_min,
            max_score=self.score_max,
        )
        
        # Extract regions above threshold
        regions = per_residue.to_regions(
            threshold=self._threshold,
            min_length=self.config.min_region_length,
            merge_gap=self.config.merge_gap,
        )
        
        # Calculate amyloid probability
        # Based on fraction of sequence in APRs and max score
        if regions:
            apr_coverage = sum(r.length for r in regions) / len(sequence)
            max_score = max(r.score for r in regions if r.score)
            # Sigmoid-like transformation
            excess = max_score - self._threshold
            probability = min(0.5 + 0.5 * np.tanh(excess / 2) + 0.2 * apr_coverage, 1.0)
        else:
            probability = max(0, (np.mean(scores) - self._threshold + 2) / 4)
            probability = min(max(probability, 0), 0.5)
        
        is_amyloidogenic = len(regions) > 0
        
        return PredictionResult(
            sequence_id="",
            sequence=sequence,
            predictor_name=self.name,
            predictor_version=self.version,
            per_residue_scores=per_residue,
            predicted_regions=regions,
            is_amyloidogenic=is_amyloidogenic,
            amyloid_probability=float(probability),
            raw_output={
                "scale": self.scale_name,
                "threshold": self._threshold,
                "mean_score": float(np.mean(scores)),
                "max_score": float(np.max(scores)),
                "use_triple_window": self.use_triple_window,
            } if self.config.return_raw_output else None,
        )


# =============================================================================
# Convenience functions
# =============================================================================

def predict_with_foldamyloid(
    sequence: str,
    window_size: int = 5,
    threshold: Optional[float] = None,
) -> PredictionResult:
    """
    Quick FoldAmyloid prediction without creating explicit objects.
    
    Args:
        sequence: Protein sequence
        window_size: Sliding window size
        threshold: Custom threshold (None = use default 21.4)
        
    Returns:
        PredictionResult with FoldAmyloid scores
        
    Example:
        >>> result = predict_with_foldamyloid("KLVFFAEDVGSNK")
        >>> print(f"Amyloidogenic: {result.is_amyloidogenic}")
    """
    from amyloidbench.core.models import ProteinRecord
    
    config = PredictorConfig(
        threshold=threshold,
        use_cache=False,
    )
    config.window_size = window_size
    
    predictor = FoldAmyloidPredictor(config=config)
    protein = ProteinRecord(id="query", sequence=sequence)
    
    return predictor.predict(protein)


def get_packing_density_profile(sequence: str) -> list[tuple[str, float]]:
    """
    Get the packing density profile for a sequence.
    
    Useful for understanding which residues contribute most to
    aggregation propensity.
    
    Args:
        sequence: Protein sequence
        
    Returns:
        List of (residue, packing_density) tuples
        
    Example:
        >>> profile = get_packing_density_profile("KLVFFAE")
        >>> for aa, density in profile:
        ...     print(f"{aa}: {density:.2f}")
        K: 20.31
        L: 23.66
        V: 23.96
        F: 25.52
        F: 25.52
        A: 20.90
        E: 18.94
    """
    return [(aa, PACKING_DENSITY.get(aa, MEAN_PACKING_DENSITY)) for aa in sequence.upper()]
