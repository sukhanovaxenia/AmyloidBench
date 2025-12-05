"""
Standardized output formats for amyloid predictors.

This module defines unified output structures that all predictors (web-based,
local reimplementations, and standalone tools) must produce. This ensures
consistent downstream analysis, visualization, and benchmarking regardless
of the prediction source.

Output Components
-----------------
1. **Per-residue scores**: Continuous aggregation propensity values
2. **Per-residue classification**: Binary amyloidogenic/non-amyloidogenic labels
3. **Region annotations**: APR boundaries with metadata
4. **Visualization data**: Structures suitable for plotting
5. **Raw tool output**: Original output for debugging/validation

Design Rationale
----------------
Different predictors use different scoring schemes (energies, probabilities,
z-scores, percentages). The standardized format includes both the raw scores
and normalized scores (0-1 scale) to enable cross-predictor comparison while
preserving original information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union
import json
import csv
from datetime import datetime

import numpy as np


class ScoreType(str, Enum):
    """Types of scoring schemes used by predictors."""
    PROBABILITY = "probability"      # 0-1 range, higher = more amyloidogenic
    ENERGY = "energy"                # kcal/mol, lower = more amyloidogenic
    ZSCORE = "zscore"                # Standard deviations from mean
    PERCENTAGE = "percentage"        # 0-100 range
    RAW = "raw"                       # Arbitrary scale
    BINARY = "binary"                # 0 or 1


class ClassificationLabel(str, Enum):
    """Per-residue classification labels."""
    AMYLOIDOGENIC = "amyloidogenic"
    NON_AMYLOIDOGENIC = "non_amyloidogenic"
    UNCERTAIN = "uncertain"
    NOT_ANALYZED = "not_analyzed"     # e.g., terminal residues


@dataclass
class ResidueScore:
    """Score and classification for a single residue."""
    position: int                     # 1-indexed position
    residue: str                      # Single-letter amino acid
    raw_score: float                  # Original predictor score
    normalized_score: float           # 0-1 normalized (higher = more amyloidogenic)
    classification: ClassificationLabel
    confidence: Optional[float] = None  # Confidence in classification (0-1)
    
    def to_dict(self) -> dict:
        return {
            'position': self.position,
            'residue': self.residue,
            'raw_score': self.raw_score,
            'normalized_score': self.normalized_score,
            'classification': self.classification.value,
            'confidence': self.confidence,
        }


@dataclass
class PredictedRegion:
    """An aggregation-prone region with detailed annotation."""
    start: int                        # 1-indexed start position
    end: int                          # 1-indexed end position (inclusive)
    sequence: str                     # Region sequence
    mean_score: float                 # Mean raw score
    max_score: float                  # Maximum raw score
    mean_normalized: float            # Mean normalized score
    classification: ClassificationLabel = ClassificationLabel.AMYLOIDOGENIC
    region_type: Optional[str] = None  # e.g., "steric_zipper", "beta_arch"
    confidence: Optional[float] = None
    
    @property
    def length(self) -> int:
        return self.end - self.start + 1
    
    def to_dict(self) -> dict:
        return {
            'start': self.start,
            'end': self.end,
            'length': self.length,
            'sequence': self.sequence,
            'mean_score': self.mean_score,
            'max_score': self.max_score,
            'mean_normalized': self.mean_normalized,
            'classification': self.classification.value,
            'region_type': self.region_type,
            'confidence': self.confidence,
        }


@dataclass
class PredictorOutput:
    """
    Standardized output from any amyloid predictor.
    
    This unified format enables:
    - Cross-predictor comparison and consensus building
    - Consistent visualization across different tools
    - Standardized benchmarking metrics
    - Interoperability with downstream analysis pipelines
    
    Attributes:
        predictor_name: Name of the prediction tool
        predictor_version: Tool version for reproducibility
        sequence_id: Identifier for the input sequence
        sequence: The analyzed protein sequence
        residue_scores: Per-residue detailed scores and classifications
        predicted_regions: Identified aggregation-prone regions
        overall_classification: Whole-sequence classification
        overall_score: Summary score for the sequence
        score_type: Type of scoring used by the predictor
        threshold: Threshold used for classification
        execution_time: Time taken for prediction (seconds)
        timestamp: When the prediction was made
        source: 'web', 'local', or 'standalone'
        raw_output: Original tool output (for debugging)
        error: Error message if prediction failed
    """
    predictor_name: str
    predictor_version: str
    sequence_id: str
    sequence: str
    residue_scores: list[ResidueScore]
    predicted_regions: list[PredictedRegion]
    overall_classification: ClassificationLabel
    overall_score: float
    overall_probability: float        # Always 0-1 regardless of score_type
    score_type: ScoreType
    threshold: float
    execution_time: Optional[float] = None
    timestamp: Optional[str] = None
    source: str = "unknown"           # 'web', 'local', 'standalone'
    raw_output: Optional[dict] = None
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    @property
    def is_amyloidogenic(self) -> bool:
        return self.overall_classification == ClassificationLabel.AMYLOIDOGENIC
    
    @property
    def n_amyloidogenic_residues(self) -> int:
        return sum(1 for r in self.residue_scores 
                   if r.classification == ClassificationLabel.AMYLOIDOGENIC)
    
    @property
    def amyloidogenic_fraction(self) -> float:
        if not self.residue_scores:
            return 0.0
        return self.n_amyloidogenic_residues / len(self.residue_scores)
    
    def get_scores_array(self, normalized: bool = True) -> np.ndarray:
        """Get scores as numpy array."""
        if normalized:
            return np.array([r.normalized_score for r in self.residue_scores])
        return np.array([r.raw_score for r in self.residue_scores])
    
    def get_classification_array(self) -> np.ndarray:
        """Get binary classification array (1=amyloidogenic, 0=non-amyloidogenic)."""
        return np.array([
            1 if r.classification == ClassificationLabel.AMYLOIDOGENIC else 0
            for r in self.residue_scores
        ])
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'predictor_name': self.predictor_name,
            'predictor_version': self.predictor_version,
            'sequence_id': self.sequence_id,
            'sequence': self.sequence,
            'sequence_length': len(self.sequence),
            'overall_classification': self.overall_classification.value,
            'overall_score': self.overall_score,
            'overall_probability': self.overall_probability,
            'is_amyloidogenic': self.is_amyloidogenic,
            'n_amyloidogenic_residues': self.n_amyloidogenic_residues,
            'amyloidogenic_fraction': self.amyloidogenic_fraction,
            'score_type': self.score_type.value,
            'threshold': self.threshold,
            'n_regions': len(self.predicted_regions),
            'regions': [r.to_dict() for r in self.predicted_regions],
            'residue_scores': [r.to_dict() for r in self.residue_scores],
            'execution_time': self.execution_time,
            'timestamp': self.timestamp,
            'source': self.source,
            'error': self.error,
            'warnings': self.warnings,
        }
    
    def to_json(self, path: Optional[Union[str, Path]] = None, indent: int = 2) -> str:
        """Export to JSON format."""
        json_str = json.dumps(self.to_dict(), indent=indent)
        if path:
            Path(path).write_text(json_str)
        return json_str
    
    def to_csv(self, path: Union[str, Path]) -> None:
        """Export per-residue scores to CSV."""
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow([
                'position', 'residue', 'raw_score', 'normalized_score',
                'classification', 'confidence', 'in_region'
            ])
            
            # Determine which residues are in regions
            in_region = set()
            for region in self.predicted_regions:
                for pos in range(region.start, region.end + 1):
                    in_region.add(pos)
            
            # Data rows
            for r in self.residue_scores:
                writer.writerow([
                    r.position,
                    r.residue,
                    f"{r.raw_score:.4f}",
                    f"{r.normalized_score:.4f}",
                    r.classification.value,
                    f"{r.confidence:.4f}" if r.confidence else "",
                    "yes" if r.position in in_region else "no"
                ])
    
    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        import pandas as pd
        
        in_region = set()
        for region in self.predicted_regions:
            for pos in range(region.start, region.end + 1):
                in_region.add(pos)
        
        data = []
        for r in self.residue_scores:
            data.append({
                'position': r.position,
                'residue': r.residue,
                'raw_score': r.raw_score,
                'normalized_score': r.normalized_score,
                'classification': r.classification.value,
                'confidence': r.confidence,
                'in_region': r.position in in_region,
            })
        
        return pd.DataFrame(data)


@dataclass
class MultiPredictorOutput:
    """
    Combined output from multiple predictors for consensus analysis.
    """
    sequence_id: str
    sequence: str
    predictor_outputs: dict[str, PredictorOutput]
    consensus_scores: Optional[np.ndarray] = None
    consensus_classification: Optional[list[ClassificationLabel]] = None
    consensus_regions: Optional[list[PredictedRegion]] = None
    
    @property
    def n_predictors(self) -> int:
        return len(self.predictor_outputs)
    
    @property
    def successful_predictors(self) -> list[str]:
        return [name for name, out in self.predictor_outputs.items() 
                if out.error is None]
    
    @property
    def failed_predictors(self) -> list[str]:
        return [name for name, out in self.predictor_outputs.items() 
                if out.error is not None]
    
    def calculate_consensus(self, min_agreement: int = 3) -> None:
        """Calculate consensus from individual predictions."""
        n = len(self.sequence)
        successful = [out for out in self.predictor_outputs.values() 
                      if out.error is None]
        
        if not successful:
            return
        
        # Vote counting per residue
        votes = np.zeros(n)
        score_sum = np.zeros(n)
        
        for out in successful:
            for r in out.residue_scores:
                if r.classification == ClassificationLabel.AMYLOIDOGENIC:
                    votes[r.position - 1] += 1
                score_sum[r.position - 1] += r.normalized_score
        
        # Consensus scores (average normalized)
        self.consensus_scores = score_sum / len(successful)
        
        # Consensus classification
        self.consensus_classification = [
            ClassificationLabel.AMYLOIDOGENIC if votes[i] >= min_agreement
            else ClassificationLabel.NON_AMYLOIDOGENIC
            for i in range(n)
        ]
        
        # Extract consensus regions
        self.consensus_regions = []
        in_region = False
        region_start = 0
        
        for i, cls in enumerate(self.consensus_classification):
            if cls == ClassificationLabel.AMYLOIDOGENIC:
                if not in_region:
                    region_start = i
                    in_region = True
            else:
                if in_region:
                    self._add_consensus_region(region_start, i - 1, votes)
                    in_region = False
        
        if in_region:
            self._add_consensus_region(region_start, n - 1, votes)
    
    def _add_consensus_region(self, start: int, end: int, votes: np.ndarray) -> None:
        """Add a consensus region."""
        region_votes = votes[start:end + 1]
        region_scores = self.consensus_scores[start:end + 1]
        
        self.consensus_regions.append(PredictedRegion(
            start=start + 1,  # 1-indexed
            end=end + 1,
            sequence=self.sequence[start:end + 1],
            mean_score=float(np.mean(region_votes)),
            max_score=float(np.max(region_votes)),
            mean_normalized=float(np.mean(region_scores)),
            confidence=float(np.mean(region_votes) / self.n_predictors),
        ))
    
    def to_comparison_dataframe(self):
        """Create DataFrame comparing all predictors."""
        import pandas as pd
        
        data = []
        for i, aa in enumerate(self.sequence):
            row = {
                'position': i + 1,
                'residue': aa,
            }
            
            for name, out in self.predictor_outputs.items():
                if out.error is None and i < len(out.residue_scores):
                    r = out.residue_scores[i]
                    row[f'{name}_score'] = r.normalized_score
                    row[f'{name}_class'] = r.classification.value
                else:
                    row[f'{name}_score'] = None
                    row[f'{name}_class'] = None
            
            if self.consensus_scores is not None:
                row['consensus_score'] = self.consensus_scores[i]
                row['consensus_class'] = self.consensus_classification[i].value
            
            data.append(row)
        
        return pd.DataFrame(data)


# =============================================================================
# SCORE NORMALIZATION UTILITIES
# =============================================================================

def normalize_scores(
    scores: list[float],
    score_type: ScoreType,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> list[float]:
    """
    Normalize scores to 0-1 range where higher = more amyloidogenic.
    
    Different predictors use different scoring conventions:
    - Probability: Already 0-1, higher = more amyloidogenic
    - Energy: Lower = more stable aggregate = more amyloidogenic
    - Z-score: Higher = more aggregation prone
    - Percentage: 0-100, needs scaling
    - Raw: Arbitrary, use min-max normalization
    """
    scores_arr = np.array(scores)
    
    if score_type == ScoreType.PROBABILITY:
        # Already normalized
        return np.clip(scores_arr, 0, 1).tolist()
    
    elif score_type == ScoreType.PERCENTAGE:
        # Scale 0-100 to 0-1
        return (scores_arr / 100).clip(0, 1).tolist()
    
    elif score_type == ScoreType.ENERGY:
        # Lower energy = more amyloidogenic, invert
        if min_val is None:
            min_val = scores_arr.min()
        if max_val is None:
            max_val = scores_arr.max()
        
        if max_val == min_val:
            return [0.5] * len(scores)
        
        # Normalize and invert
        normalized = (scores_arr - min_val) / (max_val - min_val)
        return (1 - normalized).clip(0, 1).tolist()
    
    elif score_type == ScoreType.ZSCORE:
        # Z-scores: use sigmoid transformation
        # Z > 0 = above mean aggregation propensity
        normalized = 1 / (1 + np.exp(-scores_arr))
        return normalized.clip(0, 1).tolist()
    
    elif score_type == ScoreType.RAW:
        # Min-max normalization
        if min_val is None:
            min_val = scores_arr.min()
        if max_val is None:
            max_val = scores_arr.max()
        
        if max_val == min_val:
            return [0.5] * len(scores)
        
        normalized = (scores_arr - min_val) / (max_val - min_val)
        return normalized.clip(0, 1).tolist()
    
    elif score_type == ScoreType.BINARY:
        return scores_arr.astype(float).tolist()
    
    else:
        raise ValueError(f"Unknown score type: {score_type}")


def classify_residues(
    normalized_scores: list[float],
    threshold: float = 0.5,
    uncertain_margin: float = 0.1,
) -> list[ClassificationLabel]:
    """
    Classify residues based on normalized scores.
    
    Args:
        normalized_scores: 0-1 normalized scores
        threshold: Classification threshold
        uncertain_margin: Margin around threshold for uncertain classification
    """
    classifications = []
    
    for score in normalized_scores:
        if score >= threshold + uncertain_margin:
            classifications.append(ClassificationLabel.AMYLOIDOGENIC)
        elif score <= threshold - uncertain_margin:
            classifications.append(ClassificationLabel.NON_AMYLOIDOGENIC)
        else:
            classifications.append(ClassificationLabel.UNCERTAIN)
    
    return classifications
