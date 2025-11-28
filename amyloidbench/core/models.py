"""
Core data models for AmyloidBench.

This module defines the fundamental data structures used throughout the pipeline,
including sequence representations, prediction results, and amyloidogenic regions.
All models use Pydantic for validation and serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class AmyloidPolymorph(str, Enum):
    """
    Classification of amyloid structural polymorphs.
    
    Based on the structural architecture of the amyloid core:
    - CROSS_BETA_PARALLEL: Parallel in-register β-sheets (most disease amyloids)
    - CROSS_BETA_ANTIPARALLEL: Antiparallel β-sheet arrangement
    - BETA_SOLENOID: β-helix/solenoid architecture (HET-s prion, curli)
    - CROSS_ALPHA: α-helical amyloid structure (PSMα3)
    - AMORPHOUS: Non-fibrillar aggregates
    - UNKNOWN: Structure not determined or ambiguous
    """
    CROSS_BETA_PARALLEL = "cross_beta_parallel"
    CROSS_BETA_ANTIPARALLEL = "cross_beta_antiparallel"
    BETA_SOLENOID = "beta_solenoid"
    CROSS_ALPHA = "cross_alpha"
    AMORPHOUS = "amorphous"
    UNKNOWN = "unknown"


class PredictionConfidence(str, Enum):
    """Confidence levels for predictions."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


class AminoAcidProperties(BaseModel):
    """
    Biophysical properties of amino acids relevant to amyloidogenicity.
    
    These properties form the basis of the fallback predictor's feature space.
    """
    hydrophobicity_kd: float = Field(..., description="Kyte-Doolittle hydrophobicity")
    hydrophobicity_eisenberg: float = Field(..., description="Eisenberg consensus scale")
    beta_propensity: float = Field(..., description="Chou-Fasman β-sheet propensity")
    alpha_propensity: float = Field(..., description="Chou-Fasman α-helix propensity")
    charge: float = Field(..., description="Net charge at pH 7.0")
    volume: float = Field(..., description="Side chain volume (Å³)")
    aromaticity: bool = Field(..., description="Contains aromatic ring")
    is_gatekeeper: bool = Field(..., description="Acts as aggregation gatekeeper (P, K, R, E, D)")


class Region(BaseModel):
    """
    Represents a contiguous region within a protein sequence.
    
    Used for annotating amyloidogenic regions (APRs), aggregation-prone regions,
    and other sequence features.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    start: int = Field(..., ge=0, description="0-indexed start position (inclusive)")
    end: int = Field(..., ge=0, description="0-indexed end position (exclusive)")
    sequence: str = Field(..., min_length=1, description="Amino acid sequence of the region")
    
    # Optional metadata
    score: Optional[float] = Field(None, description="Associated score (tool-dependent)")
    confidence: Optional[PredictionConfidence] = None
    annotation: Optional[str] = Field(None, description="Additional annotation/notes")
    
    @field_validator("end")
    @classmethod
    def end_after_start(cls, v: int, info) -> int:
        if "start" in info.data and v <= info.data["start"]:
            raise ValueError("end must be greater than start")
        return v
    
    @property
    def length(self) -> int:
        """Length of the region in residues."""
        return self.end - self.start
    
    def overlaps(self, other: Region) -> bool:
        """Check if this region overlaps with another."""
        return not (self.end <= other.start or other.end <= self.start)
    
    def merge(self, other: Region) -> Region:
        """Merge two overlapping regions."""
        if not self.overlaps(other):
            raise ValueError("Cannot merge non-overlapping regions")
        new_start = min(self.start, other.start)
        new_end = max(self.end, other.end)
        # Sequence needs to be provided from the parent sequence
        return Region(
            start=new_start,
            end=new_end,
            sequence="",  # Must be filled by caller
            score=max(self.score or 0, other.score or 0) if (self.score or other.score) else None,
        )


class PerResidueScores(BaseModel):
    """
    Per-residue prediction scores across a protein sequence.
    
    This is the fundamental output format for most amyloidogenicity predictors,
    allowing visualization of score profiles and identification of APRs.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    scores: list[float] = Field(..., description="Score for each residue position")
    sequence: str = Field(..., description="The protein sequence")
    predictor: str = Field(..., description="Name of the predictor")
    
    # Score metadata
    score_type: str = Field("raw", description="Type of score: raw, normalized, probability")
    threshold: Optional[float] = Field(None, description="Threshold for positive prediction")
    min_score: Optional[float] = Field(None, description="Theoretical minimum score")
    max_score: Optional[float] = Field(None, description="Theoretical maximum score")
    
    @field_validator("scores")
    @classmethod
    def validate_scores_length(cls, v: list[float], info) -> list[float]:
        if "sequence" in info.data and len(v) != len(info.data["sequence"]):
            raise ValueError(
                f"scores length ({len(v)}) must match sequence length ({len(info.data['sequence'])})"
            )
        return v
    
    def normalize(self, method: str = "minmax") -> PerResidueScores:
        """
        Normalize scores to [0, 1] range.
        
        Args:
            method: Normalization method - "minmax" or "zscore"
        
        Returns:
            New PerResidueScores with normalized values
        """
        arr = np.array(self.scores)
        
        if method == "minmax":
            min_val = self.min_score if self.min_score is not None else arr.min()
            max_val = self.max_score if self.max_score is not None else arr.max()
            if max_val - min_val > 0:
                normalized = (arr - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(arr)
        elif method == "zscore":
            mean, std = arr.mean(), arr.std()
            if std > 0:
                normalized = (arr - mean) / std
            else:
                normalized = np.zeros_like(arr)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return PerResidueScores(
            scores=normalized.tolist(),
            sequence=self.sequence,
            predictor=self.predictor,
            score_type="normalized",
            threshold=None,  # Threshold no longer valid after normalization
            min_score=0.0 if method == "minmax" else None,
            max_score=1.0 if method == "minmax" else None,
        )
    
    def to_regions(
        self, 
        threshold: Optional[float] = None,
        min_length: int = 5,
        merge_gap: int = 2
    ) -> list[Region]:
        """
        Convert per-residue scores to discrete regions above threshold.
        
        Args:
            threshold: Score threshold (uses self.threshold if None)
            min_length: Minimum region length to report
            merge_gap: Merge regions separated by ≤ this many residues
        
        Returns:
            List of Region objects representing APRs
        """
        thresh = threshold if threshold is not None else self.threshold
        if thresh is None:
            raise ValueError("No threshold specified")
        
        regions = []
        in_region = False
        start = 0
        
        for i, score in enumerate(self.scores):
            if score >= thresh and not in_region:
                in_region = True
                start = i
            elif score < thresh and in_region:
                in_region = False
                if i - start >= min_length:
                    regions.append(Region(
                        start=start,
                        end=i,
                        sequence=self.sequence[start:i],
                        score=float(np.mean(self.scores[start:i])),
                    ))
        
        # Handle region extending to end
        if in_region and len(self.scores) - start >= min_length:
            regions.append(Region(
                start=start,
                end=len(self.scores),
                sequence=self.sequence[start:],
                score=float(np.mean(self.scores[start:])),
            ))
        
        # Merge nearby regions
        if merge_gap > 0 and len(regions) > 1:
            merged = [regions[0]]
            for region in regions[1:]:
                if region.start - merged[-1].end <= merge_gap:
                    # Merge regions
                    merged[-1] = Region(
                        start=merged[-1].start,
                        end=region.end,
                        sequence=self.sequence[merged[-1].start:region.end],
                        score=float(np.mean(self.scores[merged[-1].start:region.end])),
                    )
                else:
                    merged.append(region)
            regions = merged
        
        return regions


class PredictionResult(BaseModel):
    """
    Complete prediction result from a single predictor.
    
    Contains both the raw per-residue scores and derived regions,
    along with metadata about the prediction.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Input identification
    sequence_id: str = Field(..., description="Identifier for the input sequence")
    sequence: str = Field(..., description="The protein sequence")
    
    # Predictor info
    predictor_name: str = Field(..., description="Name of the predictor tool")
    predictor_version: Optional[str] = Field(None, description="Version of the predictor")
    
    # Results
    per_residue_scores: Optional[PerResidueScores] = None
    predicted_regions: list[Region] = Field(default_factory=list)
    
    # Binary classification
    is_amyloidogenic: Optional[bool] = Field(None, description="Overall amyloid classification")
    amyloid_probability: Optional[float] = Field(None, ge=0, le=1)
    
    # Structural classification
    predicted_polymorph: Optional[AmyloidPolymorph] = None
    
    # Metadata
    raw_output: Optional[dict[str, Any]] = Field(None, description="Raw predictor output")
    runtime_seconds: Optional[float] = Field(None, ge=0)
    error_message: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Whether the prediction completed successfully."""
        return self.error_message is None


class ConsensusResult(BaseModel):
    """
    Consensus prediction aggregated from multiple predictors.
    
    Combines results using weighted voting or meta-prediction to
    provide a unified amyloidogenicity assessment.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Input
    sequence_id: str
    sequence: str
    
    # Individual predictions
    individual_results: dict[str, PredictionResult] = Field(
        ..., description="Results from each predictor, keyed by predictor name"
    )
    
    # Consensus scores
    consensus_per_residue: Optional[PerResidueScores] = Field(
        None, description="Meta-score combining all predictors"
    )
    consensus_regions: list[Region] = Field(default_factory=list)
    
    # Consensus classification
    consensus_is_amyloidogenic: Optional[bool] = None
    consensus_probability: Optional[float] = Field(None, ge=0, le=1)
    consensus_polymorph: Optional[AmyloidPolymorph] = None
    
    # Voting statistics
    n_predictors_agree_positive: int = 0
    n_predictors_agree_negative: int = 0
    n_predictors_total: int = 0
    
    # Method metadata
    consensus_method: str = Field("majority_vote", description="Method used for consensus")
    predictor_weights: Optional[dict[str, float]] = None
    
    def agreement_ratio(self) -> float:
        """Calculate the ratio of predictors agreeing with the consensus."""
        if self.n_predictors_total == 0:
            return 0.0
        agreeing = max(self.n_predictors_agree_positive, self.n_predictors_agree_negative)
        return agreeing / self.n_predictors_total


class ProteinRecord(BaseModel):
    """
    Complete protein record with sequence, structure, and annotations.
    
    This is the primary input object for the prediction pipeline.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Identification
    id: str = Field(..., description="Unique identifier (e.g., UniProt accession)")
    name: Optional[str] = Field(None, description="Protein name")
    organism: Optional[str] = None
    
    # Sequence
    sequence: str = Field(..., min_length=1)
    
    # Structure (optional)
    structure_path: Optional[Path] = Field(None, description="Path to PDB/mmCIF file")
    structure_source: Optional[str] = Field(None, description="PDB, AlphaFold, ESMFold, etc.")
    
    # Known annotations
    known_amyloid_regions: list[Region] = Field(
        default_factory=list, description="Experimentally validated APRs"
    )
    is_known_amyloid: Optional[bool] = Field(None, description="Known amyloid-forming protein")
    known_polymorph: Optional[AmyloidPolymorph] = None
    
    # Source database
    source_database: Optional[str] = Field(None, description="WALTZ-DB, Cross-Beta DB, etc.")
    
    @property
    def sequence_length(self) -> int:
        """Length of the protein sequence."""
        return len(self.sequence)
    
    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, v: str) -> str:
        """Validate that sequence contains only valid amino acid characters."""
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        ambiguous = set("BXZJUO")  # Allow ambiguous codes
        allowed = valid_aa | ambiguous
        
        v = v.upper().replace(" ", "").replace("\n", "")
        invalid = set(v) - allowed
        
        if invalid:
            raise ValueError(f"Invalid amino acid characters: {invalid}")
        
        return v


@dataclass
class BenchmarkDataset:
    """
    Dataset for benchmarking predictor performance.
    
    Contains both positive (amyloidogenic) and negative examples
    with ground truth annotations.
    """
    name: str
    description: str
    proteins: list[ProteinRecord] = field(default_factory=list)
    source_url: Optional[str] = None
    citation: Optional[str] = None
    
    @property
    def n_positive(self) -> int:
        """Number of positive (amyloidogenic) examples."""
        return sum(1 for p in self.proteins if p.is_known_amyloid)
    
    @property
    def n_negative(self) -> int:
        """Number of negative examples."""
        return sum(1 for p in self.proteins if p.is_known_amyloid is False)
    
    @property
    def n_total(self) -> int:
        """Total number of proteins."""
        return len(self.proteins)
    
    def get_positive_examples(self) -> list[ProteinRecord]:
        """Get all positive examples."""
        return [p for p in self.proteins if p.is_known_amyloid]
    
    def get_negative_examples(self) -> list[ProteinRecord]:
        """Get all negative examples."""
        return [p for p in self.proteins if p.is_known_amyloid is False]
