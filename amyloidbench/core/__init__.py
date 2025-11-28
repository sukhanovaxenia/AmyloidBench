"""
Core data structures and utilities for AmyloidBench.

This module provides the foundational components for working with protein
sequences, structures, and prediction results. The design follows the
principle of separating biological data representation from computational
methods.

Modules:
    models: Pydantic-based data models for sequences, regions, and results
    sequence: Sequence parsing, validation, and composition analysis
    structure: PDB/mmCIF handling and structural feature extraction
"""

from .models import (
    AmyloidPolymorph,
    BenchmarkDataset,
    ConsensusResult,
    PerResidueScores,
    PredictionConfidence,
    PredictionResult,
    ProteinRecord,
    Region,
)
from .sequence import (
    AA_CLASSES,
    STANDARD_AA,
    SequenceError,
    SequenceValidator,
    calculate_class_composition,
    calculate_composition,
    extract_region_context,
    find_low_complexity_regions,
    find_motifs,
    parse_fasta,
    sequence_hash,
    sliding_window,
    to_fasta,
)

__all__ = [
    # Models
    "ProteinRecord",
    "Region",
    "PerResidueScores",
    "PredictionResult",
    "ConsensusResult",
    "AmyloidPolymorph",
    "PredictionConfidence",
    "BenchmarkDataset",
    # Sequence utilities
    "SequenceValidator",
    "SequenceError",
    "parse_fasta",
    "to_fasta",
    "sequence_hash",
    "calculate_composition",
    "calculate_class_composition",
    "find_motifs",
    "find_low_complexity_regions",
    "extract_region_context",
    "sliding_window",
    "STANDARD_AA",
    "AA_CLASSES",
]
