"""
AmyloidBench: Comprehensive consensus meta-predictor for protein amyloidogenicity.

This package provides a unified framework for predicting amyloidogenic regions
in proteins using multiple established algorithms, with proper benchmarking
against curated databases and consensus scoring methodologies.

The biological basis for amyloid prediction rests on the observation that
certain amino acid sequence patterns—particularly hydrophobic segments with
high β-sheet propensity—are intrinsically prone to forming the cross-β
architecture characteristic of amyloid fibrils. These aggregation-prone
regions (APRs) are evolutionarily suppressed in most proteins through
"gatekeeper" residues (P, K, R, E, D) but remain detectable through
computational methods.

Key components:
    - core: Sequence and structure handling, data models
    - predictors: Individual predictor implementations
    - features: Feature extraction for the fallback predictor
    - classification: Binary and polymorph classification
    - benchmark: Database loaders and metrics calculation
    - visualization: Score profiles and comparison plots
    - cli: Command-line interface

Basic usage:
    >>> from amyloidbench import predict
    >>> from amyloidbench.core.models import ProteinRecord
    >>> 
    >>> protein = ProteinRecord(
    ...     id="test",
    ...     sequence="MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
    ... )
    >>> result = predict(protein)
    >>> print(f"Amyloidogenic: {result.is_amyloidogenic}")
    >>> for region in result.predicted_regions:
    ...     print(f"  APR: {region.start}-{region.end} ({region.sequence})")

For more information, see the documentation at https://amyloidbench.readthedocs.io

Author: Xenia (ARRIAM)
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Xenia"
__email__ = "xenia@arriam.ru"

from .core.models import (
    AmyloidPolymorph,
    ConsensusResult,
    PerResidueScores,
    PredictionResult,
    ProteinRecord,
    Region,
)
from .core.sequence import parse_fasta, sequence_hash
from .predictors.base import (
    BasePredictor,
    PredictorConfig,
    PredictorType,
    get_predictor,
    list_predictors,
)

# Import concrete predictors to register them
from .predictors.local import aggrescan3d


def predict(
    protein: ProteinRecord,
    predictors: list[str] | None = None,
    consensus: bool = True,
) -> PredictionResult | ConsensusResult:
    """
    Run amyloidogenicity prediction on a protein.
    
    This is the main high-level interface for predictions. For more control,
    use individual predictor classes directly.
    
    Args:
        protein: ProteinRecord with sequence (and optionally structure)
        predictors: List of predictor names to use (default: all available)
        consensus: If True and multiple predictors, return consensus result
    
    Returns:
        PredictionResult if single predictor, ConsensusResult if consensus
    
    Example:
        >>> from amyloidbench import predict
        >>> from amyloidbench.core.models import ProteinRecord
        >>> 
        >>> protein = ProteinRecord(id="PrP", sequence="MANLGCWMLVLFVATWSDLGLC...")
        >>> result = predict(protein, predictors=["Aggrescan3D"])
        >>> print(f"Found {len(result.predicted_regions)} APRs")
    """
    available = {p["name"]: p for p in list_predictors()}
    
    if predictors is None:
        predictors = list(available.keys())
    
    if len(predictors) == 1:
        pred = get_predictor(predictors[0])
        return pred.predict(protein)
    
    # Multiple predictors
    results = {}
    for pred_name in predictors:
        if pred_name not in available:
            continue
        pred = get_predictor(pred_name)
        results[pred_name] = pred.predict(protein)
    
    if not consensus:
        # Return first successful result
        for result in results.values():
            if result.success:
                return result
        return list(results.values())[0]  # Return first even if failed
    
    # Build consensus result
    # (Full consensus implementation in Phase 4)
    from .core.models import ConsensusResult
    
    n_positive = sum(1 for r in results.values() if r.is_amyloidogenic)
    n_negative = sum(1 for r in results.values() if r.is_amyloidogenic is False)
    
    return ConsensusResult(
        sequence_id=protein.id,
        sequence=protein.sequence,
        individual_results=results,
        consensus_is_amyloidogenic=n_positive > n_negative,
        n_predictors_agree_positive=n_positive,
        n_predictors_agree_negative=n_negative,
        n_predictors_total=len(results),
        consensus_method="majority_vote",
    )


__all__ = [
    # Version
    "__version__",
    # Main function
    "predict",
    # Models
    "ProteinRecord",
    "Region",
    "PredictionResult",
    "ConsensusResult",
    "PerResidueScores",
    "AmyloidPolymorph",
    # Sequence utilities
    "parse_fasta",
    "sequence_hash",
    # Predictor system
    "BasePredictor",
    "PredictorConfig",
    "PredictorType",
    "get_predictor",
    "list_predictors",
]
