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
from .predictors.local.reimplemented import foldamyloid
from .predictors.local import fallback

# Import consensus engine
from .consensus import ConsensusEngine, ConsensusMethod, quick_consensus


def predict(
    protein: ProteinRecord,
    predictors: list[str] | None = None,
    consensus: bool = True,
    method: str = "weighted_average",
) -> PredictionResult | ConsensusResult:
    """
    Run amyloidogenicity prediction on a protein.
    
    This is the main high-level interface for predictions. For more control,
    use individual predictor classes or the ConsensusEngine directly.
    
    Args:
        protein: ProteinRecord with sequence (and optionally structure)
        predictors: List of predictor names to use (default: all available)
        consensus: If True and multiple predictors, return consensus result
        method: Consensus method ('majority_vote', 'weighted_average', 
                'intersection', 'union')
    
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
    
    # Multiple predictors - use consensus engine
    if consensus:
        # Map string method to enum
        method_map = {
            "majority_vote": ConsensusMethod.MAJORITY_VOTE,
            "weighted_average": ConsensusMethod.WEIGHTED_AVERAGE,
            "intersection": ConsensusMethod.INTERSECTION,
            "union": ConsensusMethod.UNION,
        }
        consensus_method = method_map.get(method, ConsensusMethod.WEIGHTED_AVERAGE)
        
        return quick_consensus(protein, predictors, method=consensus_method)
    
    # No consensus - return first successful result
    for pred_name in predictors:
        if pred_name not in available:
            continue
        try:
            pred = get_predictor(pred_name)
            result = pred.predict(protein)
            if result.success:
                return result
        except Exception:
            continue
    
    # All failed - return last attempt
    pred = get_predictor(predictors[0])
    return pred.predict(protein)


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
    # Consensus
    "ConsensusEngine",
    "ConsensusMethod",
    "quick_consensus",
]
