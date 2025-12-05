"""
Locally-running amyloidogenicity predictors.

This submodule contains predictors that execute locally without requiring
internet connectivity. These fall into three categories:

1. **Standalone package wrappers**: Interfaces to published tools that
   provide downloadable implementations. Example: Aggrescan3D standalone.

2. **Re-implementations**: Our implementations of published algorithms
   where standalone versions are unavailable but the methodology is
   documented. These follow the original publications as closely as
   possible while enabling local execution and batch processing.

3. **Fallback predictor**: A biophysics-based predictor using fundamental
   principles of amyloid formation. Provides interpretable predictions
   and serves as a baseline for benchmarking.

The biological advantage of local predictors includes reproducibility
(no server changes affect results) and the ability to process large
datasets without web interface limitations.

Available predictors:
    - Aggrescan3DPredictor: Structure-based aggregation prediction
    - FoldAmyloidPredictor: Contact density-based prediction (re-implemented)
    - FallbackPredictor: Biophysics-based with optional ML enhancement
    - WaltzPredictor: PSSM-based hexapeptide amyloid detection
    - TangoPredictor: Statistical mechanics β-aggregation model
    - PastaPredictor: Pairwise β-strand pairing energy
    - AggrescanPredictor: a3v scale aggregation hot spots
    - ZyggregatorPredictor: Combined intrinsic property predictor
    - CrossBetaPredictor: β-sheet structural compatibility
    - ArchCandyPredictor: β-arch detection for β-arcade amyloids
    - APPNNPredictor: Neural network feature approximation
"""

from .aggrescan3d import Aggrescan3DPredictor, predict_with_a3d
from .fallback import FallbackPredictor, calculate_tango_like_score, ScoringWeights
from .reimplemented import (
    FoldAmyloidPredictor,
    predict_with_foldamyloid,
    get_packing_density_profile,
)
from .algorithms import (
    # Predictors
    WaltzPredictor,
    TangoPredictor,
    PastaPredictor,
    AggrescanPredictor,
    ZyggregatorPredictor,
    CrossBetaPredictor,
    ArchCandyPredictor,
    APPNNPredictor,
    # Convenience functions
    predict_with_waltz,
    predict_with_tango,
    predict_with_pasta,
    predict_with_archcandy,
    predict_with_appnn,
    predict_with_all,
    get_consensus_prediction,
    # Amino acid scales (useful for custom analysis)
    KYTE_DOOLITTLE,
    BETA_PROPENSITY_CF,
    AGGRESCAN_A3V,
)

__all__ = [
    # Wrapper predictors
    "Aggrescan3DPredictor",
    "predict_with_a3d",
    # Re-implemented predictors
    "FoldAmyloidPredictor", 
    "predict_with_foldamyloid",
    "get_packing_density_profile",
    # Algorithm-based predictors
    "WaltzPredictor",
    "TangoPredictor",
    "PastaPredictor",
    "AggrescanPredictor",
    "ZyggregatorPredictor",
    "CrossBetaPredictor",
    "ArchCandyPredictor",
    "APPNNPredictor",
    # Convenience functions
    "predict_with_waltz",
    "predict_with_tango",
    "predict_with_pasta",
    "predict_with_archcandy",
    "predict_with_appnn",
    "predict_with_all",
    "get_consensus_prediction",
    # Fallback
    "FallbackPredictor",
    "calculate_tango_like_score",
    "ScoringWeights",
    # Amino acid scales
    "KYTE_DOOLITTLE",
    "BETA_PROPENSITY_CF",
    "AGGRESCAN_A3V",
]
