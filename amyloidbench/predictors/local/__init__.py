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
"""

from .aggrescan3d import Aggrescan3DPredictor, predict_with_a3d
from .fallback import FallbackPredictor, calculate_tango_like_score, ScoringWeights
from .reimplemented import (
    FoldAmyloidPredictor,
    predict_with_foldamyloid,
    get_packing_density_profile,
)

__all__ = [
    "Aggrescan3DPredictor",
    "predict_with_a3d",
    "FoldAmyloidPredictor", 
    "predict_with_foldamyloid",
    "get_packing_density_profile",
    "FallbackPredictor",
    "calculate_tango_like_score",
    "ScoringWeights",
]
