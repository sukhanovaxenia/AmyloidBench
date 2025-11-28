"""
Locally-running amyloidogenicity predictors.

This submodule contains predictors that execute locally without requiring
internet connectivity. These fall into two categories:

1. **Standalone package wrappers**: Interfaces to published tools that
   provide downloadable implementations. Example: Aggrescan3D standalone.

2. **Re-implementations**: Our implementations of published algorithms
   where standalone versions are unavailable but the methodology is
   documented. These follow the original publications as closely as
   possible while enabling local execution and batch processing.

The biological advantage of local predictors includes reproducibility
(no server changes affect results) and the ability to process large
datasets without web interface limitations.

Available predictors:
    - Aggrescan3DPredictor: Structure-based aggregation prediction
    
Planned predictors (to be implemented):
    - FoldAmyloidPredictor: Contact density-based prediction
    - AggrescanSeqPredictor: Original sequence-based AGGRESCAN
    - FallbackPredictor: Our biophysical/contextual predictor
"""

from .aggrescan3d import Aggrescan3DPredictor, predict_with_a3d

__all__ = [
    "Aggrescan3DPredictor",
    "predict_with_a3d",
]
