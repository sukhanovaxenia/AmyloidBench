"""
Web-based amyloidogenicity predictors accessed via web automation.

This submodule contains wrappers for prediction tools that are only
available as web servers without APIs or downloadable implementations.
These predictors use asynchronous web automation (Playwright) to submit
sequences and parse results.

**Important considerations for web-based predictors:**

1. **Rate limiting**: Web servers impose limits on request frequency.
   Excessive requests may result in IP blocking. Default delays between
   requests are implemented to respect server policies.

2. **Reproducibility**: Web servers may update their algorithms without
   notice. Results may differ across time due to server-side changes.
   We recommend caching results for reproducibility.

3. **Availability**: Web servers may be temporarily unavailable for
   maintenance. The predictors implement retry logic with exponential
   backoff.

4. **Batch processing**: Large-scale analyses should be conducted with
   appropriate delays. Consider running overnight for proteome-scale
   studies.

**Biological note**: Many foundational amyloid predictors were developed
before the current emphasis on software reproducibility. The original
algorithms (e.g., WALTZ, PASTA) are well-documented in publications but
lack standalone implementations. Web automation provides access to these
established methods until proper re-implementations can be validated.

Available predictors:
    - WaltzPredictor: Hexapeptide position-specific scoring (Maurer-Stroh et al., 2010)
    - Pasta2Predictor: Pairwise energy + threading (Walsh et al., 2014)
    
Local approximations available when servers unavailable:
    - predict_waltz_local: Simplified WALTZ-like PSSM scoring
    - predict_pasta_local: Simplified pairwise energy calculation
"""

from .base import (
    WebPredictorBase,
    WebPredictorConfig,
    parse_score_table,
    parse_highlighted_regions,
    extract_job_id,
    PLAYWRIGHT_AVAILABLE,
)

# Always export local approximations (no dependencies required)
from .waltz import predict_waltz_local, calculate_waltz_like_score
from .pasta2 import predict_pasta_local, calculate_pasta_like_energy

__all__ = [
    # Base classes
    "WebPredictorBase",
    "WebPredictorConfig",
    # Utility functions
    "parse_score_table",
    "parse_highlighted_regions",
    "extract_job_id",
    "PLAYWRIGHT_AVAILABLE",
    # Local approximations
    "predict_waltz_local",
    "calculate_waltz_like_score",
    "predict_pasta_local",
    "calculate_pasta_like_energy",
]

# Import web predictors only if Playwright is available
if PLAYWRIGHT_AVAILABLE:
    from .waltz import WaltzPredictor
    from .pasta2 import Pasta2Predictor
    
    __all__ += [
        "WaltzPredictor",
        "Pasta2Predictor",
    ]
