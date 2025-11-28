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
    (To be implemented in Phase 2)
    - WaltzPredictor: Hexapeptide amyloidogenicity
    - Pasta2Predictor: Pairwise energy + threading
    - AmylPred2Predictor: Consensus of 11 predictors
    - TapassPredictor: Aggregation-prone motifs
    - FoldAmyloidWebPredictor: Contact density prediction
"""

# Web predictors will be implemented in Phase 2
# from .waltz import WaltzPredictor
# from .pasta2 import Pasta2Predictor
# from .amylpred2 import AmylPred2Predictor

__all__ = [
    # "WaltzPredictor",
    # "Pasta2Predictor",
    # "AmylPred2Predictor",
]
