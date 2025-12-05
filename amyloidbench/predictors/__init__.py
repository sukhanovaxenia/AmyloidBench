"""
Amyloidogenicity predictor implementations.

This module contains both locally-running predictors and wrappers for
web-based tools. The predictor architecture follows the Strategy pattern,
allowing seamless swapping of prediction algorithms while maintaining
a consistent interface for downstream analysis.

Predictor Categories:

**Sequence-based heuristic predictors**
Methods that evaluate amino acid sequences using experimentally-derived
propensity scales. Examples: AGGRESCAN (aggregation), FoldAmyloid (contact
density), WALTZ (hexapeptide amyloidogenicity). These capture the intrinsic
aggregation tendency encoded in primary sequence.

**Sequence-based machine learning predictors**
Methods trained on labeled datasets to recognize amyloidogenic patterns.
Examples: APPNN, FISH Amyloid, AmyloGram. These can capture complex,
non-linear relationships between sequence features and amyloid formation.

**Structure-based predictors**
Methods incorporating 3D structural information, particularly solvent
accessibility. Examples: Aggrescan3D, Zyggregator3D. These distinguish
between exposed APRs (aggregation-competent) and buried hydrophobic cores
(aggregation-protected in native state).

**Threading/template-based predictors**
Methods that model query sequences against known amyloid structures.
Examples: PATH, PASTA 2.0. These evaluate structural compatibility with
the steric zipper architecture characteristic of amyloid cores.

**Consensus predictors**
Meta-methods combining multiple individual predictors. Examples: AmylPred2,
MetAmyl. These reduce false positive rates through agreement voting and
can capture complementary aspects of amyloidogenicity.

Submodules:
    base: Abstract base classes and predictor registry
    local: Locally-running predictors (standalone packages)
    remote: Web-based predictors (require internet access)
"""

from .base import (
    AsyncBasePredictor,
    BasePredictor,
    PredictorCapability,
    PredictorConfig,
    PredictorError,
    PredictorTimeoutError,
    PredictorType,
    PredictorUnavailableError,
    TrainablePredictor,
    get_predictor,
    list_predictors,
    register_predictor,
)

from .output_models import (
    PredictorOutput,
    ResidueScore,
    PredictedRegion,
    ClassificationLabel,
    ScoreType,
    MultiPredictorOutput,
    normalize_scores,
    classify_residues,
)

from .export import (
    export_to_tsv,
    export_regions_to_tsv,
    export_batch_results,
    plot_prediction_result,
    plot_multi_predictor_comparison,
    plot_sequence_heatmap,
    ResultExporter,
)

__all__ = [
    # Base classes
    "BasePredictor",
    "AsyncBasePredictor",
    "TrainablePredictor",
    # Configuration and types
    "PredictorConfig",
    "PredictorType",
    "PredictorCapability",
    # Exceptions
    "PredictorError",
    "PredictorTimeoutError",
    "PredictorUnavailableError",
    # Registry functions
    "register_predictor",
    "get_predictor",
    "list_predictors",
    # Output models
    "PredictorOutput",
    "ResidueScore",
    "PredictedRegion",
    "ClassificationLabel",
    "ScoreType",
    "MultiPredictorOutput",
    "normalize_scores",
    "classify_residues",
    # Export functions
    "export_to_tsv",
    "export_regions_to_tsv",
    "export_batch_results",
    "plot_prediction_result",
    "plot_multi_predictor_comparison",
    "plot_sequence_heatmap",
    "ResultExporter",
]
