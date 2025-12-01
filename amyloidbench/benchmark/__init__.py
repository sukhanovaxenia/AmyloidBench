"""
Benchmarking framework for amyloidogenicity predictor evaluation.

This module provides comprehensive benchmarking against experimentally-validated
databases with rigorous metric calculation and comparison tools.

Supported Databases
-------------------

**WALTZ-DB** (Louros et al., 2020)
Hexapeptide-resolution data on amyloid-forming and non-forming sequences.
Contains experimentally validated examples from ThT binding assays.

**Cross-Beta DB** (Sawaya et al., 2021)
Curated from PDB structures with confirmed cross-β amyloid architecture.
Contains structure-validated amyloidogenic regions.

**AmyPro** (Varadi et al., 2018)
Database of amyloid-forming proteins with annotated regions.

**Reference Datasets** (AmyloidBench curated)
Gold-standard sequences including canonical peptides (GNNQQNY, KLVFFA, VQIVYK),
disease proteins (Aβ42, α-synuclein, tau), functional amyloids (curli, HET-s),
and negative controls (ubiquitin, lysozyme, GFP).

**Synthetic datasets**
Built-in datasets for testing based on canonical amyloid sequences.

Evaluation Metrics
------------------
- Sensitivity, Specificity, Precision, F1 score
- Matthews Correlation Coefficient (MCC)
- AUC-ROC, AUC-PR
- Per-residue SOV score, Jaccard index
- Bootstrap confidence intervals

Statistical Comparison
----------------------
- McNemar's test for paired classification comparison
- Wilcoxon signed-rank test for paired scores
- DeLong test for AUC-ROC comparison
- Friedman test with Nemenyi post-hoc for multiple predictors
- Multiple testing correction (Bonferroni, Holm, Benjamini-Hochberg)

Polymorph-Aware Benchmarking
----------------------------
- Per-fold performance (steric zipper, β-solenoid, β-arcade)
- Per-geometry analysis (parallel, antiparallel)
- Cross-polymorph generalization testing

Quick Start
-----------
    >>> from amyloidbench.benchmark import quick_benchmark, load_benchmark_dataset
    >>> 
    >>> # Quick benchmark with synthetic data
    >>> results = quick_benchmark(dataset="synthetic")
    >>> for r in results:
    ...     print(f"{r.predictor_name}: MCC={r.classification_metrics.mcc:.3f}")
    >>> 
    >>> # Full benchmark with runner
    >>> from amyloidbench.benchmark import BenchmarkRunner
    >>> runner = BenchmarkRunner()
    >>> runner.add_all_predictors()
    >>> dataset = load_benchmark_dataset("synthetic")
    >>> results = runner.run(dataset)
    >>> report = runner.generate_report(results)
    >>> print(report.summary())
    
    >>> # Statistical comparison
    >>> from amyloidbench.benchmark import compare_benchmark_results
    >>> comparison = compare_benchmark_results(results)
    >>> print(comparison.summary())
    
    >>> # Reference dataset benchmarking
    >>> from amyloidbench.benchmark import create_comprehensive_dataset
    >>> ref_dataset = create_comprehensive_dataset()
    >>> results = runner.run(ref_dataset)
"""

from .datasets import (
    # Enums
    AmyloidStatus,
    ExperimentalMethod,
    # Data classes
    BenchmarkEntry,
    BenchmarkDataset,
    # Loaders
    WaltzDBLoader,
    CrossBetaDBLoader,
    AmyProLoader,
    FASTALoader,
    # Functions
    load_benchmark_dataset,
    get_available_datasets,
)

from .metrics import (
    # Data classes
    ConfusionMatrix,
    ClassificationMetrics,
    PerResidueMetrics,
    BenchmarkResult,
    # Calculation functions
    calculate_classification_metrics,
    calculate_per_residue_metrics,
    calculate_mcc,
    calculate_auc_roc,
    calculate_auc_pr,
    calculate_sov,
    find_optimal_threshold,
    bootstrap_confidence_interval,
    # Comparison functions
    compare_predictors,
    rank_predictors,
)

from .runner import (
    # Configuration
    RunnerConfig,
    # Main class
    BenchmarkRunner,
    BenchmarkReport,
    # Convenience
    quick_benchmark,
)

from .statistics import (
    # Data classes
    StatisticalComparison,
    MultipleComparisonResult,
    # Pairwise tests
    mcnemar_test,
    wilcoxon_signed_rank_test,
    paired_t_test,
    delong_test,
    # Multiple comparison
    friedman_test,
    correct_pvalues,
    # Convenience
    compare_benchmark_results,
    pairwise_compare_predictors,
)

from .reference_datasets import (
    # Data classes
    CanonicalPeptide,
    DiseaseProtein,
    FunctionalAmyloid,
    NegativeControl,
    # Dataset builders
    create_canonical_peptide_dataset,
    create_disease_protein_dataset,
    create_functional_amyloid_dataset,
    create_negative_control_dataset,
    create_comprehensive_dataset,
    # Access functions
    get_canonical_peptides,
    get_disease_proteins,
    get_functional_amyloids,
    get_negative_controls,
    get_sequence_by_name,
    # Raw data
    CANONICAL_PEPTIDES,
    DISEASE_PROTEINS,
    FUNCTIONAL_AMYLOIDS,
    NEGATIVE_CONTROLS,
)

from .polymorph_benchmark import (
    # Data classes
    PolymorphBenchmarkEntry,
    PolymorphBenchmarkResult,
    RegionPolymorphAnalysis,
    # Functions
    annotate_dataset_with_polymorphs,
    analyze_regions_with_polymorphs,
    test_cross_polymorph_generalization,
    # Runner
    PolymorphBenchmarkRunner,
    # Convenience
    quick_polymorph_benchmark,
    get_polymorph_specific_performance,
)

__all__ = [
    # Dataset enums
    "AmyloidStatus",
    "ExperimentalMethod",
    # Dataset classes
    "BenchmarkEntry",
    "BenchmarkDataset",
    # Dataset loaders
    "WaltzDBLoader",
    "CrossBetaDBLoader",
    "AmyProLoader",
    "FASTALoader",
    # Dataset functions
    "load_benchmark_dataset",
    "get_available_datasets",
    # Metric classes
    "ConfusionMatrix",
    "ClassificationMetrics",
    "PerResidueMetrics",
    "BenchmarkResult",
    # Metric functions
    "calculate_classification_metrics",
    "calculate_per_residue_metrics",
    "calculate_mcc",
    "calculate_auc_roc",
    "calculate_auc_pr",
    "calculate_sov",
    "find_optimal_threshold",
    "bootstrap_confidence_interval",
    "compare_predictors",
    "rank_predictors",
    # Runner
    "RunnerConfig",
    "BenchmarkRunner",
    "BenchmarkReport",
    "quick_benchmark",
    # Statistical comparison
    "StatisticalComparison",
    "MultipleComparisonResult",
    "mcnemar_test",
    "wilcoxon_signed_rank_test",
    "paired_t_test",
    "delong_test",
    "friedman_test",
    "correct_pvalues",
    "compare_benchmark_results",
    "pairwise_compare_predictors",
    # Reference datasets
    "CanonicalPeptide",
    "DiseaseProtein",
    "FunctionalAmyloid",
    "NegativeControl",
    "create_canonical_peptide_dataset",
    "create_disease_protein_dataset",
    "create_functional_amyloid_dataset",
    "create_negative_control_dataset",
    "create_comprehensive_dataset",
    "get_canonical_peptides",
    "get_disease_proteins",
    "get_functional_amyloids",
    "get_negative_controls",
    "get_sequence_by_name",
    "CANONICAL_PEPTIDES",
    "DISEASE_PROTEINS",
    "FUNCTIONAL_AMYLOIDS",
    "NEGATIVE_CONTROLS",
    # Polymorph benchmarking
    "PolymorphBenchmarkEntry",
    "PolymorphBenchmarkResult",
    "RegionPolymorphAnalysis",
    "PolymorphBenchmarkRunner",
    "annotate_dataset_with_polymorphs",
    "analyze_regions_with_polymorphs",
    "test_cross_polymorph_generalization",
    "quick_polymorph_benchmark",
    "get_polymorph_specific_performance",
]
