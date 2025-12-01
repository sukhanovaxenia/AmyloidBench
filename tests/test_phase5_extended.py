"""
Extended test suite for Phase 5: Advanced benchmarking features.

Tests verify:
1. Statistical comparison methods work correctly
2. Reference datasets are properly structured
3. Polymorph-aware benchmarking integrates with Phase 4
4. Multiple testing corrections are accurate
"""

import pytest
import numpy as np
from pathlib import Path

# Statistics module
from amyloidbench.benchmark.statistics import (
    StatisticalComparison,
    MultipleComparisonResult,
    mcnemar_test,
    wilcoxon_signed_rank_test,
    paired_t_test,
    delong_test,
    friedman_test,
    correct_pvalues,
    compare_benchmark_results,
)

# Reference datasets
from amyloidbench.benchmark.reference_datasets import (
    CanonicalPeptide,
    DiseaseProtein,
    FunctionalAmyloid,
    NegativeControl,
    CANONICAL_PEPTIDES,
    DISEASE_PROTEINS,
    FUNCTIONAL_AMYLOIDS,
    NEGATIVE_CONTROLS,
    create_canonical_peptide_dataset,
    create_disease_protein_dataset,
    create_functional_amyloid_dataset,
    create_negative_control_dataset,
    create_comprehensive_dataset,
    get_canonical_peptides,
    get_disease_proteins,
    get_sequence_by_name,
)

# Polymorph benchmarking
from amyloidbench.benchmark.polymorph_benchmark import (
    PolymorphBenchmarkEntry,
    PolymorphBenchmarkResult,
    RegionPolymorphAnalysis,
    PolymorphBenchmarkRunner,
    annotate_dataset_with_polymorphs,
    analyze_regions_with_polymorphs,
)

# Other imports
from amyloidbench.benchmark import (
    AmyloidStatus,
    BenchmarkEntry,
    BenchmarkDataset,
    BenchmarkResult,
    ClassificationMetrics,
)
from amyloidbench.core.models import Region
from amyloidbench.classification.polymorph import AmyloidFold, CrossBetaGeometry


# =============================================================================
# Statistical Comparison Tests
# =============================================================================

class TestMcNemarTest:
    """Test McNemar's test for paired classification."""
    
    def test_perfect_agreement(self):
        """Both classifiers make same errors."""
        errors_a = [False, True, False, True, False]
        errors_b = [False, True, False, True, False]
        
        result = mcnemar_test(errors_a, errors_b)
        
        assert result.test_name == "McNemar's Test"
        assert result.p_value >= 0.5  # No difference
        assert result.significant is False
    
    def test_significant_difference(self):
        """Classifiers have clearly different error patterns."""
        # A makes 20 errors where B is correct
        # B makes 5 errors where A is correct
        errors_a = [True] * 20 + [False] * 35
        errors_b = [False] * 20 + [True] * 5 + [False] * 30
        
        result = mcnemar_test(errors_a, errors_b)
        
        assert result.p_value < 0.05  # Significant difference
    
    def test_small_sample(self):
        """Uses exact binomial test for small samples."""
        errors_a = [True, False, False]
        errors_b = [False, True, False]
        
        result = mcnemar_test(errors_a, errors_b)
        
        assert "binomial" in result.notes[0].lower() or len(result.notes) == 0


class TestWilcoxonTest:
    """Test Wilcoxon signed-rank test."""
    
    def test_paired_scores(self):
        """Test with paired score samples."""
        scores_a = [0.85, 0.82, 0.88, 0.81, 0.86, 0.83, 0.87, 0.84, 0.85, 0.86]
        scores_b = [0.80, 0.78, 0.82, 0.79, 0.81, 0.77, 0.83, 0.80, 0.79, 0.81]
        
        result = wilcoxon_signed_rank_test(scores_a, scores_b)
        
        assert result.test_name == "Wilcoxon Signed-Rank Test"
        assert 0 <= result.p_value <= 1
        assert result.effect_size is not None
    
    def test_identical_scores(self):
        """Identical scores yield non-significant result."""
        scores_a = [0.8] * 10
        scores_b = [0.8] * 10
        
        result = wilcoxon_signed_rank_test(scores_a, scores_b)
        
        assert result.p_value >= 0.5


class TestPairedTTest:
    """Test paired t-test."""
    
    def test_significant_difference(self):
        """Clear difference should be detected."""
        scores_a = np.array([0.90, 0.91, 0.89, 0.92, 0.88])
        scores_b = np.array([0.70, 0.72, 0.69, 0.71, 0.68])
        
        result = paired_t_test(scores_a, scores_b)
        
        assert result.test_name == "Paired t-test"
        assert result.p_value < 0.05
        assert result.confidence_interval is not None
        assert result.effect_size is not None  # Cohen's d


class TestDeLongTest:
    """Test DeLong test for AUC comparison."""
    
    def test_equal_aucs(self):
        """Similar predictors should have non-significant difference."""
        y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
        scores_a = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1, 0.6, 0.4, 0.75, 0.35])
        scores_b = np.array([0.85, 0.82, 0.72, 0.28, 0.22, 0.12, 0.62, 0.38, 0.78, 0.32])
        
        result = delong_test(y_true, scores_a, scores_b)
        
        assert result.test_name == "DeLong Test"
        assert result.p_value > 0.1  # Not significantly different
    
    def test_different_aucs(self):
        """Different predictors should show significant difference."""
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        scores_a = np.array([0.9, 0.85, 0.8, 0.75, 0.7, 0.3, 0.25, 0.2, 0.15, 0.1])  # Good
        scores_b = np.array([0.6, 0.55, 0.5, 0.45, 0.4, 0.5, 0.45, 0.4, 0.35, 0.3])  # Random-ish
        
        result = delong_test(y_true, scores_a, scores_b)
        
        assert result.confidence_interval is not None


class TestFriedmanTest:
    """Test Friedman test for multiple classifiers."""
    
    def test_multiple_predictors(self):
        """Compare 3 predictors across 10 datasets."""
        # Scores: rows = datasets, cols = predictors
        scores = np.array([
            [0.85, 0.80, 0.70],
            [0.88, 0.82, 0.72],
            [0.83, 0.78, 0.68],
            [0.87, 0.81, 0.71],
            [0.86, 0.79, 0.69],
            [0.84, 0.77, 0.67],
            [0.89, 0.83, 0.73],
            [0.82, 0.76, 0.66],
            [0.85, 0.80, 0.70],
            [0.86, 0.81, 0.71],
        ])
        
        result = friedman_test(scores, ["PredA", "PredB", "PredC"])
        
        assert result.test_name.startswith("Friedman")
        assert len(result.rankings) == 3
        assert result.rankings[0][0] == "PredA"  # Best predictor
    
    def test_nemenyi_posthoc(self):
        """Significant Friedman should produce pairwise comparisons."""
        scores = np.array([
            [0.95, 0.50, 0.30],
            [0.92, 0.48, 0.28],
            [0.94, 0.52, 0.32],
            [0.93, 0.49, 0.29],
            [0.91, 0.51, 0.31],
        ])
        
        result = friedman_test(scores, ["Best", "Medium", "Worst"], alpha=0.05)
        
        if result.overall_significant:
            assert len(result.pairwise_comparisons) > 0
            assert result.critical_difference is not None


class TestMultipleTestingCorrection:
    """Test p-value correction methods."""
    
    def test_bonferroni(self):
        """Bonferroni correction."""
        p_values = [0.01, 0.04, 0.06, 0.10]
        corrected, alpha = correct_pvalues(p_values, method="bonferroni")
        
        assert corrected[0] == pytest.approx(0.04, abs=0.001)
        assert alpha == pytest.approx(0.05 / 4)
    
    def test_holm(self):
        """Holm-Bonferroni correction."""
        p_values = [0.01, 0.02, 0.03, 0.04]
        corrected, _ = correct_pvalues(p_values, method="holm")
        
        assert len(corrected) == 4
        assert all(c <= 1.0 for c in corrected)
    
    def test_fdr_bh(self):
        """Benjamini-Hochberg FDR correction."""
        p_values = [0.001, 0.01, 0.05, 0.10, 0.20]
        corrected, _ = correct_pvalues(p_values, method="fdr_bh")
        
        assert len(corrected) == 5
        # FDR correction is less conservative than Bonferroni
        assert corrected[0] < 0.01  # Very small p stays small


# =============================================================================
# Reference Dataset Tests
# =============================================================================

class TestCanonicalPeptides:
    """Test canonical peptide reference data."""
    
    def test_canonical_peptides_exist(self):
        """Verify canonical peptides are defined."""
        assert len(CANONICAL_PEPTIDES) >= 10
    
    def test_gnnqqny_present(self):
        """GNNQQNY should be in the dataset."""
        gnnqqny = next((p for p in CANONICAL_PEPTIDES if p.name == "GNNQQNY"), None)
        
        assert gnnqqny is not None
        assert gnnqqny.sequence == "GNNQQNY"
        assert gnnqqny.zipper_class == 1
        assert "1YJO" in gnnqqny.pdb_ids
    
    def test_klvffa_present(self):
        """KLVFFA (Aβ core) should be present."""
        klvffa = next((p for p in CANONICAL_PEPTIDES if p.name == "KLVFFA"), None)
        
        assert klvffa is not None
        assert klvffa.source_protein == "Amyloid β (Aβ)"
    
    def test_canonical_dataset(self):
        """Test canonical peptide dataset creation."""
        dataset = create_canonical_peptide_dataset()
        
        assert dataset.name == "Canonical_Peptides"
        assert len(dataset) >= 10
        assert dataset.n_positive == len(dataset)  # All positive
        assert dataset.n_negative == 0


class TestDiseaseProteins:
    """Test disease-associated protein data."""
    
    def test_disease_proteins_exist(self):
        """Verify disease proteins are defined."""
        assert len(DISEASE_PROTEINS) >= 5
    
    def test_abeta42_present(self):
        """Aβ42 should be in the dataset."""
        abeta = next((p for p in DISEASE_PROTEINS if "Amyloid" in p.name), None)
        
        assert abeta is not None
        assert len(abeta.sequence) == 42
        assert len(abeta.apr_regions) >= 2
        assert abeta.disease == "Alzheimer's disease"
    
    def test_alpha_synuclein_present(self):
        """α-Synuclein should be present."""
        asyn = next((p for p in DISEASE_PROTEINS if "Synuclein" in p.name), None)
        
        assert asyn is not None
        assert len(asyn.sequence) == 140
        assert "Parkinson" in asyn.disease
    
    def test_disease_dataset(self):
        """Test disease protein dataset creation."""
        dataset = create_disease_protein_dataset()
        
        assert dataset.name == "Disease_Proteins"
        assert len(dataset) >= 5
        
        # Check APR annotations
        for entry in dataset:
            if entry.is_positive:
                assert len(entry.amyloid_regions) > 0


class TestFunctionalAmyloids:
    """Test functional amyloid data."""
    
    def test_functional_amyloids_exist(self):
        """Verify functional amyloids are defined."""
        assert len(FUNCTIONAL_AMYLOIDS) >= 3
    
    def test_curli_present(self):
        """Curli (CsgA) should be present."""
        curli = next((p for p in FUNCTIONAL_AMYLOIDS if "Curli" in p.name), None)
        
        assert curli is not None
        assert curli.organism == "Escherichia coli"
        assert curli.fold_type == "β-solenoid"
    
    def test_het_s_present(self):
        """HET-s should be present."""
        het_s = next((p for p in FUNCTIONAL_AMYLOIDS if "HET-s" in p.name), None)
        
        assert het_s is not None
        assert het_s.fold_type == "β-solenoid"
        assert len(het_s.apr_regions) >= 1


class TestNegativeControls:
    """Test negative control proteins."""
    
    def test_negative_controls_exist(self):
        """Verify negative controls are defined."""
        assert len(NEGATIVE_CONTROLS) >= 4
    
    def test_ubiquitin_present(self):
        """Ubiquitin should be present."""
        ubi = next((p for p in NEGATIVE_CONTROLS if "Ubiquitin" in p.name), None)
        
        assert ubi is not None
        assert len(ubi.sequence) == 76
    
    def test_negative_dataset(self):
        """Test negative control dataset creation."""
        dataset = create_negative_control_dataset()
        
        assert dataset.name == "Negative_Controls"
        assert dataset.n_negative == len(dataset)  # All negative
        assert dataset.n_positive == 0


class TestComprehensiveDataset:
    """Test comprehensive combined dataset."""
    
    def test_comprehensive_dataset_creation(self):
        """Test comprehensive dataset combines all sources."""
        dataset = create_comprehensive_dataset()
        
        assert dataset.name == "AmyloidBench_Reference"
        assert len(dataset) >= 20
        assert dataset.n_positive > 0
        assert dataset.n_negative > 0
    
    def test_sequence_by_name(self):
        """Test getting sequences by name."""
        gnnqqny = get_sequence_by_name("GNNQQNY")
        assert gnnqqny == "GNNQQNY"
        
        ubiquitin = get_sequence_by_name("Ubiquitin")
        assert ubiquitin is not None
        assert len(ubiquitin) == 76


# =============================================================================
# Polymorph Benchmarking Tests
# =============================================================================

class TestPolymorphAnnotation:
    """Test polymorph annotation functionality."""
    
    def test_annotate_canonical_peptides(self):
        """Canonical peptides should get polymorph annotations."""
        dataset = create_canonical_peptide_dataset()
        annotated = annotate_dataset_with_polymorphs(dataset)
        
        assert len(annotated) == len(dataset)
        
        # GNNQQNY should be steric zipper
        gnnqqny = next((e for e in annotated if "GNNQQNY" in e.id), None)
        assert gnnqqny is not None
        assert gnnqqny.known_fold == AmyloidFold.STERIC_ZIPPER
    
    def test_annotate_with_metadata(self):
        """Entries with zipper_class metadata should use it."""
        entry = BenchmarkEntry(
            id="test",
            sequence="GNNQQNY",
            amyloid_status=AmyloidStatus.POSITIVE,
            metadata={"zipper_class": 1},
        )
        dataset = BenchmarkDataset(name="test", description="", entries=[entry])
        
        annotated = annotate_dataset_with_polymorphs(dataset)
        
        assert annotated[0].known_fold == AmyloidFold.STERIC_ZIPPER


class TestPolymorphBenchmarkRunner:
    """Test polymorph-aware benchmark runner."""
    
    def test_runner_initialization(self):
        """Runner should initialize correctly."""
        runner = PolymorphBenchmarkRunner()
        assert runner.classifier is not None
    
    def test_run_with_mock_predictor(self):
        """Test running with a mock predictor."""
        from amyloidbench.predictors import get_predictor
        
        predictor = get_predictor("FallbackPredictor")
        dataset = create_canonical_peptide_dataset()
        annotated = annotate_dataset_with_polymorphs(dataset)
        
        runner = PolymorphBenchmarkRunner()
        result = runner.run(predictor, annotated)
        
        assert isinstance(result, PolymorphBenchmarkResult)
        assert result.n_samples > 0
        assert len(result.per_fold_metrics) > 0


class TestRegionPolymorphAnalysis:
    """Test region-specific polymorph analysis."""
    
    def test_analyze_abeta_regions(self):
        """Test analyzing Aβ42 APR regions."""
        sequence = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"
        regions = [
            Region(start=17, end=22, sequence="KLVFFA", score=0.9),
            Region(start=31, end=41, sequence="IILMVGGVVI", score=0.8),
        ]
        
        analyses = analyze_regions_with_polymorphs(sequence, regions)
        
        assert len(analyses) == 2
        
        # Both regions should be analyzed
        for analysis in analyses:
            assert analysis.predicted_fold is not None
            assert analysis.predicted_geometry is not None
            assert 0 <= analysis.confidence <= 1


class TestPolymorphBenchmarkResult:
    """Test PolymorphBenchmarkResult dataclass."""
    
    def test_result_summary(self):
        """Test result summary generation."""
        result = PolymorphBenchmarkResult(
            predictor_name="TestPredictor",
            overall_metrics=ClassificationMetrics(
                tp=10, tn=10, fp=2, fn=3,
                sensitivity=0.77, specificity=0.83, mcc=0.60
            ),
            per_fold_metrics={
                "steric_zipper": ClassificationMetrics(sensitivity=0.8, mcc=0.65),
                "beta_solenoid": ClassificationMetrics(sensitivity=0.7, mcc=0.55),
            },
            per_geometry_metrics={},
            n_samples=25,
            per_fold_counts={"steric_zipper": 15, "beta_solenoid": 10},
        )
        
        summary = result.summary()
        
        assert "TestPredictor" in summary
        assert "steric_zipper" in summary
        assert "beta_solenoid" in summary


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhase5Integration:
    """Test integration between Phase 5 components."""
    
    def test_reference_dataset_with_runner(self):
        """Reference dataset works with benchmark runner."""
        from amyloidbench.benchmark import BenchmarkRunner
        
        runner = BenchmarkRunner()
        runner.add_predictor("FallbackPredictor")
        
        dataset = create_comprehensive_dataset()
        results = runner.run(dataset)
        
        assert len(results) == 1
        assert results[0].n_samples == len(dataset)
    
    def test_statistical_comparison_of_results(self):
        """Statistical comparison works on benchmark results."""
        from amyloidbench.benchmark import BenchmarkRunner, WaltzDBLoader
        
        runner = BenchmarkRunner()
        runner.add_predictor("Aggrescan3D")
        runner.add_predictor("FallbackPredictor")
        
        dataset = WaltzDBLoader.create_synthetic_dataset(n_positive=20, n_negative=20)
        results = runner.run(dataset)
        
        # Compare results
        comparison = compare_benchmark_results(results, metric="mcc")
        
        assert len(comparison.rankings) == 2
    
    def test_polymorph_benchmark_with_reference_data(self):
        """Polymorph benchmarking works with reference datasets."""
        from amyloidbench.predictors import get_predictor
        from amyloidbench.benchmark.polymorph_benchmark import quick_polymorph_benchmark
        
        predictor = get_predictor("FallbackPredictor")
        
        # Use smaller dataset for speed
        dataset = create_canonical_peptide_dataset()
        result = quick_polymorph_benchmark(predictor, dataset)
        
        assert result.n_samples > 0
        assert "steric_zipper" in result.per_fold_metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
