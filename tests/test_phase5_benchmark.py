"""
Test suite for Phase 5: Benchmarking framework.

Tests verify:
1. Dataset loaders work correctly
2. Metrics calculations are accurate
3. Benchmark runner integrates all components
4. Report generation produces valid output
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from amyloidbench.benchmark import (
    # Dataset components
    AmyloidStatus,
    ExperimentalMethod,
    BenchmarkEntry,
    BenchmarkDataset,
    WaltzDBLoader,
    CrossBetaDBLoader,
    FASTALoader,
    load_benchmark_dataset,
    get_available_datasets,
    # Metrics
    ConfusionMatrix,
    ClassificationMetrics,
    PerResidueMetrics,
    BenchmarkResult,
    calculate_classification_metrics,
    calculate_per_residue_metrics,
    calculate_mcc,
    calculate_auc_roc,
    calculate_auc_pr,
    calculate_sov,
    find_optimal_threshold,
    compare_predictors,
    rank_predictors,
    # Runner
    BenchmarkRunner,
    BenchmarkReport,
    quick_benchmark,
)


class TestBenchmarkEntry:
    """Test BenchmarkEntry dataclass."""
    
    def test_entry_creation(self):
        """Test basic entry creation."""
        entry = BenchmarkEntry(
            id="TEST001",
            sequence="KLVFFA",
            amyloid_status=AmyloidStatus.POSITIVE,
        )
        
        assert entry.id == "TEST001"
        assert entry.sequence == "KLVFFA"
        assert entry.is_positive is True
        assert entry.is_negative is False
    
    def test_entry_with_regions(self):
        """Test entry with APR regions."""
        from amyloidbench.core.models import Region
        
        entry = BenchmarkEntry(
            id="ABETA",
            sequence="DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA",
            amyloid_status=AmyloidStatus.POSITIVE,
            amyloid_regions=[
                Region(start=16, end=21, sequence="KLVFF"),
            ],
        )
        
        assert len(entry.amyloid_regions) == 1
        assert entry.amyloid_regions[0].sequence == "KLVFF"
    
    def test_to_protein_record(self):
        """Test conversion to ProteinRecord."""
        entry = BenchmarkEntry(
            id="TEST",
            sequence="GNNQQNY",
            amyloid_status=AmyloidStatus.POSITIVE,
        )
        
        protein = entry.to_protein_record()
        
        assert protein.id == "TEST"
        assert protein.sequence == "GNNQQNY"
        assert protein.is_known_amyloid is True


class TestBenchmarkDataset:
    """Test BenchmarkDataset class."""
    
    def test_dataset_creation(self):
        """Test basic dataset creation."""
        entries = [
            BenchmarkEntry(id="1", sequence="AAA", amyloid_status=AmyloidStatus.POSITIVE),
            BenchmarkEntry(id="2", sequence="BBB", amyloid_status=AmyloidStatus.NEGATIVE),
        ]
        
        dataset = BenchmarkDataset(
            name="Test",
            description="Test dataset",
            entries=entries,
        )
        
        assert len(dataset) == 2
        assert dataset.n_positive == 1
        assert dataset.n_negative == 1
    
    def test_dataset_iteration(self):
        """Test dataset iteration."""
        entries = [
            BenchmarkEntry(id=str(i), sequence="AAA", amyloid_status=AmyloidStatus.POSITIVE)
            for i in range(5)
        ]
        
        dataset = BenchmarkDataset(name="Test", description="", entries=entries)
        
        count = sum(1 for _ in dataset)
        assert count == 5
    
    def test_filter_by_status(self):
        """Test filtering by amyloid status."""
        entries = [
            BenchmarkEntry(id="1", sequence="AAA", amyloid_status=AmyloidStatus.POSITIVE),
            BenchmarkEntry(id="2", sequence="BBB", amyloid_status=AmyloidStatus.POSITIVE),
            BenchmarkEntry(id="3", sequence="CCC", amyloid_status=AmyloidStatus.NEGATIVE),
        ]
        
        dataset = BenchmarkDataset(name="Test", description="", entries=entries)
        
        positives = dataset.filter_by_status(AmyloidStatus.POSITIVE)
        assert len(positives) == 2
        
        negatives = dataset.filter_by_status(AmyloidStatus.NEGATIVE)
        assert len(negatives) == 1
    
    def test_summary(self):
        """Test summary generation."""
        entries = [
            BenchmarkEntry(id="1", sequence="AAA", amyloid_status=AmyloidStatus.POSITIVE),
        ]
        
        dataset = BenchmarkDataset(
            name="Test",
            description="Test dataset",
            entries=entries,
            citation="Test et al., 2024",
        )
        
        summary = dataset.summary()
        assert "Test" in summary
        assert "1" in summary


class TestDatasetLoaders:
    """Test database loaders."""
    
    def test_waltz_synthetic(self):
        """Test WALTZ-DB synthetic dataset creation."""
        dataset = WaltzDBLoader.create_synthetic_dataset(n_positive=10, n_negative=10)
        
        assert len(dataset) == 20
        assert dataset.n_positive == 10
        assert dataset.n_negative == 10
        assert dataset.name == "WALTZ-DB_synthetic"
    
    def test_crossbeta_synthetic(self):
        """Test Cross-Beta DB synthetic dataset."""
        dataset = CrossBetaDBLoader.create_synthetic_dataset()
        
        assert len(dataset) >= 5  # Known canonical sequences
        assert dataset.n_positive >= 5  # All positive
        
        # Check for Aβ42
        abeta = next((e for e in dataset if "ABETA" in e.id), None)
        assert abeta is not None
        assert len(abeta.amyloid_regions) > 0
    
    def test_load_benchmark_dataset_synthetic(self):
        """Test load_benchmark_dataset with synthetic data."""
        dataset = load_benchmark_dataset("synthetic")
        
        assert len(dataset) > 0
        assert dataset.n_positive > 0
    
    def test_load_benchmark_dataset_waltz(self):
        """Test loading WALTZ synthetic."""
        dataset = load_benchmark_dataset("waltz")
        
        assert len(dataset) > 0
        assert "WALTZ" in dataset.name
    
    def test_get_available_datasets(self):
        """Test available datasets listing."""
        datasets = get_available_datasets()
        
        assert len(datasets) >= 3
        names = [d["name"] for d in datasets]
        assert any("WALTZ" in n for n in names)
        assert any("Synthetic" in n for n in names)
    
    def test_fasta_loader(self):
        """Test FASTA file loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(">SEQ1|POSITIVE\n")
            f.write("KLVFFA\n")
            f.write(">SEQ2|NEGATIVE\n")
            f.write("KKKDDD\n")
            f.flush()
            
            loader = FASTALoader()
            dataset = loader.load(Path(f.name))
        
        assert len(dataset) == 2
        assert dataset.entries[0].is_positive
        assert dataset.entries[1].is_negative


class TestMetrics:
    """Test metric calculation functions."""
    
    def test_confusion_matrix(self):
        """Test ConfusionMatrix properties."""
        cm = ConfusionMatrix(tp=10, tn=20, fp=5, fn=3)
        
        assert cm.total == 38
        assert cm.n_positive == 13  # tp + fn
        assert cm.n_negative == 25  # tn + fp
    
    def test_calculate_mcc_perfect(self):
        """Test MCC for perfect prediction."""
        mcc = calculate_mcc(tp=10, tn=10, fp=0, fn=0)
        assert mcc == 1.0
    
    def test_calculate_mcc_random(self):
        """Test MCC for random prediction."""
        # Equal errors in all cells -> MCC near 0
        mcc = calculate_mcc(tp=5, tn=5, fp=5, fn=5)
        assert -0.1 <= mcc <= 0.1
    
    def test_calculate_mcc_worst(self):
        """Test MCC for perfectly wrong prediction."""
        mcc = calculate_mcc(tp=0, tn=0, fp=10, fn=10)
        assert mcc == -1.0
    
    def test_calculate_classification_metrics(self):
        """Test full classification metrics."""
        y_true = [True, True, True, False, False]
        y_pred = [True, True, False, False, True]
        
        metrics = calculate_classification_metrics(y_true, y_pred)
        
        # TP=2, TN=1, FP=1, FN=1
        assert metrics.tp == 2
        assert metrics.tn == 1
        assert metrics.fp == 1
        assert metrics.fn == 1
        
        assert metrics.sensitivity == pytest.approx(2/3, abs=0.01)
        assert metrics.specificity == pytest.approx(1/2, abs=0.01)
    
    def test_calculate_classification_with_scores(self):
        """Test metrics with prediction scores."""
        y_true = [True, True, False, False]
        y_pred = [True, True, False, False]
        y_scores = [0.9, 0.8, 0.2, 0.1]
        
        metrics = calculate_classification_metrics(y_true, y_pred, y_scores)
        
        assert metrics.auc_roc is not None
        assert 0.9 <= metrics.auc_roc <= 1.0  # Perfect separation
    
    def test_calculate_auc_roc(self):
        """Test AUC-ROC calculation."""
        y_true = np.array([True, True, False, False])
        y_scores = np.array([0.9, 0.8, 0.3, 0.1])
        
        auc = calculate_auc_roc(y_true, y_scores)
        
        assert 0.9 <= auc <= 1.0
    
    def test_calculate_auc_pr(self):
        """Test AUC-PR calculation."""
        y_true = np.array([True, True, False, False])
        y_scores = np.array([0.9, 0.8, 0.3, 0.1])
        
        auc = calculate_auc_pr(y_true, y_scores)
        
        assert 0.9 <= auc <= 1.0
    
    def test_find_optimal_threshold(self):
        """Test optimal threshold finding."""
        y_true = np.array([True, True, True, False, False, False])
        y_scores = np.array([0.9, 0.7, 0.6, 0.4, 0.3, 0.1])
        
        threshold = find_optimal_threshold(y_true, y_scores)
        
        # Optimal should be around 0.5 (between positive and negative clusters)
        assert 0.4 <= threshold <= 0.6
    
    def test_per_residue_metrics(self):
        """Test per-residue metric calculation."""
        true_labels = [False, False, True, True, True, False, False]
        pred_labels = [False, True, True, True, False, False, False]
        
        metrics = calculate_per_residue_metrics(true_labels, pred_labels)
        
        assert metrics.n_residues == 7
        assert metrics.residue_sensitivity > 0
        assert metrics.residue_specificity > 0
    
    def test_calculate_sov(self):
        """Test SOV score calculation."""
        # Perfect prediction
        true_labels = np.array([False, True, True, True, False])
        pred_labels = np.array([False, True, True, True, False])
        
        sov = calculate_sov(true_labels, pred_labels)
        assert sov == 1.0
        
        # No overlap
        true_labels = np.array([True, True, False, False])
        pred_labels = np.array([False, False, True, True])
        
        sov = calculate_sov(true_labels, pred_labels)
        assert sov == 0.0


class TestBenchmarkRunner:
    """Test the benchmark runner."""
    
    def test_runner_initialization(self):
        """Test runner initialization."""
        runner = BenchmarkRunner()
        assert len(runner.predictors) == 0
    
    def test_add_predictor(self):
        """Test adding predictors."""
        runner = BenchmarkRunner()
        runner.add_predictor("Aggrescan3D")
        
        assert "Aggrescan3D" in runner.predictors
    
    def test_add_all_predictors(self):
        """Test adding all local predictors."""
        runner = BenchmarkRunner()
        runner.add_all_predictors(local_only=True)
        
        assert len(runner.predictors) >= 1
    
    def test_run_benchmark(self):
        """Test running benchmark on synthetic data."""
        runner = BenchmarkRunner()
        runner.add_predictor("FallbackPredictor")
        
        dataset = load_benchmark_dataset("synthetic")
        results = runner.run(dataset)
        
        assert len(results) == 1
        assert results[0].predictor_name == "FallbackPredictor"
        assert results[0].n_samples > 0
    
    def test_run_multiple_predictors(self):
        """Test running with multiple predictors."""
        runner = BenchmarkRunner()
        runner.add_predictor("Aggrescan3D")
        runner.add_predictor("FallbackPredictor")
        
        dataset = WaltzDBLoader.create_synthetic_dataset(n_positive=20, n_negative=20)
        results = runner.run(dataset)
        
        assert len(results) == 2
        predictor_names = [r.predictor_name for r in results]
        assert "Aggrescan3D" in predictor_names
        assert "FallbackPredictor" in predictor_names
    
    def test_generate_report(self):
        """Test report generation."""
        runner = BenchmarkRunner()
        runner.add_predictor("FallbackPredictor")
        
        dataset = load_benchmark_dataset("synthetic")
        results = runner.run(dataset)
        
        report = runner.generate_report(results)
        
        assert isinstance(report, BenchmarkReport)
        assert len(report.results) == 1
    
    def test_report_summary(self):
        """Test report summary generation."""
        runner = BenchmarkRunner()
        runner.add_predictor("FallbackPredictor")
        
        dataset = WaltzDBLoader.create_synthetic_dataset(n_positive=10, n_negative=10)
        results = runner.run(dataset)
        
        report = runner.generate_report(results)
        summary = report.summary()
        
        assert "FallbackPredictor" in summary
        assert "Sens" in summary
        assert "MCC" in summary
    
    def test_report_to_dict(self):
        """Test report serialization."""
        runner = BenchmarkRunner()
        runner.add_predictor("FallbackPredictor")
        
        dataset = WaltzDBLoader.create_synthetic_dataset(n_positive=5, n_negative=5)
        results = runner.run(dataset)
        
        report = runner.generate_report(results)
        data = report.to_dict()
        
        assert "generated_at" in data
        assert "datasets" in data


class TestQuickBenchmark:
    """Test convenience functions."""
    
    def test_quick_benchmark(self):
        """Test quick_benchmark function."""
        results = quick_benchmark(
            predictor_names=["FallbackPredictor"],
            dataset="synthetic",
        )
        
        assert len(results) >= 1
        assert results[0].n_samples > 0


class TestComparisons:
    """Test predictor comparison functions."""
    
    def test_compare_predictors(self):
        """Test predictor comparison."""
        runner = BenchmarkRunner()
        runner.add_predictor("Aggrescan3D")
        runner.add_predictor("FallbackPredictor")
        
        dataset = WaltzDBLoader.create_synthetic_dataset(n_positive=10, n_negative=10)
        results = runner.run(dataset)
        
        comparison = compare_predictors(results)
        
        assert "Aggrescan3D" in comparison
        assert "FallbackPredictor" in comparison
        assert "mcc" in comparison["Aggrescan3D"]
    
    def test_rank_predictors(self):
        """Test predictor ranking."""
        runner = BenchmarkRunner()
        runner.add_predictor("Aggrescan3D")
        runner.add_predictor("FallbackPredictor")
        
        dataset = WaltzDBLoader.create_synthetic_dataset(n_positive=10, n_negative=10)
        results = runner.run(dataset)
        
        comparison = compare_predictors(results)
        rankings = rank_predictors(comparison, by="mcc")
        
        assert len(rankings) == 2
        # Rankings should be sorted descending
        assert rankings[0][1] >= rankings[1][1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
