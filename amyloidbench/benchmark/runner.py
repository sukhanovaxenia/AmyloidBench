"""
Benchmark runner for systematic predictor evaluation.

This module provides the BenchmarkRunner class which orchestrates
the evaluation of amyloidogenicity predictors against curated
benchmark datasets.

Usage
-----
Basic benchmarking:

    >>> runner = BenchmarkRunner()
    >>> runner.add_predictor("Aggrescan3D")
    >>> runner.add_predictor("FoldAmyloid")
    >>> 
    >>> dataset = load_benchmark_dataset("synthetic")
    >>> results = runner.run(dataset)
    >>> 
    >>> for result in results:
    ...     print(result.summary())

Full evaluation pipeline:

    >>> runner = BenchmarkRunner(parallel=True)
    >>> runner.add_all_predictors()
    >>> 
    >>> # Run on multiple datasets
    >>> datasets = ["waltz", "crossbeta"]
    >>> all_results = runner.run_all_datasets(datasets)
    >>> 
    >>> # Generate comparison report
    >>> report = runner.generate_report(all_results)
    >>> report.save("benchmark_results.html")

Cross-validation:

    >>> runner = BenchmarkRunner()
    >>> runner.add_predictor("FallbackPredictor")
    >>> 
    >>> cv_results = runner.cross_validate(dataset, n_folds=5)
    >>> print(f"Mean AUC: {cv_results.mean_auc:.3f} ± {cv_results.std_auc:.3f}")
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np

from amyloidbench.core.models import PredictionResult, ProteinRecord, Region
from amyloidbench.predictors.base import BasePredictor, get_predictor, list_predictors

from .datasets import (
    AmyloidStatus,
    BenchmarkDataset,
    BenchmarkEntry,
    load_benchmark_dataset,
)
from .metrics import (
    BenchmarkResult,
    ClassificationMetrics,
    PerResidueMetrics,
    calculate_classification_metrics,
    calculate_per_residue_metrics,
    compare_predictors,
    rank_predictors,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Benchmark Runner
# =============================================================================

@dataclass
class RunnerConfig:
    """
    Configuration for benchmark runner.
    
    Attributes:
        threshold: Classification threshold (or 'optimal' for auto)
        min_region_length: Minimum APR length for region matching
        max_workers: Maximum parallel workers (0 = sequential)
        timeout: Per-prediction timeout in seconds
        cache_predictions: Whether to cache predictor outputs
        include_per_residue: Whether to compute per-residue metrics
        save_individual_results: Whether to store per-sample predictions
    """
    threshold: Union[float, str] = 0.5
    min_region_length: int = 5
    max_workers: int = 0
    timeout: float = 60.0
    cache_predictions: bool = True
    include_per_residue: bool = True
    save_individual_results: bool = True


class BenchmarkRunner:
    """
    Orchestrates predictor evaluation against benchmark datasets.
    
    The BenchmarkRunner manages:
    1. Predictor registration and configuration
    2. Dataset loading and preprocessing
    3. Prediction generation (sequential or parallel)
    4. Metric calculation at multiple levels
    5. Result aggregation and reporting
    
    Attributes:
        predictors: Dictionary of registered predictors
        config: Runner configuration
        results_cache: Cache of computed results
    """
    
    def __init__(
        self,
        config: Optional[RunnerConfig] = None,
        parallel: bool = False,
    ):
        """
        Initialize benchmark runner.
        
        Args:
            config: Runner configuration
            parallel: Enable parallel prediction (overrides config.max_workers)
        """
        self.config = config or RunnerConfig()
        if parallel:
            self.config.max_workers = 4
        
        self.predictors: dict[str, BasePredictor] = {}
        self.results_cache: dict[str, list[BenchmarkResult]] = {}
    
    def add_predictor(
        self,
        name: str,
        predictor: Optional[BasePredictor] = None,
        **config_kwargs,
    ):
        """
        Add a predictor to the benchmark.
        
        Args:
            name: Predictor name (must be registered if predictor not provided)
            predictor: Optional predictor instance
            **config_kwargs: Configuration overrides
        """
        if predictor is None:
            predictor = get_predictor(name, **config_kwargs)
        
        self.predictors[name] = predictor
        logger.info(f"Added predictor: {name}")
    
    def add_all_predictors(self, local_only: bool = True):
        """
        Add all available predictors.
        
        Args:
            local_only: Only add predictors that run locally
        """
        for pred_info in list_predictors():
            if local_only and pred_info.get("type") in ["web", "remote"]:
                continue
            
            try:
                self.add_predictor(pred_info["name"])
            except Exception as e:
                logger.warning(f"Could not add {pred_info['name']}: {e}")
    
    def remove_predictor(self, name: str):
        """Remove a predictor from the benchmark."""
        if name in self.predictors:
            del self.predictors[name]
    
    def run(
        self,
        dataset: BenchmarkDataset,
        predictors: Optional[list[str]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> list[BenchmarkResult]:
        """
        Run benchmark on a dataset.
        
        Args:
            dataset: Benchmark dataset to evaluate
            predictors: Optional subset of predictors to run
            progress_callback: Optional callback(current, total, predictor_name)
            
        Returns:
            List of BenchmarkResult objects, one per predictor
        """
        predictors_to_run = predictors or list(self.predictors.keys())
        
        if not predictors_to_run:
            raise ValueError("No predictors registered. Call add_predictor() first.")
        
        results = []
        
        for pred_name in predictors_to_run:
            if pred_name not in self.predictors:
                logger.warning(f"Predictor not registered: {pred_name}")
                continue
            
            logger.info(f"Evaluating {pred_name} on {dataset.name}...")
            
            result = self._evaluate_predictor(
                pred_name,
                dataset,
                progress_callback,
            )
            results.append(result)
        
        return results
    
    def _evaluate_predictor(
        self,
        pred_name: str,
        dataset: BenchmarkDataset,
        progress_callback: Optional[Callable] = None,
    ) -> BenchmarkResult:
        """Evaluate a single predictor on a dataset."""
        predictor = self.predictors[pred_name]
        
        start_time = time.time()
        
        # Collect predictions
        y_true = []
        y_pred = []
        y_scores = []
        per_sample = []
        
        all_true_labels = []
        all_pred_labels = []
        
        entries = [e for e in dataset.entries 
                   if e.amyloid_status in [AmyloidStatus.POSITIVE, AmyloidStatus.NEGATIVE]]
        
        for i, entry in enumerate(entries):
            if progress_callback:
                progress_callback(i + 1, len(entries), pred_name)
            
            try:
                # Get prediction
                protein = entry.to_protein_record()
                result = predictor.predict(protein)
                
                # Binary classification
                true_label = entry.is_positive
                pred_label = result.is_amyloidogenic or False
                score = result.amyloid_probability or (1.0 if pred_label else 0.0)
                
                y_true.append(true_label)
                y_pred.append(pred_label)
                y_scores.append(score)
                
                # Per-residue labels if available
                if entry.has_residue_labels and result.per_residue_scores:
                    true_residue_labels = entry.residue_labels
                    
                    # Get predicted residue labels
                    threshold = result.per_residue_scores.threshold or self.config.threshold
                    if isinstance(threshold, str):
                        threshold = 0.5
                    
                    pred_residue_labels = [
                        s >= threshold 
                        for s in result.per_residue_scores.scores
                    ]
                    
                    # Align lengths
                    min_len = min(len(true_residue_labels), len(pred_residue_labels))
                    all_true_labels.extend(true_residue_labels[:min_len])
                    all_pred_labels.extend(pred_residue_labels[:min_len])
                
                # Store individual result
                if self.config.save_individual_results:
                    per_sample.append({
                        "id": entry.id,
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "score": score,
                        "n_regions": len(result.predicted_regions),
                    })
                    
            except Exception as e:
                logger.warning(f"Prediction failed for {entry.id}: {e}")
                continue
        
        runtime = time.time() - start_time
        
        # Calculate classification metrics
        if not y_true:
            logger.error(f"No valid predictions for {pred_name}")
            return BenchmarkResult(
                predictor_name=pred_name,
                dataset_name=dataset.name,
                n_samples=0,
                classification_metrics=ClassificationMetrics(),
                runtime_seconds=runtime,
            )
        
        # Handle threshold optimization
        threshold = self.config.threshold
        if threshold == "optimal" and y_scores:
            from .metrics import find_optimal_threshold
            threshold = find_optimal_threshold(
                np.array(y_true),
                np.array(y_scores),
            )
            # Re-classify with optimal threshold
            y_pred = [s >= threshold for s in y_scores]
        
        classification_metrics = calculate_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_scores=y_scores if y_scores else None,
            threshold=threshold if isinstance(threshold, float) else 0.5,
        )
        
        # Calculate per-residue metrics if available
        per_residue_metrics = None
        if all_true_labels and all_pred_labels and self.config.include_per_residue:
            per_residue_metrics = calculate_per_residue_metrics(
                all_true_labels,
                all_pred_labels,
            )
        
        return BenchmarkResult(
            predictor_name=pred_name,
            dataset_name=dataset.name,
            n_samples=len(y_true),
            classification_metrics=classification_metrics,
            per_residue_metrics=per_residue_metrics,
            per_sample_results=per_sample,
            runtime_seconds=runtime,
        )
    
    def run_all_datasets(
        self,
        dataset_sources: list[str],
        data_paths: Optional[dict[str, Path]] = None,
    ) -> dict[str, list[BenchmarkResult]]:
        """
        Run benchmark on multiple datasets.
        
        Args:
            dataset_sources: List of dataset names
            data_paths: Optional paths for datasets requiring files
            
        Returns:
            Dictionary mapping dataset name to results list
        """
        data_paths = data_paths or {}
        all_results = {}
        
        for source in dataset_sources:
            logger.info(f"Loading dataset: {source}")
            
            try:
                dataset = load_benchmark_dataset(
                    source,
                    path=data_paths.get(source),
                )
            except Exception as e:
                logger.error(f"Failed to load {source}: {e}")
                continue
            
            results = self.run(dataset)
            all_results[dataset.name] = results
        
        return all_results
    
    def cross_validate(
        self,
        dataset: BenchmarkDataset,
        predictor_name: str,
        n_folds: int = 5,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """
        Perform cross-validation for a trainable predictor.
        
        Args:
            dataset: Benchmark dataset
            predictor_name: Name of predictor to evaluate
            n_folds: Number of CV folds
            random_state: Random seed
            
        Returns:
            Dictionary with CV results
        """
        np.random.seed(random_state)
        
        predictor = self.predictors.get(predictor_name)
        if predictor is None:
            raise ValueError(f"Predictor not found: {predictor_name}")
        
        # Get positive/negative entries
        positives, negatives = dataset.get_positive_negative_split()
        
        # Shuffle
        np.random.shuffle(positives)
        np.random.shuffle(negatives)
        
        # Create folds (stratified)
        fold_results = []
        
        pos_fold_size = len(positives) // n_folds
        neg_fold_size = len(negatives) // n_folds
        
        for fold in range(n_folds):
            # Create test set for this fold
            pos_start = fold * pos_fold_size
            pos_end = pos_start + pos_fold_size if fold < n_folds - 1 else len(positives)
            neg_start = fold * neg_fold_size
            neg_end = neg_start + neg_fold_size if fold < n_folds - 1 else len(negatives)
            
            test_entries = positives[pos_start:pos_end] + negatives[neg_start:neg_end]
            
            test_dataset = BenchmarkDataset(
                name=f"{dataset.name}_fold{fold}",
                description=f"CV fold {fold}",
                entries=test_entries,
            )
            
            # Evaluate
            result = self._evaluate_predictor(predictor_name, test_dataset)
            fold_results.append(result)
        
        # Aggregate results
        metrics = [r.classification_metrics for r in fold_results]
        
        return {
            "n_folds": n_folds,
            "mean_accuracy": np.mean([m.accuracy for m in metrics]),
            "std_accuracy": np.std([m.accuracy for m in metrics]),
            "mean_sensitivity": np.mean([m.sensitivity for m in metrics]),
            "std_sensitivity": np.std([m.sensitivity for m in metrics]),
            "mean_specificity": np.mean([m.specificity for m in metrics]),
            "std_specificity": np.std([m.specificity for m in metrics]),
            "mean_mcc": np.mean([m.mcc for m in metrics]),
            "std_mcc": np.std([m.mcc for m in metrics]),
            "mean_auc": np.mean([m.auc_roc or 0 for m in metrics]),
            "std_auc": np.std([m.auc_roc or 0 for m in metrics]),
            "fold_results": fold_results,
        }
    
    def generate_report(
        self,
        results: Union[list[BenchmarkResult], dict[str, list[BenchmarkResult]]],
        output_path: Optional[Path] = None,
    ) -> BenchmarkReport:
        """
        Generate a comprehensive benchmark report.
        
        Args:
            results: Benchmark results (single dataset or multiple)
            output_path: Optional path to save report
            
        Returns:
            BenchmarkReport object
        """
        # Normalize to dict format
        if isinstance(results, list):
            results = {"default": results}
        
        report = BenchmarkReport(results)
        
        if output_path:
            report.save(output_path)
        
        return report


@dataclass
class BenchmarkReport:
    """
    Comprehensive benchmark report with comparison tables.
    """
    results: dict[str, list[BenchmarkResult]]
    generated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Compute summary statistics."""
        self._compute_summary()
    
    def _compute_summary(self):
        """Compute summary tables."""
        self.comparisons = {}
        self.rankings = {}
        
        for dataset_name, dataset_results in self.results.items():
            comparison = compare_predictors(dataset_results)
            self.comparisons[dataset_name] = comparison
            self.rankings[dataset_name] = rank_predictors(comparison, by="mcc")
    
    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            "=" * 70,
            "AMYLOIDBENCH BENCHMARK REPORT",
            f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            "",
        ]
        
        for dataset_name, dataset_results in self.results.items():
            lines.append(f"Dataset: {dataset_name}")
            lines.append("-" * 50)
            
            # Summary table
            lines.append(f"{'Predictor':<20} {'Sens':>8} {'Spec':>8} {'MCC':>8} {'AUC':>8}")
            lines.append("-" * 50)
            
            for result in sorted(dataset_results, 
                                key=lambda r: r.classification_metrics.mcc, 
                                reverse=True):
                m = result.classification_metrics
                lines.append(
                    f"{result.predictor_name:<20} "
                    f"{m.sensitivity:>8.3f} "
                    f"{m.specificity:>8.3f} "
                    f"{m.mcc:>8.3f} "
                    f"{m.auc_roc or 0:>8.3f}"
                )
            
            lines.append("")
            
            # Best predictor
            if self.rankings.get(dataset_name):
                best_name, best_mcc = self.rankings[dataset_name][0]
                lines.append(f"Best predictor (by MCC): {best_name} ({best_mcc:.3f})")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "datasets": {
                dataset_name: {
                    "results": [
                        {
                            "predictor": r.predictor_name,
                            "n_samples": r.n_samples,
                            "metrics": {
                                "accuracy": r.classification_metrics.accuracy,
                                "sensitivity": r.classification_metrics.sensitivity,
                                "specificity": r.classification_metrics.specificity,
                                "precision": r.classification_metrics.precision,
                                "f1_score": r.classification_metrics.f1_score,
                                "mcc": r.classification_metrics.mcc,
                                "auc_roc": r.classification_metrics.auc_roc,
                            },
                            "runtime": r.runtime_seconds,
                        }
                        for r in results
                    ],
                    "ranking": self.rankings.get(dataset_name, []),
                }
                for dataset_name, results in self.results.items()
            },
        }
    
    def save(self, path: Path):
        """
        Save report to file.
        
        Supports .txt, .json, .html formats based on extension.
        """
        path = Path(path)
        
        if path.suffix == ".txt":
            path.write_text(self.summary())
        
        elif path.suffix == ".json":
            import json
            path.write_text(json.dumps(self.to_dict(), indent=2))
        
        elif path.suffix == ".html":
            html = self._to_html()
            path.write_text(html)
        
        else:
            # Default to text
            path.write_text(self.summary())
        
        logger.info(f"Report saved to {path}")
    
    def _to_html(self) -> str:
        """Generate HTML report."""
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>AmyloidBench Benchmark Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 40px; }",
            "table { border-collapse: collapse; margin: 20px 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }",
            "th { background-color: #4CAF50; color: white; }",
            "tr:nth-child(even) { background-color: #f2f2f2; }",
            ".best { font-weight: bold; color: #2e7d32; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>AmyloidBench Benchmark Report</h1>",
            f"<p>Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>",
        ]
        
        for dataset_name, dataset_results in self.results.items():
            html.append(f"<h2>Dataset: {dataset_name}</h2>")
            
            # Results table
            html.append("<table>")
            html.append("<tr><th>Predictor</th><th>Sens</th><th>Spec</th>"
                       "<th>Prec</th><th>F1</th><th>MCC</th><th>AUC</th></tr>")
            
            # Find best MCC for highlighting
            best_mcc = max(r.classification_metrics.mcc for r in dataset_results)
            
            for result in sorted(dataset_results,
                                key=lambda r: r.classification_metrics.mcc,
                                reverse=True):
                m = result.classification_metrics
                css_class = "best" if m.mcc == best_mcc else ""
                
                html.append(
                    f"<tr class='{css_class}'>"
                    f"<td style='text-align:left'>{result.predictor_name}</td>"
                    f"<td>{m.sensitivity:.3f}</td>"
                    f"<td>{m.specificity:.3f}</td>"
                    f"<td>{m.precision:.3f}</td>"
                    f"<td>{m.f1_score:.3f}</td>"
                    f"<td>{m.mcc:.3f}</td>"
                    f"<td>{m.auc_roc or 0:.3f}</td>"
                    "</tr>"
                )
            
            html.append("</table>")
        
        html.extend([
            "</body>",
            "</html>",
        ])
        
        return "\n".join(html)


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_benchmark(
    predictor_names: Optional[list[str]] = None,
    dataset: str = "synthetic",
) -> list[BenchmarkResult]:
    """
    Quick benchmark with minimal configuration.
    
    Args:
        predictor_names: Predictors to evaluate (None = all local)
        dataset: Dataset name or 'synthetic'
        
    Returns:
        List of BenchmarkResult objects
    """
    runner = BenchmarkRunner()
    
    if predictor_names:
        for name in predictor_names:
            runner.add_predictor(name)
    else:
        runner.add_all_predictors(local_only=True)
    
    benchmark_dataset = load_benchmark_dataset(dataset)
    
    return runner.run(benchmark_dataset)
