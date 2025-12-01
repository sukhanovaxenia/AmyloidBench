"""
Performance metrics for amyloidogenicity prediction evaluation.

This module provides comprehensive metrics for evaluating prediction
algorithms at multiple levels:

1. **Binary classification**: Protein-level amyloid vs non-amyloid
2. **Per-residue prediction**: Position-specific APR detection
3. **Region overlap**: How well predicted APRs match known regions

Metrics Implemented
-------------------

**Classification Metrics**:
- Accuracy: (TP + TN) / (TP + TN + FP + FN)
- Sensitivity (Recall): TP / (TP + FN) - How many amyloids are detected
- Specificity: TN / (TN + FP) - How many non-amyloids are rejected
- Precision (PPV): TP / (TP + FP) - How many predictions are correct
- F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
- MCC: Matthews Correlation Coefficient - balanced metric
- AUC-ROC: Area under ROC curve - threshold-independent

**Per-residue Metrics**:
- Residue-level sensitivity/specificity
- Segment Overlap (SOV): Measures region boundary accuracy
- Jaccard Index: Intersection/Union of predicted vs true regions

**Threshold-independent Metrics**:
- AUC-ROC: Overall discrimination ability
- AUC-PR: Area under Precision-Recall curve (better for imbalanced data)
- Optimal threshold: Youden's J statistic maximization

Statistical Considerations
--------------------------
Amyloidogenicity datasets are often imbalanced (more negatives than
positives in hexapeptide studies, or vice versa in disease-focused
collections). This module provides:

- Balanced accuracy: (Sensitivity + Specificity) / 2
- MCC: Robust to class imbalance
- Confidence intervals via bootstrapping

References
----------
- Baldi et al. (2000) - MCC for bioinformatics
- Zemla et al. (1999) - SOV score for secondary structure
- Davis & Goadrich (2006) - AUC-PR for imbalanced classification
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ConfusionMatrix:
    """
    Binary classification confusion matrix.
    
    Attributes:
        tp: True positives
        tn: True negatives
        fp: False positives
        fn: False negatives
    """
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    
    @property
    def total(self) -> int:
        return self.tp + self.tn + self.fp + self.fn
    
    @property
    def n_positive(self) -> int:
        """Actual positives."""
        return self.tp + self.fn
    
    @property
    def n_negative(self) -> int:
        """Actual negatives."""
        return self.tn + self.fp
    
    @property
    def n_predicted_positive(self) -> int:
        """Predicted positives."""
        return self.tp + self.fp
    
    @property
    def n_predicted_negative(self) -> int:
        """Predicted negatives."""
        return self.tn + self.fn
    
    def __add__(self, other: ConfusionMatrix) -> ConfusionMatrix:
        return ConfusionMatrix(
            tp=self.tp + other.tp,
            tn=self.tn + other.tn,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
        )


@dataclass
class ClassificationMetrics:
    """
    Complete set of binary classification metrics.
    
    All metrics are computed from the confusion matrix and
    score distributions for threshold-dependent analysis.
    """
    # Confusion matrix counts
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    
    # Derived metrics
    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    sensitivity: float = 0.0  # Recall, TPR
    specificity: float = 0.0  # TNR
    precision: float = 0.0    # PPV
    npv: float = 0.0          # Negative predictive value
    f1_score: float = 0.0
    mcc: float = 0.0          # Matthews Correlation Coefficient
    
    # Threshold-independent
    auc_roc: Optional[float] = None
    auc_pr: Optional[float] = None
    
    # Threshold
    threshold: float = 0.5
    optimal_threshold: Optional[float] = None
    
    # Confidence intervals (from bootstrapping)
    sensitivity_ci: Optional[tuple[float, float]] = None
    specificity_ci: Optional[tuple[float, float]] = None
    auc_ci: Optional[tuple[float, float]] = None
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Classification Metrics",
            "=" * 40,
            f"Confusion Matrix:",
            f"  TP: {self.tp:5d}  FN: {self.fn:5d}  (Actual Positive: {self.tp + self.fn})",
            f"  FP: {self.fp:5d}  TN: {self.tn:5d}  (Actual Negative: {self.fp + self.tn})",
            "",
            f"Accuracy:          {self.accuracy:.3f}",
            f"Balanced Accuracy: {self.balanced_accuracy:.3f}",
            f"Sensitivity:       {self.sensitivity:.3f}",
            f"Specificity:       {self.specificity:.3f}",
            f"Precision:         {self.precision:.3f}",
            f"F1 Score:          {self.f1_score:.3f}",
            f"MCC:               {self.mcc:.3f}",
        ]
        
        if self.auc_roc is not None:
            lines.append(f"AUC-ROC:           {self.auc_roc:.3f}")
        if self.auc_pr is not None:
            lines.append(f"AUC-PR:            {self.auc_pr:.3f}")
        if self.optimal_threshold is not None:
            lines.append(f"Optimal Threshold: {self.optimal_threshold:.3f}")
        
        return "\n".join(lines)


@dataclass
class PerResidueMetrics:
    """
    Metrics for per-residue prediction accuracy.
    
    Evaluates how well predicted APR positions match
    experimentally determined positions.
    """
    # Basic counts
    n_residues: int = 0
    n_true_positive_residues: int = 0
    n_true_apr_residues: int = 0
    n_predicted_apr_residues: int = 0
    
    # Classification at residue level
    residue_sensitivity: float = 0.0
    residue_specificity: float = 0.0
    residue_precision: float = 0.0
    residue_f1: float = 0.0
    residue_mcc: float = 0.0
    
    # Region overlap metrics
    sov_score: float = 0.0  # Segment OVerlap
    jaccard_index: float = 0.0
    
    # Boundary accuracy
    mean_boundary_error: float = 0.0  # Average offset in residues
    
    def summary(self) -> str:
        """Generate summary."""
        lines = [
            "Per-Residue Metrics",
            "=" * 40,
            f"Total residues:     {self.n_residues}",
            f"True APR residues:  {self.n_true_apr_residues}",
            f"Pred APR residues:  {self.n_predicted_apr_residues}",
            f"Correct APR res:    {self.n_true_positive_residues}",
            "",
            f"Residue Sensitivity: {self.residue_sensitivity:.3f}",
            f"Residue Specificity: {self.residue_specificity:.3f}",
            f"Residue Precision:   {self.residue_precision:.3f}",
            f"Residue F1:          {self.residue_f1:.3f}",
            f"Residue MCC:         {self.residue_mcc:.3f}",
            "",
            f"SOV Score:           {self.sov_score:.3f}",
            f"Jaccard Index:       {self.jaccard_index:.3f}",
        ]
        return "\n".join(lines)


@dataclass
class BenchmarkResult:
    """
    Complete benchmark results for a predictor on a dataset.
    
    Attributes:
        predictor_name: Name of evaluated predictor
        dataset_name: Name of benchmark dataset
        n_samples: Number of sequences evaluated
        classification_metrics: Protein-level metrics
        per_residue_metrics: Position-specific metrics (if available)
        per_sample_results: Individual predictions for analysis
    """
    predictor_name: str
    dataset_name: str
    n_samples: int
    classification_metrics: ClassificationMetrics
    per_residue_metrics: Optional[PerResidueMetrics] = None
    per_sample_results: list[dict] = field(default_factory=list)
    runtime_seconds: float = 0.0
    
    def summary(self) -> str:
        """Generate complete summary."""
        lines = [
            f"Benchmark Results: {self.predictor_name} on {self.dataset_name}",
            "=" * 60,
            f"Samples evaluated: {self.n_samples}",
            f"Runtime: {self.runtime_seconds:.2f} seconds",
            "",
            self.classification_metrics.summary(),
        ]
        
        if self.per_residue_metrics:
            lines.append("")
            lines.append(self.per_residue_metrics.summary())
        
        return "\n".join(lines)


# =============================================================================
# Metric Calculation Functions
# =============================================================================

def calculate_classification_metrics(
    y_true: Sequence[bool],
    y_pred: Sequence[bool],
    y_scores: Optional[Sequence[float]] = None,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """
    Calculate complete classification metrics.
    
    Args:
        y_true: True labels (True = amyloid)
        y_pred: Predicted labels
        y_scores: Optional prediction scores for AUC calculation
        threshold: Classification threshold used
        
    Returns:
        ClassificationMetrics object
    """
    y_true = np.array(y_true, dtype=bool)
    y_pred = np.array(y_pred, dtype=bool)
    
    # Confusion matrix
    tp = np.sum(y_true & y_pred)
    tn = np.sum(~y_true & ~y_pred)
    fp = np.sum(~y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)
    
    # Basic metrics
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    balanced_accuracy = (sensitivity + specificity) / 2
    
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    # MCC
    mcc = calculate_mcc(tp, tn, fp, fn)
    
    metrics = ClassificationMetrics(
        tp=int(tp),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        npv=npv,
        f1_score=f1,
        mcc=mcc,
        threshold=threshold,
    )
    
    # AUC if scores provided
    if y_scores is not None:
        y_scores = np.array(y_scores)
        metrics.auc_roc = calculate_auc_roc(y_true, y_scores)
        metrics.auc_pr = calculate_auc_pr(y_true, y_scores)
        metrics.optimal_threshold = find_optimal_threshold(y_true, y_scores)
    
    return metrics


def calculate_mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    Calculate Matthews Correlation Coefficient.
    
    MCC ranges from -1 (total disagreement) to +1 (perfect prediction).
    It is particularly useful for imbalanced datasets.
    
    MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    """
    numerator = tp * tn - fp * fn
    denominator = np.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def calculate_auc_roc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Calculate Area Under ROC Curve.
    
    Uses trapezoidal integration over sorted scores.
    """
    # Sort by scores descending
    order = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[order]
    
    # Calculate TPR and FPR at each threshold
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.5  # Undefined, return random baseline
    
    tpr = np.cumsum(y_true_sorted) / n_pos
    fpr = np.cumsum(~y_true_sorted) / n_neg
    
    # Add (0,0) point
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    
    # Trapezoidal integration
    auc = np.trapezoid(tpr, fpr)
    
    return float(auc)


def calculate_auc_pr(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Calculate Area Under Precision-Recall Curve.
    
    More informative than AUC-ROC for imbalanced datasets.
    """
    # Sort by scores descending
    order = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[order]
    
    n_pos = np.sum(y_true)
    if n_pos == 0:
        return 0.0
    
    # Calculate precision and recall at each threshold
    tp_cumsum = np.cumsum(y_true_sorted)
    precision = tp_cumsum / np.arange(1, len(y_true) + 1)
    recall = tp_cumsum / n_pos
    
    # Add (0, 1) starting point for PR curve
    precision = np.concatenate([[1], precision])
    recall = np.concatenate([[0], recall])
    
    # Trapezoidal integration
    auc = np.trapezoid(precision, recall)
    
    return float(auc)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric: str = "youden",
) -> float:
    """
    Find optimal classification threshold.
    
    Args:
        y_true: True labels
        y_scores: Prediction scores
        metric: Optimization criterion
            - 'youden': Maximize Youden's J = Sensitivity + Specificity - 1
            - 'f1': Maximize F1 score
            - 'mcc': Maximize MCC
            
    Returns:
        Optimal threshold value
    """
    unique_scores = np.unique(y_scores)
    
    if len(unique_scores) < 2:
        return 0.5
    
    # Test thresholds at midpoints between unique scores
    thresholds = (unique_scores[:-1] + unique_scores[1:]) / 2
    
    best_threshold = 0.5
    best_metric = -float('inf')
    
    for thresh in thresholds:
        y_pred = y_scores >= thresh
        
        tp = np.sum(y_true & y_pred)
        tn = np.sum(~y_true & ~y_pred)
        fp = np.sum(~y_true & y_pred)
        fn = np.sum(y_true & ~y_pred)
        
        if metric == "youden":
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            value = sens + spec - 1
        elif metric == "f1":
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            value = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        elif metric == "mcc":
            value = calculate_mcc(tp, tn, fp, fn)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if value > best_metric:
            best_metric = value
            best_threshold = thresh
    
    return float(best_threshold)


def calculate_per_residue_metrics(
    true_labels: Sequence[bool],
    pred_labels: Sequence[bool],
) -> PerResidueMetrics:
    """
    Calculate per-residue prediction metrics.
    
    Args:
        true_labels: True per-residue labels (True = in APR)
        pred_labels: Predicted per-residue labels
        
    Returns:
        PerResidueMetrics object
    """
    true_labels = np.array(true_labels, dtype=bool)
    pred_labels = np.array(pred_labels, dtype=bool)
    
    n = len(true_labels)
    
    # Basic counts
    tp = np.sum(true_labels & pred_labels)
    tn = np.sum(~true_labels & ~pred_labels)
    fp = np.sum(~true_labels & pred_labels)
    fn = np.sum(true_labels & ~pred_labels)
    
    # Classification metrics at residue level
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    mcc = calculate_mcc(tp, tn, fp, fn)
    
    # Jaccard index (intersection over union)
    intersection = np.sum(true_labels & pred_labels)
    union = np.sum(true_labels | pred_labels)
    jaccard = intersection / union if union > 0 else 0
    
    # SOV score
    sov = calculate_sov(true_labels, pred_labels)
    
    return PerResidueMetrics(
        n_residues=n,
        n_true_positive_residues=int(tp),
        n_true_apr_residues=int(np.sum(true_labels)),
        n_predicted_apr_residues=int(np.sum(pred_labels)),
        residue_sensitivity=sensitivity,
        residue_specificity=specificity,
        residue_precision=precision,
        residue_f1=f1,
        residue_mcc=mcc,
        sov_score=sov,
        jaccard_index=jaccard,
    )


def calculate_sov(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
) -> float:
    """
    Calculate Segment OVerlap (SOV) score.
    
    SOV measures how well predicted segments overlap with true segments,
    accounting for segment boundaries. Originally developed for secondary
    structure prediction but applicable to APR detection.
    
    SOV = (1/N) * Σ [(min_overlap(s1,s2) + δ(s1,s2)) / max_extent(s1,s2)] * len(s1)
    """
    # Find segments in true labels
    true_segments = _find_segments(true_labels)
    pred_segments = _find_segments(pred_labels)
    
    if not true_segments:
        return 1.0 if not pred_segments else 0.0
    
    # Calculate SOV
    total_length = sum(s[1] - s[0] for s in true_segments)
    if total_length == 0:
        return 0.0
    
    sov_sum = 0.0
    
    for t_start, t_end in true_segments:
        t_len = t_end - t_start
        
        # Find overlapping predicted segments
        for p_start, p_end in pred_segments:
            # Check overlap
            overlap_start = max(t_start, p_start)
            overlap_end = min(t_end, p_end)
            
            if overlap_start < overlap_end:
                overlap = overlap_end - overlap_start
                
                # Max extent
                max_start = min(t_start, p_start)
                max_end = max(t_end, p_end)
                max_extent = max_end - max_start
                
                # Delta (bonus for good boundary alignment)
                delta = min(
                    max_extent - overlap,
                    overlap,
                    t_len // 2,
                    (p_end - p_start) // 2,
                )
                
                sov_sum += ((overlap + delta) / max_extent) * t_len
                break  # Only count best overlap per true segment
    
    return sov_sum / total_length


def _find_segments(labels: np.ndarray) -> list[tuple[int, int]]:
    """Find contiguous True segments in a boolean array."""
    segments = []
    in_segment = False
    start = 0
    
    for i, val in enumerate(labels):
        if val and not in_segment:
            in_segment = True
            start = i
        elif not val and in_segment:
            in_segment = False
            segments.append((start, i))
    
    if in_segment:
        segments.append((start, len(labels)))
    
    return segments


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """
    Calculate confidence interval via bootstrapping.
    
    Args:
        y_true: True labels
        y_scores: Prediction scores
        metric_fn: Function to compute metric (y_true, y_scores) -> float
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        random_state: Random seed
        
    Returns:
        (lower_bound, upper_bound) tuple
    """
    np.random.seed(random_state)
    n = len(y_true)
    
    bootstrap_values = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        y_true_boot = y_true[indices]
        y_scores_boot = y_scores[indices]
        
        # Calculate metric
        try:
            value = metric_fn(y_true_boot, y_scores_boot)
            bootstrap_values.append(value)
        except Exception:
            continue
    
    if not bootstrap_values:
        return (0.0, 1.0)
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_values, 100 * alpha / 2)
    upper = np.percentile(bootstrap_values, 100 * (1 - alpha / 2))
    
    return (float(lower), float(upper))


# =============================================================================
# Comparison Functions
# =============================================================================

def compare_predictors(
    results: list[BenchmarkResult],
) -> dict[str, dict[str, float]]:
    """
    Compare multiple predictors on the same dataset.
    
    Args:
        results: List of BenchmarkResult objects
        
    Returns:
        Dictionary of predictor -> metric -> value
    """
    comparison = {}
    
    for result in results:
        metrics = result.classification_metrics
        comparison[result.predictor_name] = {
            "accuracy": metrics.accuracy,
            "balanced_accuracy": metrics.balanced_accuracy,
            "sensitivity": metrics.sensitivity,
            "specificity": metrics.specificity,
            "precision": metrics.precision,
            "f1_score": metrics.f1_score,
            "mcc": metrics.mcc,
            "auc_roc": metrics.auc_roc or 0,
        }
    
    return comparison


def rank_predictors(
    comparison: dict[str, dict[str, float]],
    by: str = "mcc",
) -> list[tuple[str, float]]:
    """
    Rank predictors by a specific metric.
    
    Args:
        comparison: Output from compare_predictors()
        by: Metric to rank by
        
    Returns:
        List of (predictor_name, metric_value) sorted descending
    """
    rankings = [
        (name, metrics.get(by, 0))
        for name, metrics in comparison.items()
    ]
    
    return sorted(rankings, key=lambda x: x[1], reverse=True)
