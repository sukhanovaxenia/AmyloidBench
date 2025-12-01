"""
Benchmark result visualization.

This module provides functions for visualizing predictor performance
from benchmark evaluations, including ROC curves, precision-recall
curves, and metric comparison charts.

Plot Types
----------

**ROC Curves**
Receiver Operating Characteristic curves showing the tradeoff between
sensitivity (true positive rate) and 1-specificity (false positive rate).
Includes AUC values and confidence bands.

**Precision-Recall Curves**
More informative than ROC for imbalanced datasets. Shows precision vs
recall with Average Precision (AP) scores.

**Metric Comparison**
Bar charts comparing multiple predictors across metrics like
sensitivity, specificity, MCC, and AUC.

**Score Distributions**
Histograms showing the distribution of prediction scores for positive
and negative samples, useful for threshold selection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


# =============================================================================
# Color Schemes
# =============================================================================

PREDICTOR_COLORS = [
    "#e41a1c",  # Red
    "#377eb8",  # Blue
    "#4daf4a",  # Green
    "#984ea3",  # Purple
    "#ff7f00",  # Orange
    "#a65628",  # Brown
    "#f781bf",  # Pink
    "#999999",  # Gray
]

METRIC_COLORS = {
    "sensitivity": "#e41a1c",
    "specificity": "#377eb8",
    "precision": "#4daf4a",
    "f1_score": "#984ea3",
    "mcc": "#ff7f00",
    "auc_roc": "#a65628",
}


# =============================================================================
# ROC and PR Curves
# =============================================================================

def plot_roc_curve(
    y_true: Sequence[bool],
    y_scores: Sequence[float],
    predictor_name: str = "Predictor",
    color: Optional[str] = None,
    ax: Optional[Any] = None,
    show_auc: bool = True,
    show_diagonal: bool = True,
) -> Any:
    """
    Plot ROC curve for a single predictor.
    
    Args:
        y_true: True binary labels
        y_scores: Prediction scores
        predictor_name: Name for legend
        color: Line color
        ax: Optional axes to plot on
        show_auc: Whether to show AUC in legend
        show_diagonal: Whether to show random baseline diagonal
        
    Returns:
        matplotlib Figure object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")
    
    y_true = np.array(y_true, dtype=bool)
    y_scores = np.array(y_scores)
    
    # Calculate ROC curve points
    fpr, tpr, auc = _calculate_roc_points(y_true, y_scores)
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure
    
    color = color or PREDICTOR_COLORS[0]
    
    # Plot ROC curve
    label = f"{predictor_name} (AUC = {auc:.3f})" if show_auc else predictor_name
    ax.plot(fpr, tpr, color=color, linewidth=2, label=label)
    
    # Random baseline
    if show_diagonal:
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='lower right')
    ax.set_aspect('equal')
    
    return fig


def plot_multi_roc(
    results: dict[str, tuple[Sequence[bool], Sequence[float]]],
    figsize: tuple[float, float] = (8, 8),
    title: str = "ROC Curve Comparison",
) -> Any:
    """
    Plot ROC curves for multiple predictors.
    
    Args:
        results: Dict mapping predictor names to (y_true, y_scores) tuples
        figsize: Figure size
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, (name, (y_true, y_scores)) in enumerate(results.items()):
        color = PREDICTOR_COLORS[i % len(PREDICTOR_COLORS)]
        
        y_true = np.array(y_true, dtype=bool)
        y_scores = np.array(y_scores)
        
        fpr, tpr, auc = _calculate_roc_points(y_true, y_scores)
        
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC = {auc:.3f})")
    
    # Random baseline
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='lower right')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig


def plot_pr_curve(
    y_true: Sequence[bool],
    y_scores: Sequence[float],
    predictor_name: str = "Predictor",
    color: Optional[str] = None,
    ax: Optional[Any] = None,
    show_ap: bool = True,
) -> Any:
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True binary labels
        y_scores: Prediction scores
        predictor_name: Name for legend
        color: Line color
        ax: Optional axes
        show_ap: Whether to show Average Precision in legend
        
    Returns:
        matplotlib Figure object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")
    
    y_true = np.array(y_true, dtype=bool)
    y_scores = np.array(y_scores)
    
    # Calculate PR curve points
    precision, recall, ap = _calculate_pr_points(y_true, y_scores)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure
    
    color = color or PREDICTOR_COLORS[0]
    
    label = f"{predictor_name} (AP = {ap:.3f})" if show_ap else predictor_name
    ax.plot(recall, precision, color=color, linewidth=2, label=label)
    
    # Random baseline (prevalence)
    prevalence = np.mean(y_true)
    ax.axhline(y=prevalence, color='gray', linestyle='--', alpha=0.5, label=f'Random ({prevalence:.2f})')
    
    ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='lower left')
    
    return fig


def plot_multi_pr(
    results: dict[str, tuple[Sequence[bool], Sequence[float]]],
    figsize: tuple[float, float] = (8, 8),
    title: str = "Precision-Recall Curve Comparison",
) -> Any:
    """
    Plot PR curves for multiple predictors.
    
    Args:
        results: Dict mapping predictor names to (y_true, y_scores) tuples
        figsize: Figure size
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get prevalence from first result for baseline
    first_y_true = list(results.values())[0][0]
    prevalence = np.mean(first_y_true)
    
    for i, (name, (y_true, y_scores)) in enumerate(results.items()):
        color = PREDICTOR_COLORS[i % len(PREDICTOR_COLORS)]
        
        y_true = np.array(y_true, dtype=bool)
        y_scores = np.array(y_scores)
        
        precision, recall, ap = _calculate_pr_points(y_true, y_scores)
        
        ax.plot(recall, precision, color=color, linewidth=2, label=f"{name} (AP = {ap:.3f})")
    
    # Random baseline
    ax.axhline(y=prevalence, color='gray', linestyle='--', alpha=0.5, label=f'Random ({prevalence:.2f})')
    
    ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='lower left')
    
    plt.tight_layout()
    return fig


# =============================================================================
# Metric Comparison Charts
# =============================================================================

def plot_metric_comparison(
    metrics_dict: dict[str, dict[str, float]],
    metrics_to_plot: Optional[list[str]] = None,
    figsize: tuple[float, float] = (12, 6),
    title: str = "Predictor Performance Comparison",
) -> Any:
    """
    Plot bar chart comparing predictors across metrics.
    
    Args:
        metrics_dict: Dict mapping predictor names to metric dictionaries
        metrics_to_plot: List of metrics to include (default: common set)
        figsize: Figure size
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")
    
    if metrics_to_plot is None:
        metrics_to_plot = ['sensitivity', 'specificity', 'precision', 'f1_score', 'mcc']
    
    predictor_names = list(metrics_dict.keys())
    n_predictors = len(predictor_names)
    n_metrics = len(metrics_to_plot)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(n_predictors)
    width = 0.8 / n_metrics
    
    for i, metric in enumerate(metrics_to_plot):
        values = [metrics_dict[pred].get(metric, 0) for pred in predictor_names]
        color = METRIC_COLORS.get(metric, PREDICTOR_COLORS[i % len(PREDICTOR_COLORS)])
        
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric.replace('_', ' ').title(), color=color, alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.2f}',
                ha='center', va='bottom',
                fontsize=8,
            )
    
    ax.set_xlabel('Predictor', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(predictor_names, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.15)
    
    plt.tight_layout()
    return fig


def plot_metric_heatmap(
    metrics_dict: dict[str, dict[str, float]],
    metrics_to_plot: Optional[list[str]] = None,
    figsize: tuple[float, float] = (10, 6),
    cmap: str = 'RdYlGn',
    annot: bool = True,
) -> Any:
    """
    Plot heatmap of metrics across predictors.
    
    Args:
        metrics_dict: Dict mapping predictor names to metric dictionaries
        metrics_to_plot: List of metrics to include
        figsize: Figure size
        cmap: Colormap
        annot: Whether to annotate cells with values
        
    Returns:
        matplotlib Figure object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")
    
    if metrics_to_plot is None:
        metrics_to_plot = ['sensitivity', 'specificity', 'precision', 'f1_score', 'mcc', 'auc_roc']
    
    predictor_names = list(metrics_dict.keys())
    
    # Build matrix
    matrix = np.zeros((len(predictor_names), len(metrics_to_plot)))
    for i, pred in enumerate(predictor_names):
        for j, metric in enumerate(metrics_to_plot):
            matrix[i, j] = metrics_dict[pred].get(metric, 0)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    # Labels
    ax.set_xticks(range(len(metrics_to_plot)))
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_plot], rotation=45, ha='right')
    ax.set_yticks(range(len(predictor_names)))
    ax.set_yticklabels(predictor_names)
    
    # Annotations
    if annot:
        for i in range(len(predictor_names)):
            for j in range(len(metrics_to_plot)):
                val = matrix[i, j]
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=10)
    
    plt.colorbar(im, ax=ax, label='Score')
    ax.set_title('Performance Metrics Heatmap', fontsize=14)
    
    plt.tight_layout()
    return fig


# =============================================================================
# Score Distribution Plots
# =============================================================================

def plot_score_distribution(
    y_true: Sequence[bool],
    y_scores: Sequence[float],
    predictor_name: str = "Predictor",
    threshold: Optional[float] = None,
    figsize: tuple[float, float] = (10, 6),
    bins: int = 50,
) -> Any:
    """
    Plot histogram of prediction scores for positive and negative classes.
    
    Args:
        y_true: True binary labels
        y_scores: Prediction scores
        predictor_name: Name for title
        threshold: Optional threshold line to show
        figsize: Figure size
        bins: Number of histogram bins
        
    Returns:
        matplotlib Figure object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")
    
    y_true = np.array(y_true, dtype=bool)
    y_scores = np.array(y_scores)
    
    pos_scores = y_scores[y_true]
    neg_scores = y_scores[~y_true]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Histograms
    ax.hist(neg_scores, bins=bins, alpha=0.6, color=PREDICTOR_COLORS[1], label='Negative', density=True)
    ax.hist(pos_scores, bins=bins, alpha=0.6, color=PREDICTOR_COLORS[0], label='Positive', density=True)
    
    # Threshold line
    if threshold is not None:
        ax.axvline(x=threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.2f})')
    
    ax.set_xlabel('Prediction Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Score Distribution - {predictor_name}', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_multi_score_distribution(
    results: dict[str, tuple[Sequence[bool], Sequence[float]]],
    figsize: tuple[float, float] = (14, 4),
    bins: int = 30,
) -> Any:
    """
    Plot score distributions for multiple predictors in subplots.
    
    Args:
        results: Dict mapping predictor names to (y_true, y_scores) tuples
        figsize: Figure size (will be scaled by number of predictors)
        bins: Number of histogram bins
        
    Returns:
        matplotlib Figure object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")
    
    n_predictors = len(results)
    fig, axes = plt.subplots(1, n_predictors, figsize=(figsize[0], figsize[1]))
    
    if n_predictors == 1:
        axes = [axes]
    
    for ax, (name, (y_true, y_scores)) in zip(axes, results.items()):
        y_true = np.array(y_true, dtype=bool)
        y_scores = np.array(y_scores)
        
        pos_scores = y_scores[y_true]
        neg_scores = y_scores[~y_true]
        
        ax.hist(neg_scores, bins=bins, alpha=0.6, color=PREDICTOR_COLORS[1], label='Negative', density=True)
        ax.hist(pos_scores, bins=bins, alpha=0.6, color=PREDICTOR_COLORS[0], label='Positive', density=True)
        
        ax.set_xlabel('Score')
        ax.set_title(name)
        ax.legend(fontsize=8)
    
    axes[0].set_ylabel('Density')
    fig.suptitle('Score Distributions by Predictor', fontsize=14)
    
    plt.tight_layout()
    return fig


# =============================================================================
# Ranking Visualization
# =============================================================================

def plot_predictor_ranking(
    ranking: list[tuple[str, float]],
    metric_name: str = "MCC",
    figsize: tuple[float, float] = (10, 6),
) -> Any:
    """
    Plot horizontal bar chart of predictor rankings.
    
    Args:
        ranking: List of (predictor_name, score) tuples, sorted
        metric_name: Name of ranking metric for label
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")
    
    names = [r[0] for r in ranking]
    values = [r[1] for r in ranking]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(names))
    colors = [PREDICTOR_COLORS[i % len(PREDICTOR_COLORS)] for i in range(len(names))]
    
    bars = ax.barh(y_pos, values, color=colors, alpha=0.8)
    
    # Value labels
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f'{val:.3f}', va='center', fontsize=10)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel(metric_name, fontsize=12)
    ax.set_title(f'Predictor Ranking by {metric_name}', fontsize=14)
    ax.set_xlim(0, max(values) * 1.15)
    
    # Highlight best
    ax.barh(y_pos[0], values[0], color=PREDICTOR_COLORS[0], edgecolor='gold', linewidth=3)
    
    plt.tight_layout()
    return fig


# =============================================================================
# Helper Functions
# =============================================================================

def _calculate_roc_points(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Calculate ROC curve points and AUC."""
    # Sort by scores descending
    order = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[order]
    
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return np.array([0, 1]), np.array([0, 1]), 0.5
    
    tpr = np.cumsum(y_true_sorted) / n_pos
    fpr = np.cumsum(~y_true_sorted) / n_neg
    
    # Add origin
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    
    # Calculate AUC
    auc = np.trapz(tpr, fpr)
    
    return fpr, tpr, float(auc)


def _calculate_pr_points(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Calculate PR curve points and Average Precision."""
    # Sort by scores descending
    order = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[order]
    
    n_pos = np.sum(y_true)
    if n_pos == 0:
        return np.array([1, 0]), np.array([0, 1]), 0.0
    
    tp_cumsum = np.cumsum(y_true_sorted)
    precision = tp_cumsum / np.arange(1, len(y_true) + 1)
    recall = tp_cumsum / n_pos
    
    # Add starting point
    precision = np.concatenate([[1], precision])
    recall = np.concatenate([[0], recall])
    
    # Average Precision
    ap = np.trapz(precision, recall)
    
    return precision, recall, float(ap)


def create_benchmark_figure_set(
    benchmark_results: list,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Create a complete set of benchmark visualization figures.
    
    Args:
        benchmark_results: List of BenchmarkResult objects
        output_dir: Optional directory to save figures
        
    Returns:
        Dictionary of figure names to Figure objects
    """
    figures = {}
    
    # Prepare data
    metrics_dict = {}
    roc_data = {}
    
    for result in benchmark_results:
        name = result.predictor_name
        m = result.classification_metrics
        
        metrics_dict[name] = {
            'sensitivity': m.sensitivity,
            'specificity': m.specificity,
            'precision': m.precision,
            'f1_score': m.f1_score,
            'mcc': m.mcc,
            'auc_roc': m.auc_roc or 0,
        }
    
    # Metric comparison
    figures['metric_comparison'] = plot_metric_comparison(metrics_dict)
    
    # Metric heatmap
    figures['metric_heatmap'] = plot_metric_heatmap(metrics_dict)
    
    # Rankings
    ranking = sorted(
        [(name, m['mcc']) for name, m in metrics_dict.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    figures['ranking'] = plot_predictor_ranking(ranking)
    
    # Save if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, fig in figures.items():
            fig.savefig(output_dir / f"{name}.png", dpi=150, bbox_inches='tight')
            logger.info(f"Saved {name}.png")
    
    return figures
