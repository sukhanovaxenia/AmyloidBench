"""
Statistical comparison visualizations for AmyloidBench.

This module provides publication-quality visualizations for statistical
predictor comparisons, integrating with the Phase 5 statistical comparison
framework.

Plot Types
----------

**Critical Difference Diagrams**
Demšar-style diagrams for visualizing Friedman/Nemenyi test results,
showing predictor rankings with critical difference bars connecting
statistically indistinguishable predictors.

**P-Value Heatmaps**
Symmetric matrices showing pairwise statistical comparison p-values,
with significance indicated by color intensity.

**Effect Size Plots**
Forest plots showing effect sizes (Cohen's d, odds ratios) with
confidence intervals for pairwise comparisons.

**Polymorph Performance**
Radar charts and grouped bar plots showing predictor performance
stratified by amyloid polymorph type.

References
----------
Demšar J. (2006) Statistical Comparisons of Classifiers over Multiple
Data Sets. JMLR 7:1-30.
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
    from matplotlib.patches import FancyBboxPatch, Rectangle
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


# =============================================================================
# Color Schemes for Statistical Plots
# =============================================================================

SIGNIFICANCE_COLORS = {
    "highly_significant": "#d62728",    # p < 0.001
    "significant": "#ff7f0e",            # p < 0.01
    "marginally_significant": "#ffbb78", # p < 0.05
    "not_significant": "#98df8a",        # p >= 0.05
}

POLYMORPH_COLORS = {
    "steric_zipper": "#1f77b4",
    "beta_arcade": "#ff7f0e",
    "beta_solenoid": "#2ca02c",
    "greek_key": "#d62728",
    "cross_alpha": "#9467bd",
    "beta_helix": "#8c564b",
    "unknown": "#7f7f7f",
}

GEOMETRY_COLORS = {
    "parallel_in_register": "#1f77b4",
    "antiparallel": "#ff7f0e",
    "out_of_register": "#2ca02c",
    "unknown": "#7f7f7f",
}


# =============================================================================
# Critical Difference Diagram
# =============================================================================

def plot_critical_difference_diagram(
    rankings: list[tuple[str, float]],
    critical_difference: float,
    title: str = "Critical Difference Diagram",
    figsize: tuple[float, float] = (10, 4),
    highlight_best: bool = True,
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """
    Create Demšar-style critical difference diagram.
    
    This visualization shows predictor rankings from a Friedman test with
    horizontal bars connecting predictors that are not significantly
    different according to the Nemenyi post-hoc test.
    
    Args:
        rankings: List of (predictor_name, mean_rank) tuples, sorted by rank
        critical_difference: CD value from Nemenyi test
        title: Plot title
        figsize: Figure dimensions
        highlight_best: Whether to highlight the best-ranked predictor
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> rankings = [('PredA', 1.2), ('PredB', 2.1), ('PredC', 2.8)]
        >>> fig = plot_critical_difference_diagram(rankings, cd=0.8)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")
    
    n_predictors = len(rankings)
    
    # Sort by rank (best = lowest rank = leftmost position)
    sorted_rankings = sorted(rankings, key=lambda x: x[1])
    names = [r[0] for r in sorted_rankings]
    ranks = np.array([r[1] for r in sorted_rankings])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up the axis
    min_rank = max(1, ranks.min() - 0.5)
    max_rank = ranks.max() + 0.5
    ax.set_xlim(min_rank - 0.3, max_rank + 0.3)
    ax.set_ylim(-0.5, n_predictors + 1)
    
    # Draw the rank axis
    ax.hlines(y=n_predictors, xmin=min_rank, xmax=max_rank, color='black', linewidth=2)
    
    # Draw tick marks
    for r in range(int(min_rank), int(max_rank) + 1):
        ax.vlines(x=r, ymin=n_predictors - 0.1, ymax=n_predictors + 0.1, color='black', linewidth=1)
        ax.text(r, n_predictors + 0.3, str(r), ha='center', va='bottom', fontsize=10)
    
    # Draw predictor labels and connecting lines
    y_positions = np.linspace(n_predictors - 1, 0.5, n_predictors)
    
    for i, (name, rank) in enumerate(zip(names, ranks)):
        y = y_positions[i]
        
        # Color best predictor differently
        color = '#1f77b4' if (highlight_best and i == 0) else 'black'
        fontweight = 'bold' if (highlight_best and i == 0) else 'normal'
        
        # Draw connecting line from name to rank position
        ax.hlines(y=y, xmin=min_rank - 0.2, xmax=rank, color='gray', linewidth=0.5, linestyle=':')
        
        # Draw marker at rank position
        ax.plot(rank, y, 'o', color=color, markersize=8, zorder=5)
        ax.vlines(x=rank, ymin=y, ymax=n_predictors, color='gray', linewidth=0.5, linestyle=':')
        
        # Draw predictor name
        ax.text(min_rank - 0.25, y, name, ha='right', va='center', fontsize=11, 
                color=color, fontweight=fontweight)
    
    # Draw critical difference bars for groups of non-significant predictors
    # Find groups where predictors are within CD of each other
    groups = _find_cd_groups(ranks, critical_difference)
    
    bar_y = n_predictors + 0.6
    for group_start, group_end in groups:
        if group_end > group_start:  # Only draw bar if group has multiple members
            ax.hlines(y=bar_y, xmin=ranks[group_start], xmax=ranks[group_end], 
                     color='#2ca02c', linewidth=3, zorder=4)
            # Draw end caps
            ax.vlines(x=ranks[group_start], ymin=bar_y - 0.05, ymax=bar_y + 0.05, 
                     color='#2ca02c', linewidth=3)
            ax.vlines(x=ranks[group_end], ymin=bar_y - 0.05, ymax=bar_y + 0.05, 
                     color='#2ca02c', linewidth=3)
            bar_y += 0.25
    
    # Add CD value annotation
    cd_x = max_rank - critical_difference / 2
    cd_y = -0.3
    ax.annotate('', xy=(cd_x - critical_difference / 2, cd_y), 
                xytext=(cd_x + critical_difference / 2, cd_y),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(cd_x, cd_y - 0.15, f'CD = {critical_difference:.2f}', ha='center', 
            va='top', fontsize=10, color='red')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved critical difference diagram to {save_path}")
    
    return fig


def _find_cd_groups(ranks: np.ndarray, cd: float) -> list[tuple[int, int]]:
    """Find groups of predictors within critical difference of each other."""
    n = len(ranks)
    groups = []
    
    i = 0
    while i < n:
        # Find extent of current group
        j = i
        while j < n - 1 and (ranks[j + 1] - ranks[i]) < cd:
            j += 1
        
        if j > i:  # Group has multiple members
            groups.append((i, j))
        
        i = j + 1
    
    return groups


# =============================================================================
# P-Value Heatmap
# =============================================================================

def plot_pvalue_heatmap(
    predictor_names: list[str],
    pvalue_matrix: np.ndarray,
    alpha: float = 0.05,
    title: str = "Pairwise Statistical Comparison",
    figsize: tuple[float, float] = (8, 6),
    annotate: bool = True,
    cmap: str = "RdYlGn_r",
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """
    Plot heatmap of pairwise p-values from statistical comparisons.
    
    Args:
        predictor_names: Names of predictors
        pvalue_matrix: Symmetric matrix of p-values (n x n)
        alpha: Significance threshold for highlighting
        title: Plot title
        figsize: Figure dimensions
        annotate: Whether to show p-values in cells
        cmap: Colormap name
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")
    
    n = len(predictor_names)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create masked array for diagonal
    masked_matrix = np.ma.masked_where(np.eye(n, dtype=bool), pvalue_matrix)
    
    # Plot heatmap
    im = ax.imshow(masked_matrix, cmap=cmap, vmin=0, vmax=1, aspect='equal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='p-value', shrink=0.8)
    
    # Add significance threshold line to colorbar
    cbar.ax.axhline(y=alpha, color='black', linestyle='--', linewidth=2)
    cbar.ax.text(1.5, alpha, f'α={alpha}', va='center', fontsize=9)
    
    # Set ticks
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(predictor_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(predictor_names, fontsize=10)
    
    # Add p-value annotations
    if annotate:
        for i in range(n):
            for j in range(n):
                if i != j:
                    p = pvalue_matrix[i, j]
                    # Format p-value
                    if p < 0.001:
                        text = '<.001'
                    elif p < 0.01:
                        text = f'{p:.3f}'
                    else:
                        text = f'{p:.2f}'
                    
                    # Color based on significance
                    color = 'white' if p < 0.3 else 'black'
                    fontweight = 'bold' if p < alpha else 'normal'
                    
                    ax.text(j, i, text, ha='center', va='center', 
                           color=color, fontsize=9, fontweight=fontweight)
    
    # Highlight significant cells
    for i in range(n):
        for j in range(n):
            if i != j and pvalue_matrix[i, j] < alpha:
                rect = Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                 fill=False, edgecolor='black', linewidth=2)
                ax.add_patch(rect)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved p-value heatmap to {save_path}")
    
    return fig


def create_pvalue_matrix_from_comparisons(
    predictor_names: list[str],
    comparisons: list,  # List of StatisticalComparison objects
) -> np.ndarray:
    """
    Create symmetric p-value matrix from pairwise comparison results.
    
    Args:
        predictor_names: List of predictor names
        comparisons: List of StatisticalComparison objects
        
    Returns:
        Symmetric n x n matrix of p-values
    """
    n = len(predictor_names)
    matrix = np.ones((n, n))
    
    name_to_idx = {name: i for i, name in enumerate(predictor_names)}
    
    for comp in comparisons:
        if hasattr(comp, 'predictor_a') and hasattr(comp, 'predictor_b'):
            i = name_to_idx.get(comp.predictor_a)
            j = name_to_idx.get(comp.predictor_b)
            if i is not None and j is not None:
                matrix[i, j] = comp.p_value
                matrix[j, i] = comp.p_value
    
    return matrix


# =============================================================================
# Effect Size Forest Plot
# =============================================================================

def plot_effect_sizes(
    comparisons: list[tuple[str, str, float, tuple[float, float]]],
    title: str = "Effect Sizes with 95% CI",
    figsize: tuple[float, float] = (10, 6),
    reference_line: float = 0.0,
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """
    Create forest plot of effect sizes with confidence intervals.
    
    Args:
        comparisons: List of (name_a, name_b, effect_size, (ci_low, ci_high)) tuples
        title: Plot title
        figsize: Figure dimensions
        reference_line: X-position for reference line (typically 0 for no effect)
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")
    
    n = len(comparisons)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_positions = np.arange(n)[::-1]
    
    for i, (name_a, name_b, effect, ci) in enumerate(comparisons):
        y = y_positions[i]
        ci_low, ci_high = ci
        
        # Determine significance color
        if ci_low > reference_line or ci_high < reference_line:
            color = '#d62728'  # Significant
            marker = 's'
        else:
            color = '#1f77b4'  # Not significant
            marker = 'o'
        
        # Draw CI line
        ax.hlines(y=y, xmin=ci_low, xmax=ci_high, color=color, linewidth=2)
        
        # Draw CI caps
        ax.vlines(x=ci_low, ymin=y - 0.1, ymax=y + 0.1, color=color, linewidth=2)
        ax.vlines(x=ci_high, ymin=y - 0.1, ymax=y + 0.1, color=color, linewidth=2)
        
        # Draw effect size point
        ax.plot(effect, y, marker, color=color, markersize=10, zorder=5)
        
        # Label
        label = f"{name_a} vs {name_b}"
        ax.text(ax.get_xlim()[0] - 0.1, y, label, ha='right', va='center', fontsize=10)
    
    # Reference line
    ax.axvline(x=reference_line, color='gray', linestyle='--', linewidth=1, zorder=0)
    
    ax.set_yticks([])
    ax.set_xlabel('Effect Size (Cohen\'s d)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#d62728', 
               markersize=10, label='Significant'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', 
               markersize=10, label='Not Significant'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved effect size plot to {save_path}")
    
    return fig


# =============================================================================
# Polymorph Performance Radar Chart
# =============================================================================

def plot_polymorph_radar(
    predictor_performances: dict[str, dict[str, float]],
    metric: str = "mcc",
    title: str = "Performance by Amyloid Polymorph",
    figsize: tuple[float, float] = (8, 8),
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """
    Create radar chart comparing predictor performance across polymorph types.
    
    Args:
        predictor_performances: Dict mapping predictor names to dicts of
            {polymorph_type: metric_value}
        metric: Name of metric being plotted (for label)
        title: Plot title
        figsize: Figure dimensions
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> performances = {
        ...     'PredA': {'steric_zipper': 0.8, 'beta_solenoid': 0.7},
        ...     'PredB': {'steric_zipper': 0.6, 'beta_solenoid': 0.9},
        ... }
        >>> fig = plot_polymorph_radar(performances)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")
    
    # Get all polymorph types
    all_polymorphs = set()
    for perfs in predictor_performances.values():
        all_polymorphs.update(perfs.keys())
    polymorphs = sorted(all_polymorphs)
    
    n_polymorphs = len(polymorphs)
    if n_polymorphs < 3:
        logger.warning("Radar chart requires at least 3 categories, falling back to bar chart")
        return plot_polymorph_bars(predictor_performances, metric, title, figsize, save_path)
    
    # Set up radar chart
    angles = np.linspace(0, 2 * np.pi, n_polymorphs, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    # Plot each predictor
    colors = plt.cm.Set2(np.linspace(0, 1, len(predictor_performances)))
    
    for (pred_name, perfs), color in zip(predictor_performances.items(), colors):
        values = [perfs.get(p, 0) for p in polymorphs]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, color=color, label=pred_name)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([p.replace('_', '\n') for p in polymorphs], fontsize=10)
    
    # Set radial limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    
    ax.set_title(title, fontsize=14, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved polymorph radar chart to {save_path}")
    
    return fig


def plot_polymorph_bars(
    predictor_performances: dict[str, dict[str, float]],
    metric: str = "mcc",
    title: str = "Performance by Amyloid Polymorph",
    figsize: tuple[float, float] = (12, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """
    Create grouped bar chart comparing predictor performance across polymorph types.
    
    Args:
        predictor_performances: Dict mapping predictor names to dicts of
            {polymorph_type: metric_value}
        metric: Name of metric being plotted (for label)
        title: Plot title
        figsize: Figure dimensions
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")
    
    # Get all polymorph types
    all_polymorphs = set()
    for perfs in predictor_performances.values():
        all_polymorphs.update(perfs.keys())
    polymorphs = sorted(all_polymorphs)
    
    predictors = list(predictor_performances.keys())
    n_predictors = len(predictors)
    n_polymorphs = len(polymorphs)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(n_polymorphs)
    width = 0.8 / n_predictors
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_predictors))
    
    for i, (pred_name, color) in enumerate(zip(predictors, colors)):
        perfs = predictor_performances[pred_name]
        values = [perfs.get(p, 0) for p in polymorphs]
        offset = (i - n_predictors / 2 + 0.5) * width
        
        bars = ax.bar(x + offset, values, width, label=pred_name, color=color, edgecolor='white')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    ax.set_xlabel('Polymorph Type', fontsize=12)
    ax.set_ylabel(f'{metric.upper()}', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace('_', '\n') for p in polymorphs], fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(title='Predictor')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved polymorph bar chart to {save_path}")
    
    return fig


# =============================================================================
# Reference Dataset Performance Summary
# =============================================================================

def plot_reference_dataset_summary(
    results: dict[str, dict[str, float]],
    dataset_names: list[str] = None,
    metrics: list[str] = None,
    title: str = "Reference Dataset Performance Summary",
    figsize: tuple[float, float] = (12, 8),
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """
    Create summary plot of predictor performance across reference datasets.
    
    Args:
        results: Dict mapping predictor names to dicts of
            {dataset_name: {metric: value}}
        dataset_names: Optional list of dataset names to include
        metrics: Optional list of metrics to show
        title: Plot title
        figsize: Figure dimensions
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")
    
    predictors = list(results.keys())
    
    # Get all datasets and metrics
    if dataset_names is None:
        dataset_names = set()
        for pred_results in results.values():
            dataset_names.update(pred_results.keys())
        dataset_names = sorted(dataset_names)
    
    if metrics is None:
        metrics = ['sensitivity', 'specificity', 'mcc', 'auc_roc']
    
    n_datasets = len(dataset_names)
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)
    if n_metrics == 1:
        axes = [axes]
    
    x = np.arange(len(predictors))
    width = 0.8 / n_datasets
    
    colors = plt.cm.Paired(np.linspace(0, 1, n_datasets))
    
    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        
        for ds_idx, (dataset, color) in enumerate(zip(dataset_names, colors)):
            values = []
            for pred in predictors:
                pred_results = results.get(pred, {})
                ds_results = pred_results.get(dataset, {})
                values.append(ds_results.get(metric, 0))
            
            offset = (ds_idx - n_datasets / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=dataset if ax_idx == 0 else "", 
                  color=color, edgecolor='white')
        
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=0.5)
        ax.set_xlim(-0.5, len(predictors) - 0.5)
    
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(predictors, rotation=45, ha='right', fontsize=10)
    axes[-1].set_xlabel('Predictor', fontsize=12)
    
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    axes[0].legend(title='Dataset', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved reference dataset summary to {save_path}")
    
    return fig


# =============================================================================
# Statistical Comparison from Phase 5 Integration
# =============================================================================

def visualize_multiple_comparison_result(
    result,  # MultipleComparisonResult object
    output_dir: Optional[Union[str, Path]] = None,
    prefix: str = "comparison",
) -> dict[str, Any]:
    """
    Generate comprehensive visualizations from MultipleComparisonResult.
    
    Creates:
    - Critical difference diagram (if Friedman test)
    - P-value heatmap
    - Effect size plot (if available)
    
    Args:
        result: MultipleComparisonResult from statistical comparison
        output_dir: Optional directory to save figures
        prefix: Filename prefix for saved figures
        
    Returns:
        Dict mapping plot names to Figure objects
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")
    
    figures = {}
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Critical difference diagram
    if hasattr(result, 'rankings') and result.rankings:
        cd = result.critical_difference if hasattr(result, 'critical_difference') else 0.5
        save_path = output_dir / f"{prefix}_cd_diagram.png" if output_dir else None
        figures['cd_diagram'] = plot_critical_difference_diagram(
            rankings=result.rankings,
            critical_difference=cd,
            title=f"Critical Difference Diagram\n({result.test_name}, p={result.overall_p_value:.4f})",
            save_path=save_path,
        )
    
    # P-value heatmap from pairwise comparisons
    if hasattr(result, 'pairwise_comparisons') and result.pairwise_comparisons:
        # Extract predictor names
        pred_names = set()
        for comp in result.pairwise_comparisons:
            pred_names.add(comp.predictor_a)
            pred_names.add(comp.predictor_b)
        pred_names = sorted(pred_names)
        
        # Create p-value matrix
        pvalue_matrix = create_pvalue_matrix_from_comparisons(
            pred_names, result.pairwise_comparisons
        )
        
        save_path = output_dir / f"{prefix}_pvalue_heatmap.png" if output_dir else None
        figures['pvalue_heatmap'] = plot_pvalue_heatmap(
            predictor_names=pred_names,
            pvalue_matrix=pvalue_matrix,
            title="Pairwise Statistical Comparison (p-values)",
            save_path=save_path,
        )
    
    # Effect size plot
    effect_data = []
    if hasattr(result, 'pairwise_comparisons'):
        for comp in result.pairwise_comparisons:
            if comp.effect_size is not None and comp.confidence_interval:
                effect_data.append((
                    comp.predictor_a,
                    comp.predictor_b,
                    comp.effect_size,
                    comp.confidence_interval,
                ))
    
    if effect_data:
        save_path = output_dir / f"{prefix}_effect_sizes.png" if output_dir else None
        figures['effect_sizes'] = plot_effect_sizes(
            comparisons=effect_data,
            title="Effect Sizes with 95% Confidence Intervals",
            save_path=save_path,
        )
    
    return figures


def visualize_polymorph_benchmark_result(
    result,  # PolymorphBenchmarkResult object
    output_dir: Optional[Union[str, Path]] = None,
    prefix: str = "polymorph",
) -> dict[str, Any]:
    """
    Generate visualizations from PolymorphBenchmarkResult.
    
    Args:
        result: PolymorphBenchmarkResult from polymorph-aware benchmarking
        output_dir: Optional directory to save figures
        prefix: Filename prefix for saved figures
        
    Returns:
        Dict mapping plot names to Figure objects
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")
    
    figures = {}
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Per-fold performance
    if hasattr(result, 'per_fold_metrics') and result.per_fold_metrics:
        performances = {
            result.predictor_name: {
                fold: metrics.mcc for fold, metrics in result.per_fold_metrics.items()
            }
        }
        
        save_path = output_dir / f"{prefix}_fold_performance.png" if output_dir else None
        figures['fold_performance'] = plot_polymorph_bars(
            predictor_performances=performances,
            metric='mcc',
            title=f"MCC by Fold Type - {result.predictor_name}",
            save_path=save_path,
        )
    
    # Per-geometry performance
    if hasattr(result, 'per_geometry_metrics') and result.per_geometry_metrics:
        geom_performances = {
            result.predictor_name: {
                geom: metrics.mcc for geom, metrics in result.per_geometry_metrics.items()
            }
        }
        
        save_path = output_dir / f"{prefix}_geometry_performance.png" if output_dir else None
        figures['geometry_performance'] = plot_polymorph_bars(
            predictor_performances=geom_performances,
            metric='mcc',
            title=f"MCC by β-Sheet Geometry - {result.predictor_name}",
            save_path=save_path,
        )
    
    return figures


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_comparison_plot(
    predictor_names: list[str],
    metric_values: dict[str, list[float]],
    title: str = "Predictor Comparison",
    figsize: tuple[float, float] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """
    Create quick comparison plot for multiple predictors and metrics.
    
    Args:
        predictor_names: List of predictor names
        metric_values: Dict mapping metric names to lists of values
        title: Plot title
        figsize: Figure dimensions
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")
    
    n_predictors = len(predictor_names)
    n_metrics = len(metric_values)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(n_predictors)
    width = 0.8 / n_metrics
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))
    
    for i, (metric_name, values) in enumerate(metric_values.items()):
        offset = (i - n_metrics / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=metric_name, color=colors[i])
    
    ax.set_xlabel('Predictor', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(predictor_names, rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
