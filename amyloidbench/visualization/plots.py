"""
Visualization module for amyloidogenicity prediction results.

This module generates publication-quality figures for analyzing and
communicating amyloid prediction results. All visualizations are
designed with scientific rigor in mind, following best practices
for data visualization in structural biology.

Visualization Types
-------------------

**Score Profiles**
Per-residue amyloidogenicity scores plotted along the sequence,
enabling identification of aggregation-prone regions (APRs) and
their boundaries. Critical for:
- Identifying local aggregation hotspots
- Comparing predictor sensitivity to different sequence features
- Validating predictions against known APR boundaries

**Predictor Comparisons**
Side-by-side comparison of multiple predictors on the same sequence,
revealing algorithmic biases and consensus regions. Useful for:
- Meta-predictor development
- Understanding predictor complementarity
- Identifying robust vs. algorithm-specific predictions

**Agreement Heatmaps**
Pairwise agreement between predictors across a dataset, quantifying
methodological consistency. Important for:
- Assessing predictor independence
- Identifying redundant methods
- Guiding ensemble construction

**Polymorph Distributions**
Probability distributions over structural classifications,
communicating uncertainty in fold-type predictions.

**Benchmark Comparisons**
Performance metrics across predictors and datasets with statistical
confidence intervals, enabling rigorous method comparison.

Design Principles
-----------------
1. Colorblind-friendly palettes (viridis, cividis)
2. Consistent styling across all plot types
3. Publication-ready resolution (300 DPI default)
4. Minimal chartjunk, maximum data-ink ratio
5. Explicit uncertainty representation
6. Accessible without color (patterns, annotations)

References
----------
- Tufte (2001) - Visual display of quantitative information
- Wickham (2010) - Layered grammar of graphics
- Rougier et al. (2014) - Ten rules for better figures
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

# Check matplotlib availability
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available. Visualization features disabled.")


# =============================================================================
# Color Schemes and Styling
# =============================================================================

# Colorblind-friendly palette for categorical data
CATEGORICAL_COLORS = [
    '#0077BB',  # Blue
    '#EE7733',  # Orange
    '#009988',  # Teal
    '#CC3311',  # Red
    '#33BBEE',  # Cyan
    '#EE3377',  # Magenta
    '#BBBBBB',  # Grey
    '#000000',  # Black
]

# Sequential colormap for scores (low to high amyloidogenicity)
AMYLOID_CMAP_COLORS = [
    '#F7FCF5',  # Very low (white-green)
    '#C7E9C0',  # Low
    '#74C476',  # Medium-low
    '#31A354',  # Medium
    '#006D2C',  # Medium-high
    '#00441B',  # High (dark green)
]

# Diverging colormap for comparisons
COMPARISON_CMAP_COLORS = [
    '#2166AC',  # Negative (blue)
    '#67A9CF',
    '#D1E5F0',
    '#F7F7F7',  # Neutral (white)
    '#FDDBC7',
    '#EF8A62',
    '#B2182B',  # Positive (red)
]


@dataclass
class PlotStyle:
    """
    Unified styling configuration for all plots.
    
    Attributes:
        figsize: Default figure size (width, height) in inches
        dpi: Resolution for saved figures
        font_family: Font family for text
        title_size: Title font size
        label_size: Axis label font size
        tick_size: Tick label font size
        legend_size: Legend font size
        line_width: Default line width
        marker_size: Default marker size
        grid_alpha: Grid line transparency
        spine_width: Axis spine width
    """
    figsize: tuple[float, float] = (10, 4)
    dpi: int = 300
    font_family: str = 'sans-serif'
    title_size: int = 14
    label_size: int = 12
    tick_size: int = 10
    legend_size: int = 10
    line_width: float = 1.5
    marker_size: float = 6
    grid_alpha: float = 0.3
    spine_width: float = 1.0
    
    def apply(self):
        """Apply style to matplotlib."""
        if not HAS_MATPLOTLIB:
            return
        
        plt.rcParams.update({
            'figure.figsize': self.figsize,
            'figure.dpi': self.dpi,
            'font.family': self.font_family,
            'font.size': self.label_size,
            'axes.titlesize': self.title_size,
            'axes.labelsize': self.label_size,
            'xtick.labelsize': self.tick_size,
            'ytick.labelsize': self.tick_size,
            'legend.fontsize': self.legend_size,
            'lines.linewidth': self.line_width,
            'lines.markersize': self.marker_size,
            'grid.alpha': self.grid_alpha,
            'axes.linewidth': self.spine_width,
            'axes.spines.top': False,
            'axes.spines.right': False,
        })


# Default style
DEFAULT_STYLE = PlotStyle()


# =============================================================================
# Score Profile Visualization
# =============================================================================

def plot_score_profile(
    scores: Sequence[float],
    sequence: Optional[str] = None,
    threshold: float = 0.5,
    predictor_name: str = "Predictor",
    known_regions: Optional[list[tuple[int, int]]] = None,
    highlight_apr: bool = True,
    ax: Optional[Any] = None,
    style: Optional[PlotStyle] = None,
) -> Optional[Figure]:
    """
    Plot per-residue amyloidogenicity scores along a protein sequence.
    
    This visualization is fundamental for identifying aggregation-prone
    regions (APRs) and understanding their sequence context. The plot
    shows score values with optional threshold line and APR highlighting.
    
    Args:
        scores: Per-residue amyloidogenicity scores
        sequence: Optional amino acid sequence for x-axis annotation
        threshold: Classification threshold for APR detection
        predictor_name: Name for plot title/legend
        known_regions: Optional list of (start, end) tuples for validated APRs
        highlight_apr: Whether to shade regions above threshold
        ax: Optional matplotlib axes (creates new figure if None)
        style: Plot styling configuration
        
    Returns:
        matplotlib Figure object (None if matplotlib unavailable)
        
    Example:
        >>> result = predictor.predict(protein)
        >>> fig = plot_score_profile(
        ...     result.per_residue_scores.scores,
        ...     sequence=protein.sequence,
        ...     threshold=result.per_residue_scores.threshold,
        ...     predictor_name="Aggrescan3D"
        ... )
        >>> fig.savefig("abeta_profile.png")
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib required for visualization")
        return None
    
    style = style or DEFAULT_STYLE
    style.apply()
    
    scores = np.array(scores)
    positions = np.arange(1, len(scores) + 1)
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=style.figsize)
    else:
        fig = ax.get_figure()
    
    # Plot scores
    ax.plot(positions, scores, color=CATEGORICAL_COLORS[0], 
            linewidth=style.line_width, label=predictor_name)
    
    # Threshold line
    ax.axhline(y=threshold, color='#666666', linestyle='--', 
               linewidth=1, alpha=0.7, label=f'Threshold ({threshold:.2f})')
    
    # Highlight APRs (regions above threshold)
    if highlight_apr:
        apr_mask = scores >= threshold
        ax.fill_between(positions, 0, scores, where=apr_mask,
                       alpha=0.3, color=CATEGORICAL_COLORS[0])
    
    # Mark known APR regions
    if known_regions:
        for start, end in known_regions:
            ax.axvspan(start, end, alpha=0.2, color=CATEGORICAL_COLORS[3],
                      label='Known APR' if start == known_regions[0][0] else '')
    
    # Styling
    ax.set_xlabel('Residue Position')
    ax.set_ylabel('Amyloidogenicity Score')
    ax.set_title(f'{predictor_name} Score Profile')
    ax.set_xlim(1, len(scores))
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=style.grid_alpha)
    
    # Add sequence annotation if short enough
    if sequence and len(sequence) <= 50:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        tick_positions = np.arange(1, len(sequence) + 1, max(1, len(sequence) // 10))
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels([sequence[i-1] for i in tick_positions], fontsize=8)
        ax2.set_xlabel('Sequence')
    
    plt.tight_layout()
    return fig


def plot_multi_predictor_profile(
    scores_dict: dict[str, Sequence[float]],
    sequence: Optional[str] = None,
    thresholds: Optional[dict[str, float]] = None,
    consensus_threshold: Optional[float] = None,
    ax: Optional[Any] = None,
    style: Optional[PlotStyle] = None,
) -> Optional[Figure]:
    """
    Compare score profiles from multiple predictors on the same plot.
    
    This visualization reveals predictor agreement and disagreement
    across the sequence, essential for meta-predictor development
    and understanding algorithmic biases.
    
    Args:
        scores_dict: Dictionary mapping predictor names to score arrays
        sequence: Optional amino acid sequence
        thresholds: Per-predictor thresholds
        consensus_threshold: Optional line showing consensus threshold
        ax: Optional matplotlib axes
        style: Plot styling configuration
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> scores = {
        ...     "Aggrescan3D": result1.per_residue_scores.scores,
        ...     "FoldAmyloid": result2.per_residue_scores.scores,
        ... }
        >>> fig = plot_multi_predictor_profile(scores)
    """
    if not HAS_MATPLOTLIB:
        return None
    
    style = style or DEFAULT_STYLE
    style.apply()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(style.figsize[0], style.figsize[1] * 1.2))
    else:
        fig = ax.get_figure()
    
    thresholds = thresholds or {}
    
    # Plot each predictor
    for i, (name, scores) in enumerate(scores_dict.items()):
        scores = np.array(scores)
        positions = np.arange(1, len(scores) + 1)
        color = CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)]
        
        ax.plot(positions, scores, color=color, linewidth=style.line_width,
                label=name, alpha=0.8)
        
        # Individual threshold
        if name in thresholds:
            ax.axhline(y=thresholds[name], color=color, linestyle=':',
                      linewidth=1, alpha=0.5)
    
    # Consensus threshold
    if consensus_threshold is not None:
        ax.axhline(y=consensus_threshold, color='black', linestyle='--',
                  linewidth=1.5, label=f'Consensus ({consensus_threshold:.2f})')
    
    ax.set_xlabel('Residue Position')
    ax.set_ylabel('Amyloidogenicity Score')
    ax.set_title('Multi-Predictor Score Comparison')
    ax.set_xlim(1, max(len(s) for s in scores_dict.values()))
    ax.legend(loc='upper right', framealpha=0.9, ncol=2)
    ax.grid(True, alpha=style.grid_alpha)
    
    plt.tight_layout()
    return fig


# =============================================================================
# Agreement and Comparison Visualizations
# =============================================================================

def plot_agreement_heatmap(
    agreement_matrix: np.ndarray,
    predictor_names: list[str],
    metric_name: str = "Agreement",
    annotate: bool = True,
    ax: Optional[Any] = None,
    style: Optional[PlotStyle] = None,
) -> Optional[Figure]:
    """
    Plot pairwise predictor agreement as a heatmap.
    
    Agreement matrices reveal methodological similarities between
    predictors, guiding ensemble construction and identifying
    redundant methods.
    
    Args:
        agreement_matrix: Square matrix of pairwise agreement values
        predictor_names: Names for row/column labels
        metric_name: Name of agreement metric (e.g., "Cohen's κ", "Jaccard")
        annotate: Whether to show values in cells
        ax: Optional matplotlib axes
        style: Plot styling configuration
        
    Returns:
        matplotlib Figure object
    """
    if not HAS_MATPLOTLIB:
        return None
    
    style = style or DEFAULT_STYLE
    style.apply()
    
    n = len(predictor_names)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(n * 1.2 + 2, n * 1.0 + 1))
    else:
        fig = ax.get_figure()
    
    # Create heatmap
    im = ax.imshow(agreement_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric_name)
    
    # Ticks and labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(predictor_names, rotation=45, ha='right')
    ax.set_yticklabels(predictor_names)
    
    # Annotations
    if annotate:
        for i in range(n):
            for j in range(n):
                value = agreement_matrix[i, j]
                text_color = 'white' if value < 0.5 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                       color=text_color, fontsize=style.tick_size)
    
    ax.set_title(f'Predictor {metric_name}')
    
    plt.tight_layout()
    return fig


def plot_benchmark_comparison(
    results: list[dict],
    metrics: list[str] = None,
    ax: Optional[Any] = None,
    style: Optional[PlotStyle] = None,
) -> Optional[Figure]:
    """
    Bar chart comparing predictor performance metrics.
    
    Args:
        results: List of dicts with 'predictor' and metric values
        metrics: Metrics to display (default: sensitivity, specificity, MCC)
        ax: Optional matplotlib axes
        style: Plot styling configuration
        
    Returns:
        matplotlib Figure object
    """
    if not HAS_MATPLOTLIB:
        return None
    
    style = style or DEFAULT_STYLE
    style.apply()
    
    metrics = metrics or ['sensitivity', 'specificity', 'mcc']
    predictor_names = [r['predictor'] for r in results]
    n_predictors = len(predictor_names)
    n_metrics = len(metrics)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(n_predictors * 1.5 + 2, 5))
    else:
        fig = ax.get_figure()
    
    x = np.arange(n_predictors)
    width = 0.8 / n_metrics
    
    for i, metric in enumerate(metrics):
        values = [r.get(metric, 0) for r in results]
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, 
                     label=metric.replace('_', ' ').title(),
                     color=CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)])
        
        # Value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Predictor')
    ax.set_ylabel('Score')
    ax.set_title('Benchmark Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(predictor_names, rotation=45, ha='right')
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=style.grid_alpha, axis='y')
    
    plt.tight_layout()
    return fig


# =============================================================================
# Polymorph Classification Visualization
# =============================================================================

def plot_polymorph_probabilities(
    fold_probs: dict[str, float],
    geometry_probs: Optional[dict[str, float]] = None,
    title: str = "Polymorph Classification",
    ax: Optional[Any] = None,
    style: Optional[PlotStyle] = None,
) -> Optional[Figure]:
    """
    Visualize polymorph classification probabilities.
    
    Shows probability distribution over fold types and optionally
    cross-β geometries, communicating prediction uncertainty.
    
    Args:
        fold_probs: Dictionary of fold type probabilities
        geometry_probs: Optional geometry probabilities
        title: Plot title
        ax: Optional matplotlib axes
        style: Plot styling configuration
        
    Returns:
        matplotlib Figure object
    """
    if not HAS_MATPLOTLIB:
        return None
    
    style = style or DEFAULT_STYLE
    style.apply()
    
    n_plots = 2 if geometry_probs else 1
    
    if ax is None:
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]
    else:
        fig = ax.get_figure()
        axes = [ax]
    
    # Fold type probabilities
    fold_names = list(fold_probs.keys())
    fold_values = list(fold_probs.values())
    
    colors = [CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)] 
              for i in range(len(fold_names))]
    
    bars = axes[0].barh(fold_names, fold_values, color=colors)
    axes[0].set_xlabel('Probability')
    axes[0].set_title('Fold Type')
    axes[0].set_xlim(0, 1)
    
    # Highlight best prediction
    max_idx = np.argmax(fold_values)
    bars[max_idx].set_edgecolor('black')
    bars[max_idx].set_linewidth(2)
    
    # Geometry probabilities
    if geometry_probs and n_plots > 1:
        geom_names = list(geometry_probs.keys())
        geom_values = list(geometry_probs.values())
        
        axes[1].barh(geom_names, geom_values, 
                    color=CATEGORICAL_COLORS[1:len(geom_names)+1])
        axes[1].set_xlabel('Probability')
        axes[1].set_title('Cross-β Geometry')
        axes[1].set_xlim(0, 1)
    
    fig.suptitle(title, fontsize=style.title_size)
    plt.tight_layout()
    return fig


# =============================================================================
# Region Visualization
# =============================================================================

def plot_region_diagram(
    sequence_length: int,
    predicted_regions: list[tuple[int, int, str]],
    known_regions: Optional[list[tuple[int, int, str]]] = None,
    title: str = "APR Region Diagram",
    ax: Optional[Any] = None,
    style: Optional[PlotStyle] = None,
) -> Optional[Figure]:
    """
    Schematic diagram showing predicted vs known APR regions.
    
    Provides a clear visual comparison of prediction accuracy
    at the region level, useful for evaluating boundary detection.
    
    Args:
        sequence_length: Total sequence length
        predicted_regions: List of (start, end, label) for predictions
        known_regions: Optional list of (start, end, label) for ground truth
        title: Plot title
        ax: Optional matplotlib axes
        style: Plot styling configuration
        
    Returns:
        matplotlib Figure object
    """
    if not HAS_MATPLOTLIB:
        return None
    
    style = style or DEFAULT_STYLE
    style.apply()
    
    n_tracks = 2 if known_regions else 1
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(style.figsize[0], 1.5 * n_tracks + 1))
    else:
        fig = ax.get_figure()
    
    # Sequence backbone
    y_pred = 0.7
    y_known = 0.3 if known_regions else None
    
    ax.axhline(y=y_pred, color='#CCCCCC', linewidth=8, solid_capstyle='round')
    ax.text(-sequence_length * 0.02, y_pred, 'Predicted', ha='right', va='center',
           fontsize=style.tick_size)
    
    if known_regions:
        ax.axhline(y=y_known, color='#CCCCCC', linewidth=8, solid_capstyle='round')
        ax.text(-sequence_length * 0.02, y_known, 'Known', ha='right', va='center',
               fontsize=style.tick_size)
    
    # Predicted regions
    for i, (start, end, label) in enumerate(predicted_regions):
        rect = mpatches.FancyBboxPatch(
            (start, y_pred - 0.08), end - start, 0.16,
            boxstyle="round,pad=0.01", facecolor=CATEGORICAL_COLORS[0],
            edgecolor='black', linewidth=1
        )
        ax.add_patch(rect)
        ax.text((start + end) / 2, y_pred, label, ha='center', va='center',
               fontsize=7, color='white', fontweight='bold')
    
    # Known regions
    if known_regions:
        for i, (start, end, label) in enumerate(known_regions):
            rect = mpatches.FancyBboxPatch(
                (start, y_known - 0.08), end - start, 0.16,
                boxstyle="round,pad=0.01", facecolor=CATEGORICAL_COLORS[3],
                edgecolor='black', linewidth=1
            )
            ax.add_patch(rect)
            ax.text((start + end) / 2, y_known, label, ha='center', va='center',
                   fontsize=7, color='white', fontweight='bold')
    
    # Styling
    ax.set_xlim(-sequence_length * 0.15, sequence_length * 1.05)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Residue Position')
    ax.set_title(title)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    
    # Position markers
    tick_interval = max(10, sequence_length // 10)
    ax.set_xticks(np.arange(0, sequence_length + 1, tick_interval))
    
    plt.tight_layout()
    return fig


# =============================================================================
# Utility Functions
# =============================================================================

def save_figure(
    fig: Figure,
    path: Union[str, Path],
    dpi: int = 300,
    transparent: bool = False,
    bbox_inches: str = 'tight',
) -> Path:
    """
    Save figure with publication-quality settings.
    
    Args:
        fig: matplotlib Figure object
        path: Output file path
        dpi: Resolution (300 for print, 150 for web)
        transparent: Transparent background
        bbox_inches: Bounding box handling
        
    Returns:
        Path to saved file
    """
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib required")
    
    path = Path(path)
    fig.savefig(path, dpi=dpi, transparent=transparent, 
                bbox_inches=bbox_inches, facecolor='white')
    logger.info(f"Figure saved to {path}")
    return path


def figure_to_base64(fig: Figure, format: str = 'png') -> str:
    """
    Convert figure to base64 string for embedding in HTML.
    
    Args:
        fig: matplotlib Figure object
        format: Image format ('png', 'svg')
        
    Returns:
        Base64-encoded string
    """
    if not HAS_MATPLOTLIB:
        raise RuntimeError("matplotlib required")
    
    import base64
    
    buf = io.BytesIO()
    fig.savefig(buf, format=format, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/{format};base64,{encoded}"


def create_summary_figure(
    sequence: str,
    scores_dict: dict[str, Sequence[float]],
    consensus_regions: list[tuple[int, int]],
    fold_probs: Optional[dict[str, float]] = None,
    style: Optional[PlotStyle] = None,
) -> Optional[Figure]:
    """
    Create a comprehensive summary figure with multiple panels.
    
    Combines score profiles, region diagram, and optionally
    polymorph classification in a single figure.
    
    Args:
        sequence: Protein sequence
        scores_dict: Predictor scores
        consensus_regions: Identified APR regions
        fold_probs: Optional polymorph probabilities
        style: Plot styling configuration
        
    Returns:
        matplotlib Figure with multiple panels
    """
    if not HAS_MATPLOTLIB:
        return None
    
    style = style or DEFAULT_STYLE
    style.apply()
    
    n_rows = 3 if fold_probs else 2
    fig = plt.figure(figsize=(12, 3 * n_rows))
    
    # Panel A: Score profiles
    ax1 = fig.add_subplot(n_rows, 1, 1)
    plot_multi_predictor_profile(scores_dict, sequence=sequence, ax=ax1, style=style)
    ax1.set_title('A. Predictor Score Profiles')
    
    # Panel B: Region diagram
    ax2 = fig.add_subplot(n_rows, 1, 2)
    regions_with_labels = [(s, e, sequence[s:e][:6]) for s, e in consensus_regions]
    plot_region_diagram(len(sequence), regions_with_labels, ax=ax2, style=style)
    ax2.set_title('B. Consensus APR Regions')
    
    # Panel C: Polymorph classification
    if fold_probs:
        ax3 = fig.add_subplot(n_rows, 1, 3)
        # Convert to horizontal bar chart format
        names = list(fold_probs.keys())
        values = list(fold_probs.values())
        colors = [CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)] for i in range(len(names))]
        ax3.barh(names, values, color=colors)
        ax3.set_xlabel('Probability')
        ax3.set_xlim(0, 1)
        ax3.set_title('C. Polymorph Classification')
    
    plt.tight_layout()
    return fig
