"""
Per-residue score profile visualization.

This module provides functions for visualizing amyloidogenicity prediction
scores along protein sequences, the primary output for interpreting results.

Plot Types
----------

**Score Profile Plot**
Shows per-residue scores as a line plot with threshold marking and
optional APR region highlighting. Best for single-predictor analysis.

**Multi-Predictor Profile**
Overlays multiple predictors on the same axes for direct comparison.
Includes consensus track showing agreement regions.

**Annotated Profile**
Adds biological annotations including:
- Secondary structure (if available)
- Domain boundaries
- Known APRs for validation
- Sequence conservation

Design Principles
-----------------
- Colorblind-friendly palettes (default: colorbrewer qualitative)
- Clear threshold visualization
- Consistent styling for publication
- Optional matplotlib/plotly backends
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


# =============================================================================
# Color Schemes
# =============================================================================

# Colorblind-friendly palette (ColorBrewer Set2)
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

# APR highlighting colors
APR_PREDICTED_COLOR = "#ff9999"  # Light red
APR_KNOWN_COLOR = "#99ccff"      # Light blue
APR_OVERLAP_COLOR = "#cc99ff"    # Light purple

# Score colormap (low to high aggregation propensity)
SCORE_CMAP_COLORS = [
    "#2166ac",  # Blue (low)
    "#67a9cf",
    "#d1e5f0",
    "#f7f7f7",  # White (neutral)
    "#fddbc7",
    "#ef8a62",
    "#b2182b",  # Red (high)
]


@dataclass
class ProfilePlotConfig:
    """
    Configuration for score profile plots.
    
    Attributes:
        figsize: Figure size in inches (width, height)
        dpi: Resolution for saved figures
        line_width: Width of score lines
        threshold_style: Style for threshold line ('dashed', 'solid', 'dotted')
        show_threshold: Whether to show threshold line
        show_apr_regions: Whether to highlight APR regions
        show_sequence: Whether to show sequence track
        show_legend: Whether to show legend
        title: Optional plot title
        colors: Custom color list for predictors
        font_size: Base font size
        save_format: Default format for saving ('png', 'svg', 'pdf')
    """
    figsize: tuple[float, float] = (12, 4)
    dpi: int = 150
    line_width: float = 1.5
    threshold_style: str = "dashed"
    show_threshold: bool = True
    show_apr_regions: bool = True
    show_sequence: bool = False
    show_legend: bool = True
    title: Optional[str] = None
    colors: Optional[list[str]] = None
    font_size: int = 10
    save_format: str = "png"


# =============================================================================
# Main Plotting Functions
# =============================================================================

def plot_score_profile(
    scores: Sequence[float],
    threshold: float = 0.5,
    sequence: Optional[str] = None,
    predicted_regions: Optional[list[tuple[int, int]]] = None,
    known_regions: Optional[list[tuple[int, int]]] = None,
    predictor_name: str = "Predictor",
    config: Optional[ProfilePlotConfig] = None,
    ax: Optional[Any] = None,
) -> Any:
    """
    Plot per-residue amyloidogenicity scores.
    
    Args:
        scores: Per-residue prediction scores
        threshold: Classification threshold
        sequence: Optional protein sequence for annotations
        predicted_regions: List of (start, end) predicted APR regions
        known_regions: List of (start, end) known APR regions
        predictor_name: Name for legend
        config: Plot configuration
        ax: Optional matplotlib axes to plot on
        
    Returns:
        matplotlib Figure object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
    
    config = config or ProfilePlotConfig()
    colors = config.colors or PREDICTOR_COLORS
    
    # Create figure if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    else:
        fig = ax.figure
    
    scores = np.array(scores)
    positions = np.arange(len(scores))
    
    # Plot score line
    ax.plot(
        positions, scores,
        color=colors[0],
        linewidth=config.line_width,
        label=predictor_name,
    )
    
    # Add threshold line
    if config.show_threshold:
        ax.axhline(
            y=threshold,
            color='gray',
            linestyle='--' if config.threshold_style == 'dashed' else config.threshold_style,
            linewidth=1,
            label=f'Threshold ({threshold:.2f})',
            alpha=0.7,
        )
    
    # Highlight APR regions
    if config.show_apr_regions:
        # Known APRs (background)
        if known_regions:
            for start, end in known_regions:
                ax.axvspan(
                    start, end,
                    alpha=0.3,
                    color=APR_KNOWN_COLOR,
                    label='Known APR' if start == known_regions[0][0] else None,
                )
        
        # Predicted APRs
        if predicted_regions:
            for start, end in predicted_regions:
                ax.axvspan(
                    start, end,
                    alpha=0.3,
                    color=APR_PREDICTED_COLOR,
                    label='Predicted APR' if start == predicted_regions[0][0] else None,
                )
    
    # Style
    ax.set_xlabel('Residue Position', fontsize=config.font_size)
    ax.set_ylabel('Amyloidogenicity Score', fontsize=config.font_size)
    ax.set_xlim(0, len(scores) - 1)
    ax.set_ylim(0, max(1.0, np.max(scores) * 1.1))
    
    if config.title:
        ax.set_title(config.title, fontsize=config.font_size + 2)
    
    if config.show_legend:
        ax.legend(loc='upper right', fontsize=config.font_size - 2)
    
    # Add sequence track
    if config.show_sequence and sequence:
        _add_sequence_track(ax, sequence, config)
    
    plt.tight_layout()
    return fig


def plot_multi_predictor_profile(
    scores_dict: dict[str, Sequence[float]],
    thresholds: Optional[dict[str, float]] = None,
    sequence: Optional[str] = None,
    known_regions: Optional[list[tuple[int, int]]] = None,
    config: Optional[ProfilePlotConfig] = None,
) -> Any:
    """
    Plot multiple predictor scores on the same axes.
    
    Args:
        scores_dict: Dictionary mapping predictor names to score arrays
        thresholds: Optional dictionary of per-predictor thresholds
        sequence: Optional protein sequence
        known_regions: Known APR regions for validation
        config: Plot configuration
        
    Returns:
        matplotlib Figure object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")
    
    config = config or ProfilePlotConfig()
    colors = config.colors or PREDICTOR_COLORS
    thresholds = thresholds or {}
    
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    
    # Plot each predictor
    for i, (name, scores) in enumerate(scores_dict.items()):
        scores = np.array(scores)
        positions = np.arange(len(scores))
        color = colors[i % len(colors)]
        
        ax.plot(
            positions, scores,
            color=color,
            linewidth=config.line_width,
            label=name,
            alpha=0.8,
        )
        
        # Individual threshold if provided
        if name in thresholds and config.show_threshold:
            ax.axhline(
                y=thresholds[name],
                color=color,
                linestyle=':',
                linewidth=0.8,
                alpha=0.5,
            )
    
    # Global threshold (average or 0.5)
    if config.show_threshold and not thresholds:
        ax.axhline(
            y=0.5,
            color='gray',
            linestyle='--',
            linewidth=1,
            label='Threshold',
            alpha=0.7,
        )
    
    # Known APRs
    if config.show_apr_regions and known_regions:
        for i, (start, end) in enumerate(known_regions):
            ax.axvspan(
                start, end,
                alpha=0.2,
                color=APR_KNOWN_COLOR,
                label='Known APR' if i == 0 else None,
            )
    
    # Get max length from scores
    max_len = max(len(s) for s in scores_dict.values())
    
    ax.set_xlabel('Residue Position', fontsize=config.font_size)
    ax.set_ylabel('Amyloidogenicity Score', fontsize=config.font_size)
    ax.set_xlim(0, max_len - 1)
    
    if config.title:
        ax.set_title(config.title, fontsize=config.font_size + 2)
    
    if config.show_legend:
        ax.legend(loc='upper right', fontsize=config.font_size - 2)
    
    plt.tight_layout()
    return fig


def plot_consensus_profile(
    prediction_result: Any,
    show_individual: bool = True,
    config: Optional[ProfilePlotConfig] = None,
) -> Any:
    """
    Plot consensus prediction with individual predictor contributions.
    
    Args:
        prediction_result: ConsensusPredictionResult from consensus engine
        show_individual: Whether to show individual predictor lines
        config: Plot configuration
        
    Returns:
        matplotlib Figure object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")
    
    config = config or ProfilePlotConfig()
    colors = config.colors or PREDICTOR_COLORS
    
    # Create figure with subplots
    n_plots = 2 if show_individual else 1
    fig, axes = plt.subplots(
        n_plots, 1,
        figsize=(config.figsize[0], config.figsize[1] * n_plots),
        dpi=config.dpi,
        sharex=True,
    )
    
    if n_plots == 1:
        axes = [axes]
    
    # Get consensus scores
    if hasattr(prediction_result, 'consensus_per_residue_scores'):
        consensus_scores = prediction_result.consensus_per_residue_scores.scores
    else:
        # Fallback: use first available predictor
        consensus_scores = None
        for r in prediction_result.individual_results.values():
            if r.per_residue_scores:
                consensus_scores = r.per_residue_scores.scores
                break
    
    if consensus_scores is None:
        raise ValueError("No per-residue scores available")
    
    positions = np.arange(len(consensus_scores))
    
    # Main consensus plot
    ax_main = axes[0]
    ax_main.fill_between(
        positions, consensus_scores,
        alpha=0.3,
        color=colors[0],
        label='Consensus',
    )
    ax_main.plot(
        positions, consensus_scores,
        color=colors[0],
        linewidth=config.line_width + 0.5,
    )
    
    # Highlight consensus regions
    if hasattr(prediction_result, 'consensus_regions'):
        for region in prediction_result.consensus_regions:
            ax_main.axvspan(
                region.start, region.end,
                alpha=0.3,
                color=APR_PREDICTED_COLOR,
            )
    
    ax_main.set_ylabel('Consensus Score', fontsize=config.font_size)
    ax_main.set_title(config.title or 'Consensus Prediction', fontsize=config.font_size + 2)
    ax_main.legend(loc='upper right')
    
    # Individual predictor plot
    if show_individual and len(axes) > 1:
        ax_ind = axes[1]
        
        for i, (name, result) in enumerate(prediction_result.individual_results.items()):
            if result.per_residue_scores:
                scores = result.per_residue_scores.scores
                color = colors[(i + 1) % len(colors)]
                ax_ind.plot(
                    np.arange(len(scores)), scores,
                    color=color,
                    linewidth=config.line_width,
                    label=name,
                    alpha=0.7,
                )
        
        ax_ind.set_ylabel('Individual Scores', fontsize=config.font_size)
        ax_ind.legend(loc='upper right', fontsize=config.font_size - 2)
    
    axes[-1].set_xlabel('Residue Position', fontsize=config.font_size)
    
    plt.tight_layout()
    return fig


def _add_sequence_track(ax: Any, sequence: str, config: ProfilePlotConfig):
    """Add sequence annotation track below plot."""
    # Get current y-limits
    ymin, ymax = ax.get_ylim()
    
    # Add sequence as x-tick labels for sparse positions
    step = max(1, len(sequence) // 50)  # Show max ~50 labels
    
    tick_positions = list(range(0, len(sequence), step))
    tick_labels = [sequence[i] if i < len(sequence) else '' for i in tick_positions]
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=config.font_size - 2, family='monospace')


# =============================================================================
# Heatmap Visualization
# =============================================================================

def plot_score_heatmap(
    scores_dict: dict[str, Sequence[float]],
    sequence: Optional[str] = None,
    figsize: tuple[float, float] = (14, 6),
    cmap: str = 'RdYlBu_r',
    vmin: float = 0,
    vmax: float = 1,
) -> Any:
    """
    Plot heatmap of per-residue scores from multiple predictors.
    
    Args:
        scores_dict: Dictionary mapping predictor names to score arrays
        sequence: Optional protein sequence
        figsize: Figure size
        cmap: Colormap name
        vmin, vmax: Score range for color mapping
        
    Returns:
        matplotlib Figure object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")
    
    predictor_names = list(scores_dict.keys())
    n_predictors = len(predictor_names)
    
    # Build score matrix
    max_len = max(len(s) for s in scores_dict.values())
    score_matrix = np.zeros((n_predictors, max_len))
    
    for i, name in enumerate(predictor_names):
        scores = scores_dict[name]
        score_matrix[i, :len(scores)] = scores
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(
        score_matrix,
        aspect='auto',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest',
    )
    
    # Labels
    ax.set_yticks(range(n_predictors))
    ax.set_yticklabels(predictor_names)
    ax.set_xlabel('Residue Position')
    ax.set_ylabel('Predictor')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Score')
    
    # Add sequence track if provided
    if sequence:
        ax.set_xticks(range(0, len(sequence), max(1, len(sequence) // 20)))
    
    plt.tight_layout()
    return fig


# =============================================================================
# Region Visualization
# =============================================================================

def plot_apr_comparison(
    predicted_regions: dict[str, list[tuple[int, int]]],
    known_regions: Optional[list[tuple[int, int]]] = None,
    sequence_length: int = 100,
    figsize: tuple[float, float] = (12, 4),
) -> Any:
    """
    Plot comparison of APR regions from different predictors.
    
    Creates a track-style visualization where each predictor has a row
    showing its predicted regions as colored blocks.
    
    Args:
        predicted_regions: Dict mapping predictor names to region lists
        known_regions: Optional ground truth regions
        sequence_length: Length of sequence
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting")
    
    colors = PREDICTOR_COLORS
    
    # Add known regions row if provided
    all_tracks = {}
    if known_regions:
        all_tracks['Known APRs'] = known_regions
    all_tracks.update(predicted_regions)
    
    n_tracks = len(all_tracks)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each track
    for i, (name, regions) in enumerate(all_tracks.items()):
        y = n_tracks - i - 1
        color = APR_KNOWN_COLOR if name == 'Known APRs' else colors[i % len(colors)]
        
        for start, end in regions:
            rect = mpatches.Rectangle(
                (start, y - 0.4), end - start, 0.8,
                facecolor=color,
                edgecolor='black',
                linewidth=0.5,
                alpha=0.7,
            )
            ax.add_patch(rect)
        
        # Track label
        ax.text(-5, y, name, ha='right', va='center', fontsize=10)
    
    ax.set_xlim(-sequence_length * 0.15, sequence_length)
    ax.set_ylim(-0.5, n_tracks - 0.5)
    ax.set_xlabel('Residue Position')
    ax.set_yticks([])
    
    # Add position markers
    ax.set_xticks(range(0, sequence_length + 1, max(1, sequence_length // 10)))
    
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


# =============================================================================
# Utility Functions
# =============================================================================

def save_figure(
    fig: Any,
    path: Union[str, Path],
    dpi: int = 150,
    transparent: bool = False,
):
    """
    Save figure to file.
    
    Args:
        fig: matplotlib Figure object
        path: Output path
        dpi: Resolution
        transparent: Whether to use transparent background
    """
    path = Path(path)
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches='tight',
        transparent=transparent,
    )
    logger.info(f"Saved figure to {path}")


def figure_to_base64(fig: Any, format: str = 'png') -> str:
    """
    Convert figure to base64-encoded string for embedding.
    
    Args:
        fig: matplotlib Figure object
        format: Image format
        
    Returns:
        Base64-encoded image string
    """
    import base64
    
    buf = BytesIO()
    fig.savefig(buf, format=format, bbox_inches='tight')
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')
