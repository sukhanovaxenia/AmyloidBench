"""
Unified Result Export and Visualization Module.

This module provides consistent export and visualization functions for all amyloid
predictors, producing outputs that match the format of the APPNN R script:
- Per-residue TSV tables with scores and classifications
- Publication-quality visualizations with hotspot highlighting
- Batch export capabilities

The standardized output format enables:
- Cross-predictor comparison using identical data structures
- Integration with downstream analysis pipelines
- Reproducible figure generation for publications
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import PatchCollection
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from .output_models import (
    PredictorOutput,
    ResidueScore,
    PredictedRegion,
    ClassificationLabel,
    MultiPredictorOutput,
)


logger = logging.getLogger(__name__)


# ============================================================================
# DATA EXPORT FUNCTIONS
# ============================================================================

def export_to_tsv(
    output: PredictorOutput,
    filepath: Union[str, Path],
    include_raw: bool = True,
) -> Path:
    """
    Export prediction results to TSV format.
    
    Produces output matching the APPNN R script format:
    - id: Sequence identifier
    - pos: 1-indexed residue position
    - aa: Single-letter amino acid
    - score: Raw predictor score
    - normalized: Normalized score (0-1, higher = more amyloidogenic)
    - is_hotspot: Boolean classification
    - overall: Overall sequence score
    
    Args:
        output: PredictorOutput object from any predictor
        filepath: Output TSV file path
        include_raw: Include raw_output dict as JSON column
        
    Returns:
        Path to created file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Build data rows
    rows = []
    for r in output.residue_scores:
        row = {
            'id': output.sequence_id,
            'pos': r.position,
            'aa': r.residue,
            'score': r.raw_score,
            'normalized': r.normalized_score,
            'is_hotspot': r.classification == ClassificationLabel.AMYLOIDOGENIC,
            'overall': output.overall_score,
        }
        rows.append(row)
    
    if PANDAS_AVAILABLE:
        df = pd.DataFrame(rows)
        df.to_csv(filepath, sep='\t', index=False)
    else:
        # Manual TSV writing
        with open(filepath, 'w') as f:
            headers = ['id', 'pos', 'aa', 'score', 'normalized', 'is_hotspot', 'overall']
            f.write('\t'.join(headers) + '\n')
            for row in rows:
                values = [str(row[h]) for h in headers]
                f.write('\t'.join(values) + '\n')
    
    logger.info(f"Exported {len(rows)} residues to {filepath}")
    return filepath


def export_regions_to_tsv(
    output: PredictorOutput,
    filepath: Union[str, Path],
) -> Path:
    """
    Export predicted regions (hotspots/APRs) to TSV format.
    
    Columns:
    - id: Sequence identifier
    - start: 1-indexed start position
    - end: 1-indexed end position (inclusive)
    - length: Region length
    - sequence: Region sequence
    - mean_score: Mean raw score
    - max_score: Maximum raw score
    - mean_normalized: Mean normalized score
    - region_type: Type annotation (e.g., "hot_spot", "steric_zipper")
    
    Args:
        output: PredictorOutput object
        filepath: Output TSV file path
        
    Returns:
        Path to created file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for region in output.predicted_regions:
        rows.append({
            'id': output.sequence_id,
            'start': region.start,
            'end': region.end,
            'length': region.length,
            'sequence': region.sequence,
            'mean_score': region.mean_score,
            'max_score': region.max_score,
            'mean_normalized': region.mean_normalized,
            'region_type': region.region_type or 'apr',
        })
    
    if PANDAS_AVAILABLE:
        df = pd.DataFrame(rows)
        df.to_csv(filepath, sep='\t', index=False)
    else:
        with open(filepath, 'w') as f:
            headers = ['id', 'start', 'end', 'length', 'sequence', 
                      'mean_score', 'max_score', 'mean_normalized', 'region_type']
            f.write('\t'.join(headers) + '\n')
            for row in rows:
                values = [str(row[h]) for h in headers]
                f.write('\t'.join(values) + '\n')
    
    logger.info(f"Exported {len(rows)} regions to {filepath}")
    return filepath


def export_batch_results(
    outputs: List[PredictorOutput],
    output_dir: Union[str, Path],
    save_plots: bool = True,
    plot_format: str = 'png',
    dpi: int = 300,
) -> Dict[str, Dict[str, Path]]:
    """
    Export multiple prediction results to a directory.
    
    Creates:
    - Individual TSV files for each sequence
    - Individual region TSV files
    - Individual plots (optional)
    - Summary TSV with overall results
    
    Args:
        outputs: List of PredictorOutput objects
        output_dir: Output directory
        save_plots: Generate visualization plots
        plot_format: Plot format ('png', 'pdf', 'svg')
        dpi: Plot resolution
        
    Returns:
        Dictionary mapping sequence_id to {tsv, regions, plot} paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    summary_rows = []
    
    for output in outputs:
        # Safe filename
        safe_id = _safe_filename(output.sequence_id)
        
        # Export per-residue TSV
        tsv_path = export_to_tsv(
            output, 
            output_dir / f"{safe_id}.tsv"
        )
        
        # Export regions TSV
        regions_path = None
        if output.predicted_regions:
            regions_path = export_regions_to_tsv(
                output,
                output_dir / f"{safe_id}_regions.tsv"
            )
        
        # Generate plot
        plot_path = None
        if save_plots and MATPLOTLIB_AVAILABLE:
            plot_path = plot_prediction_result(
                output,
                output_dir / f"{safe_id}.{plot_format}",
                dpi=dpi,
            )
        
        results[output.sequence_id] = {
            'tsv': tsv_path,
            'regions': regions_path,
            'plot': plot_path,
        }
        
        # Summary row
        summary_rows.append({
            'id': output.sequence_id,
            'predictor': output.predictor_name,
            'length': len(output.sequence),
            'overall_score': output.overall_score,
            'overall_probability': output.overall_probability,
            'is_amyloidogenic': output.is_amyloidogenic,
            'n_regions': len(output.predicted_regions),
            'n_hotspot_residues': output.n_amyloidogenic_residues,
            'hotspot_fraction': output.amyloidogenic_fraction,
        })
    
    # Write summary
    summary_path = output_dir / "summary.tsv"
    if PANDAS_AVAILABLE:
        pd.DataFrame(summary_rows).to_csv(summary_path, sep='\t', index=False)
    else:
        with open(summary_path, 'w') as f:
            if summary_rows:
                headers = list(summary_rows[0].keys())
                f.write('\t'.join(headers) + '\n')
                for row in summary_rows:
                    f.write('\t'.join(str(row[h]) for h in headers) + '\n')
    
    logger.info(f"Batch export complete: {len(outputs)} sequences to {output_dir}")
    return results


def _safe_filename(name: str) -> str:
    """Convert string to safe filename."""
    return re.sub(r'[^A-Za-z0-9._-]', '_', name)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_prediction_result(
    output: PredictorOutput,
    filepath: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 5),
    dpi: int = 300,
    tick_step: int = 10,
    minor_step: int = 5,
    band_alpha: float = 0.2,
    show_threshold: bool = True,
    color_hotspot: str = '#D55E00',
    color_normal: str = '#000000',
    color_band: str = '#D55E00',
) -> Optional[Path]:
    """
    Generate publication-quality visualization of prediction results.
    
    Matches the output style of the APPNN R script:
    - Line plot of per-residue scores
    - Points colored by classification
    - Shaded bands for hotspot regions
    - Threshold line (optional)
    - Title with overall score
    
    Args:
        output: PredictorOutput object
        filepath: Output file path (None to display)
        figsize: Figure size (width, height) in inches
        dpi: Resolution for raster formats
        tick_step: Major tick interval on x-axis
        minor_step: Minor tick interval
        band_alpha: Transparency for hotspot bands
        show_threshold: Draw threshold line
        color_hotspot: Color for hotspot residues
        color_normal: Color for non-hotspot residues
        color_band: Color for hotspot bands
        
    Returns:
        Path to saved figure, or None if displayed
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available, skipping plot")
        return None
    
    # Extract data
    positions = [r.position for r in output.residue_scores]
    scores = [r.normalized_score for r in output.residue_scores]
    is_hotspot = [r.classification == ClassificationLabel.AMYLOIDOGENIC 
                  for r in output.residue_scores]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw hotspot bands first (behind everything)
    for region in output.predicted_regions:
        rect = mpatches.Rectangle(
            (region.start - 0.5, 0),
            region.length,
            1.0,
            alpha=band_alpha,
            facecolor=color_band,
            edgecolor='none',
            zorder=1,
        )
        ax.add_patch(rect)
    
    # Draw line
    ax.plot(positions, scores, '-', color='#333333', linewidth=1, zorder=2)
    
    # Draw points with colors based on classification
    colors = [color_hotspot if h else color_normal for h in is_hotspot]
    ax.scatter(positions, scores, c=colors, s=20, zorder=3)
    
    # Threshold line
    if show_threshold and output.threshold is not None:
        # Convert threshold to normalized scale if needed
        threshold_normalized = _normalize_threshold(output)
        ax.axhline(y=threshold_normalized, color='#888888', linestyle='--', 
                   linewidth=1, label='Threshold', zorder=1)
    
    # Formatting
    ax.set_xlim(0, len(output.sequence) + 1)
    ax.set_ylim(0, 1.05)
    
    # X-axis ticks
    xmax = len(output.sequence)
    ax.set_xticks(range(0, xmax + 1, tick_step))
    if minor_step:
        ax.set_xticks(range(0, xmax + 1, minor_step), minor=True)
    
    ax.set_xlabel('Residue Position', fontsize=11)
    ax.set_ylabel('Aggregation Propensity (normalized)', fontsize=11)
    
    # Title with overall score
    title = f"{output.sequence_id} ({output.predictor_name})"
    if output.overall_probability is not None:
        title += f"  [overall={output.overall_probability:.3f}]"
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=color_hotspot, label='Amyloidogenic'),
        mpatches.Patch(facecolor=color_normal, label='Non-amyloidogenic'),
    ]
    if output.predicted_regions:
        legend_elements.append(
            mpatches.Patch(facecolor=color_band, alpha=band_alpha, label='Hotspot Region')
        )
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    
    # Save or show
    if filepath:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved plot to {filepath}")
        return filepath
    else:
        plt.show()
        return None


def plot_multi_predictor_comparison(
    multi_output: MultiPredictorOutput,
    filepath: Optional[Union[str, Path]] = None,
    figsize: tuple = (14, 8),
    dpi: int = 300,
) -> Optional[Path]:
    """
    Generate comparison plot showing results from multiple predictors.
    
    Creates a multi-panel figure with:
    - Individual traces for each predictor
    - Consensus score overlay
    - Heatmap of per-predictor classifications
    
    Args:
        multi_output: MultiPredictorOutput with results from multiple predictors
        filepath: Output file path
        figsize: Figure size
        dpi: Resolution
        
    Returns:
        Path to saved figure
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available, skipping plot")
        return None
    
    n_predictors = len(multi_output.predictor_outputs)
    if n_predictors == 0:
        logger.warning("No predictor outputs to plot")
        return None
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                     height_ratios=[3, 1],
                                     sharex=True)
    
    positions = list(range(1, len(multi_output.sequence) + 1))
    colors = plt.cm.tab10(np.linspace(0, 1, n_predictors))
    
    # Top panel: Score traces
    for i, (name, output) in enumerate(multi_output.predictor_outputs.items()):
        if output.error is None:
            scores = [r.normalized_score for r in output.residue_scores]
            ax1.plot(positions[:len(scores)], scores, '-', 
                    color=colors[i], label=name, alpha=0.7, linewidth=1.5)
    
    # Consensus if available
    if multi_output.consensus_scores is not None:
        ax1.plot(positions, multi_output.consensus_scores, 'k-', 
                linewidth=2.5, label='Consensus', zorder=10)
    
    ax1.set_ylabel('Normalized Score', fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='upper right', fontsize=9, ncol=2)
    ax1.set_title(f"Multi-predictor Analysis: {multi_output.sequence_id}", 
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: Classification heatmap
    classification_matrix = []
    predictor_names = []
    
    for name, output in multi_output.predictor_outputs.items():
        if output.error is None:
            classifications = [
                1 if r.classification == ClassificationLabel.AMYLOIDOGENIC else 0
                for r in output.residue_scores
            ]
            # Pad if necessary
            while len(classifications) < len(multi_output.sequence):
                classifications.append(0)
            classification_matrix.append(classifications)
            predictor_names.append(name)
    
    if classification_matrix:
        im = ax2.imshow(classification_matrix, aspect='auto', cmap='RdYlGn_r',
                       vmin=0, vmax=1, extent=[0.5, len(multi_output.sequence)+0.5, 
                                               len(predictor_names)-0.5, -0.5])
        ax2.set_yticks(range(len(predictor_names)))
        ax2.set_yticklabels(predictor_names, fontsize=9)
        ax2.set_xlabel('Residue Position', fontsize=11)
    
    plt.tight_layout()
    
    if filepath:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved multi-predictor plot to {filepath}")
        return filepath
    else:
        plt.show()
        return None


def plot_sequence_heatmap(
    outputs: List[PredictorOutput],
    filepath: Optional[Union[str, Path]] = None,
    figsize: Optional[tuple] = None,
    dpi: int = 300,
    cmap: str = 'YlOrRd',
) -> Optional[Path]:
    """
    Generate heatmap showing predictions across multiple sequences.
    
    Useful for comparing predictions across a protein family or
    visualizing predictions from multiple predictors.
    
    Args:
        outputs: List of PredictorOutput objects
        filepath: Output file path
        figsize: Figure size (auto-calculated if None)
        dpi: Resolution
        cmap: Colormap name
        
    Returns:
        Path to saved figure
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    if not outputs:
        return None
    
    # Build score matrix
    max_len = max(len(o.sequence) for o in outputs)
    n_seqs = len(outputs)
    
    if figsize is None:
        figsize = (max(10, max_len / 10), max(4, n_seqs / 2))
    
    matrix = np.zeros((n_seqs, max_len))
    seq_labels = []
    
    for i, output in enumerate(outputs):
        for r in output.residue_scores:
            if r.position <= max_len:
                matrix[i, r.position - 1] = r.normalized_score
        seq_labels.append(f"{output.sequence_id} ({output.predictor_name})")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0, vmax=1)
    
    ax.set_yticks(range(n_seqs))
    ax.set_yticklabels(seq_labels, fontsize=9)
    ax.set_xlabel('Residue Position', fontsize=11)
    ax.set_title('Aggregation Propensity Heatmap', fontsize=12, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Normalized Score', fontsize=10)
    
    plt.tight_layout()
    
    if filepath:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return filepath
    else:
        plt.show()
        return None


def _normalize_threshold(output: PredictorOutput) -> float:
    """Convert threshold to normalized scale based on score type."""
    from .output_models import ScoreType
    
    threshold = output.threshold
    
    if output.score_type == ScoreType.PROBABILITY:
        return threshold
    elif output.score_type == ScoreType.PERCENTAGE:
        return threshold / 100.0
    elif output.score_type == ScoreType.ENERGY:
        # For energy, lower = more amyloidogenic
        # Typical range: -10 to 0, threshold around -5
        return max(0, min(1, (threshold + 10) / 10))
    elif output.score_type == ScoreType.RAW:
        # AGGRESCAN-like: range -0.6 to 0.8, threshold -0.02
        return max(0, min(1, (threshold + 0.6) / 1.4))
    else:
        return 0.5


# ============================================================================
# UNIFIED RESULT EXPORTER CLASS
# ============================================================================

class ResultExporter:
    """
    Unified exporter for amyloid prediction results.
    
    Provides consistent output format across all predictors, matching
    the APPNN R script conventions:
    
    - TSV files with per-residue data
    - Region annotations in separate TSV
    - Publication-quality visualizations
    - Batch processing capabilities
    
    Example:
        >>> exporter = ResultExporter(output_dir="results/")
        >>> exporter.export(prediction_result)
        >>> exporter.export_batch([result1, result2, result3])
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path] = "./exports",
        save_plots: bool = True,
        plot_format: str = 'png',
        dpi: int = 300,
    ):
        """
        Initialize the exporter.
        
        Args:
            output_dir: Default output directory
            save_plots: Generate plots by default
            plot_format: Default plot format
            dpi: Default plot resolution
        """
        self.output_dir = Path(output_dir)
        self.save_plots = save_plots
        self.plot_format = plot_format
        self.dpi = dpi
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export(
        self,
        output: PredictorOutput,
        subdir: Optional[str] = None,
    ) -> Dict[str, Path]:
        """
        Export a single prediction result.
        
        Args:
            output: PredictorOutput to export
            subdir: Optional subdirectory within output_dir
            
        Returns:
            Dictionary with paths to created files
        """
        export_dir = self.output_dir
        if subdir:
            export_dir = export_dir / subdir
        export_dir.mkdir(parents=True, exist_ok=True)
        
        safe_id = _safe_filename(output.sequence_id)
        
        result = {}
        
        # TSV export
        result['tsv'] = export_to_tsv(
            output,
            export_dir / f"{safe_id}.tsv"
        )
        
        # Regions export
        if output.predicted_regions:
            result['regions'] = export_regions_to_tsv(
                output,
                export_dir / f"{safe_id}_regions.tsv"
            )
        
        # Plot
        if self.save_plots:
            result['plot'] = plot_prediction_result(
                output,
                export_dir / f"{safe_id}.{self.plot_format}",
                dpi=self.dpi,
            )
        
        # JSON metadata
        result['json'] = export_dir / f"{safe_id}.json"
        output.to_json(result['json'])
        
        return result
    
    def export_batch(
        self,
        outputs: List[PredictorOutput],
        subdir: Optional[str] = None,
    ) -> Dict[str, Dict[str, Path]]:
        """
        Export multiple prediction results.
        
        Args:
            outputs: List of PredictorOutput objects
            subdir: Optional subdirectory
            
        Returns:
            Dictionary mapping sequence_id to file paths
        """
        export_dir = self.output_dir
        if subdir:
            export_dir = export_dir / subdir
        
        return export_batch_results(
            outputs,
            export_dir,
            save_plots=self.save_plots,
            plot_format=self.plot_format,
            dpi=self.dpi,
        )
    
    def export_comparison(
        self,
        multi_output: MultiPredictorOutput,
        subdir: Optional[str] = None,
    ) -> Dict[str, Path]:
        """
        Export multi-predictor comparison results.
        
        Args:
            multi_output: MultiPredictorOutput with multiple predictor results
            subdir: Optional subdirectory
            
        Returns:
            Dictionary with paths to created files
        """
        export_dir = self.output_dir
        if subdir:
            export_dir = export_dir / subdir
        export_dir.mkdir(parents=True, exist_ok=True)
        
        safe_id = _safe_filename(multi_output.sequence_id)
        result = {}
        
        # Comparison TSV
        if PANDAS_AVAILABLE:
            df = multi_output.to_comparison_dataframe()
            tsv_path = export_dir / f"{safe_id}_comparison.tsv"
            df.to_csv(tsv_path, sep='\t', index=False)
            result['comparison_tsv'] = tsv_path
        
        # Comparison plot
        if self.save_plots:
            result['comparison_plot'] = plot_multi_predictor_comparison(
                multi_output,
                export_dir / f"{safe_id}_comparison.{self.plot_format}",
                dpi=self.dpi,
            )
        
        # Individual exports
        for name, output in multi_output.predictor_outputs.items():
            if output.error is None:
                predictor_dir = export_dir / name
                predictor_dir.mkdir(exist_ok=True)
                result[name] = self.export(output, subdir=str(predictor_dir.relative_to(self.output_dir)))
        
        return result
