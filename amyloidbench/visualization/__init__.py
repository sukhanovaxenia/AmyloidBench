"""
Visualization tools for amyloidogenicity prediction analysis.

This module provides publication-ready visualizations for prediction
results, benchmark evaluations, and comprehensive HTML reports.

Visualization Types
-------------------

**Score Profiles**
Per-residue amyloidogenicity scores along sequences with:
- APR highlighting and threshold lines
- Multi-predictor overlays for comparison
- Known region annotations

**Agreement Heatmaps**
Pairwise predictor agreement matrices showing:
- Methodological consistency
- Algorithmic complementarity

**Benchmark Comparisons**
Performance metric visualizations including:
- Bar charts for metric comparison
- Progress bars for rankings

**Polymorph Distributions**
Probability distributions over structural classifications:
- Fold type probabilities
- Cross-β geometry predictions

**HTML Reports**
Self-contained reports with:
- Embedded figures
- Interactive collapsible sections
- Sequence highlighting
- Publication-ready formatting

Quick Start
-----------
    >>> from amyloidbench.visualization import plot_score_profile, generate_sequence_report
    >>> 
    >>> # Plot prediction scores
    >>> fig = plot_score_profile(
    ...     scores=result.per_residue_scores.scores,
    ...     threshold=0.5,
    ...     predictor_name="Aggrescan3D"
    ... )
    >>> fig.savefig("score_profile.png", dpi=300)
    >>> 
    >>> # Generate HTML report
    >>> html = generate_sequence_report(
    ...     sequence=protein.sequence,
    ...     sequence_id=protein.id,
    ...     prediction_results={"Aggrescan3D": result}
    ... )

Design Principles
-----------------
1. Colorblind-friendly palettes (viridis-based)
2. Publication-ready defaults (300 DPI)
3. Minimal external dependencies
4. Consistent styling across plot types
"""

from .plots import (
    # Availability check
    HAS_MATPLOTLIB,
    # Styling
    PlotStyle,
    CATEGORICAL_COLORS,
    # Score profiles
    plot_score_profile,
    plot_multi_predictor_profile,
    # Agreement and comparison
    plot_agreement_heatmap,
    plot_benchmark_comparison,
    # Polymorph classification
    plot_polymorph_probabilities,
    # Region diagrams
    plot_region_diagram,
    # Summary figures
    create_summary_figure,
    # Utilities
    save_figure,
    figure_to_base64,
)

from .reports import (
    # Configuration
    ReportConfig,
    # Generators
    SequenceReportGenerator,
    BenchmarkReportGenerator,
    # Convenience functions
    generate_sequence_report,
    generate_benchmark_report,
)

__all__ = [
    # Availability
    "HAS_MATPLOTLIB",
    # Styling
    "PlotStyle",
    "CATEGORICAL_COLORS",
    # Score profiles
    "plot_score_profile",
    "plot_multi_predictor_profile",
    # Comparisons
    "plot_agreement_heatmap",
    "plot_benchmark_comparison",
    # Polymorph
    "plot_polymorph_probabilities",
    # Regions
    "plot_region_diagram",
    # Summary
    "create_summary_figure",
    # Utilities
    "save_figure",
    "figure_to_base64",
    # Report configuration
    "ReportConfig",
    # Report generators
    "SequenceReportGenerator",
    "BenchmarkReportGenerator",
    # Report functions
    "generate_sequence_report",
    "generate_benchmark_report",
]
