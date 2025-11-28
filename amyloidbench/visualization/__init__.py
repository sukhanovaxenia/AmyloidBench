"""
Visualization tools for amyloidogenicity prediction analysis.

Effective visualization is crucial for interpreting prediction results
and communicating findings. This module provides publication-ready plots
for various aspects of amyloidogenicity analysis.

**Per-residue score profiles**

The primary visualization shows prediction scores along the protein
sequence, highlighting regions exceeding the amyloidogenicity threshold.
Key design features:
- Multiple predictors on same axis for direct comparison
- Threshold line clearly marked
- Known APRs annotated if available (for validation)
- Secondary structure track when available
- Domain annotations from UniProt

**Multi-predictor comparison**

Heatmaps showing agreement/disagreement between predictors:
- Rows: residue positions
- Columns: predictors
- Color: score intensity or binary prediction
- Consensus track highlighting regions with high agreement

**Benchmark result visualization**

For performance evaluation:
- ROC curves with confidence bands
- Precision-recall curves
- Metric comparison bar charts with error bars
- Score distribution histograms for positive/negative classes

**Structural visualization**

When 3D structure is available:
- Surface coloring by aggregation propensity
- APR highlighting in PyMOL-ready format
- Integration with py3Dmol for interactive web views

**Design principles**

1. *Clarity*: Unambiguous representation of biological meaning
2. *Accessibility*: Colorblind-friendly palettes (viridis default)
3. *Reproducibility*: All plots generated with specified seeds
4. *Flexibility*: Customizable for publication requirements
5. *Interactivity*: Optional Plotly versions for exploration

Submodules:
    profiles: Per-residue score visualization
    heatmaps: Multi-predictor comparison matrices
    roc: Performance evaluation curves
    structure: 3D structure coloring and rendering
"""

# Visualization module will be implemented in Phase 6
__all__ = []
