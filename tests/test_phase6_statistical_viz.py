"""
Tests for Phase 6 extended statistical visualization module.

This module tests the statistical comparison visualizations including
critical difference diagrams, p-value heatmaps, effect size plots,
and polymorph performance charts.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

# Check matplotlib availability
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for testing
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# Critical Difference Diagram Tests
# =============================================================================

class TestCriticalDifferenceDiagram:
    """Test critical difference diagram generation."""
    
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_basic_cd_diagram(self):
        """Test basic critical difference diagram creation."""
        from amyloidbench.visualization.statistical_plots import plot_critical_difference_diagram
        
        rankings = [
            ('PredA', 1.2),
            ('PredB', 2.1),
            ('PredC', 2.8),
            ('PredD', 3.5),
        ]
        
        fig = plot_critical_difference_diagram(rankings, critical_difference=0.8)
        
        assert fig is not None
        assert len(fig.get_axes()) == 1
        plt.close(fig)
    
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_cd_diagram_with_groups(self):
        """Test CD diagram with multiple non-significant groups."""
        from amyloidbench.visualization.statistical_plots import plot_critical_difference_diagram
        
        # Predictors 1-2 and 3-4 should form groups (within CD of each other)
        rankings = [
            ('Best', 1.0),
            ('Good', 1.3),
            ('Medium', 2.5),
            ('Worst', 2.8),
        ]
        
        fig = plot_critical_difference_diagram(
            rankings, 
            critical_difference=0.5,
            highlight_best=True
        )
        
        assert fig is not None
        plt.close(fig)
    
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_cd_diagram_single_predictor(self):
        """Test CD diagram with single predictor."""
        from amyloidbench.visualization.statistical_plots import plot_critical_difference_diagram
        
        rankings = [('OnlyPredictor', 1.0)]
        
        fig = plot_critical_difference_diagram(rankings, critical_difference=0.5)
        assert fig is not None
        plt.close(fig)


# =============================================================================
# P-Value Heatmap Tests
# =============================================================================

class TestPValueHeatmap:
    """Test p-value heatmap generation."""
    
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_basic_heatmap(self):
        """Test basic p-value heatmap creation."""
        from amyloidbench.visualization.statistical_plots import plot_pvalue_heatmap
        
        predictor_names = ['PredA', 'PredB', 'PredC']
        pvalue_matrix = np.array([
            [1.0, 0.02, 0.15],
            [0.02, 1.0, 0.08],
            [0.15, 0.08, 1.0],
        ])
        
        fig = plot_pvalue_heatmap(predictor_names, pvalue_matrix)
        
        assert fig is not None
        plt.close(fig)
    
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_heatmap_with_significance(self):
        """Test heatmap highlighting significant comparisons."""
        from amyloidbench.visualization.statistical_plots import plot_pvalue_heatmap
        
        predictor_names = ['A', 'B', 'C', 'D']
        # Create matrix with mix of significant and non-significant
        pvalue_matrix = np.array([
            [1.0, 0.001, 0.04, 0.12],
            [0.001, 1.0, 0.02, 0.08],
            [0.04, 0.02, 1.0, 0.30],
            [0.12, 0.08, 0.30, 1.0],
        ])
        
        fig = plot_pvalue_heatmap(
            predictor_names, 
            pvalue_matrix, 
            alpha=0.05,
            annotate=True
        )
        
        assert fig is not None
        plt.close(fig)
    
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_create_pvalue_matrix(self):
        """Test p-value matrix creation from comparison objects."""
        from amyloidbench.visualization.statistical_plots import create_pvalue_matrix_from_comparisons
        
        # Mock comparison objects
        class MockComparison:
            def __init__(self, a, b, p):
                self.predictor_a = a
                self.predictor_b = b
                self.p_value = p
        
        comparisons = [
            MockComparison('A', 'B', 0.01),
            MockComparison('A', 'C', 0.05),
            MockComparison('B', 'C', 0.20),
        ]
        
        matrix = create_pvalue_matrix_from_comparisons(['A', 'B', 'C'], comparisons)
        
        assert matrix.shape == (3, 3)
        assert matrix[0, 1] == 0.01
        assert matrix[1, 0] == 0.01  # Symmetric
        assert matrix[0, 2] == 0.05


# =============================================================================
# Effect Size Plot Tests
# =============================================================================

class TestEffectSizePlot:
    """Test effect size forest plot generation."""
    
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_basic_effect_size_plot(self):
        """Test basic effect size plot creation."""
        from amyloidbench.visualization.statistical_plots import plot_effect_sizes
        
        comparisons = [
            ('PredA', 'PredB', 0.5, (0.2, 0.8)),    # Significant (CI doesn't include 0)
            ('PredA', 'PredC', 0.2, (-0.1, 0.5)),   # Not significant
            ('PredB', 'PredC', -0.3, (-0.6, 0.0)),  # Borderline
        ]
        
        fig = plot_effect_sizes(comparisons)
        
        assert fig is not None
        plt.close(fig)
    
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_effect_size_single_comparison(self):
        """Test effect size plot with single comparison."""
        from amyloidbench.visualization.statistical_plots import plot_effect_sizes
        
        comparisons = [('A', 'B', 1.0, (0.5, 1.5))]
        
        fig = plot_effect_sizes(comparisons)
        assert fig is not None
        plt.close(fig)


# =============================================================================
# Polymorph Performance Visualization Tests
# =============================================================================

class TestPolymorphVisualization:
    """Test polymorph performance visualization."""
    
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_polymorph_radar_chart(self):
        """Test radar chart for polymorph performance."""
        from amyloidbench.visualization.statistical_plots import plot_polymorph_radar
        
        performances = {
            'PredA': {
                'steric_zipper': 0.85,
                'beta_solenoid': 0.72,
                'beta_arcade': 0.68,
                'greek_key': 0.55,
            },
            'PredB': {
                'steric_zipper': 0.70,
                'beta_solenoid': 0.88,
                'beta_arcade': 0.75,
                'greek_key': 0.60,
            },
        }
        
        fig = plot_polymorph_radar(performances)
        
        assert fig is not None
        plt.close(fig)
    
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_polymorph_bars(self):
        """Test bar chart for polymorph performance."""
        from amyloidbench.visualization.statistical_plots import plot_polymorph_bars
        
        performances = {
            'PredA': {'steric_zipper': 0.8, 'beta_solenoid': 0.7},
            'PredB': {'steric_zipper': 0.6, 'beta_solenoid': 0.9},
        }
        
        fig = plot_polymorph_bars(performances, metric='mcc')
        
        assert fig is not None
        plt.close(fig)
    
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_polymorph_radar_fallback(self):
        """Test radar chart falls back to bars with <3 categories."""
        from amyloidbench.visualization.statistical_plots import plot_polymorph_radar
        
        # Only 2 categories - should fall back to bar chart
        performances = {
            'PredA': {'steric_zipper': 0.8, 'beta_solenoid': 0.7},
        }
        
        fig = plot_polymorph_radar(performances)
        assert fig is not None
        plt.close(fig)


# =============================================================================
# Reference Dataset Summary Tests
# =============================================================================

class TestReferenceDatasetSummary:
    """Test reference dataset performance summary visualization."""
    
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_basic_summary(self):
        """Test basic reference dataset summary plot."""
        from amyloidbench.visualization.statistical_plots import plot_reference_dataset_summary
        
        results = {
            'PredA': {
                'canonical': {'sensitivity': 0.9, 'specificity': 0.8, 'mcc': 0.7, 'auc_roc': 0.85},
                'disease': {'sensitivity': 0.85, 'specificity': 0.75, 'mcc': 0.6, 'auc_roc': 0.80},
            },
            'PredB': {
                'canonical': {'sensitivity': 0.8, 'specificity': 0.85, 'mcc': 0.65, 'auc_roc': 0.82},
                'disease': {'sensitivity': 0.75, 'specificity': 0.80, 'mcc': 0.55, 'auc_roc': 0.78},
            },
        }
        
        fig = plot_reference_dataset_summary(results)
        
        assert fig is not None
        plt.close(fig)
    
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_summary_with_specific_metrics(self):
        """Test summary with specific metrics selection."""
        from amyloidbench.visualization.statistical_plots import plot_reference_dataset_summary
        
        results = {
            'PredA': {
                'canonical': {'sensitivity': 0.9, 'mcc': 0.7},
            },
        }
        
        fig = plot_reference_dataset_summary(
            results,
            metrics=['sensitivity', 'mcc']
        )
        
        assert fig is not None
        plt.close(fig)


# =============================================================================
# Integration with Phase 5 Tests
# =============================================================================

class TestPhase5Integration:
    """Test integration with Phase 5 statistical comparison module."""
    
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_visualize_friedman_result(self):
        """Test visualization of Friedman test result."""
        from amyloidbench.visualization.statistical_plots import visualize_multiple_comparison_result
        
        # Create mock MultipleComparisonResult
        class MockComparison:
            def __init__(self, a, b, p, effect, ci):
                self.predictor_a = a
                self.predictor_b = b
                self.p_value = p
                self.effect_size = effect
                self.confidence_interval = ci
        
        class MockResult:
            test_name = "Friedman"
            overall_statistic = 10.5
            overall_p_value = 0.005
            rankings = [('PredA', 1.2), ('PredB', 2.1), ('PredC', 2.7)]
            critical_difference = 0.8
            pairwise_comparisons = [
                MockComparison('PredA', 'PredB', 0.02, 0.5, (0.2, 0.8)),
                MockComparison('PredA', 'PredC', 0.01, 0.8, (0.4, 1.2)),
                MockComparison('PredB', 'PredC', 0.15, 0.3, (-0.1, 0.7)),
            ]
        
        result = MockResult()
        figures = visualize_multiple_comparison_result(result)
        
        assert 'cd_diagram' in figures
        assert 'pvalue_heatmap' in figures
        assert 'effect_sizes' in figures
        
        for fig in figures.values():
            plt.close(fig)
    
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_visualize_polymorph_benchmark_result(self):
        """Test visualization of polymorph benchmark result."""
        from amyloidbench.visualization.statistical_plots import visualize_polymorph_benchmark_result
        from amyloidbench.benchmark.metrics import ClassificationMetrics
        
        # Create mock PolymorphBenchmarkResult
        class MockResult:
            predictor_name = "TestPredictor"
            overall_metrics = ClassificationMetrics(
                sensitivity=0.8, specificity=0.75, precision=0.78,
                f1_score=0.79, mcc=0.55, accuracy=0.77
            )
            per_fold_metrics = {
                'steric_zipper': ClassificationMetrics(
                    sensitivity=0.85, specificity=0.80, precision=0.82,
                    f1_score=0.83, mcc=0.65, accuracy=0.82
                ),
                'beta_solenoid': ClassificationMetrics(
                    sensitivity=0.70, specificity=0.65, precision=0.68,
                    f1_score=0.69, mcc=0.35, accuracy=0.67
                ),
            }
            per_geometry_metrics = {
                'parallel': ClassificationMetrics(
                    sensitivity=0.82, specificity=0.78, precision=0.80,
                    f1_score=0.81, mcc=0.60, accuracy=0.80
                ),
                'antiparallel': ClassificationMetrics(
                    sensitivity=0.75, specificity=0.70, precision=0.72,
                    f1_score=0.73, mcc=0.45, accuracy=0.72
                ),
            }
            per_fold_counts = {'steric_zipper': 10, 'beta_solenoid': 5}
        
        result = MockResult()
        figures = visualize_polymorph_benchmark_result(result)
        
        assert 'fold_performance' in figures
        assert 'geometry_performance' in figures
        
        for fig in figures.values():
            plt.close(fig)


# =============================================================================
# Quick Plotting Tests
# =============================================================================

class TestQuickPlotting:
    """Test quick plotting convenience functions."""
    
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_quick_comparison_plot(self):
        """Test quick comparison plot."""
        from amyloidbench.visualization.statistical_plots import quick_comparison_plot
        
        predictor_names = ['Aggrescan3D', 'FoldAmyloid', 'WALTZ']
        metric_values = {
            'Sensitivity': [0.85, 0.78, 0.82],
            'Specificity': [0.80, 0.85, 0.77],
            'MCC': [0.65, 0.62, 0.58],
        }
        
        fig = quick_comparison_plot(predictor_names, metric_values)
        
        assert fig is not None
        plt.close(fig)


# =============================================================================
# Color Scheme Tests
# =============================================================================

class TestColorSchemes:
    """Test color scheme definitions."""
    
    def test_significance_colors_defined(self):
        """Test significance color scheme is properly defined."""
        from amyloidbench.visualization.statistical_plots import SIGNIFICANCE_COLORS
        
        assert 'highly_significant' in SIGNIFICANCE_COLORS
        assert 'significant' in SIGNIFICANCE_COLORS
        assert 'marginally_significant' in SIGNIFICANCE_COLORS
        assert 'not_significant' in SIGNIFICANCE_COLORS
    
    def test_polymorph_colors_defined(self):
        """Test polymorph color scheme is properly defined."""
        from amyloidbench.visualization.statistical_plots import POLYMORPH_COLORS
        
        assert 'steric_zipper' in POLYMORPH_COLORS
        assert 'beta_solenoid' in POLYMORPH_COLORS
        assert 'beta_arcade' in POLYMORPH_COLORS
    
    def test_geometry_colors_defined(self):
        """Test geometry color scheme is properly defined."""
        from amyloidbench.visualization.statistical_plots import GEOMETRY_COLORS
        
        assert 'parallel_in_register' in GEOMETRY_COLORS
        assert 'antiparallel' in GEOMETRY_COLORS


# =============================================================================
# Biological Validation Tests
# =============================================================================

class TestBiologicalValidation:
    """Test visualizations with biologically meaningful data."""
    
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_disease_vs_functional_amyloid_comparison(self):
        """Test visualization comparing disease vs functional amyloid performance."""
        from amyloidbench.visualization.statistical_plots import plot_polymorph_bars
        
        # Realistic scenario: predictor performance on different amyloid types
        performances = {
            'WALTZ': {
                'Aβ_steric_zipper': 0.82,
                'Tau_steric_zipper': 0.78,
                'αSyn_greek_key': 0.65,
                'HET-s_beta_solenoid': 0.55,
                'Curli_beta_solenoid': 0.50,
            },
            'Aggrescan3D': {
                'Aβ_steric_zipper': 0.75,
                'Tau_steric_zipper': 0.72,
                'αSyn_greek_key': 0.70,
                'HET-s_beta_solenoid': 0.68,
                'Curli_beta_solenoid': 0.65,
            },
        }
        
        fig = plot_polymorph_bars(
            performances,
            metric='MCC',
            title='Predictor Performance: Disease vs Functional Amyloids'
        )
        
        assert fig is not None
        plt.close(fig)
    
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_cross_beta_geometry_performance(self):
        """Test visualization of performance by cross-β geometry."""
        from amyloidbench.visualization.statistical_plots import plot_polymorph_radar
        
        # Performance varies by β-sheet arrangement
        performances = {
            'FoldAmyloid': {
                'parallel_in_register': 0.85,  # Best for steric zippers
                'antiparallel': 0.78,
                'out_of_register': 0.65,
                'mixed': 0.70,
            },
            'PASTA2': {
                'parallel_in_register': 0.80,
                'antiparallel': 0.82,
                'out_of_register': 0.75,
                'mixed': 0.78,
            },
        }
        
        fig = plot_polymorph_radar(
            performances,
            title='Predictor Performance by β-Sheet Geometry'
        )
        
        assert fig is not None
        plt.close(fig)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_empty_rankings(self):
        """Test CD diagram with empty rankings."""
        from amyloidbench.visualization.statistical_plots import plot_critical_difference_diagram
        
        # Empty list should handle gracefully
        try:
            fig = plot_critical_difference_diagram([], critical_difference=0.5)
            plt.close(fig)
        except (ValueError, IndexError):
            pass  # Expected to fail gracefully
    
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_single_predictor_heatmap(self):
        """Test heatmap with single predictor."""
        from amyloidbench.visualization.statistical_plots import plot_pvalue_heatmap
        
        pvalue_matrix = np.array([[1.0]])
        
        fig = plot_pvalue_heatmap(['Single'], pvalue_matrix)
        assert fig is not None
        plt.close(fig)
    
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
    def test_missing_polymorph_data(self):
        """Test polymorph plot with missing data for some predictors."""
        from amyloidbench.visualization.statistical_plots import plot_polymorph_bars
        
        performances = {
            'PredA': {'steric_zipper': 0.8, 'beta_solenoid': 0.7, 'beta_arcade': 0.6},
            'PredB': {'steric_zipper': 0.7},  # Missing other types
        }
        
        fig = plot_polymorph_bars(performances)
        assert fig is not None
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
