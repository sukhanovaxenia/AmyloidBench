"""
Test suite for Phase 6: Visualization and HTML reports.

Tests verify:
1. Plot generation functions work correctly
2. HTML report generation produces valid output
3. Figure utilities work as expected
4. Integration with prediction results
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from amyloidbench.visualization import (
    # Availability
    HAS_MATPLOTLIB,
    # Styling
    PlotStyle,
    CATEGORICAL_COLORS,
    # Plotting functions
    plot_score_profile,
    plot_multi_predictor_profile,
    plot_agreement_heatmap,
    plot_benchmark_comparison,
    plot_polymorph_probabilities,
    plot_region_diagram,
    create_summary_figure,
    # Utilities
    save_figure,
    figure_to_base64,
    # Reports
    ReportConfig,
    SequenceReportGenerator,
    BenchmarkReportGenerator,
    generate_sequence_report,
    generate_benchmark_report,
)


# Skip all visualization tests if matplotlib not available
pytestmark = pytest.mark.skipif(
    not HAS_MATPLOTLIB,
    reason="matplotlib not installed"
)


class TestPlotStyle:
    """Test plot styling configuration."""
    
    def test_default_style(self):
        """Test default style values."""
        style = PlotStyle()
        assert style.figsize == (10, 4)
        assert style.dpi == 300
        assert style.line_width == 1.5
    
    def test_custom_style(self):
        """Test custom style configuration."""
        style = PlotStyle(figsize=(12, 6), dpi=150)
        assert style.figsize == (12, 6)
        assert style.dpi == 150
    
    def test_apply_style(self):
        """Test style application."""
        style = PlotStyle()
        style.apply()  # Should not raise


class TestColorSchemes:
    """Test color configurations."""
    
    def test_categorical_colors(self):
        """Test categorical color palette."""
        assert len(CATEGORICAL_COLORS) >= 8
        # All should be valid hex colors
        for color in CATEGORICAL_COLORS:
            assert color.startswith('#')
            assert len(color) == 7


class TestScoreProfilePlots:
    """Test score profile plotting."""
    
    def test_plot_score_profile_basic(self):
        """Test basic score profile plot."""
        scores = np.random.rand(50)
        fig = plot_score_profile(scores)
        
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_plot_score_profile_with_sequence(self):
        """Test score profile with sequence annotation."""
        sequence = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"
        scores = np.random.rand(len(sequence))
        
        fig = plot_score_profile(
            scores,
            sequence=sequence,
            predictor_name="TestPredictor"
        )
        
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_plot_score_profile_with_threshold(self):
        """Test score profile with custom threshold."""
        scores = np.random.rand(30)
        
        fig = plot_score_profile(
            scores,
            threshold=0.7,
            highlight_apr=True
        )
        
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_plot_score_profile_with_known_regions(self):
        """Test score profile with known APR regions."""
        scores = np.random.rand(42)
        known_regions = [(16, 21), (30, 40)]
        
        fig = plot_score_profile(
            scores,
            known_regions=known_regions
        )
        
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestMultiPredictorPlots:
    """Test multi-predictor comparison plots."""
    
    def test_plot_multi_predictor_profile(self):
        """Test multi-predictor overlay plot."""
        scores_dict = {
            "Predictor1": np.random.rand(50),
            "Predictor2": np.random.rand(50),
            "Predictor3": np.random.rand(50),
        }
        
        fig = plot_multi_predictor_profile(scores_dict)
        
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_plot_multi_predictor_with_thresholds(self):
        """Test multi-predictor with individual thresholds."""
        scores_dict = {
            "A": np.random.rand(30),
            "B": np.random.rand(30),
        }
        thresholds = {"A": 0.5, "B": 0.6}
        
        fig = plot_multi_predictor_profile(
            scores_dict,
            thresholds=thresholds
        )
        
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestAgreementHeatmap:
    """Test agreement heatmap plotting."""
    
    def test_plot_agreement_heatmap(self):
        """Test basic agreement heatmap."""
        matrix = np.array([
            [1.0, 0.8, 0.6],
            [0.8, 1.0, 0.7],
            [0.6, 0.7, 1.0]
        ])
        names = ["A", "B", "C"]
        
        fig = plot_agreement_heatmap(matrix, names)
        
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_plot_agreement_heatmap_annotated(self):
        """Test annotated agreement heatmap."""
        matrix = np.random.rand(4, 4)
        np.fill_diagonal(matrix, 1.0)
        names = ["P1", "P2", "P3", "P4"]
        
        fig = plot_agreement_heatmap(
            matrix,
            names,
            annotate=True,
            metric_name="Cohen's κ"
        )
        
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestBenchmarkComparison:
    """Test benchmark comparison plotting."""
    
    def test_plot_benchmark_comparison(self):
        """Test benchmark comparison bar chart."""
        results = [
            {"predictor": "A", "sensitivity": 0.8, "specificity": 0.9, "mcc": 0.7},
            {"predictor": "B", "sensitivity": 0.7, "specificity": 0.85, "mcc": 0.6},
            {"predictor": "C", "sensitivity": 0.75, "specificity": 0.88, "mcc": 0.65},
        ]
        
        fig = plot_benchmark_comparison(results)
        
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_plot_benchmark_custom_metrics(self):
        """Test benchmark with custom metrics."""
        results = [
            {"predictor": "X", "auc": 0.85, "f1_score": 0.78},
            {"predictor": "Y", "auc": 0.82, "f1_score": 0.75},
        ]
        
        fig = plot_benchmark_comparison(
            results,
            metrics=["auc", "f1_score"]
        )
        
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestPolymorphPlots:
    """Test polymorph classification plots."""
    
    def test_plot_polymorph_probabilities(self):
        """Test polymorph probability bar chart."""
        fold_probs = {
            "steric_zipper": 0.45,
            "beta_solenoid": 0.25,
            "beta_arcade": 0.20,
            "greek_key": 0.10,
        }
        
        fig = plot_polymorph_probabilities(fold_probs)
        
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_plot_polymorph_with_geometry(self):
        """Test polymorph with geometry probabilities."""
        fold_probs = {
            "steric_zipper": 0.6,
            "beta_arcade": 0.4,
        }
        geometry_probs = {
            "parallel_in_register": 0.7,
            "antiparallel": 0.3,
        }
        
        fig = plot_polymorph_probabilities(
            fold_probs,
            geometry_probs=geometry_probs
        )
        
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestRegionDiagram:
    """Test region diagram plotting."""
    
    def test_plot_region_diagram(self):
        """Test basic region diagram."""
        predicted_regions = [
            (10, 20, "KLVFFA"),
            (30, 40, "VGGVVIA"),
        ]
        
        fig = plot_region_diagram(
            sequence_length=50,
            predicted_regions=predicted_regions
        )
        
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_plot_region_diagram_with_known(self):
        """Test region diagram with known regions."""
        predicted = [(15, 25, "pred1")]
        known = [(16, 22, "known1")]
        
        fig = plot_region_diagram(
            sequence_length=50,
            predicted_regions=predicted,
            known_regions=known
        )
        
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestSummaryFigure:
    """Test summary figure creation."""
    
    def test_create_summary_figure(self):
        """Test multi-panel summary figure."""
        sequence = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"
        scores_dict = {
            "A": np.random.rand(len(sequence)),
            "B": np.random.rand(len(sequence)),
        }
        consensus_regions = [(16, 21), (30, 42)]
        
        fig = create_summary_figure(
            sequence=sequence,
            scores_dict=scores_dict,
            consensus_regions=consensus_regions
        )
        
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_create_summary_with_polymorph(self):
        """Test summary figure with polymorph panel."""
        sequence = "KLVFFAEDVGSNK"
        scores_dict = {"Test": np.random.rand(len(sequence))}
        consensus_regions = [(0, 6)]
        fold_probs = {"steric_zipper": 0.7, "beta_arcade": 0.3}
        
        fig = create_summary_figure(
            sequence=sequence,
            scores_dict=scores_dict,
            consensus_regions=consensus_regions,
            fold_probs=fold_probs
        )
        
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestFigureUtilities:
    """Test figure utility functions."""
    
    def test_save_figure(self):
        """Test figure saving."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = save_figure(fig, f.name)
            assert Path(path).exists()
            assert Path(path).stat().st_size > 0
        
        plt.close(fig)
    
    def test_figure_to_base64(self):
        """Test base64 conversion."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        b64 = figure_to_base64(fig)
        
        assert b64.startswith("data:image/png;base64,")
        assert len(b64) > 100  # Should have substantial content
        
        plt.close(fig)


class TestReportConfig:
    """Test report configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ReportConfig()
        assert config.include_figures == True
        assert config.figure_format == "png"
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ReportConfig(
            title="Custom Report",
            include_figures=False
        )
        assert config.title == "Custom Report"
        assert config.include_figures == False


class TestSequenceReportGenerator:
    """Test sequence report generation."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        gen = SequenceReportGenerator()
        assert gen.config is not None
    
    def test_generate_minimal_report(self):
        """Test minimal report generation."""
        gen = SequenceReportGenerator()
        
        html = gen.generate(
            sequence="KLVFFA",
            sequence_id="test",
            prediction_results={}
        )
        
        assert "<html" in html
        assert "test" in html
        assert "KLVFFA" in html
    
    def test_generate_with_consensus(self):
        """Test report with consensus data."""
        gen = SequenceReportGenerator()
        
        consensus = {
            "is_amyloidogenic": True,
            "probability": 0.85,
            "agreement": "3/3",
            "regions": [
                {"start": 0, "end": 6, "sequence": "KLVFFA", "score": 0.9}
            ]
        }
        
        html = gen.generate(
            sequence="KLVFFAEDVGSNK",
            sequence_id="test_consensus",
            prediction_results={},
            consensus_result=consensus
        )
        
        assert "Amyloidogenic" in html
        assert "85" in html or "0.85" in html
    
    def test_generate_with_polymorph(self):
        """Test report with polymorph data."""
        gen = SequenceReportGenerator()
        
        polymorph = {
            "fold": "steric_zipper",
            "geometry": "parallel_in_register",
            "zipper_class": "class_1",
            "confidence": 0.75,
            "fold_probabilities": {"steric_zipper": 0.6, "beta_arcade": 0.4}
        }
        
        html = gen.generate(
            sequence="GNNQQNY",
            sequence_id="sup35",
            prediction_results={},
            polymorph_result=polymorph
        )
        
        assert "Steric Zipper" in html or "steric_zipper" in html
        assert "75" in html or "0.75" in html
    
    def test_generate_with_features(self):
        """Test report with extracted features."""
        gen = SequenceReportGenerator()
        
        features = {
            "hydrophobicity": 0.65,
            "beta_sheet_propensity": 1.2,
            "net_charge": -2,
        }
        
        html = gen.generate(
            sequence="VQIVYK",
            sequence_id="tau",
            prediction_results={},
            features=features
        )
        
        assert "Features" in html
        assert "hydrophobicity" in html.lower() or "Hydrophobicity" in html
    
    def test_save_report(self):
        """Test saving report to file."""
        gen = SequenceReportGenerator()
        
        html = gen.generate(
            sequence="TEST",
            sequence_id="test",
            prediction_results={}
        )
        
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            path = gen.save(html, f.name)
            assert Path(path).exists()
            
            content = Path(path).read_text()
            assert "<html" in content


class TestBenchmarkReportGenerator:
    """Test benchmark report generation."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        gen = BenchmarkReportGenerator()
        assert gen.config is not None
    
    def test_generate_benchmark_report(self):
        """Test benchmark report generation."""
        gen = BenchmarkReportGenerator()
        
        results = [
            {"predictor": "A", "sensitivity": 0.8, "specificity": 0.9,
             "precision": 0.85, "f1_score": 0.82, "mcc": 0.75, "auc_roc": 0.88},
            {"predictor": "B", "sensitivity": 0.7, "specificity": 0.85,
             "precision": 0.78, "f1_score": 0.74, "mcc": 0.65, "auc_roc": 0.80},
        ]
        
        html = gen.generate(results, "Test Dataset")
        
        assert "<html" in html
        assert "Test Dataset" in html
        assert "Predictor A" in html or "A" in html
    
    def test_generate_with_dataset_info(self):
        """Test benchmark report with dataset info."""
        gen = BenchmarkReportGenerator()
        
        results = [
            {"predictor": "Test", "mcc": 0.7, "sensitivity": 0.8,
             "specificity": 0.85, "precision": 0.82, "f1_score": 0.81, "auc_roc": 0.85}
        ]
        
        dataset_info = {
            "n_samples": 100,
            "n_positive": 40,
            "n_negative": 60,
            "description": "Test benchmark dataset"
        }
        
        html = gen.generate(
            results,
            "TestDB",
            dataset_info=dataset_info
        )
        
        assert "100" in html
        assert "40" in html


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_generate_sequence_report(self):
        """Test generate_sequence_report function."""
        html = generate_sequence_report(
            sequence="KLVFFA",
            sequence_id="test",
            prediction_results={}
        )
        
        assert "<html" in html
    
    def test_generate_sequence_report_to_file(self):
        """Test generate_sequence_report with output path."""
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            html = generate_sequence_report(
                sequence="GNNQQNY",
                sequence_id="sup35",
                prediction_results={},
                output_path=Path(f.name)
            )
            
            assert Path(f.name).exists()
    
    def test_generate_benchmark_report(self):
        """Test generate_benchmark_report function."""
        results = [
            {"predictor": "X", "mcc": 0.8, "sensitivity": 0.85,
             "specificity": 0.90, "precision": 0.87, "f1_score": 0.86, "auc_roc": 0.92}
        ]
        
        html = generate_benchmark_report(results, "TestSet")
        
        assert "<html" in html
        assert "TestSet" in html


class TestIntegrationWithPredictors:
    """Test integration with actual predictor outputs."""
    
    def test_report_with_real_prediction(self):
        """Test report generation with real prediction result."""
        from amyloidbench import predict, ProteinRecord
        from amyloidbench.predictors.base import get_predictor
        
        protein = ProteinRecord(
            id="Abeta_test",
            sequence="DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"
        )
        
        # Get real prediction using individual predictor
        predictor = get_predictor("FallbackPredictor")
        result = predictor.predict(protein)
        
        # Generate report with actual results
        prediction_results = {"FallbackPredictor": result}
        
        html = generate_sequence_report(
            sequence=protein.sequence,
            sequence_id=protein.id,
            prediction_results=prediction_results,
            consensus_result={
                "is_amyloidogenic": result.is_amyloidogenic or False,
                "probability": result.amyloid_probability or 0,
                "agreement": "1/1",
                "regions": [
                    {"start": r.start, "end": r.end, "sequence": r.sequence, "score": r.score or 0}
                    for r in result.predicted_regions
                ]
            }
        )
        
        assert "Abeta_test" in html
        assert "<html" in html


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
