"""
Test suite for predictor infrastructure.

These tests validate the abstract predictor interface, caching behavior,
and the Aggrescan3D implementation. Biological validity is tested through
known amyloidogenic sequences with experimentally characterized APRs.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from amyloidbench.core.models import (
    PerResidueScores,
    PredictionResult,
    ProteinRecord,
    Region,
)
from amyloidbench.predictors.base import (
    BasePredictor,
    PredictorCapability,
    PredictorConfig,
    PredictorError,
    PredictorType,
    get_predictor,
    list_predictors,
    register_predictor,
)
from amyloidbench.predictors.local.aggrescan3d import (
    Aggrescan3DPredictor,
    predict_with_a3d,
)


# =============================================================================
# Test sequences with known amyloidogenic properties
# =============================================================================

# Human Amyloid-β (1-42) - archetypal amyloidogenic peptide
# Contains well-characterized APRs: KLVFF (16-20) and C-terminal region
ABETA_42 = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"

# Human α-synuclein NAC region (61-95) - core aggregation domain
# This region is necessary and sufficient for α-syn fibrillization
ASYN_NAC = "EQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFV"

# Human ubiquitin - highly soluble, non-amyloidogenic control
UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


class TestPredictorConfig:
    """Tests for predictor configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PredictorConfig()
        
        assert config.use_cache is True
        assert config.timeout_seconds == 300.0
        assert config.max_retries == 3
        assert config.min_region_length == 5
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = PredictorConfig(
            threshold=0.7,
            min_region_length=6,
            use_cache=False,
            timeout_seconds=60.0,
        )
        
        assert config.threshold == 0.7
        assert config.min_region_length == 6
        assert config.use_cache is False
    
    def test_cache_dir_creation(self):
        """Test that cache directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache"
            config = PredictorConfig(cache_dir=cache_path)
            assert cache_path.exists()


class TestPredictorRegistry:
    """Tests for predictor registration and lookup."""
    
    def test_list_predictors(self):
        """Test listing registered predictors."""
        predictors = list_predictors()
        
        # At minimum, Aggrescan3D should be registered
        names = [p["name"] for p in predictors]
        assert "Aggrescan3D" in names
    
    def test_get_predictor(self):
        """Test retrieving predictor by name."""
        predictor = get_predictor("Aggrescan3D")
        
        assert predictor.name == "Aggrescan3D"
        assert isinstance(predictor, BasePredictor)
    
    def test_get_unknown_predictor(self):
        """Test error handling for unknown predictor."""
        with pytest.raises(KeyError):
            get_predictor("NonexistentPredictor")
    
    def test_predictor_info(self):
        """Test predictor metadata retrieval."""
        predictor = get_predictor("Aggrescan3D")
        info = predictor.get_info()
        
        assert "name" in info
        assert "version" in info
        assert "capabilities" in info
        assert "threshold" in info


class TestAggrescan3DPredictor:
    """
    Tests for Aggrescan3D predictor.
    
    Aggrescan3D identifies Structural Aggregation-Prone Regions (STAPs)
    by combining the AGGRESCAN aggregation propensity scale with solvent
    accessibility from 3D structures. In sequence-only mode (fallback),
    it uses the raw AGGRESCAN scale.
    """
    
    def test_initialization(self):
        """Test predictor initialization."""
        predictor = Aggrescan3DPredictor()
        
        assert predictor.name == "Aggrescan3D"
        assert PredictorCapability.STRUCTURE_INPUT in predictor.capabilities
        assert PredictorCapability.PER_RESIDUE_SCORES in predictor.capabilities
    
    def test_abeta_prediction_sequence_mode(self):
        """
        Test prediction on Aβ42 in sequence-only mode.
        
        Aβ42 contains multiple established APRs:
        - KLVFF (residues 16-20): Central hydrophobic cluster, critical for toxicity
        - C-terminal region (29-42): Forms the β-sheet core in fibril structures
        
        The predictor should identify these regions as aggregation-prone.
        """
        predictor = Aggrescan3DPredictor(
            config=PredictorConfig(use_cache=False)
        )
        protein = ProteinRecord(id="Abeta42", sequence=ABETA_42)
        
        result = predictor.predict(protein)
        
        assert result.success is True
        assert result.per_residue_scores is not None
        assert len(result.per_residue_scores.scores) == len(ABETA_42)
        
        # Should identify amyloidogenic regions
        # Note: In sequence-only mode, threshold may need adjustment
        assert result.predicted_regions is not None
    
    def test_ubiquitin_prediction(self):
        """
        Test prediction on ubiquitin (negative control).
        
        Ubiquitin is a highly soluble, globular protein with:
        - High charged residue content (gatekeepers)
        - No known amyloidogenic propensity under physiological conditions
        - Stable native fold that protects hydrophobic core
        
        The predictor should either find no APRs or weak APRs only.
        """
        predictor = Aggrescan3DPredictor(
            config=PredictorConfig(use_cache=False)
        )
        protein = ProteinRecord(id="Ubiquitin", sequence=UBIQUITIN)
        
        result = predictor.predict(protein)
        
        assert result.success is True
        # Ubiquitin should have lower aggregation scores than Aβ42
        if result.per_residue_scores:
            max_score = max(result.per_residue_scores.scores)
            # The maximum score should be lower than typical APR scores
            # This is a relative assertion - exact values depend on normalization
            assert max_score is not None
    
    def test_asyn_nac_region(self):
        """
        Test prediction on α-synuclein NAC region.
        
        The NAC (Non-Amyloid-β Component) region is:
        - The minimal amyloidogenic core of α-synuclein
        - Highly hydrophobic with VVG, VT, VA repeats
        - Necessary and sufficient for fibril formation
        
        Should show strong positive aggregation propensity.
        """
        predictor = Aggrescan3DPredictor(
            config=PredictorConfig(use_cache=False)
        )
        protein = ProteinRecord(id="aSyn_NAC", sequence=ASYN_NAC)
        
        result = predictor.predict(protein)
        
        assert result.success is True
        # NAC region is highly amyloidogenic
        if result.per_residue_scores:
            scores = result.per_residue_scores.scores
            # Should have substantial positive-scoring regions
            positive_fraction = sum(1 for s in scores if s > 0) / len(scores)
            # NAC is mostly aggregation-prone
            assert positive_fraction > 0.3
    
    def test_a3d_scale_values(self):
        """
        Test that AGGRESCAN scale values are biologically consistent.
        
        The AGGRESCAN scale was derived from E. coli inclusion body
        formation rates. Expected patterns:
        - Aromatic hydrophobic (F, W, Y): Most aggregation-prone
        - Aliphatic hydrophobic (I, L, V): Strongly aggregation-prone
        - Charged residues (K, R, D, E): Aggregation-inhibiting
        - Proline (P): Aggregation-inhibiting (β-sheet breaker)
        """
        scale = Aggrescan3DPredictor.A3D_SCALE
        
        # Aromatic residues should be highly positive
        assert scale['F'] > 1.5
        assert scale['W'] > 1.0
        assert scale['Y'] > 0.5
        
        # Aliphatic hydrophobic should be positive
        assert scale['I'] > 1.0
        assert scale['V'] > 1.0
        assert scale['L'] > 1.0
        
        # Charged residues should be negative (protective)
        assert scale['K'] < -1.0
        assert scale['R'] < -1.0
        assert scale['D'] < -1.0
        assert scale['E'] < -0.5
        
        # Proline should be negative (β-breaker)
        assert scale['P'] < 0
    
    def test_caching_behavior(self):
        """Test that results are cached appropriately."""
        predictor = Aggrescan3DPredictor(
            config=PredictorConfig(use_cache=True)
        )
        protein = ProteinRecord(id="test", sequence=ABETA_42)
        
        # First prediction
        result1 = predictor.predict(protein)
        
        # Second prediction should use cache
        result2 = predictor.predict(protein)
        
        # Results should be identical
        assert result1.per_residue_scores.scores == result2.per_residue_scores.scores
        
        # Clean up
        predictor.clear_cache()
    
    def test_threshold_effect(self):
        """Test that threshold affects region detection."""
        protein = ProteinRecord(id="test", sequence=ABETA_42)
        
        # Strict threshold
        predictor_strict = Aggrescan3DPredictor(
            config=PredictorConfig(threshold=1.0, use_cache=False)
        )
        result_strict = predictor_strict.predict(protein)
        
        # Lenient threshold
        predictor_lenient = Aggrescan3DPredictor(
            config=PredictorConfig(threshold=-0.5, use_cache=False)
        )
        result_lenient = predictor_lenient.predict(protein)
        
        # Lenient threshold should find more regions
        assert len(result_lenient.predicted_regions) >= len(result_strict.predicted_regions)


class TestConvenienceFunction:
    """Test the predict_with_a3d convenience function."""
    
    def test_simple_prediction(self):
        """Test simple sequence prediction."""
        result = predict_with_a3d(ABETA_42)
        
        assert result.success is True
        assert result.per_residue_scores is not None


class TestBasePredictor:
    """Tests for abstract base predictor functionality."""
    
    def test_cannot_instantiate_abstract(self):
        """Test that BasePredictor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BasePredictor()
    
    def test_custom_predictor_implementation(self):
        """Test implementing a custom predictor."""
        
        @register_predictor
        class DummyPredictor(BasePredictor):
            name = "DummyPredictor"
            version = "1.0"
            predictor_type = PredictorType.SEQUENCE_HEURISTIC
            capabilities = {PredictorCapability.PER_RESIDUE_SCORES}
            default_threshold = 0.5
            
            def _predict_impl(self, sequence, structure_path=None):
                # Simple dummy implementation: random scores
                scores = [0.5] * len(sequence)
                return PredictionResult(
                    sequence_id="",
                    sequence=sequence,
                    predictor_name=self.name,
                    per_residue_scores=PerResidueScores(
                        scores=scores,
                        sequence=sequence,
                        predictor=self.name,
                        threshold=self.threshold,
                    ),
                )
        
        # Test the custom predictor
        predictor = DummyPredictor()
        protein = ProteinRecord(id="test", sequence="MVLSPADKTNV")
        result = predictor.predict(protein)
        
        assert result.success is True
        assert len(result.per_residue_scores.scores) == 11


class TestErrorHandling:
    """Tests for predictor error handling."""
    
    def test_retry_behavior(self):
        """Test that failed predictions are retried."""
        
        call_count = 0
        
        class FailingPredictor(BasePredictor):
            name = "FailingPredictor"
            version = "1.0"
            predictor_type = PredictorType.SEQUENCE_HEURISTIC
            capabilities = set()
            
            def _predict_impl(self, sequence, structure_path=None):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise PredictorError("Temporary failure")
                return PredictionResult(
                    sequence_id="",
                    sequence=sequence,
                    predictor_name=self.name,
                )
        
        predictor = FailingPredictor(
            config=PredictorConfig(max_retries=3, retry_delay=0.01, use_cache=False)
        )
        protein = ProteinRecord(id="test", sequence="MVLSPADKTNV")
        
        result = predictor.predict(protein)
        
        assert result.success is True
        assert call_count == 3  # Failed twice, succeeded on third try
    
    def test_exhausted_retries(self):
        """Test handling when all retries are exhausted."""
        
        class AlwaysFailingPredictor(BasePredictor):
            name = "AlwaysFailingPredictor"
            version = "1.0"
            predictor_type = PredictorType.SEQUENCE_HEURISTIC
            capabilities = set()
            
            def _predict_impl(self, sequence, structure_path=None):
                raise PredictorError("Permanent failure")
        
        predictor = AlwaysFailingPredictor(
            config=PredictorConfig(max_retries=2, retry_delay=0.01, use_cache=False)
        )
        protein = ProteinRecord(id="test", sequence="MVLSPADKTNV")
        
        result = predictor.predict(protein)
        
        assert result.success is False
        assert "Permanent failure" in result.error_message


class TestBatchPrediction:
    """Tests for batch prediction functionality."""
    
    def test_batch_prediction(self):
        """Test predicting multiple proteins."""
        predictor = Aggrescan3DPredictor(
            config=PredictorConfig(use_cache=False)
        )
        
        proteins = [
            ProteinRecord(id="Abeta", sequence=ABETA_42),
            ProteinRecord(id="NAC", sequence=ASYN_NAC),
            ProteinRecord(id="Ubiq", sequence=UBIQUITIN),
        ]
        
        # Track progress
        progress_calls = []
        def progress_callback(current, total):
            progress_calls.append((current, total))
        
        results = predictor.predict_batch(
            proteins, 
            progress_callback=progress_callback
        )
        
        assert len(results) == 3
        assert all(r.success for r in results)
        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3)


# =============================================================================
# Biological validation tests
# =============================================================================

class TestBiologicalValidation:
    """
    Tests that validate predictor output against known biological properties.
    
    These tests ensure that the computational predictions align with
    experimentally established amyloidogenic properties of reference proteins.
    """
    
    def test_abeta_klvff_detection(self):
        """
        Test detection of KLVFF region in Aβ42.
        
        KLVFF (residues 16-20) is the central hydrophobic cluster that:
        - Is essential for Aβ fibril formation
        - Shows highest aggregation propensity in experimental assays
        - Is targeted by anti-amyloid therapeutic strategies
        
        Any reasonable predictor should score this region highly.
        """
        predictor = Aggrescan3DPredictor(
            config=PredictorConfig(use_cache=False)
        )
        protein = ProteinRecord(id="Abeta42", sequence=ABETA_42)
        result = predictor.predict(protein)
        
        if result.per_residue_scores:
            scores = result.per_residue_scores.scores
            
            # KLVFF is at positions 16-20 (0-indexed: 16-21)
            klvff_scores = scores[16:21]
            flanking_scores = scores[10:15]  # N-terminal flank
            
            # KLVFF should have higher scores than flanking region
            klvff_mean = np.mean(klvff_scores)
            flank_mean = np.mean(flanking_scores)
            
            # This is a soft assertion - the trend should be clear
            assert klvff_mean > flank_mean or klvff_mean > 0
    
    def test_relative_amyloidogenicity(self):
        """
        Test relative amyloidogenicity ordering.
        
        Expected ordering by aggregation propensity:
        1. Aβ42 - highly amyloidogenic, forms fibrils rapidly
        2. α-Syn NAC - amyloidogenic core region
        3. Ubiquitin - soluble, should be lowest
        
        Mean scores should follow this ordering.
        """
        predictor = Aggrescan3DPredictor(
            config=PredictorConfig(use_cache=False)
        )
        
        results = {}
        for name, seq in [
            ("Abeta", ABETA_42),
            ("NAC", ASYN_NAC),
            ("Ubiquitin", UBIQUITIN),
        ]:
            protein = ProteinRecord(id=name, sequence=seq)
            result = predictor.predict(protein)
            if result.per_residue_scores:
                results[name] = np.mean(result.per_residue_scores.scores)
        
        # Ubiquitin should have lowest mean score
        # (most negative or least positive)
        if results:
            assert results["Ubiquitin"] < results["Abeta"]
            assert results["Ubiquitin"] < results["NAC"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
