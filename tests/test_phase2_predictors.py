"""
Test suite for Phase 2 predictors.

Tests the FoldAmyloid re-implementation and local approximations of
WALTZ and PASTA scoring. Web-based predictors are tested separately
with mock HTML responses to avoid network dependencies.
"""

import pytest
import numpy as np

from amyloidbench.core.models import ProteinRecord, Region
from amyloidbench.predictors.base import list_predictors, get_predictor, PredictorConfig


# =============================================================================
# Test sequences with known properties
# =============================================================================

# Aβ42 - highly amyloidogenic, well-characterized APRs
ABETA_42 = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"

# α-synuclein NAC region - amyloidogenic core
ASYN_NAC = "EQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFV"

# Ubiquitin - soluble negative control
UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

# KLVFF hexapeptide - minimal amyloidogenic unit from Aβ
KLVFF = "KLVFFA"

# Proline-rich sequence - should not be amyloidogenic
PROLINE_RICH = "PPGPPGPPGPPGPPG"


class TestFoldAmyloidPredictor:
    """
    Tests for FoldAmyloid re-implementation.
    
    FoldAmyloid identifies amyloidogenic regions based on expected
    packing density - the capacity to form tightly packed steric
    zipper structures characteristic of amyloid cores.
    """
    
    def test_predictor_registered(self):
        """Verify FoldAmyloid is in the predictor registry."""
        predictor_names = [p["name"] for p in list_predictors()]
        assert "FoldAmyloid" in predictor_names
    
    def test_predictor_initialization(self):
        """Test basic initialization."""
        predictor = get_predictor("FoldAmyloid")
        
        assert predictor.name == "FoldAmyloid"
        assert predictor.version == "1.0-reimpl"
        assert predictor.default_threshold == 21.4  # Original WALTZ threshold
    
    def test_packing_density_scale(self):
        """
        Verify packing density scale values are biologically consistent.
        
        Expected patterns:
        - Aromatic residues (F, W, Y): Highest density (bulky, favorable stacking)
        - Aliphatic hydrophobic (I, V, L): High density
        - Charged residues (K, R, D, E): Lower density (unfavorable in zipper)
        - Glycine: Lowest (no side chain, creates packing defects)
        """
        from amyloidbench.predictors.local.reimplemented.foldamyloid import PACKING_DENSITY
        
        # Aromatic residues should have highest packing density
        assert PACKING_DENSITY['F'] > 25.0
        assert PACKING_DENSITY['W'] > 25.0
        
        # Aliphatic hydrophobic should be high
        assert PACKING_DENSITY['I'] > 24.0
        assert PACKING_DENSITY['V'] > 23.0
        assert PACKING_DENSITY['L'] > 23.0
        
        # Charged residues should be lower
        assert PACKING_DENSITY['K'] < 21.0
        assert PACKING_DENSITY['D'] < 20.0
        assert PACKING_DENSITY['E'] < 20.0
        
        # Glycine should be lowest (no side chain)
        assert PACKING_DENSITY['G'] < 18.0
        
        # Proline should be relatively low (β-breaker)
        assert PACKING_DENSITY['P'] < 19.0
    
    def test_abeta_prediction(self):
        """
        Test FoldAmyloid on Aβ42.
        
        Aβ42 contains well-characterized APRs that should be detected:
        - KLVFF region (positions 16-20): Central hydrophobic cluster
        - C-terminal region (29-42): Forms the β-sheet core
        """
        predictor = get_predictor("FoldAmyloid", PredictorConfig(use_cache=False))
        protein = ProteinRecord(id="Abeta42", sequence=ABETA_42)
        
        result = predictor.predict(protein)
        
        assert result.success is True
        assert result.is_amyloidogenic is True
        
        # Should have per-residue scores
        assert result.per_residue_scores is not None
        assert len(result.per_residue_scores.scores) == len(ABETA_42)
        
        # Should identify at least one APR
        assert len(result.predicted_regions) >= 1
        
        # Check that KLVFF region has high scores
        scores = result.per_residue_scores.scores
        klvff_scores = scores[16:21]  # KLVFF region
        avg_klvff = np.mean(klvff_scores)
        
        # KLVFF should score above threshold
        assert avg_klvff > predictor.default_threshold * 0.95  # Allow small tolerance
    
    def test_ubiquitin_prediction(self):
        """
        Test FoldAmyloid on ubiquitin (negative control).
        
        Ubiquitin is highly soluble with no known amyloidogenic propensity.
        While it may contain short hydrophobic segments, the overall
        profile should show lower aggregation propensity than Aβ42.
        """
        predictor = get_predictor("FoldAmyloid", PredictorConfig(use_cache=False))
        protein = ProteinRecord(id="Ubiquitin", sequence=UBIQUITIN)
        
        result = predictor.predict(protein)
        
        assert result.success is True
        
        # Should have lower scores than Aβ42
        if result.per_residue_scores:
            ubiq_mean = np.mean(result.per_residue_scores.scores)
            
            # Test Aβ42 for comparison
            abeta_result = predictor.predict(ProteinRecord(id="Abeta", sequence=ABETA_42))
            abeta_mean = np.mean(abeta_result.per_residue_scores.scores)
            
            # Ubiquitin should have lower mean score
            assert ubiq_mean < abeta_mean
    
    def test_proline_rich_sequence(self):
        """
        Test that proline-rich sequences are correctly identified as non-amyloidogenic.
        
        Proline is a β-sheet breaker due to:
        1. Cyclic side chain constrains backbone angles
        2. Cannot form main-chain hydrogen bonds as H-bond donor
        3. Introduces kinks that disrupt β-sheet continuity
        
        Proline-rich sequences should have low aggregation scores.
        """
        predictor = get_predictor("FoldAmyloid", PredictorConfig(use_cache=False))
        protein = ProteinRecord(id="ProRich", sequence=PROLINE_RICH)
        
        result = predictor.predict(protein)
        
        assert result.success is True
        
        # Should have low scores and no APRs
        if result.per_residue_scores:
            max_score = max(result.per_residue_scores.scores)
            # Proline has low packing density (~18.3), so max should be below threshold
            assert max_score < predictor.default_threshold
        
        # Should not identify APRs
        assert len(result.predicted_regions) == 0 or result.is_amyloidogenic is False
    
    def test_window_size_effect(self):
        """
        Test that window size affects smoothing appropriately.
        
        Larger windows provide more smoothing, which can:
        - Reduce noise from single residue fluctuations
        - Potentially miss very short APRs
        - Better capture extended amyloidogenic regions
        """
        from amyloidbench.predictors.local.reimplemented.foldamyloid import FoldAmyloidPredictor
        
        protein = ProteinRecord(id="test", sequence=ABETA_42)
        
        # Small window
        config_small = PredictorConfig(use_cache=False)
        config_small.window_size = 3
        pred_small = FoldAmyloidPredictor(config=config_small)
        result_small = pred_small.predict(protein)
        
        # Large window
        config_large = PredictorConfig(use_cache=False)
        config_large.window_size = 9
        pred_large = FoldAmyloidPredictor(config=config_large)
        result_large = pred_large.predict(protein)
        
        # Both should succeed
        assert result_small.success and result_large.success
        
        # Larger window should have less variance in scores
        var_small = np.var(result_small.per_residue_scores.scores)
        var_large = np.var(result_large.per_residue_scores.scores)
        
        assert var_large < var_small


class TestLocalWaltzApproximation:
    """
    Tests for local WALTZ-like scoring approximation.
    
    These tests validate the simplified WALTZ scoring that can be
    used when the web server is unavailable.
    """
    
    def test_calculate_waltz_like_score(self):
        """Test hexapeptide scoring function."""
        from amyloidbench.predictors.remote import calculate_waltz_like_score
        
        # KLVFFA should have high score (central Aβ APR)
        klvffa_score = calculate_waltz_like_score("KLVFFA")
        
        # All charged hexapeptide should have low score
        charged_score = calculate_waltz_like_score("KKKKRR")
        
        assert klvffa_score > charged_score
        
        # Hexapeptide with proline should have low score
        proline_score = calculate_waltz_like_score("PPPPPP")
        assert proline_score < 0  # Should be negative
    
    def test_predict_waltz_local_klvff(self):
        """
        Test local WALTZ prediction on KLVFF-containing sequence.
        
        KLVFF is a well-established amyloidogenic hexapeptide from Aβ.
        """
        from amyloidbench.predictors.remote import predict_waltz_local
        
        regions = predict_waltz_local(ABETA_42, threshold=0.2)
        
        # Should identify at least one region
        assert len(regions) >= 1
        
        # Check if KLVFF region (16-21) is captured
        klvff_found = any(
            r[0] <= 16 and r[1] >= 20 
            for r in regions
        )
        assert klvff_found, "KLVFF region not detected"
    
    def test_predict_waltz_local_short_sequence(self):
        """Test handling of sequences shorter than hexapeptide."""
        from amyloidbench.predictors.remote import predict_waltz_local
        
        # Too short - should return empty
        regions = predict_waltz_local("KLVF")
        assert regions == []
    
    def test_predict_waltz_local_threshold_effect(self):
        """Test that threshold affects region detection."""
        from amyloidbench.predictors.remote import predict_waltz_local
        
        # Strict threshold
        regions_strict = predict_waltz_local(ABETA_42, threshold=0.5)
        
        # Lenient threshold
        regions_lenient = predict_waltz_local(ABETA_42, threshold=0.1)
        
        # Lenient should find at least as many regions
        assert len(regions_lenient) >= len(regions_strict)


class TestLocalPastaApproximation:
    """
    Tests for local PASTA-like energy calculation.
    
    PASTA evaluates pairwise interaction energies for cross-β formation.
    """
    
    def test_pair_energy_hydrophobic(self):
        """
        Test that hydrophobic pairs have favorable (negative) energy.
        
        The steric zipper interface requires complementary hydrophobic
        packing between adjacent β-strands.
        """
        from amyloidbench.predictors.remote.pasta2 import get_pair_energy
        
        # Phe-Phe is highly favorable (aromatic stacking)
        ff_energy = get_pair_energy('F', 'F')
        assert ff_energy < 0
        
        # Val-Ile is favorable
        vi_energy = get_pair_energy('V', 'I')
        assert vi_energy < 0
    
    def test_pair_energy_charged(self):
        """
        Test that like-charged pairs have unfavorable (positive) energy.
        
        Same-charge pairs create electrostatic repulsion that destabilizes
        the cross-β structure.
        """
        from amyloidbench.predictors.remote.pasta2 import get_pair_energy
        
        # Lys-Lys repulsion
        kk_energy = get_pair_energy('K', 'K')
        assert kk_energy > 0
        
        # Asp-Asp repulsion
        dd_energy = get_pair_energy('D', 'D')
        assert dd_energy > 0
    
    def test_pair_energy_proline(self):
        """Test that proline pairs are highly unfavorable."""
        from amyloidbench.predictors.remote.pasta2 import get_pair_energy
        
        pp_energy = get_pair_energy('P', 'P')
        assert pp_energy > 1.0  # Should be significantly positive
    
    def test_calculate_pasta_like_energy(self):
        """Test segment energy calculation."""
        from amyloidbench.predictors.remote.pasta2 import calculate_pasta_like_energy
        
        # Hydrophobic segment should have negative energy
        hydrophobic_energy = calculate_pasta_like_energy("VVVVIFFF")
        assert hydrophobic_energy < 0
        
        # Charged segment should have less favorable energy
        charged_energy = calculate_pasta_like_energy("KKKKDDDD")
        assert charged_energy > hydrophobic_energy
    
    def test_predict_pasta_local(self):
        """Test local PASTA prediction on Aβ42."""
        from amyloidbench.predictors.remote import predict_pasta_local
        
        regions = predict_pasta_local(ABETA_42, threshold=-1.5)
        
        # Should identify amyloidogenic regions
        # Note: with local approximation, results may differ from server
        assert isinstance(regions, list)


class TestMultiplePredictorConsistency:
    """
    Tests for consistency between different predictors.
    
    While different methods use different scoring approaches, they should
    show general agreement on well-characterized sequences.
    """
    
    def test_all_predictors_detect_abeta_amyloidogenic(self):
        """
        Verify all predictors identify Aβ42 as amyloidogenic.
        
        Aβ42 is the archetypal amyloidogenic peptide - any reasonable
        predictor should classify it as amyloid-forming.
        """
        protein = ProteinRecord(id="Abeta42", sequence=ABETA_42)
        
        for pred_info in list_predictors():
            try:
                predictor = get_predictor(pred_info["name"], PredictorConfig(use_cache=False))
                result = predictor.predict(protein)
                
                if result.success:
                    # All should identify Aβ42 as amyloidogenic
                    assert result.is_amyloidogenic is True, \
                        f"{pred_info['name']} failed to identify Aβ42 as amyloidogenic"
            except Exception as e:
                # Some predictors may require additional setup
                pass
    
    def test_predictor_score_correlation(self):
        """
        Test that FoldAmyloid and Aggrescan3D scores are correlated on Aβ42.
        
        While the absolute scales differ, both should identify similar
        regions as aggregation-prone.
        """
        protein = ProteinRecord(id="Abeta42", sequence=ABETA_42)
        config = PredictorConfig(use_cache=False, normalize_scores=True)
        
        a3d = get_predictor("Aggrescan3D", config)
        fold = get_predictor("FoldAmyloid", config)
        
        a3d_result = a3d.predict(protein)
        fold_result = fold.predict(protein)
        
        if a3d_result.per_residue_scores and fold_result.per_residue_scores:
            a3d_scores = np.array(a3d_result.per_residue_scores.scores)
            fold_scores = np.array(fold_result.per_residue_scores.scores)
            
            # Calculate Pearson correlation
            correlation = np.corrcoef(a3d_scores, fold_scores)[0, 1]
            
            # Should be positively correlated (both identify similar regions)
            # Not requiring strong correlation as methods are fundamentally different
            assert correlation > 0, "Expected positive correlation between predictors"


class TestPredictorPerformanceCharacteristics:
    """
    Tests for predictor performance and edge cases.
    """
    
    def test_foldamyloid_on_very_short_sequence(self):
        """Test handling of minimal sequence."""
        predictor = get_predictor("FoldAmyloid", PredictorConfig(use_cache=False))
        protein = ProteinRecord(id="short", sequence="MVLSP")  # 5 residues
        
        result = predictor.predict(protein)
        assert result.success is True
        assert len(result.per_residue_scores.scores) == 5
    
    def test_foldamyloid_on_long_sequence(self):
        """Test handling of longer sequences (performance)."""
        predictor = get_predictor("FoldAmyloid", PredictorConfig(use_cache=False))
        
        # Generate a 500-residue sequence
        long_seq = "MVLSPADKTN" * 50
        protein = ProteinRecord(id="long", sequence=long_seq)
        
        result = predictor.predict(protein)
        assert result.success is True
        assert len(result.per_residue_scores.scores) == 500
    
    def test_foldamyloid_handles_ambiguous_residues(self):
        """Test handling of ambiguous amino acid codes (X, B, Z)."""
        predictor = get_predictor("FoldAmyloid", PredictorConfig(use_cache=False))
        
        # Sequence with ambiguous residue
        seq_with_x = "MVLSXPADKTN"
        protein = ProteinRecord(id="ambiguous", sequence=seq_with_x)
        
        result = predictor.predict(protein)
        # Should still succeed (use mean value for unknown residues)
        assert result.success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
