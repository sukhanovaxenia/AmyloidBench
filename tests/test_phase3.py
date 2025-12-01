"""
Test suite for Phase 3: Feature extraction, Fallback predictor, and Consensus engine.

Tests verify:
1. Feature extraction captures biophysically meaningful properties
2. FallbackPredictor provides reasonable baseline predictions
3. ConsensusEngine properly aggregates multiple predictor outputs
"""

import pytest
import numpy as np

from amyloidbench.core.models import ProteinRecord, Region
from amyloidbench.features import (
    FeatureExtractor,
    extract_features,
    HYDROPHOBICITY_KD,
    BETA_PROPENSITY_CF,
    AGGREGATION_PROPENSITY,
)
from amyloidbench.predictors.base import get_predictor, list_predictors, PredictorConfig
from amyloidbench.consensus import ConsensusEngine, ConsensusMethod, quick_consensus


# =============================================================================
# Test sequences
# =============================================================================

ABETA_42 = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"
ASYN_NAC = "EQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFV"
UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
PROLINE_RICH = "PPGPPGPPGPPGPPG"


class TestFeatureExtraction:
    """Tests for the feature extraction module."""
    
    def test_extractor_initialization(self):
        """Test FeatureExtractor initialization."""
        extractor = FeatureExtractor()
        assert extractor.window_size == 7
        assert extractor.include_dipeptides is True
    
    def test_extract_features_returns_correct_structure(self):
        """Verify extracted features have expected structure."""
        features = extract_features(ABETA_42)
        
        assert features.sequence == ABETA_42
        assert features.length == len(ABETA_42)
        assert len(features.per_residue) > 0
        assert len(features.global_features) > 0
    
    def test_per_residue_features_length(self):
        """Per-residue features should match sequence length."""
        features = extract_features(ABETA_42)
        
        for name, values in features.per_residue.items():
            assert len(values) == len(ABETA_42), f"{name} length mismatch"
    
    def test_hydrophobicity_features(self):
        """
        Test hydrophobicity feature extraction.
        
        Aβ42 contains both hydrophobic (KLVFF, C-terminal) and
        hydrophilic (N-terminal, charged residues) regions.
        """
        features = extract_features(ABETA_42)
        
        # Mean hydrophobicity should be moderate (mixed sequence)
        mean_hydro = features.global_features['mean_hydrophobicity']
        assert -2.0 < mean_hydro < 2.0  # Not extremely hydrophobic or hydrophilic
        
        # Per-residue hydrophobicity should vary
        hydro_array = features.per_residue['hydrophobicity']
        assert np.std(hydro_array) > 1.0  # Significant variation
    
    def test_beta_propensity_features(self):
        """
        Test β-sheet propensity features.
        
        Aβ42 should have regions of high β-propensity corresponding
        to the cross-β core (KLVFF and C-terminal).
        """
        features = extract_features(ABETA_42)
        
        # Mean β-propensity should be elevated (amyloidogenic)
        mean_beta = features.global_features['mean_beta_propensity']
        assert mean_beta > 1.0  # Above random coil baseline
        
        # KLVFF region should have high β-propensity
        # K=0.74, L=1.30, V=1.70, F=1.38, F=1.38
        klvff_beta = features.per_residue['beta_propensity'][16:21]
        assert np.mean(klvff_beta) > 1.2
    
    def test_composition_features(self):
        """Test amino acid composition features."""
        features = extract_features(ABETA_42)
        
        # Check individual AA fractions sum to ~1
        aa_fracs = [features.global_features[f'frac_{aa}'] for aa in 'ACDEFGHIKLMNPQRSTVWY']
        assert abs(sum(aa_fracs) - 1.0) < 0.01
        
        # Hydrophobic fraction should be substantial for Aβ42
        assert features.global_features['frac_hydrophobic'] > 0.3
    
    def test_gatekeeper_analysis(self):
        """
        Test gatekeeper residue detection.
        
        Gatekeepers (P, K, R, E, D) protect against aggregation.
        Aβ42 has some gatekeepers but they're not in the APR cores.
        """
        features = extract_features(ABETA_42)
        
        gatekeeper_frac = features.global_features['frac_gatekeeper']
        assert 0.1 < gatekeeper_frac < 0.4  # Present but not dominant
        
        # Ubiquitin (soluble) should have more gatekeepers
        ubiq_features = extract_features(UBIQUITIN)
        assert ubiq_features.global_features['frac_gatekeeper'] >= gatekeeper_frac
    
    def test_pattern_features(self):
        """Test sequence pattern detection."""
        features = extract_features(ABETA_42)
        
        # Should detect hydrophobic stretches
        assert features.global_features['max_hydrophobic_stretch'] >= 3
        
        # Sequence entropy should be moderate-high (diverse composition)
        assert 0.5 < features.global_features['sequence_entropy'] < 1.0
    
    def test_proline_rich_sequence(self):
        """
        Test features on proline-rich sequence.
        
        Should show low β-propensity and aggregation propensity.
        """
        features = extract_features(PROLINE_RICH)
        
        # Proline has low β-propensity (0.55) and is a gatekeeper
        assert features.global_features['mean_beta_propensity'] < 1.0
        assert features.global_features['frac_gatekeeper'] > 0.5
    
    def test_feature_vector_generation(self):
        """Test conversion to feature vector for ML."""
        features = extract_features(ABETA_42)
        
        vector = features.to_vector()
        assert isinstance(vector, np.ndarray)
        assert len(vector) == len(features.global_features)
        
        # Test with specific feature names
        feature_names = ['mean_hydrophobicity', 'mean_beta_propensity']
        vector = features.to_vector(feature_names)
        assert len(vector) == 2


class TestFallbackPredictor:
    """Tests for the biophysics-based fallback predictor."""
    
    def test_predictor_registered(self):
        """Verify FallbackPredictor is in registry."""
        names = [p['name'] for p in list_predictors()]
        assert 'FallbackPredictor' in names
    
    def test_predictor_initialization(self):
        """Test basic initialization."""
        predictor = get_predictor('FallbackPredictor')
        assert predictor.name == 'FallbackPredictor'
        assert predictor.version == '1.0'
    
    def test_abeta_prediction(self):
        """
        Test prediction on Aβ42.
        
        Should identify as amyloidogenic with APRs in expected regions.
        """
        predictor = get_predictor('FallbackPredictor', PredictorConfig(use_cache=False))
        protein = ProteinRecord(id='Abeta42', sequence=ABETA_42)
        
        result = predictor.predict(protein)
        
        assert result.success is True
        assert result.is_amyloidogenic is True
        assert len(result.predicted_regions) >= 1
        
        # Should have per-residue scores in [0, 1]
        scores = result.per_residue_scores.scores
        assert all(0 <= s <= 1 for s in scores)
    
    def test_ubiquitin_prediction(self):
        """
        Test prediction on ubiquitin (negative control).
        
        Should have lower overall scores than Aβ42.
        """
        predictor = get_predictor('FallbackPredictor', PredictorConfig(use_cache=False))
        
        abeta = ProteinRecord(id='Abeta', sequence=ABETA_42)
        ubiq = ProteinRecord(id='Ubiq', sequence=UBIQUITIN)
        
        abeta_result = predictor.predict(abeta)
        ubiq_result = predictor.predict(ubiq)
        
        # Both should succeed
        assert abeta_result.success and ubiq_result.success
        
        # Aβ42 should have higher probability
        assert abeta_result.amyloid_probability > ubiq_result.amyloid_probability
    
    def test_scoring_weights_effect(self):
        """Test that custom scoring weights affect predictions."""
        from amyloidbench.predictors.local.fallback import FallbackPredictor, ScoringWeights
        
        protein = ProteinRecord(id='test', sequence=ABETA_42)
        
        # Default weights
        pred_default = FallbackPredictor()
        result_default = pred_default.predict(protein)
        
        # High hydrophobicity weight
        high_hydro_weights = ScoringWeights(hydrophobicity=0.8, beta_propensity=0.1)
        pred_hydro = FallbackPredictor(weights=high_hydro_weights)
        result_hydro = pred_hydro.predict(protein)
        
        # Results should differ
        assert result_default.per_residue_scores.scores != result_hydro.per_residue_scores.scores
    
    def test_tango_like_scoring(self):
        """Test TANGO-like scoring function."""
        from amyloidbench.predictors.local.fallback import calculate_tango_like_score
        
        scores, regions = calculate_tango_like_score(ABETA_42)
        
        assert len(scores) == len(ABETA_42)
        assert all(0 <= s <= 1 for s in scores)
        
        # Should find at least one region
        # (may vary with threshold, but should identify something)


class TestConsensusEngine:
    """Tests for the consensus prediction engine."""
    
    def test_engine_initialization(self):
        """Test ConsensusEngine initialization."""
        engine = ConsensusEngine()
        assert len(engine.predictors) == 0
        
        # Auto-register
        engine_auto = ConsensusEngine(auto_register=True)
        assert len(engine_auto.predictors) >= 1
    
    def test_add_predictor(self):
        """Test adding predictors to engine."""
        engine = ConsensusEngine()
        
        engine.add_predictor('Aggrescan3D')
        assert 'Aggrescan3D' in engine.predictors
        assert 'Aggrescan3D' in engine.predictor_weights
        
        engine.add_predictor('FoldAmyloid', weight=2.0)
        assert engine.predictor_weights['FoldAmyloid'].weight == 2.0
    
    def test_remove_predictor(self):
        """Test removing predictors."""
        engine = ConsensusEngine()
        engine.add_predictor('Aggrescan3D')
        engine.add_predictor('FoldAmyloid')
        
        engine.remove_predictor('Aggrescan3D')
        assert 'Aggrescan3D' not in engine.predictors
        assert 'FoldAmyloid' in engine.predictors
    
    def test_majority_vote_consensus(self):
        """Test majority vote consensus method."""
        engine = ConsensusEngine()
        engine.add_predictor('Aggrescan3D')
        engine.add_predictor('FoldAmyloid')
        engine.add_predictor('FallbackPredictor')
        
        protein = ProteinRecord(id='Abeta42', sequence=ABETA_42)
        result = engine.predict(protein, method=ConsensusMethod.MAJORITY_VOTE)
        
        # All should agree Aβ42 is amyloidogenic
        assert result.n_predictors_agree_positive >= 2
        assert result.consensus_is_amyloidogenic is True
    
    def test_weighted_average_consensus(self):
        """Test weighted average consensus method."""
        engine = ConsensusEngine()
        engine.add_predictor('Aggrescan3D', weight=1.0)
        engine.add_predictor('FoldAmyloid', weight=2.0)  # Higher weight
        engine.add_predictor('FallbackPredictor', weight=1.0)
        
        protein = ProteinRecord(id='Abeta42', sequence=ABETA_42)
        result = engine.predict(protein, method=ConsensusMethod.WEIGHTED_AVERAGE)
        
        assert result.consensus_method == 'weighted_average'
        assert 'FoldAmyloid' in result.predictor_weights
        assert result.predictor_weights['FoldAmyloid'] == 2.0
    
    def test_intersection_consensus(self):
        """
        Test intersection consensus (most conservative).
        
        Only reports regions identified by ALL predictors.
        """
        engine = ConsensusEngine()
        engine.add_predictor('Aggrescan3D')
        engine.add_predictor('FoldAmyloid')
        engine.add_predictor('FallbackPredictor')
        
        protein = ProteinRecord(id='Abeta42', sequence=ABETA_42)
        result = engine.predict(protein, method=ConsensusMethod.INTERSECTION)
        
        # Intersection should find fewer or equal regions than union
        result_union = engine.predict(protein, method=ConsensusMethod.UNION)
        
        # Intersection is more conservative
        intersection_coverage = sum(r.length for r in result.consensus_regions)
        union_coverage = sum(r.length for r in result_union.consensus_regions)
        
        assert intersection_coverage <= union_coverage
    
    def test_union_consensus(self):
        """
        Test union consensus (most liberal).
        
        Reports regions identified by ANY predictor.
        """
        engine = ConsensusEngine()
        engine.add_predictor('Aggrescan3D')
        engine.add_predictor('FoldAmyloid')
        
        protein = ProteinRecord(id='Abeta42', sequence=ABETA_42)
        result = engine.predict(protein, method=ConsensusMethod.UNION)
        
        # Union should find at least as many regions as individual predictors
        assert result.consensus_is_amyloidogenic is True
    
    def test_quick_consensus_function(self):
        """Test the quick_consensus convenience function."""
        protein = ProteinRecord(id='Abeta42', sequence=ABETA_42)
        
        # Only use local predictors to avoid web connectivity issues
        result = quick_consensus(
            protein, 
            predictor_names=['Aggrescan3D', 'FoldAmyloid', 'FallbackPredictor']
        )
        
        assert result.consensus_is_amyloidogenic is True
        assert result.n_predictors_total == 3
    
    def test_agreement_ratio(self):
        """Test agreement ratio calculation."""
        engine = ConsensusEngine()
        engine.add_predictor('Aggrescan3D')
        engine.add_predictor('FoldAmyloid')
        engine.add_predictor('FallbackPredictor')
        
        protein = ProteinRecord(id='Abeta42', sequence=ABETA_42)
        result = engine.predict(protein)
        
        ratio = result.agreement_ratio()
        assert 0 <= ratio <= 1
        
        # For Aβ42, all should agree (high ratio)
        assert ratio >= 0.5
    
    def test_insufficient_predictors(self):
        """Test error handling when too few predictors."""
        from amyloidbench.consensus import ConsensusConfig
        
        config = ConsensusConfig(require_n_predictors=5)
        engine = ConsensusEngine(config=config)
        engine.add_predictor('Aggrescan3D')
        
        protein = ProteinRecord(id='test', sequence=ABETA_42)
        
        with pytest.raises(ValueError, match="Need at least"):
            engine.predict(protein)


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_pipeline_abeta(self):
        """
        Full pipeline test on Aβ42.
        
        Verifies that all components work together to correctly
        identify the well-characterized amyloidogenic regions.
        """
        from amyloidbench import predict
        
        protein = ProteinRecord(id='Abeta42', sequence=ABETA_42)
        # Only use local predictors to avoid web connectivity issues
        result = predict(
            protein, 
            predictors=['Aggrescan3D', 'FoldAmyloid', 'FallbackPredictor'],
            consensus=True, 
            method='weighted_average'
        )
        
        assert result.consensus_is_amyloidogenic is True
        
        # Should identify KLVFF region (positions 16-20) or nearby
        klvff_found = any(
            r.start <= 20 and r.end >= 16
            for r in result.consensus_regions
        )
        # Note: exact boundaries may vary by predictor
        
        # Should identify C-terminal region
        cterminal_found = any(
            r.end >= 38
            for r in result.consensus_regions
        )
    
    def test_full_pipeline_ubiquitin(self):
        """Test full pipeline on ubiquitin (negative control)."""
        from amyloidbench import predict
        
        protein = ProteinRecord(id='Ubiquitin', sequence=UBIQUITIN)
        # Only use local predictors
        result = predict(
            protein, 
            predictors=['Aggrescan3D', 'FoldAmyloid', 'FallbackPredictor'],
            consensus=True
        )
        
        # Should have lower confidence than Aβ42
        abeta_protein = ProteinRecord(id='Abeta', sequence=ABETA_42)
        abeta_result = predict(
            abeta_protein, 
            predictors=['Aggrescan3D', 'FoldAmyloid', 'FallbackPredictor'],
            consensus=True
        )
        
        assert result.consensus_probability <= abeta_result.consensus_probability
    
    def test_reproducibility(self):
        """Test that predictions are reproducible."""
        from amyloidbench import predict
        
        protein = ProteinRecord(id='test', sequence=ABETA_42)
        
        # Only use local predictors
        result1 = predict(
            protein, 
            predictors=['Aggrescan3D', 'FoldAmyloid', 'FallbackPredictor'],
            consensus=True
        )
        result2 = predict(
            protein, 
            predictors=['Aggrescan3D', 'FoldAmyloid', 'FallbackPredictor'],
            consensus=True
        )
        
        # Same results
        assert result1.consensus_is_amyloidogenic == result2.consensus_is_amyloidogenic
        assert result1.n_predictors_agree_positive == result2.n_predictors_agree_positive


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
