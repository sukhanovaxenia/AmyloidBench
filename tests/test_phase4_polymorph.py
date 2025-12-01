"""
Test suite for Phase 4: Polymorph classification.

Tests verify:
1. Correct classification of known amyloid structures
2. Feature extraction for polymorph prediction
3. Integration with prediction pipeline
4. Confidence calibration
"""

import pytest
import numpy as np

from amyloidbench.core.models import ProteinRecord, Region
from amyloidbench.classification import (
    PolymorphClassifier,
    predict_polymorph,
    StericZipperClass,
    CrossBetaGeometry,
    AmyloidFold,
    PolymorphPrediction,
    find_similar_structures,
    KNOWN_AMYLOID_STRUCTURES,
)


# =============================================================================
# Test sequences with known structures
# =============================================================================

# Short steric zipper peptides
GNNQQNY = "GNNQQNY"  # Sup35 prion, Class 1 zipper (PDB: 1YJO)
NNQQNY = "NNQQNY"    # Sup35 variant
KLVFFA = "KLVFFA"    # Aβ core (steric zipper in crystal)
VQIVYK = "VQIVYK"    # Tau PHF6 (steric zipper, PDB: 2ON9)
NFGAIL = "NFGAIL"    # IAPP amyloid core

# Longer fibril-forming sequences  
ABETA_42 = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"
ASYN_NAC = "EQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFV"
TAU_PHF = "VQIINKKTTTDTTDSNQGFRSSSTGSIDMVDSPQLA"

# β-solenoid forming sequences
HET_S = "VIDAKLKATGANGQTNIGAKIGSNSVGWATGAATAIATALQ"
CURLI_REPEAT = "SELNIYQYGGGNSALALQTDARNSDLTITQHGGGNGADVGQGS"

# Negative controls
UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
PROLINE_RICH = "PPGPPGPPGPPGPPG"


class TestPolymorphEnums:
    """Test polymorph classification enums."""
    
    def test_steric_zipper_classes(self):
        """Test StericZipperClass properties."""
        # Class 1: parallel, parallel, face-to-face
        assert StericZipperClass.CLASS_1.strand_orientation == "parallel"
        assert StericZipperClass.CLASS_1.sheet_packing == "parallel"
        assert StericZipperClass.CLASS_1.face_orientation == "face_to_face"
        
        # Class 5: antiparallel strands
        assert StericZipperClass.CLASS_5.strand_orientation == "antiparallel"
        
        # Class 4: face-to-back
        assert StericZipperClass.CLASS_4.face_orientation == "face_to_back"
    
    def test_cross_beta_geometries(self):
        """Test CrossBetaGeometry enum values."""
        geometries = list(CrossBetaGeometry)
        assert CrossBetaGeometry.PARALLEL_IN_REGISTER in geometries
        assert CrossBetaGeometry.ANTIPARALLEL in geometries
    
    def test_amyloid_folds(self):
        """Test AmyloidFold enum values."""
        folds = list(AmyloidFold)
        assert AmyloidFold.STERIC_ZIPPER in folds
        assert AmyloidFold.BETA_SOLENOID in folds
        assert AmyloidFold.BETA_ARCADE in folds


class TestPolymorphClassifier:
    """Test the main polymorph classifier."""
    
    @pytest.fixture
    def classifier(self):
        return PolymorphClassifier()
    
    def test_classifier_initialization(self, classifier):
        """Test classifier initializes correctly."""
        assert classifier.feature_extractor is not None
    
    def test_prediction_returns_correct_type(self, classifier):
        """Test predict returns PolymorphPrediction."""
        result = classifier.predict(GNNQQNY)
        assert isinstance(result, PolymorphPrediction)
    
    def test_prediction_has_all_fields(self, classifier):
        """Test prediction has all required fields."""
        result = classifier.predict(GNNQQNY)
        
        assert result.sequence == GNNQQNY
        assert result.predicted_fold is not None
        assert isinstance(result.fold_probabilities, dict)
        assert result.predicted_geometry is not None
        assert result.steric_zipper_class is not None
        assert 0 <= result.confidence <= 1
    
    def test_gnnqqny_steric_zipper(self, classifier):
        """
        Test GNNQQNY classification.
        
        This is a canonical steric zipper (PDB: 1YJO) with:
        - Parallel in-register β-strands
        - Class 1 symmetry (face-to-face)
        """
        result = classifier.predict(GNNQQNY)
        
        # Should be classified as steric zipper
        assert result.predicted_fold == AmyloidFold.STERIC_ZIPPER
        
        # Should have reasonable confidence
        assert result.confidence > 0.3
        
        # Geometry should favor parallel
        assert result.predicted_geometry in [
            CrossBetaGeometry.PARALLEL_IN_REGISTER,
            CrossBetaGeometry.PARALLEL_OUT_OF_REGISTER,
        ]
    
    def test_klvffa_steric_zipper(self, classifier):
        """Test KLVFFA (Aβ core hexapeptide)."""
        result = classifier.predict(KLVFFA)
        
        # Short hydrophobic peptide -> steric zipper
        assert result.predicted_fold == AmyloidFold.STERIC_ZIPPER
    
    def test_vqivyk_tau_hexapeptide(self, classifier):
        """Test VQIVYK (tau PHF6 hexapeptide)."""
        result = classifier.predict(VQIVYK)
        
        # Known steric zipper (PDB: 2ON9)
        assert result.predicted_fold == AmyloidFold.STERIC_ZIPPER
    
    def test_het_s_beta_solenoid(self, classifier):
        """
        Test HET-s prion domain classification.
        
        HET-s forms a β-solenoid structure with:
        - Regular glycine turns
        - Pseudo-repeating units
        """
        result = classifier.predict(HET_S)
        
        # Should be classified as β-solenoid
        assert result.predicted_fold == AmyloidFold.BETA_SOLENOID
        
        # Glycine periodicity should be detected
        assert result.structural_features['glycine_periodicity'] > 0.5
    
    def test_abeta42_classification(self, classifier):
        """
        Test full Aβ42 classification.
        
        Full-length Aβ42 forms β-arcade in fibrils (cryo-EM),
        but classifier may call steric_zipper or beta_arcade.
        """
        result = classifier.predict(ABETA_42)
        
        # Should be amyloid-compatible fold
        assert result.predicted_fold in [
            AmyloidFold.STERIC_ZIPPER,
            AmyloidFold.BETA_ARCADE,
            AmyloidFold.GREEK_KEY,
        ]
        
        # Should have moderate-high beta propensity
        assert result.structural_features['mean_beta_propensity'] > 1.0
    
    def test_short_vs_long_sequences(self, classifier):
        """Test that sequence length affects fold prediction."""
        short_result = classifier.predict(GNNQQNY)  # 7 aa
        long_result = classifier.predict(ABETA_42)   # 42 aa
        
        # Short sequences favor steric zipper
        assert short_result.fold_probabilities[AmyloidFold.STERIC_ZIPPER.value] > \
               long_result.fold_probabilities[AmyloidFold.STERIC_ZIPPER.value]
    
    def test_probability_normalization(self, classifier):
        """Test that fold probabilities sum to 1."""
        result = classifier.predict(GNNQQNY)
        
        total_prob = sum(result.fold_probabilities.values())
        assert abs(total_prob - 1.0) < 0.01
    
    def test_geometry_probability_normalization(self, classifier):
        """Test that geometry probabilities sum to 1."""
        result = classifier.predict(GNNQQNY)
        
        total_prob = sum(result.geometry_probabilities.values())
        assert abs(total_prob - 1.0) < 0.01


class TestStructuralFeatures:
    """Test structural feature extraction for polymorph classification."""
    
    @pytest.fixture
    def classifier(self):
        return PolymorphClassifier()
    
    def test_features_extracted(self, classifier):
        """Test that structural features are extracted."""
        result = classifier.predict(GNNQQNY)
        
        assert 'parallel_preference' in result.structural_features
        assert 'alternating_score' in result.structural_features
        assert 'glycine_periodicity' in result.structural_features
        assert 'repeat_score' in result.structural_features
    
    def test_qn_rich_detection(self, classifier):
        """Test Q/N-rich sequence detection."""
        result = classifier.predict(GNNQQNY)  # High Q/N content
        
        # Should have high QN fraction
        assert result.structural_features['qn_fraction'] > 0.5
    
    def test_aromatic_fraction(self, classifier):
        """Test aromatic residue detection."""
        # KLVFFA has 2 F residues out of 6 = 33%
        result = classifier.predict(KLVFFA)
        assert result.structural_features['aromatic_fraction'] > 0.3
        
        # GNNQQNY has 1 Y (tyrosine is aromatic) out of 7 = ~14%
        result2 = classifier.predict(GNNQQNY)
        assert result2.structural_features['aromatic_fraction'] > 0.1  # Has Y
        
        # Proline-rich has no aromatics
        result3 = classifier.predict(PROLINE_RICH)
        assert result3.structural_features['aromatic_fraction'] == 0
    
    def test_glycine_periodicity_het_s(self, classifier):
        """Test glycine periodicity detection for HET-s."""
        result = classifier.predict(HET_S)
        
        # HET-s has regular glycine positions
        assert result.structural_features['glycine_periodicity'] > 0.7
    
    def test_charge_asymmetry(self, classifier):
        """Test charge distribution analysis."""
        # Create sequence with clustered charges
        charged_seq = "KKKVVVVDDD"
        result = classifier.predict(charged_seq)
        
        # Should detect charge asymmetry
        assert result.structural_features['charge_asymmetry'] > 0


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_predict_polymorph(self):
        """Test predict_polymorph convenience function."""
        result = predict_polymorph(GNNQQNY)
        
        assert isinstance(result, PolymorphPrediction)
        assert result.predicted_fold == AmyloidFold.STERIC_ZIPPER
    
    def test_find_similar_structures(self):
        """Test structure similarity search."""
        similar = find_similar_structures(KLVFFA)
        
        # Should return a list
        assert isinstance(similar, list)
    
    def test_known_structures_database(self):
        """Test known structures database exists."""
        assert isinstance(KNOWN_AMYLOID_STRUCTURES, dict)
        assert len(KNOWN_AMYLOID_STRUCTURES) > 0


class TestPolymorphPredictionMethods:
    """Test PolymorphPrediction dataclass methods."""
    
    def test_summary_generation(self):
        """Test summary() method."""
        classifier = PolymorphClassifier()
        result = classifier.predict(GNNQQNY)
        
        summary = result.summary()
        
        assert isinstance(summary, str)
        assert "steric_zipper" in summary.lower()
        assert "confidence" in summary.lower()


class TestIntegrationWithPipeline:
    """Test integration with the main prediction pipeline."""
    
    def test_classify_apr_regions(self):
        """Test classifying identified APR regions."""
        from amyloidbench import predict, ProteinRecord
        
        protein = ProteinRecord(id='Abeta42', sequence=ABETA_42)
        result = predict(protein, consensus=True)
        
        # Classify each APR
        classifier = PolymorphClassifier()
        
        for region in result.consensus_regions:
            polymorph = classifier.predict(region.sequence)
            
            assert polymorph.predicted_fold is not None
            assert 0 <= polymorph.confidence <= 1
    
    def test_batch_classification(self):
        """Test classifying multiple sequences."""
        sequences = [GNNQQNY, KLVFFA, VQIVYK, NFGAIL]
        classifier = PolymorphClassifier()
        
        results = [classifier.predict(seq) for seq in sequences]
        
        # All should be steric zippers (short peptides)
        for result in results:
            assert result.predicted_fold == AmyloidFold.STERIC_ZIPPER


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def classifier(self):
        return PolymorphClassifier()
    
    def test_very_short_sequence(self, classifier):
        """Test handling of very short sequences."""
        result = classifier.predict("AAA")
        
        assert result.predicted_fold is not None
        # Very short -> likely steric zipper
    
    def test_single_amino_acid(self, classifier):
        """Test single amino acid handling."""
        result = classifier.predict("V")
        
        assert result.predicted_fold is not None
        assert result.confidence > 0
    
    def test_proline_rich_sequence(self, classifier):
        """Test proline-rich sequence (aggregation-resistant)."""
        result = classifier.predict(PROLINE_RICH)
        
        # Should still return a prediction
        assert result.predicted_fold is not None
        # But confidence may be low
    
    def test_case_insensitivity(self, classifier):
        """Test that lowercase input works."""
        upper_result = classifier.predict(GNNQQNY)
        lower_result = classifier.predict(GNNQQNY.lower())
        
        assert upper_result.predicted_fold == lower_result.predicted_fold


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
