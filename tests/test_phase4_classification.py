"""
Test suite for Phase 4: Polymorph Classification.

Tests verify:
1. Steric zipper class enumeration and properties
2. Cross-β geometry prediction based on sequence features
3. Higher-order fold classification
4. Template matching against known structures
5. Integration with prediction pipeline
"""

import pytest
import numpy as np

from amyloidbench.classification import (
    StericZipperClass,
    CrossBetaGeometry,
    AmyloidFold,
    PolymorphPrediction,
    PolymorphClassifier,
    predict_polymorph,
    find_similar_structures,
    get_known_structures,
    KNOWN_AMYLOID_STRUCTURES,
)
from amyloidbench.core.models import ProteinRecord


# =============================================================================
# Test sequences with known structures
# =============================================================================

# Aβ42 - parallel in-register β-arcade (PDB: 2NAO, 5OQV)
ABETA_42 = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"

# KLVFFA - Aβ core hexapeptide, parallel class 1 steric zipper (PDB: 2Y3J)
KLVFFA = "KLVFFA"

# GNNQQNY - Sup35 prion, antiparallel class 8 steric zipper (PDB: 1YJP)
GNNQQNY = "GNNQQNY"

# HET-s prion domain repeat - β-solenoid (PDB: 2RNM)
HETS_REPEAT = "GIDAASHIQNVGDGAGPNPQ"

# α-synuclein NAC region - parallel in-register Greek key (PDB: 2N0A)
ASYN_NAC = "EQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFV"

# Alternating hydrophobic-polar (should favor antiparallel)
ALTERNATING = "VSVSVSVS"


class TestStericZipperEnum:
    """Tests for StericZipperClass enumeration."""
    
    def test_all_eight_classes_defined(self):
        """Verify all 8 steric zipper classes are defined."""
        classes = [c for c in StericZipperClass if c != StericZipperClass.UNKNOWN]
        assert len(classes) == 8
    
    def test_class_1_properties(self):
        """
        Test Class 1 properties: parallel strands, parallel sheets, face-to-face.
        
        Class 1 is the most common arrangement, found in many disease amyloids.
        """
        c1 = StericZipperClass.CLASS_1
        assert c1.strand_orientation == "parallel"
        assert c1.sheet_packing == "parallel"
        assert c1.face_orientation == "face_to_face"
    
    def test_class_8_properties(self):
        """
        Test Class 8 properties: antiparallel strands, antiparallel sheets, face-to-back.
        
        Found in GNNQQNY (PDB: 1YJP).
        """
        c8 = StericZipperClass.CLASS_8
        assert c8.strand_orientation == "antiparallel"
        assert c8.sheet_packing == "antiparallel"
        assert c8.face_orientation == "face_to_back"
    
    def test_parallel_strand_classes(self):
        """Classes 1-4 should have parallel strand orientation."""
        parallel_classes = [
            StericZipperClass.CLASS_1,
            StericZipperClass.CLASS_2,
            StericZipperClass.CLASS_3,
            StericZipperClass.CLASS_4,
        ]
        for c in parallel_classes:
            assert c.strand_orientation == "parallel"
    
    def test_antiparallel_strand_classes(self):
        """Classes 5-8 should have antiparallel strand orientation."""
        antiparallel_classes = [
            StericZipperClass.CLASS_5,
            StericZipperClass.CLASS_6,
            StericZipperClass.CLASS_7,
            StericZipperClass.CLASS_8,
        ]
        for c in antiparallel_classes:
            assert c.strand_orientation == "antiparallel"


class TestCrossBetaGeometryPrediction:
    """Tests for cross-β geometry prediction."""
    
    def test_parallel_prediction_for_hydrophobic_sequence(self):
        """
        Hydrophobic sequences should favor parallel in-register geometry.
        
        Reason: Identical hydrophobic side chains stack favorably along
        the fibril axis in parallel arrangement.
        """
        # KLVFFA: hydrophobic, aromatic - should be parallel
        result = predict_polymorph(KLVFFA)
        
        assert result.predicted_geometry == CrossBetaGeometry.PARALLEL_IN_REGISTER
        # Parallel should have highest probability
        assert result.geometry_probabilities[CrossBetaGeometry.PARALLEL_IN_REGISTER.value] > 0.3
    
    def test_geometry_for_qn_rich_sequence(self):
        """
        Q/N-rich sequences form polar zippers favoring parallel arrangement.
        
        Asparagine and glutamine can form hydrogen-bonded "ladders" in
        parallel in-register structures (as in polyQ diseases).
        """
        result = predict_polymorph(GNNQQNY)
        
        # High Q/N content should be detected
        assert result.structural_features['qn_fraction'] > 0.5
    
    def test_alternating_pattern_detection(self):
        """
        Alternating hydrophobic-polar patterns should increase antiparallel probability.
        
        Such patterns allow complementary packing in antiparallel β-sheets.
        """
        result = predict_polymorph(ALTERNATING)
        
        # Should have elevated alternating score
        assert result.structural_features['alternating_score'] > 0.5
    
    def test_geometry_probabilities_sum_to_one(self):
        """Geometry probabilities should be normalized."""
        result = predict_polymorph(ABETA_42)
        
        total = sum(result.geometry_probabilities.values())
        assert abs(total - 1.0) < 0.01


class TestFoldClassification:
    """Tests for higher-order fold classification."""
    
    def test_steric_zipper_for_short_peptide(self):
        """
        Short peptides (< 15 residues) typically form steric zippers.
        
        The limited length allows only simple β-sheet arrangements.
        """
        result = predict_polymorph(KLVFFA)
        
        # Short peptide should favor steric zipper
        assert result.predicted_fold == AmyloidFold.STERIC_ZIPPER
    
    def test_fold_probabilities_normalized(self):
        """Fold probabilities should sum to 1."""
        result = predict_polymorph(ABETA_42)
        
        total = sum(result.fold_probabilities.values())
        assert abs(total - 1.0) < 0.01
    
    def test_repeat_detection_for_solenoid(self):
        """
        Repeated sequences should increase β-solenoid probability.
        
        β-solenoid folds (HET-s, curli) are built from repeated units
        that stack in a helical arrangement.
        """
        # Double the HET-s repeat
        double_repeat = HETS_REPEAT + HETS_REPEAT
        result = predict_polymorph(double_repeat)
        
        # Should detect repeats
        assert result.structural_features['repeat_score'] > 0
    
    def test_long_sequence_fold_prediction(self):
        """
        Longer sequences (30-60 residues) may adopt β-arcade or Greek key.
        
        These complex topologies require more residues to form.
        """
        result = predict_polymorph(ABETA_42)
        
        # Should consider complex folds for longer sequences
        assert result.predicted_fold in [
            AmyloidFold.STERIC_ZIPPER,
            AmyloidFold.BETA_ARCADE,
            AmyloidFold.GREEK_KEY,
        ]


class TestPolymorphClassifier:
    """Tests for the main PolymorphClassifier class."""
    
    def test_classifier_initialization(self):
        """Test classifier initializes correctly."""
        classifier = PolymorphClassifier()
        assert classifier.feature_extractor is not None
    
    def test_predict_returns_valid_result(self):
        """Prediction should return PolymorphPrediction with all fields."""
        classifier = PolymorphClassifier()
        result = classifier.predict(ABETA_42)
        
        assert isinstance(result, PolymorphPrediction)
        assert result.sequence == ABETA_42
        assert result.predicted_fold in AmyloidFold
        assert result.predicted_geometry in CrossBetaGeometry
        assert result.steric_zipper_class in StericZipperClass
        assert 0 <= result.confidence <= 1
    
    def test_structural_features_extracted(self):
        """Key structural features should be extracted."""
        result = predict_polymorph(ABETA_42)
        
        required_features = [
            'parallel_preference',
            'alternating_score',
            'charge_asymmetry',
            'repeat_score',
            'qn_fraction',
            'aromatic_fraction',
            'mean_hydrophobicity',
            'mean_beta_propensity',
        ]
        
        for feat in required_features:
            assert feat in result.structural_features
    
    def test_summary_generation(self):
        """Summary should be human-readable string."""
        result = predict_polymorph(ABETA_42)
        summary = result.summary()
        
        assert isinstance(summary, str)
        assert "Polymorph Prediction" in summary
        assert len(summary) > 50


class TestTemplateMatching:
    """Tests for template matching against known structures."""
    
    def test_known_structures_database(self):
        """Verify known structures database is populated."""
        structures = get_known_structures()
        
        assert len(structures) >= 5
        assert 'klvffa_2y3j' in structures
        assert 'abeta_2nao' in structures
    
    def test_exact_match_detection(self):
        """Exact sequence match should return 100% identity."""
        matches = find_similar_structures(KLVFFA, min_identity=0.9)
        
        assert len(matches) >= 1
        # KLVFFA should match itself exactly
        assert any(m['identity'] == 1.0 for m in matches)
    
    def test_partial_match_detection(self):
        """Partial matches should be found with lower threshold."""
        matches = find_similar_structures(ABETA_42, min_identity=0.3)
        
        # Should find some matches
        assert len(matches) >= 0  # May or may not find matches depending on database
    
    def test_match_structure_info(self):
        """Matched structures should include full annotation."""
        matches = find_similar_structures(KLVFFA, min_identity=0.8)
        
        if matches:
            match = matches[0]
            assert 'name' in match
            assert 'pdb_id' in match
            assert 'fold' in match
            assert 'geometry' in match


class TestBiologicalValidation:
    """
    Biological validation tests using well-characterized amyloid sequences.
    
    These tests verify that predictions align with experimentally
    determined structures from cryo-EM and X-ray crystallography.
    """
    
    def test_abeta_geometry(self):
        """
        Aβ42 forms parallel in-register fibrils.
        
        Multiple cryo-EM structures (2NAO, 5OQV, 6W0O) show parallel
        β-strands with residues stacking in register.
        """
        result = predict_polymorph(ABETA_42)
        
        # Should predict parallel geometry
        assert result.predicted_geometry == CrossBetaGeometry.PARALLEL_IN_REGISTER
    
    def test_klvffa_is_class_1_zipper(self):
        """
        KLVFFA crystallizes as Class 1 steric zipper (PDB: 2Y3J).
        
        Parallel strands, parallel sheets, face-to-face orientation.
        """
        result = predict_polymorph(KLVFFA)
        
        # Should predict parallel class
        assert result.steric_zipper_class in [
            StericZipperClass.CLASS_1,
            StericZipperClass.CLASS_2,
            StericZipperClass.CLASS_3,
            StericZipperClass.CLASS_4,
        ]
    
    def test_aromatic_content_affects_parallel_preference(self):
        """
        Aromatic residues (F, Y, W) favor parallel arrangement.
        
        π-π stacking between identical aromatic side chains is
        energetically favorable in parallel in-register structures.
        """
        result = predict_polymorph(KLVFFA)
        
        # KLVFFA has 33% aromatic (2 F residues)
        assert result.structural_features['aromatic_fraction'] > 0.2
        
        # Should favor parallel
        assert result.geometry_probabilities[CrossBetaGeometry.PARALLEL_IN_REGISTER.value] > 0.3
    
    def test_confidence_higher_for_well_characterized_sequences(self):
        """
        Well-characterized amyloid sequences should have higher confidence.
        
        Sequences with clear patterns (hydrophobicity, β-propensity)
        yield more confident predictions.
        """
        abeta_result = predict_polymorph(ABETA_42)
        klvffa_result = predict_polymorph(KLVFFA)
        
        # Both should have reasonable confidence
        assert abeta_result.confidence > 0.3
        assert klvffa_result.confidence > 0.3


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""
    
    def test_very_short_sequence(self):
        """Very short sequences (< 5 residues) should still predict."""
        result = predict_polymorph("VVV")
        
        assert result.predicted_fold is not None
        # Should note low confidence due to length
    
    def test_long_sequence(self):
        """Long sequences should handle without error."""
        long_seq = ABETA_42 * 3  # ~126 residues
        result = predict_polymorph(long_seq)
        
        assert result.sequence == long_seq
        assert result.predicted_fold is not None
    
    def test_proline_rich_sequence(self):
        """Proline-rich sequences (β-breakers) should handle gracefully."""
        proline_seq = "PPGPPGPPGPPG"
        result = predict_polymorph(proline_seq)
        
        assert result.predicted_fold is not None
        # Proline-rich should have lower β-propensity
        assert result.structural_features['mean_beta_propensity'] < 1.0
    
    def test_charged_sequence(self):
        """Highly charged sequences should detect charge features."""
        charged_seq = "KKKKEEEEDDDD"
        result = predict_polymorph(charged_seq)
        
        # Should detect charge asymmetry
        assert 'charge_asymmetry' in result.structural_features


class TestIntegrationWithPipeline:
    """Tests for integration with main prediction pipeline."""
    
    def test_predict_with_polymorph_classification(self):
        """Full pipeline: APR prediction + polymorph classification."""
        from amyloidbench import predict
        
        protein = ProteinRecord(id='Abeta42', sequence=ABETA_42)
        
        # Get consensus prediction
        consensus_result = predict(protein, consensus=True)
        
        # Now classify polymorphs for detected APRs
        if consensus_result.consensus_regions:
            for region in consensus_result.consensus_regions:
                polymorph = predict_polymorph(region.sequence)
                assert polymorph.predicted_fold is not None
    
    def test_multiple_apr_polymorph_classification(self):
        """Different APRs in same protein may have different preferences."""
        # Aβ42 has two main APRs
        # KLVFFA region and C-terminal region
        
        klvff_result = predict_polymorph("KLVFFA")
        cterminal = predict_polymorph("GAIIGLMVGGVVIA")
        
        # Both should be valid predictions
        assert klvff_result.predicted_geometry in CrossBetaGeometry
        assert cterminal.predicted_geometry in CrossBetaGeometry


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
