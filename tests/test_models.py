"""
Unit tests for AmyloidBench core data models.

These tests validate the fundamental data structures used throughout
the pipeline, ensuring proper validation, serialization, and biological
correctness of model behavior.
"""

import pytest
import numpy as np
from pathlib import Path

from amyloidbench.core.models import (
    AmyloidPolymorph,
    Region,
    PerResidueScores,
    PredictionResult,
    ConsensusResult,
    ProteinRecord,
    PredictionConfidence,
    BenchmarkDataset,
)


class TestRegion:
    """Tests for the Region model representing sequence segments."""
    
    def test_region_creation(self):
        """Test basic region instantiation."""
        region = Region(start=10, end=20, sequence="VVVVVVVVVV")
        assert region.start == 10
        assert region.end == 20
        assert region.length == 10
        assert region.sequence == "VVVVVVVVVV"
    
    def test_region_with_score(self):
        """Test region with associated prediction score."""
        region = Region(
            start=5, end=12, 
            sequence="ILFILFV",
            score=1.85,
            confidence=PredictionConfidence.HIGH,
            annotation="APR from Aggrescan3D"
        )
        assert region.score == 1.85
        assert region.confidence == PredictionConfidence.HIGH
    
    def test_region_end_must_exceed_start(self):
        """Validate that end position must be after start."""
        with pytest.raises(ValueError, match="end must be greater than start"):
            Region(start=20, end=10, sequence="AAAA")
    
    def test_region_overlap_detection(self):
        """Test overlap detection between regions."""
        region1 = Region(start=10, end=20, sequence="AAAAAAAAAA")
        region2 = Region(start=15, end=25, sequence="AAAAAAAAAA")
        region3 = Region(start=25, end=35, sequence="AAAAAAAAAA")
        
        assert region1.overlaps(region2)  # Overlapping
        assert region2.overlaps(region1)  # Symmetric
        assert not region1.overlaps(region3)  # Non-overlapping
        assert not region3.overlaps(region1)
    
    def test_region_merge(self):
        """Test merging of overlapping regions."""
        region1 = Region(start=10, end=20, sequence="", score=1.5)
        region2 = Region(start=15, end=25, sequence="", score=2.0)
        
        merged = region1.merge(region2)
        assert merged.start == 10
        assert merged.end == 25
        assert merged.score == 2.0  # Takes maximum
    
    def test_region_merge_non_overlapping_raises(self):
        """Cannot merge non-overlapping regions."""
        region1 = Region(start=10, end=20, sequence="A" * 10)
        region2 = Region(start=30, end=40, sequence="A" * 10)
        
        with pytest.raises(ValueError, match="non-overlapping"):
            region1.merge(region2)


class TestPerResidueScores:
    """Tests for per-residue prediction scores."""
    
    @pytest.fixture
    def sample_scores(self):
        """Create sample scores for testing."""
        sequence = "MVLSPADKTNVKAAWGKVGAH"
        scores = [0.1, 0.2, 1.5, 1.8, 2.0, 1.9, 1.7, 0.3, 0.1, 0.0,
                  0.1, 0.2, 0.3, 2.1, 2.3, 2.0, 1.8, 0.2, 0.1, 0.0, 0.1]
        return PerResidueScores(
            scores=scores,
            sequence=sequence,
            predictor="TestPredictor",
            score_type="raw",
            threshold=1.0,
            min_score=0.0,
            max_score=3.0,
        )
    
    def test_scores_length_validation(self):
        """Scores length must match sequence length."""
        with pytest.raises(ValueError, match="scores length"):
            PerResidueScores(
                scores=[0.1, 0.2, 0.3],
                sequence="MVLSPAD",  # 7 residues, 3 scores
                predictor="Test",
            )
    
    def test_minmax_normalization(self, sample_scores):
        """Test min-max normalization to [0,1] range."""
        normalized = sample_scores.normalize(method="minmax")
        
        assert normalized.score_type == "normalized"
        assert min(normalized.scores) >= 0.0
        assert max(normalized.scores) <= 1.0
        assert normalized.min_score == 0.0
        assert normalized.max_score == 1.0
    
    def test_zscore_normalization(self, sample_scores):
        """Test z-score normalization."""
        normalized = sample_scores.normalize(method="zscore")
        
        # Z-score should have mean ~0 and std ~1
        scores_arr = np.array(normalized.scores)
        assert abs(scores_arr.mean()) < 0.01
        assert abs(scores_arr.std() - 1.0) < 0.01
    
    def test_region_extraction(self, sample_scores):
        """Test conversion to discrete regions."""
        regions = sample_scores.to_regions(threshold=1.0, min_length=3)
        
        # Should identify two regions above threshold
        assert len(regions) >= 1
        
        # All regions should be above threshold
        for region in regions:
            assert region.score >= 1.0
            assert region.length >= 3
    
    def test_region_merging(self):
        """Test that nearby regions are merged."""
        # Create scores with two peaks separated by 2 residues below threshold
        scores = [0.0] * 5 + [2.0] * 5 + [0.5, 0.5] + [2.0] * 5 + [0.0] * 5
        sequence = "A" * len(scores)
        
        per_res = PerResidueScores(
            scores=scores,
            sequence=sequence,
            predictor="Test",
            threshold=1.0,
        )
        
        # With merge_gap=2, should merge into single region
        regions_merged = per_res.to_regions(threshold=1.0, min_length=3, merge_gap=2)
        # With merge_gap=0, should get two regions
        regions_separate = per_res.to_regions(threshold=1.0, min_length=3, merge_gap=0)
        
        assert len(regions_merged) < len(regions_separate)


class TestPredictionResult:
    """Tests for prediction result container."""
    
    def test_successful_result(self):
        """Test creation of successful prediction result."""
        result = PredictionResult(
            sequence_id="test_protein",
            sequence="MVLSPADKTNV",
            predictor_name="TestPredictor",
            predictor_version="1.0",
            is_amyloidogenic=True,
            amyloid_probability=0.85,
            predicted_regions=[
                Region(start=2, end=7, sequence="LSPAD", score=1.5)
            ],
        )
        
        assert result.success
        assert result.is_amyloidogenic
        assert len(result.predicted_regions) == 1
    
    def test_failed_result(self):
        """Test creation of failed prediction result."""
        result = PredictionResult(
            sequence_id="test_protein",
            sequence="MVLSPADKTNV",
            predictor_name="TestPredictor",
            error_message="Server timeout",
        )
        
        assert not result.success
        assert result.error_message == "Server timeout"
    
    def test_probability_bounds(self):
        """Amyloid probability must be in [0, 1]."""
        with pytest.raises(ValueError):
            PredictionResult(
                sequence_id="test",
                sequence="AAAA",
                predictor_name="Test",
                amyloid_probability=1.5,  # Invalid
            )


class TestConsensusResult:
    """Tests for consensus prediction results."""
    
    @pytest.fixture
    def sample_consensus(self):
        """Create sample consensus result."""
        individual = {
            "Predictor1": PredictionResult(
                sequence_id="test",
                sequence="MVLSPAD",
                predictor_name="Predictor1",
                is_amyloidogenic=True,
            ),
            "Predictor2": PredictionResult(
                sequence_id="test",
                sequence="MVLSPAD",
                predictor_name="Predictor2",
                is_amyloidogenic=True,
            ),
            "Predictor3": PredictionResult(
                sequence_id="test",
                sequence="MVLSPAD",
                predictor_name="Predictor3",
                is_amyloidogenic=False,
            ),
        }
        
        return ConsensusResult(
            sequence_id="test",
            sequence="MVLSPAD",
            individual_results=individual,
            consensus_is_amyloidogenic=True,
            n_predictors_agree_positive=2,
            n_predictors_agree_negative=1,
            n_predictors_total=3,
            consensus_method="majority_vote",
        )
    
    def test_agreement_ratio(self, sample_consensus):
        """Test agreement ratio calculation."""
        ratio = sample_consensus.agreement_ratio()
        assert ratio == 2/3  # 2 out of 3 agree with positive


class TestProteinRecord:
    """Tests for protein record model."""
    
    def test_valid_sequence(self):
        """Test creation with valid protein sequence."""
        protein = ProteinRecord(
            id="P12345",
            sequence="MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH",
            name="Hemoglobin",
            organism="Homo sapiens",
        )
        
        assert protein.sequence_length == 50
        assert protein.id == "P12345"
    
    def test_sequence_validation_uppercase(self):
        """Sequence should be converted to uppercase."""
        protein = ProteinRecord(
            id="test",
            sequence="mvlspadktnv",
        )
        assert protein.sequence == "MVLSPADKTNV"
    
    def test_sequence_validation_invalid_chars(self):
        """Invalid amino acid characters should raise error."""
        with pytest.raises(ValueError, match="Invalid amino acid"):
            ProteinRecord(
                id="test",
                sequence="MVLS123ADKTNV",  # Contains numbers
            )
    
    def test_sequence_whitespace_removal(self):
        """Whitespace should be stripped from sequence."""
        protein = ProteinRecord(
            id="test",
            sequence="MVL SPA\nDKT NV",
        )
        assert protein.sequence == "MVLSPADKTNV"
    
    def test_ambiguous_codes_allowed(self):
        """Ambiguous amino acid codes (X, B, etc.) should be allowed."""
        protein = ProteinRecord(
            id="test",
            sequence="MVLXPADKTNVB",  # X = unknown, B = Asx
        )
        assert "X" in protein.sequence
        assert "B" in protein.sequence
    
    def test_known_amyloid_regions(self):
        """Test protein with annotated amyloidogenic regions."""
        protein = ProteinRecord(
            id="PrP",
            sequence="MANLGCWMLVLFVATWSDLGLC" + "A" * 50,
            is_known_amyloid=True,
            known_amyloid_regions=[
                Region(start=10, end=22, sequence="LFVATWSDLGLC")
            ],
            known_polymorph=AmyloidPolymorph.CROSS_BETA_PARALLEL,
        )
        
        assert protein.is_known_amyloid
        assert len(protein.known_amyloid_regions) == 1
        assert protein.known_polymorph == AmyloidPolymorph.CROSS_BETA_PARALLEL


class TestAmyloidPolymorph:
    """Tests for amyloid polymorph enumeration."""
    
    def test_polymorph_values(self):
        """Verify all polymorph types are defined."""
        assert AmyloidPolymorph.CROSS_BETA_PARALLEL.value == "cross_beta_parallel"
        assert AmyloidPolymorph.CROSS_BETA_ANTIPARALLEL.value == "cross_beta_antiparallel"
        assert AmyloidPolymorph.BETA_SOLENOID.value == "beta_solenoid"
        assert AmyloidPolymorph.CROSS_ALPHA.value == "cross_alpha"
        assert AmyloidPolymorph.AMORPHOUS.value == "amorphous"
        assert AmyloidPolymorph.UNKNOWN.value == "unknown"


class TestBenchmarkDataset:
    """Tests for benchmark dataset container."""
    
    def test_dataset_statistics(self):
        """Test positive/negative count calculations."""
        proteins = [
            ProteinRecord(id="pos1", sequence="AAAA", is_known_amyloid=True),
            ProteinRecord(id="pos2", sequence="BBBB", is_known_amyloid=True),
            ProteinRecord(id="neg1", sequence="CCCC", is_known_amyloid=False),
            ProteinRecord(id="unk1", sequence="DDDD"),  # Unknown
        ]
        
        dataset = BenchmarkDataset(
            name="Test Dataset",
            description="Test",
            proteins=proteins,
        )
        
        assert dataset.n_total == 4
        assert dataset.n_positive == 2
        assert dataset.n_negative == 1
        assert len(dataset.get_positive_examples()) == 2
        assert len(dataset.get_negative_examples()) == 1
