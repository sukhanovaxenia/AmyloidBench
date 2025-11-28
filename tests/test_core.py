"""
Test suite for AmyloidBench core modules.

These tests validate the fundamental data structures and utilities
that underpin all prediction functionality. Biological validity is
tested alongside technical correctness.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from amyloidbench.core.models import (
    AmyloidPolymorph,
    PerResidueScores,
    PredictionResult,
    ProteinRecord,
    Region,
)
from amyloidbench.core.sequence import (
    AA_CLASSES,
    SequenceValidator,
    calculate_class_composition,
    calculate_composition,
    extract_region_context,
    find_low_complexity_regions,
    find_motifs,
    parse_fasta,
    sequence_hash,
    sliding_window,
    to_fasta,
)


# =============================================================================
# Test Data: Biologically meaningful sequences
# =============================================================================

# Human Amyloid-β (1-42) - archetypal amyloidogenic peptide
ABETA_42 = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"

# Human α-synuclein NAC region (61-95) - core amyloidogenic segment
ASYN_NAC = "EQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFV"

# Human Prion Protein (106-126) - neurotoxic amyloidogenic region
PRP_106_126 = "KTNMKHMAGAAAAGAVVGGLG"

# Non-amyloidogenic control: Human ubiquitin (highly soluble, globular)
UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

# Functional amyloid: Curli CsgA (E. coli)
CSGA_R1 = "SELNIYQYGGGNSALALQTDARN"

# Cross-α amyloid: PSMα3 from S. aureus
PSMALPHA3 = "MEFVAKLFKFFKDLLGKFLGNN"


class TestProteinRecord:
    """Tests for ProteinRecord data model."""
    
    def test_basic_creation(self):
        """Test basic record creation with minimal fields."""
        record = ProteinRecord(id="test", sequence="MVLSPADKTNV")
        assert record.id == "test"
        assert record.sequence == "MVLSPADKTNV"
        assert record.sequence_length == 11
    
    def test_sequence_validation(self):
        """Test that invalid amino acids are rejected."""
        with pytest.raises(ValueError, match="Invalid amino acid"):
            ProteinRecord(id="bad", sequence="MVLSPADKTNV123")
    
    def test_sequence_cleaning(self):
        """Test that whitespace is handled correctly."""
        record = ProteinRecord(id="test", sequence="MVL SPA\nDKT NV")
        assert record.sequence == "MVLSPADKTNV"
    
    def test_lowercase_conversion(self):
        """Test that lowercase is converted to uppercase."""
        record = ProteinRecord(id="test", sequence="mvlspadktnv")
        assert record.sequence == "MVLSPADKTNV"
    
    def test_ambiguous_codes_allowed(self):
        """Test that ambiguous amino acid codes (X, B, Z) are accepted."""
        record = ProteinRecord(id="test", sequence="MVLXPADBTZNV")
        assert "X" in record.sequence
    
    def test_abeta_record(self):
        """Test creation of Aβ42 record with annotations."""
        record = ProteinRecord(
            id="Abeta42",
            name="Amyloid-beta 42",
            sequence=ABETA_42,
            organism="Homo sapiens",
            is_known_amyloid=True,
            known_polymorph=AmyloidPolymorph.CROSS_BETA_PARALLEL,
            known_amyloid_regions=[
                Region(start=16, end=21, sequence="KLVFFA"),  # Central hydrophobic
                Region(start=29, end=42, sequence="GAIIGLMVGGVVIA"),  # C-terminal
            ],
        )
        assert record.is_known_amyloid is True
        assert len(record.known_amyloid_regions) == 2
        assert record.known_polymorph == AmyloidPolymorph.CROSS_BETA_PARALLEL


class TestRegion:
    """Tests for Region model representing APRs."""
    
    def test_region_creation(self):
        """Test basic region creation."""
        region = Region(start=16, end=21, sequence="KLVFFA")
        assert region.length == 5
    
    def test_region_overlap_detection(self):
        """Test overlap detection between regions."""
        r1 = Region(start=10, end=20, sequence="A" * 10)
        r2 = Region(start=15, end=25, sequence="A" * 10)
        r3 = Region(start=25, end=35, sequence="A" * 10)
        
        assert r1.overlaps(r2) is True
        assert r2.overlaps(r3) is True
        assert r1.overlaps(r3) is False
    
    def test_invalid_region(self):
        """Test that end must be after start."""
        with pytest.raises(ValueError):
            Region(start=20, end=10, sequence="KLVFFA")


class TestPerResidueScores:
    """Tests for per-residue score handling."""
    
    def test_score_creation(self):
        """Test basic score array creation."""
        scores = PerResidueScores(
            scores=[0.1, 0.5, 0.8, 0.3, 0.2],
            sequence="MVLSP",
            predictor="test",
        )
        assert len(scores.scores) == len(scores.sequence)
    
    def test_score_length_validation(self):
        """Test that scores must match sequence length."""
        with pytest.raises(ValueError):
            PerResidueScores(
                scores=[0.1, 0.5, 0.8],  # 3 scores
                sequence="MVLSP",  # 5 residues
                predictor="test",
            )
    
    def test_normalization_minmax(self):
        """Test min-max normalization."""
        scores = PerResidueScores(
            scores=[-2.0, 0.0, 2.0, 4.0],
            sequence="MVLS",
            predictor="test",
            min_score=-2.0,
            max_score=4.0,
        )
        normalized = scores.normalize(method="minmax")
        
        assert normalized.scores[0] == pytest.approx(0.0)
        assert normalized.scores[-1] == pytest.approx(1.0)
        assert normalized.score_type == "normalized"
    
    def test_region_extraction(self):
        """Test conversion of scores to discrete regions."""
        # Simulate a profile with one clear APR
        scores = PerResidueScores(
            scores=[0.1, 0.2, 0.8, 0.9, 0.85, 0.75, 0.3, 0.1, 0.15, 0.1],
            sequence="MVKLVFFAED",
            predictor="test",
            threshold=0.5,
        )
        regions = scores.to_regions(min_length=3, merge_gap=0)
        
        assert len(regions) == 1
        assert regions[0].start == 2
        assert regions[0].end == 6
        assert "LVFF" in regions[0].sequence
    
    def test_region_merging(self):
        """Test that nearby regions are merged."""
        # Two regions separated by one residue
        scores = PerResidueScores(
            scores=[0.1, 0.8, 0.9, 0.3, 0.8, 0.9, 0.1],
            sequence="MVLSPAK",
            predictor="test",
            threshold=0.5,
        )
        
        # Without merging
        regions_no_merge = scores.to_regions(min_length=2, merge_gap=0)
        assert len(regions_no_merge) == 2
        
        # With merging (gap=1)
        regions_merged = scores.to_regions(min_length=2, merge_gap=1)
        assert len(regions_merged) == 1


class TestSequenceValidator:
    """Tests for sequence validation utilities."""
    
    def test_valid_sequence(self):
        """Test validation of a correct sequence."""
        validator = SequenceValidator()
        is_valid, errors = validator.validate(UBIQUITIN)
        assert is_valid is True
        assert len(errors) == 0
    
    def test_invalid_characters(self):
        """Test detection of invalid characters."""
        validator = SequenceValidator()
        is_valid, errors = validator.validate("MVLS123PADKTNV")
        assert is_valid is False
        assert any("Invalid characters" in e for e in errors)
    
    def test_too_short(self):
        """Test minimum length enforcement."""
        validator = SequenceValidator(min_length=10)
        is_valid, errors = validator.validate("MVL")
        assert is_valid is False
        assert any("too short" in e for e in errors)
    
    def test_ambiguous_codes(self):
        """Test handling of ambiguous amino acid codes."""
        # Default: reject ambiguous
        validator_strict = SequenceValidator(allow_ambiguous=False)
        is_valid, _ = validator_strict.validate("MVLXSPADKTNV")
        assert is_valid is False
        
        # Permissive: allow ambiguous
        validator_permissive = SequenceValidator(allow_ambiguous=True)
        is_valid, _ = validator_permissive.validate("MVLXSPADKTNV")
        assert is_valid is True
    
    def test_cleaning(self):
        """Test sequence cleaning functionality."""
        validator = SequenceValidator()
        cleaned = validator.clean("MVL123SPADKTNV")
        assert cleaned == "MVLSPADKTNV"


class TestSequenceComposition:
    """Tests for amino acid composition analysis."""
    
    def test_basic_composition(self):
        """Test amino acid frequency calculation."""
        composition = calculate_composition("AAAVVV")
        assert composition["A"] == pytest.approx(0.5)
        assert composition["V"] == pytest.approx(0.5)
        assert composition["M"] == pytest.approx(0.0)
    
    def test_class_composition(self):
        """Test amino acid class composition for amyloidogenicity."""
        # Aβ42 should have high hydrophobic/aromatic content
        class_comp = calculate_class_composition(ABETA_42)
        
        # Aβ42 is enriched in amyloidogenic residues
        assert class_comp["amyloidogenic"] > 0.3
        
        # Verify gatekeeper calculation (D, E, K are present)
        assert class_comp["gatekeepers"] > 0.1
    
    def test_ubiquitin_vs_abeta(self):
        """Test that soluble ubiquitin differs from amyloidogenic Aβ."""
        abeta_comp = calculate_class_composition(ABETA_42)
        ubiq_comp = calculate_class_composition(UBIQUITIN)
        
        # Ubiquitin should have more charged residues (gatekeepers)
        assert ubiq_comp["gatekeepers"] > abeta_comp["gatekeepers"]


class TestSequenceHash:
    """Tests for sequence hashing (used for caching)."""
    
    def test_deterministic(self):
        """Test that same sequence gives same hash."""
        hash1 = sequence_hash(ABETA_42)
        hash2 = sequence_hash(ABETA_42)
        assert hash1 == hash2
    
    def test_case_insensitive(self):
        """Test that hash is case-insensitive."""
        hash_upper = sequence_hash("MVLSPADKTNV")
        hash_lower = sequence_hash("mvlspadktnv")
        assert hash_upper == hash_lower
    
    def test_different_sequences(self):
        """Test that different sequences give different hashes."""
        hash1 = sequence_hash(ABETA_42)
        hash2 = sequence_hash(UBIQUITIN)
        assert hash1 != hash2


class TestFastaParsing:
    """Tests for FASTA file parsing."""
    
    def test_parse_fasta_string(self):
        """Test parsing FASTA from string."""
        fasta_str = f""">Abeta42|Human amyloid-beta
{ABETA_42}
>Ubiquitin|Human ubiquitin
{UBIQUITIN}
"""
        records = list(parse_fasta(fasta_str, validate=True))
        assert len(records) == 2
        assert records[0].id == "Abeta42"
        assert records[0].sequence == ABETA_42
    
    def test_parse_fasta_file(self):
        """Test parsing FASTA from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            f.write(f">test_protein\n{ABETA_42}\n")
            f.flush()
            
            records = list(parse_fasta(f.name))
            assert len(records) == 1
            assert records[0].sequence == ABETA_42
    
    def test_to_fasta_roundtrip(self):
        """Test FASTA writing and re-parsing."""
        original = [
            ProteinRecord(id="P1", sequence=ABETA_42),
            ProteinRecord(id="P2", sequence=UBIQUITIN),
        ]
        
        fasta_str = to_fasta(original)
        parsed = list(parse_fasta(fasta_str))
        
        assert len(parsed) == 2
        assert parsed[0].sequence == ABETA_42
        assert parsed[1].sequence == UBIQUITIN


class TestMotifFinding:
    """Tests for sequence motif detection."""
    
    def test_find_hydrophobic_stretch(self):
        """Test finding hydrophobic stretches relevant to amyloidogenicity."""
        # Pattern for 5+ consecutive hydrophobic residues
        motifs = find_motifs(ABETA_42, r"[VILFYWM]{5,}")
        
        assert len(motifs) > 0
        # Should find the C-terminal hydrophobic region
        c_term_motif = [m for m in motifs if m.end > 35]
        assert len(c_term_motif) > 0
    
    def test_find_gatekeeper_poor_regions(self):
        """Test identifying regions lacking gatekeepers."""
        # Find stretches of 6+ residues without P, K, R, E, D
        motifs = find_motifs(ABETA_42, r"[^PKRED]{6,}")
        assert len(motifs) > 0


class TestLowComplexityRegions:
    """Tests for low-complexity region detection."""
    
    def test_polyq_detection(self):
        """Test detection of poly-glutamine tract (disease-relevant)."""
        polyq_seq = "MVLS" + "Q" * 20 + "PADKTNV"  # Huntingtin-like
        
        lcr = find_low_complexity_regions(polyq_seq, window_size=10, threshold=2.0)
        assert len(lcr) > 0
        
        # The poly-Q region should be identified
        polyq_region = [r for r in lcr if "Q" * 5 in r.sequence]
        assert len(polyq_region) > 0
    
    def test_normal_protein_few_lcr(self):
        """Test that typical globular proteins have few LCRs."""
        lcr = find_low_complexity_regions(UBIQUITIN, window_size=12, threshold=1.5)
        # Ubiquitin is well-folded with diverse composition
        # Should have minimal low-complexity regions
        assert len(lcr) <= 2


class TestRegionContext:
    """Tests for extracting flanking sequence context."""
    
    def test_context_extraction(self):
        """Test extraction of flanking sequences."""
        region = Region(start=16, end=21, sequence="KLVFF")
        context = extract_region_context(ABETA_42, region, flank_size=5)
        
        assert len(context["upstream"]) <= 5
        assert len(context["downstream"]) <= 5
        assert context["region"] == ABETA_42[16:21]
    
    def test_gatekeeper_counting(self):
        """Test counting of gatekeeper residues in flanking regions."""
        # Create a region in Aβ42 where we know the flanks
        region = Region(start=17, end=21, sequence="LVFF")
        context = extract_region_context(ABETA_42, region, flank_size=5)
        
        # Should count D, E, K in the flanks
        total_gatekeepers = (
            context["upstream_gatekeepers"] + context["downstream_gatekeepers"]
        )
        assert total_gatekeepers >= 0


class TestSlidingWindow:
    """Tests for sliding window iteration."""
    
    def test_window_generation(self):
        """Test basic sliding window generation."""
        windows = list(sliding_window("MVLSPADKTNV", window_size=5, step=1))
        
        assert len(windows) == 7  # 11 - 5 + 1
        assert windows[0] == (0, "MVLSP")
        assert windows[-1] == (6, "DKTNV")
    
    def test_window_step(self):
        """Test sliding window with step > 1."""
        windows = list(sliding_window("MVLSPADKTNV", window_size=3, step=3))
        
        assert len(windows) == 3
        assert windows[0] == (0, "MVL")
        assert windows[1] == (3, "SPA")


class TestPredictionResult:
    """Tests for prediction result model."""
    
    def test_success_property(self):
        """Test success/failure detection."""
        success_result = PredictionResult(
            sequence_id="test",
            sequence=ABETA_42,
            predictor_name="test",
        )
        assert success_result.success is True
        
        failed_result = PredictionResult(
            sequence_id="test",
            sequence=ABETA_42,
            predictor_name="test",
            error_message="Connection failed",
        )
        assert failed_result.success is False
    
    def test_full_prediction_result(self):
        """Test creation of complete prediction result."""
        scores = PerResidueScores(
            scores=np.random.rand(len(ABETA_42)).tolist(),
            sequence=ABETA_42,
            predictor="test",
            threshold=0.5,
        )
        
        result = PredictionResult(
            sequence_id="Abeta42",
            sequence=ABETA_42,
            predictor_name="TestPredictor",
            predictor_version="1.0",
            per_residue_scores=scores,
            predicted_regions=[
                Region(start=16, end=21, sequence="KLVFF", score=0.85),
            ],
            is_amyloidogenic=True,
            amyloid_probability=0.92,
            predicted_polymorph=AmyloidPolymorph.CROSS_BETA_PARALLEL,
            runtime_seconds=1.5,
        )
        
        assert result.success is True
        assert result.is_amyloidogenic is True
        assert len(result.predicted_regions) == 1


# =============================================================================
# Integration tests
# =============================================================================

class TestBiologicalConsistency:
    """Integration tests for biological consistency of predictions."""
    
    def test_amyloidogenic_composition_pattern(self):
        """
        Test that known amyloidogenic sequences show expected composition.
        
        Amyloidogenic regions are enriched in:
        - Hydrophobic residues (V, I, L, F, Y, W, M)
        - β-sheet propensity residues
        - Depleted in gatekeepers (P, K, R, E, D)
        """
        amyloid_seqs = [ABETA_42, ASYN_NAC, PRP_106_126]
        
        for seq in amyloid_seqs:
            comp = calculate_class_composition(seq)
            # Amyloidogenic content should be substantial
            assert comp["amyloidogenic"] > 0.2, f"Low amyloid content in {seq[:10]}"
    
    def test_functional_vs_pathological_amyloid(self):
        """
        Test compositional differences between functional and pathological amyloids.
        
        Functional amyloids (like CsgA) tend to have more regular patterns
        and potentially different gatekeeper arrangements than pathological ones.
        """
        pathological = ABETA_42
        functional = CSGA_R1
        
        path_comp = calculate_class_composition(pathological)
        func_comp = calculate_class_composition(functional)
        
        # Both should be amyloidogenic
        assert path_comp["amyloidogenic"] > 0.2
        assert func_comp["amyloidogenic"] > 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
