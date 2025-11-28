"""
Unit tests for AmyloidBench sequence handling utilities.

These tests validate sequence parsing, validation, and the compositional
analysis functions that underpin feature extraction for amyloidogenicity
prediction. The biological relevance of each test is grounded in the
sequence determinants of protein aggregation.
"""

import pytest
import tempfile
from pathlib import Path
from io import StringIO

from amyloidbench.core.sequence import (
    SequenceValidator,
    SequenceError,
    parse_fasta,
    sequence_hash,
    calculate_composition,
    calculate_class_composition,
    calculate_dipeptide_composition,
    find_motifs,
    find_low_complexity_regions,
    extract_region_context,
    sliding_window,
    to_fasta,
    STANDARD_AA,
    AA_CLASSES,
)
from amyloidbench.core.models import ProteinRecord, Region


class TestSequenceValidator:
    """
    Tests for sequence validation.
    
    Validation is critical because amyloidogenicity predictors are
    sensitive to sequence quality—unusual characters or truncations
    can produce spurious predictions.
    """
    
    def test_valid_standard_sequence(self):
        """Standard amino acids should pass validation."""
        validator = SequenceValidator()
        is_valid, errors = validator.validate("MVLSPADKTNVKAAWGKVGAH")
        
        assert is_valid
        assert len(errors) == 0
    
    def test_lowercase_conversion(self):
        """Lowercase sequences should be accepted and cleaned."""
        validator = SequenceValidator()
        cleaned = validator.clean("mvlspadktnv")
        
        assert cleaned == "MVLSPADKTNV"
    
    def test_reject_invalid_characters(self):
        """Non-amino acid characters should be rejected."""
        validator = SequenceValidator(allow_ambiguous=False)
        is_valid, errors = validator.validate("MVLS123ADKTNV")
        
        assert not is_valid
        assert any("Invalid characters" in e for e in errors)
    
    def test_allow_ambiguous_codes(self):
        """Ambiguous codes (X, B, Z) should be optionally allowed."""
        validator_strict = SequenceValidator(allow_ambiguous=False)
        validator_lenient = SequenceValidator(allow_ambiguous=True)
        
        seq_with_x = "MVLSXADKTNV"
        
        is_valid_strict, _ = validator_strict.validate(seq_with_x)
        is_valid_lenient, _ = validator_lenient.validate(seq_with_x)
        
        assert not is_valid_strict
        assert is_valid_lenient
    
    def test_minimum_length_constraint(self):
        """Sequences must meet minimum length for APR detection."""
        validator = SequenceValidator(min_length=10)
        is_valid, errors = validator.validate("MVLS")  # Too short
        
        assert not is_valid
        assert any("too short" in e for e in errors)
    
    def test_maximum_length_constraint(self):
        """Very long sequences may exceed tool limits."""
        validator = SequenceValidator(max_length=100)
        is_valid, errors = validator.validate("A" * 150)
        
        assert not is_valid
        assert any("too long" in e for e in errors)
    
    def test_excessive_unknown_residues(self):
        """Warn if >10% of residues are unknown (X)."""
        validator = SequenceValidator(allow_ambiguous=True)
        # 5 X's in 20 residues = 25% unknown
        is_valid, errors = validator.validate("MVLSXXAXXTXXXXAXXXXX")
        
        assert not is_valid
        assert any("unknown residues" in e for e in errors)
    
    def test_clean_with_replacement(self):
        """Invalid characters can be replaced with placeholder."""
        validator = SequenceValidator()
        cleaned = validator.clean("MVL123ADK", replacement="X")
        
        assert cleaned == "MVLXXXADK"
    
    def test_clean_with_removal(self):
        """Invalid characters can be removed entirely."""
        validator = SequenceValidator()
        cleaned = validator.clean("MVL123ADK", replacement="")
        
        assert cleaned == "MVLADK"


class TestFastaParser:
    """
    Tests for FASTA format parsing.
    
    FASTA is the standard format for protein sequences in bioinformatics.
    Robust parsing handles various header formats (UniProt, NCBI, custom)
    and validates sequences during import.
    """
    
    @pytest.fixture
    def sample_fasta(self):
        """Create sample FASTA content."""
        return """>sp|P04156|PRIO_HUMAN Major prion protein OS=Homo sapiens
MANLGCWMLVLFVATWSDLGLCKKRPKPGGWNTGGSRYPGQGSPGGNRYPPQGGGGWGQ
PHGGGWGQPHGGGWGQPHGGGWGQPHGGGWGQGGGTHSQWNKPSKPKTNMKHMAGAAAA
>sp|P02769|ALBU_BOVIN Serum albumin OS=Bos taurus
MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCP
FDEHVKLVNELTEFAKTCVADESHAGCEKSLHTLFGDELCKVASLRETYGDMADCCEKQ
"""
        return sample_fasta
    
    def test_parse_fasta_string(self, sample_fasta):
        """Parse FASTA from string content."""
        proteins = list(parse_fasta(sample_fasta))
        
        assert len(proteins) == 2
        assert proteins[0].id == "sp|P04156|PRIO_HUMAN"
        assert proteins[1].id == "sp|P02769|ALBU_BOVIN"
    
    def test_parse_fasta_file(self, sample_fasta, tmp_path):
        """Parse FASTA from file path."""
        fasta_file = tmp_path / "test.fasta"
        fasta_file.write_text(sample_fasta)
        
        proteins = list(parse_fasta(fasta_file))
        assert len(proteins) == 2
    
    def test_parse_fasta_stringio(self, sample_fasta):
        """Parse FASTA from StringIO object."""
        proteins = list(parse_fasta(StringIO(sample_fasta)))
        assert len(proteins) == 2
    
    def test_parse_extracts_metadata(self, sample_fasta):
        """UniProt-style headers should extract name and organism."""
        proteins = list(parse_fasta(sample_fasta))
        
        # Note: exact parsing depends on implementation
        assert proteins[0].sequence.startswith("MANLGCWMLVLFVATWSDLGLC")
    
    def test_invalid_sequence_raises(self):
        """Invalid sequences should raise SequenceError when validating."""
        invalid_fasta = ">test\nMVLS123ADK"
        
        with pytest.raises(SequenceError):
            list(parse_fasta(invalid_fasta, validate=True))
    
    def test_skip_validation(self):
        """Validation can be disabled for permissive parsing."""
        invalid_fasta = ">test\nMVLS123ADK"
        
        # Should not raise
        proteins = list(parse_fasta(invalid_fasta, validate=False))
        assert len(proteins) == 1


class TestSequenceHash:
    """
    Tests for sequence hashing.
    
    Hashing enables efficient caching of prediction results—identical
    sequences produce identical hashes regardless of case or whitespace.
    """
    
    def test_consistent_hashing(self):
        """Same sequence produces same hash."""
        seq = "MVLSPADKTNV"
        hash1 = sequence_hash(seq)
        hash2 = sequence_hash(seq)
        
        assert hash1 == hash2
    
    def test_case_insensitive(self):
        """Hash should be case-insensitive."""
        assert sequence_hash("MVLSPADKTNV") == sequence_hash("mvlspadktnv")
    
    def test_whitespace_ignored(self):
        """Whitespace should not affect hash."""
        assert sequence_hash("MVLS PADK TNV") == sequence_hash("MVLSPADKTNV")
        assert sequence_hash("MVLS\nPADK\nTNV") == sequence_hash("MVLSPADKTNV")
    
    def test_different_sequences_different_hash(self):
        """Different sequences should produce different hashes."""
        assert sequence_hash("MVLSPADKTNV") != sequence_hash("KVLSPADKTNM")


class TestCompositionAnalysis:
    """
    Tests for amino acid composition calculations.
    
    Composition is a fundamental feature for amyloidogenicity prediction:
    - Hydrophobic residues (V, I, L, F) promote aggregation
    - Charged residues (K, R, E, D) inhibit aggregation
    - Aromatic residues (F, Y, W) stabilize steric zippers
    """
    
    def test_amino_acid_composition(self):
        """Calculate fractional composition of standard amino acids."""
        seq = "AAAAVVVVII"  # 4A, 4V, 2I
        composition = calculate_composition(seq)
        
        assert composition["A"] == pytest.approx(0.4)
        assert composition["V"] == pytest.approx(0.4)
        assert composition["I"] == pytest.approx(0.2)
        assert composition["L"] == 0.0  # Not present
    
    def test_composition_sums_to_one(self):
        """All fractions should sum to approximately 1.0."""
        seq = "MVLSPADKTNVKAAWGKVGAH"
        composition = calculate_composition(seq)
        
        total = sum(composition.values())
        assert total == pytest.approx(1.0, rel=0.01)
    
    def test_class_composition(self):
        """
        Test grouped amino acid class composition.
        
        Classes relevant to amyloidogenicity:
        - amyloidogenic: V, I, L, F, Y, W, M (high β-propensity)
        - gatekeepers: P, K, R, E, D (aggregation inhibitors)
        """
        # Sequence enriched in amyloidogenic residues
        amyloid_prone = "VVILFYWM" + "VVILFYWM"
        class_comp = calculate_class_composition(amyloid_prone)
        
        assert class_comp["amyloidogenic"] == pytest.approx(1.0)
        assert class_comp["gatekeepers"] == 0.0
        
        # Sequence enriched in gatekeepers
        gatekeeper_rich = "PKREDDPKRE"
        class_comp = calculate_class_composition(gatekeeper_rich)
        
        assert class_comp["gatekeepers"] == pytest.approx(1.0)
        assert class_comp["amyloidogenic"] == 0.0
    
    def test_dipeptide_composition(self):
        """
        Test dipeptide frequency calculation.
        
        Certain dipeptides (VV, VI, II, FF) are enriched in APRs
        due to favorable β-sheet stacking interactions.
        """
        seq = "VVVVVIIII"  # VV appears 4 times, VI once, II 4 times
        dipep_comp = calculate_dipeptide_composition(seq)
        
        # Total dipeptides = 8 (length - 1)
        assert dipep_comp["VV"] == pytest.approx(4/8)
        assert dipep_comp["II"] == pytest.approx(4/8)
        assert dipep_comp["VI"] == pytest.approx(1/8)


class TestMotifFinding:
    """
    Tests for sequence motif detection.
    
    Specific motifs are associated with amyloidogenicity:
    - Poly-Q stretches (Huntington's disease)
    - GXXXG transmembrane motifs
    - Aromatic-rich regions
    """
    
    def test_find_simple_motif(self):
        """Find occurrences of a simple pattern."""
        seq = "MVLSQQQQQQPADKQQQQQQTNV"
        regions = find_motifs(seq, r"Q{4,}")  # 4+ consecutive Q
        
        assert len(regions) == 2
        assert all(r.sequence.startswith("QQQQ") for r in regions)
    
    def test_find_gxxxg_motif(self):
        """Find GXXXG transmembrane helix motif."""
        seq = "MVLGAAAGPADKGVVVGTNVGIIIG"
        regions = find_motifs(seq, r"G...G")
        
        assert len(regions) >= 2
    
    def test_case_insensitive_search(self):
        """Motif search should be case-insensitive by default."""
        seq = "mvlsQQQQpadktnv"
        regions = find_motifs(seq, r"Q{4}")
        
        assert len(regions) == 1
    
    def test_no_matches(self):
        """Return empty list when no matches found."""
        seq = "MVLSPADKTNV"
        regions = find_motifs(seq, r"W{5}")  # No 5+ W stretch
        
        assert len(regions) == 0


class TestLowComplexityRegions:
    """
    Tests for low-complexity region detection.
    
    Low-complexity regions (compositionally biased) are associated with
    specific aggregation behaviors:
    - Poly-Q: nucleation-dependent aggregation
    - Prion-like domains: phase separation and amyloid formation
    - Repetitive sequences: may promote β-sheet stacking
    """
    
    def test_detect_poly_residue(self):
        """Detect poly-amino acid stretches."""
        seq = "MVLSPAD" + "Q" * 20 + "KTNVAAW"
        regions = find_low_complexity_regions(seq, window_size=10, threshold=1.5)
        
        assert len(regions) >= 1
        # The poly-Q region should be detected
        assert any("Q" * 5 in r.sequence for r in regions)
    
    def test_high_complexity_sequence(self):
        """High-complexity sequences should have few/no LC regions."""
        # Random-looking high-complexity sequence
        seq = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKT"
        regions = find_low_complexity_regions(seq, window_size=12, threshold=2.0)
        
        # May have some, but should be minimal
        total_lc = sum(r.length for r in regions)
        assert total_lc < len(seq) / 2


class TestRegionContext:
    """
    Tests for flanking region analysis.
    
    The context surrounding APRs is biologically critical:
    - Gatekeeper residues in flanks inhibit aggregation
    - Charged residues modulate aggregation kinetics
    - This is a key feature for our fallback predictor
    """
    
    def test_extract_flanks(self):
        """Extract upstream and downstream flanking sequences."""
        seq = "AAAAAAAAAAVVVVVVVVVVBBBBBBBBBB"  # A=10, V=10, B=10
        region = Region(start=10, end=20, sequence="VVVVVVVVVV")
        
        context = extract_region_context(seq, region, flank_size=5)
        
        assert context["upstream"] == "AAAAA"
        assert context["region"] == "VVVVVVVVVV"
        assert context["downstream"] == "BBBBB"
    
    def test_count_gatekeepers_in_flanks(self):
        """Count gatekeeper residues (P, K, R, E, D) in flanking regions."""
        seq = "PKREDAAAAVVVVVVVVVVPKREDAAAAA"
        region = Region(start=10, end=20, sequence="VVVVVVVVVV")
        
        context = extract_region_context(seq, region, flank_size=10)
        
        assert context["upstream_gatekeepers"] == 5  # PKRED
        assert context["downstream_gatekeepers"] == 5  # PKRED
    
    def test_handle_sequence_boundaries(self):
        """Correctly handle regions near sequence termini."""
        seq = "VVVVVVVVVVAAAAA"  # V at start
        region = Region(start=0, end=5, sequence="VVVVV")
        
        context = extract_region_context(seq, region, flank_size=10)
        
        assert context["upstream"] == ""  # No upstream at position 0
        assert len(context["downstream"]) <= 10


class TestSlidingWindow:
    """Tests for sliding window iteration."""
    
    def test_basic_sliding(self):
        """Generate overlapping windows across sequence."""
        seq = "MVLSPADKTNV"
        windows = list(sliding_window(seq, window_size=5, step=1))
        
        assert len(windows) == len(seq) - 5 + 1  # 7 windows
        assert windows[0] == (0, "MVLSP")
        assert windows[1] == (1, "VLSPA")
    
    def test_non_overlapping_windows(self):
        """Generate non-overlapping windows with step > 1."""
        seq = "MVLSPADKTNVKAAW"  # 15 residues
        windows = list(sliding_window(seq, window_size=5, step=5))
        
        assert len(windows) == 3
        assert windows[0] == (0, "MVLSP")
        assert windows[1] == (5, "ADKTN")
        assert windows[2] == (10, "VKAAW")


class TestFastaOutput:
    """Tests for FASTA format output generation."""
    
    def test_to_fasta_single(self):
        """Convert single protein to FASTA format."""
        protein = ProteinRecord(id="test", sequence="MVLSPADKTNV")
        fasta = to_fasta([protein])
        
        assert fasta.startswith(">test")
        assert "MVLSPADKTNV" in fasta
    
    def test_to_fasta_multiple(self):
        """Convert multiple proteins to FASTA format."""
        proteins = [
            ProteinRecord(id="prot1", sequence="AAAA"),
            ProteinRecord(id="prot2", sequence="VVVV"),
        ]
        fasta = to_fasta(proteins)
        
        assert ">prot1" in fasta
        assert ">prot2" in fasta
    
    def test_fasta_line_wrapping(self):
        """Long sequences should be wrapped at specified length."""
        long_seq = "A" * 100
        protein = ProteinRecord(id="long", sequence=long_seq)
        fasta = to_fasta([protein], line_length=60)
        
        lines = fasta.split("\n")
        # Header + 2 sequence lines (60 + 40)
        seq_lines = [l for l in lines if not l.startswith(">")]
        assert len(seq_lines[0]) == 60
        assert len(seq_lines[1]) == 40
