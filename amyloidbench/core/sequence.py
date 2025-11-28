"""
Sequence handling utilities for AmyloidBench.

This module provides comprehensive tools for working with protein sequences,
including validation, parsing from various formats, and sequence manipulation.
Proper sequence handling is critical for amyloidogenicity prediction as many
algorithms are sensitive to sequence quality and format.
"""

from __future__ import annotations

import hashlib
import re
from io import StringIO
from pathlib import Path
from typing import Iterator, Optional, Union

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from .models import ProteinRecord, Region


# Standard amino acid alphabet
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Extended alphabet including ambiguous codes
EXTENDED_AA = STANDARD_AA | set("BXZJUO*-")

# Amino acid classification for amyloidogenicity analysis
# Based on Galzitskaya et al. and cross-β propensity studies
AA_CLASSES = {
    "hydrophobic_aliphatic": set("AVILM"),
    "hydrophobic_aromatic": set("FYW"),
    "polar_uncharged": set("STNQ"),
    "positively_charged": set("KRH"),
    "negatively_charged": set("DE"),
    "special": set("CGP"),
    "amyloidogenic": set("VILFYWM"),  # High β-sheet propensity
    "gatekeepers": set("PKRED"),  # Aggregation-inhibiting residues
    "disorder_promoting": set("AGSEKRPD"),  # Intrinsically disordered
}

# Single-letter to three-letter code mapping
AA_1TO3 = {
    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
    'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
    'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
    'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
}
AA_3TO1 = {v: k for k, v in AA_1TO3.items()}


class SequenceError(Exception):
    """Exception raised for sequence-related errors."""
    pass


class SequenceValidator:
    """
    Validates protein sequences for use in amyloidogenicity prediction.
    
    Different predictors have different requirements regarding sequence
    length, allowed characters, and format. This validator ensures
    sequences meet the minimum requirements before processing.
    """
    
    MIN_LENGTH = 5  # Minimum for meaningful APR prediction
    MAX_LENGTH = 10000  # Practical limit for most web tools
    
    def __init__(
        self,
        allow_ambiguous: bool = False,
        allow_gaps: bool = False,
        min_length: int = MIN_LENGTH,
        max_length: int = MAX_LENGTH,
    ):
        """
        Initialize validator with specific constraints.
        
        Args:
            allow_ambiguous: Allow ambiguous amino acid codes (B, X, Z, J, O, U)
            allow_gaps: Allow gap characters (-, *)
            min_length: Minimum sequence length
            max_length: Maximum sequence length
        """
        self.allow_ambiguous = allow_ambiguous
        self.allow_gaps = allow_gaps
        self.min_length = min_length
        self.max_length = max_length
        
        self.allowed_chars = set(STANDARD_AA)
        if allow_ambiguous:
            self.allowed_chars |= set("BXZJUO")
        if allow_gaps:
            self.allowed_chars |= set("-*")
    
    def validate(self, sequence: str) -> tuple[bool, list[str]]:
        """
        Validate a sequence and return status with error messages.
        
        Args:
            sequence: Protein sequence to validate
        
        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []
        
        # Clean sequence
        seq = sequence.upper().replace(" ", "").replace("\n", "")
        
        # Check length
        if len(seq) < self.min_length:
            errors.append(f"Sequence too short: {len(seq)} < {self.min_length}")
        
        if len(seq) > self.max_length:
            errors.append(f"Sequence too long: {len(seq)} > {self.max_length}")
        
        # Check characters
        invalid_chars = set(seq) - self.allowed_chars
        if invalid_chars:
            errors.append(f"Invalid characters: {sorted(invalid_chars)}")
        
        # Check for suspicious patterns
        if seq.count("X") / max(len(seq), 1) > 0.1:
            errors.append("More than 10% unknown residues (X)")
        
        return len(errors) == 0, errors
    
    def clean(self, sequence: str, replacement: str = "") -> str:
        """
        Clean a sequence by removing or replacing invalid characters.
        
        Args:
            sequence: Input sequence
            replacement: Character to replace invalid chars (empty to remove)
        
        Returns:
            Cleaned sequence
        """
        seq = sequence.upper().replace(" ", "").replace("\n", "")
        
        cleaned = []
        for char in seq:
            if char in self.allowed_chars:
                cleaned.append(char)
            elif replacement:
                cleaned.append(replacement)
        
        return "".join(cleaned)


def parse_fasta(
    source: Union[str, Path, StringIO],
    validate: bool = True,
    validator: Optional[SequenceValidator] = None,
) -> Iterator[ProteinRecord]:
    """
    Parse protein sequences from FASTA format.
    
    Args:
        source: File path, FASTA string, or StringIO object
        validate: Whether to validate sequences
        validator: Custom validator (uses default if None)
    
    Yields:
        ProteinRecord objects for each sequence
    
    Raises:
        SequenceError: If validation fails and validate=True
    """
    if validator is None:
        validator = SequenceValidator(allow_ambiguous=True)
    
    # Handle different input types
    if isinstance(source, str):
        if source.startswith(">") or "\n>" in source:
            # It's a FASTA string
            handle = StringIO(source)
        else:
            # It's a file path
            handle = open(source, "r")
    elif isinstance(source, Path):
        handle = open(source, "r")
    else:
        handle = source
    
    try:
        for record in SeqIO.parse(handle, "fasta"):
            seq_str = str(record.seq)
            
            if validate:
                is_valid, errors = validator.validate(seq_str)
                if not is_valid:
                    raise SequenceError(
                        f"Sequence '{record.id}' failed validation: {'; '.join(errors)}"
                    )
            
            # Parse description for additional metadata
            description_parts = record.description.split("|")
            name = None
            organism = None
            
            if len(description_parts) > 1:
                # UniProt-style header: sp|P12345|NAME_ORGANISM
                if len(description_parts) >= 3:
                    name_org = description_parts[2].split(" ", 1)
                    if "_" in name_org[0]:
                        name, organism = name_org[0].rsplit("_", 1)
            
            yield ProteinRecord(
                id=record.id,
                name=name,
                organism=organism,
                sequence=validator.clean(seq_str) if validate else seq_str,
            )
    finally:
        if isinstance(source, (str, Path)) and not isinstance(source, str) or (
            isinstance(source, str) and not source.startswith(">")
        ):
            handle.close()


def sequence_hash(sequence: str) -> str:
    """
    Generate a unique hash for a sequence.
    
    Useful for caching prediction results and identifying duplicate sequences.
    Uses MD5 for speed (not cryptographic security).
    
    Args:
        sequence: Protein sequence
    
    Returns:
        Hexadecimal hash string
    """
    seq_clean = sequence.upper().replace(" ", "").replace("\n", "")
    return hashlib.md5(seq_clean.encode()).hexdigest()


def calculate_composition(sequence: str) -> dict[str, float]:
    """
    Calculate amino acid composition as fractions.
    
    Args:
        sequence: Protein sequence
    
    Returns:
        Dictionary mapping amino acids to their fractional content
    """
    seq = sequence.upper()
    total = len(seq)
    
    if total == 0:
        return {aa: 0.0 for aa in STANDARD_AA}
    
    composition = {}
    for aa in STANDARD_AA:
        composition[aa] = seq.count(aa) / total
    
    return composition


def calculate_class_composition(sequence: str) -> dict[str, float]:
    """
    Calculate amino acid class composition for amyloidogenicity assessment.
    
    Returns fractions for:
    - hydrophobic_aliphatic: A, V, I, L, M
    - hydrophobic_aromatic: F, Y, W (important for steric zipper)
    - polar_uncharged: S, T, N, Q
    - positively_charged: K, R, H
    - negatively_charged: D, E
    - special: C, G, P (structure-disrupting)
    - amyloidogenic: V, I, L, F, Y, W, M
    - gatekeepers: P, K, R, E, D (aggregation inhibitors)
    
    Args:
        sequence: Protein sequence
    
    Returns:
        Dictionary with class names and their fractional content
    """
    seq = sequence.upper()
    total = len(seq)
    
    if total == 0:
        return {name: 0.0 for name in AA_CLASSES}
    
    result = {}
    for class_name, aa_set in AA_CLASSES.items():
        count = sum(1 for aa in seq if aa in aa_set)
        result[class_name] = count / total
    
    return result


def calculate_dipeptide_composition(sequence: str) -> dict[str, float]:
    """
    Calculate dipeptide composition.
    
    Dipeptide frequencies capture local sequence patterns that are
    important for aggregation propensity. Certain dipeptides (VV, VI, II)
    are enriched in amyloidogenic regions.
    
    Args:
        sequence: Protein sequence
    
    Returns:
        Dictionary mapping dipeptides to their fractional content
    """
    seq = sequence.upper()
    total = max(len(seq) - 1, 1)
    
    composition = {}
    for aa1 in STANDARD_AA:
        for aa2 in STANDARD_AA:
            dipeptide = aa1 + aa2
            composition[dipeptide] = seq.count(dipeptide) / total
    
    return composition


def find_motifs(
    sequence: str,
    motif_pattern: str,
    case_sensitive: bool = False,
) -> list[Region]:
    """
    Find all occurrences of a motif pattern in a sequence.
    
    Args:
        sequence: Protein sequence
        motif_pattern: Regular expression pattern for the motif
        case_sensitive: Whether the search is case-sensitive
    
    Returns:
        List of Region objects for each match
    """
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(motif_pattern, flags)
    
    regions = []
    for match in pattern.finditer(sequence):
        regions.append(Region(
            start=match.start(),
            end=match.end(),
            sequence=match.group(),
            annotation=f"motif:{motif_pattern}",
        ))
    
    return regions


def find_low_complexity_regions(
    sequence: str,
    window_size: int = 12,
    threshold: float = 2.0,
) -> list[Region]:
    """
    Identify low-complexity regions using Shannon entropy.
    
    Low-complexity regions can influence aggregation behavior:
    - Poly-Q expansions are associated with amyloid formation
    - Repetitive sequences may promote β-sheet stacking
    
    Args:
        sequence: Protein sequence
        window_size: Size of sliding window
        threshold: Entropy threshold (below = low complexity)
    
    Returns:
        List of Region objects for low-complexity segments
    """
    import math
    
    def shannon_entropy(subseq: str) -> float:
        """Calculate Shannon entropy of a sequence."""
        freq = {}
        for aa in subseq:
            freq[aa] = freq.get(aa, 0) + 1
        
        total = len(subseq)
        entropy = 0.0
        for count in freq.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    seq = sequence.upper()
    regions = []
    in_region = False
    start = 0
    
    for i in range(len(seq) - window_size + 1):
        window = seq[i:i + window_size]
        entropy = shannon_entropy(window)
        
        if entropy < threshold and not in_region:
            in_region = True
            start = i
        elif entropy >= threshold and in_region:
            in_region = False
            regions.append(Region(
                start=start,
                end=i + window_size - 1,
                sequence=seq[start:i + window_size - 1],
                score=entropy,
                annotation="low_complexity",
            ))
    
    if in_region:
        regions.append(Region(
            start=start,
            end=len(seq),
            sequence=seq[start:],
            score=shannon_entropy(seq[start:]),
            annotation="low_complexity",
        ))
    
    return regions


def extract_region_context(
    sequence: str,
    region: Region,
    flank_size: int = 10,
) -> dict[str, str]:
    """
    Extract a region with its flanking sequence context.
    
    Flanking residues are important for amyloidogenicity prediction:
    - Gatekeeper residues (P, K, R, E, D) in flanks inhibit aggregation
    - Charged residues near APRs modulate aggregation kinetics
    
    Args:
        sequence: Full protein sequence
        region: Region of interest
        flank_size: Number of flanking residues on each side
    
    Returns:
        Dictionary with 'upstream', 'region', 'downstream', and 'full' sequences
    """
    upstream_start = max(0, region.start - flank_size)
    downstream_end = min(len(sequence), region.end + flank_size)
    
    return {
        "upstream": sequence[upstream_start:region.start],
        "region": sequence[region.start:region.end],
        "downstream": sequence[region.end:downstream_end],
        "full": sequence[upstream_start:downstream_end],
        "upstream_gatekeepers": sum(1 for aa in sequence[upstream_start:region.start] 
                                     if aa in AA_CLASSES["gatekeepers"]),
        "downstream_gatekeepers": sum(1 for aa in sequence[region.end:downstream_end] 
                                       if aa in AA_CLASSES["gatekeepers"]),
    }


def sliding_window(
    sequence: str,
    window_size: int,
    step: int = 1,
) -> Iterator[tuple[int, str]]:
    """
    Generate sliding windows over a sequence.
    
    Args:
        sequence: Input sequence
        window_size: Size of each window
        step: Step size between windows
    
    Yields:
        Tuple of (start_position, window_sequence)
    """
    for i in range(0, len(sequence) - window_size + 1, step):
        yield i, sequence[i:i + window_size]


def to_fasta(
    records: list[ProteinRecord],
    line_length: int = 60,
) -> str:
    """
    Convert ProteinRecord objects to FASTA format string.
    
    Args:
        records: List of ProteinRecord objects
        line_length: Characters per line for sequence
    
    Returns:
        FASTA-formatted string
    """
    lines = []
    for record in records:
        # Build header
        header_parts = [record.id]
        if record.name:
            header_parts.append(record.name)
        if record.organism:
            header_parts.append(f"OS={record.organism}")
        
        lines.append(f">{' '.join(header_parts)}")
        
        # Split sequence into lines
        seq = record.sequence
        for i in range(0, len(seq), line_length):
            lines.append(seq[i:i + line_length])
    
    return "\n".join(lines)
