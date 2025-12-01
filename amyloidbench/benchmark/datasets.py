"""
Database loaders for amyloidogenicity benchmarking.

This module provides loaders for curated experimental databases used to
validate amyloid prediction algorithms. Each database has different
characteristics and biases that should be considered when interpreting
benchmark results.

Supported Databases
-------------------

**Cross-Beta DB** (crossbeta.com)
Repository of experimentally validated cross-β structures from cryo-EM
and X-ray crystallography. Provides:
- Sequence-to-structure mapping
- Polymorph classifications
- Resolution and method information
Best for: Structural validation, polymorph classification benchmarking

**WALTZ-DB** (waltz-db.switchlab.org)
Curated database of hexapeptide aggregation propensities determined by
ThT fluorescence and electron microscopy. Provides:
- Binary aggregation labels (aggregating/non-aggregating)
- Quantitative ThT binding data
- Experimental conditions
Best for: Sequence-based predictor validation, threshold calibration

**AmyPro** (amypro.net)
Database of functional and pathological amyloid-forming proteins. Provides:
- Full protein sequences with annotated APRs
- Functional classification
- Disease associations
Best for: Full-length protein prediction, biological context

**Local/Custom Datasets**
Support for user-provided datasets in standardized formats (FASTA, CSV, TSV).

Data Format
-----------
All loaders return standardized BenchmarkDataset objects containing:
- Sequences with known amyloid status
- Per-residue labels where available
- Metadata (source, experimental method, conditions)

References
----------
- Cross-Beta DB: Sawaya et al. (2021), Amyloid Atlas
- WALTZ-DB: Louros et al. (2020), WALTZ-DB 2.0
- AmyPro: Varadi et al. (2018), AmyPro
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Optional, Union
from urllib.parse import urljoin

import numpy as np

from amyloidbench.core.models import ProteinRecord, Region

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

class AmyloidStatus(str, Enum):
    """Experimental amyloid classification."""
    POSITIVE = "positive"        # Forms amyloid
    NEGATIVE = "negative"        # Does not form amyloid
    AMBIGUOUS = "ambiguous"      # Conflicting evidence
    UNKNOWN = "unknown"          # Not tested


class ExperimentalMethod(str, Enum):
    """Method used to determine amyloid status."""
    CRYO_EM = "cryo_em"
    XRAY = "xray"
    NMR = "nmr"
    THT_FLUORESCENCE = "tht"
    ELECTRON_MICROSCOPY = "em"
    CONGO_RED = "congo_red"
    FTIR = "ftir"
    CD = "cd"
    SEEDING = "seeding"
    COMPUTATIONAL = "computational"
    UNKNOWN = "unknown"


@dataclass
class BenchmarkEntry:
    """
    Single entry in a benchmark dataset.
    
    Attributes:
        id: Unique identifier
        sequence: Amino acid sequence
        amyloid_status: Experimental classification
        amyloid_regions: Known APR regions (if available)
        residue_labels: Per-residue binary labels (if available)
        experimental_method: How status was determined
        source_database: Origin database
        metadata: Additional information (PDB ID, conditions, etc.)
    """
    id: str
    sequence: str
    amyloid_status: AmyloidStatus
    amyloid_regions: list[Region] = field(default_factory=list)
    residue_labels: Optional[list[bool]] = None
    experimental_method: ExperimentalMethod = ExperimentalMethod.UNKNOWN
    source_database: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_protein_record(self) -> ProteinRecord:
        """Convert to ProteinRecord for prediction."""
        return ProteinRecord(
            id=self.id,
            sequence=self.sequence,
            known_amyloid_regions=self.amyloid_regions,
            is_known_amyloid=self.amyloid_status == AmyloidStatus.POSITIVE,
        )
    
    @property
    def is_positive(self) -> bool:
        """Whether this is a positive (amyloid-forming) example."""
        return self.amyloid_status == AmyloidStatus.POSITIVE
    
    @property
    def is_negative(self) -> bool:
        """Whether this is a negative (non-amyloid) example."""
        return self.amyloid_status == AmyloidStatus.NEGATIVE
    
    @property
    def has_residue_labels(self) -> bool:
        """Whether per-residue labels are available."""
        return self.residue_labels is not None


@dataclass
class BenchmarkDataset:
    """
    Collection of benchmark entries from a single source.
    
    Attributes:
        name: Dataset name
        description: Brief description
        entries: List of benchmark entries
        source_url: Original data source
        citation: Publication reference
        version: Dataset version
    """
    name: str
    description: str
    entries: list[BenchmarkEntry]
    source_url: str = ""
    citation: str = ""
    version: str = "1.0"
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __iter__(self) -> Iterator[BenchmarkEntry]:
        return iter(self.entries)
    
    def __getitem__(self, idx: int) -> BenchmarkEntry:
        return self.entries[idx]
    
    @property
    def n_positive(self) -> int:
        """Number of positive (amyloid-forming) entries."""
        return sum(1 for e in self.entries if e.is_positive)
    
    @property
    def n_negative(self) -> int:
        """Number of negative entries."""
        return sum(1 for e in self.entries if e.is_negative)
    
    @property
    def n_with_residue_labels(self) -> int:
        """Number of entries with per-residue labels."""
        return sum(1 for e in self.entries if e.has_residue_labels)
    
    def filter_by_status(self, status: AmyloidStatus) -> BenchmarkDataset:
        """Return subset with specific amyloid status."""
        filtered = [e for e in self.entries if e.amyloid_status == status]
        return BenchmarkDataset(
            name=f"{self.name}_{status.value}",
            description=f"{self.description} (filtered: {status.value})",
            entries=filtered,
            source_url=self.source_url,
            citation=self.citation,
            version=self.version,
        )
    
    def filter_by_length(
        self, 
        min_length: int = 0, 
        max_length: int = float('inf')
    ) -> BenchmarkDataset:
        """Return subset within length range."""
        filtered = [
            e for e in self.entries 
            if min_length <= len(e.sequence) <= max_length
        ]
        return BenchmarkDataset(
            name=f"{self.name}_len{min_length}-{max_length}",
            description=f"{self.description} (length {min_length}-{max_length})",
            entries=filtered,
            source_url=self.source_url,
            citation=self.citation,
            version=self.version,
        )
    
    def get_positive_negative_split(self) -> tuple[list[BenchmarkEntry], list[BenchmarkEntry]]:
        """Split into positive and negative entries."""
        positives = [e for e in self.entries if e.is_positive]
        negatives = [e for e in self.entries if e.is_negative]
        return positives, negatives
    
    def summary(self) -> str:
        """Generate dataset summary."""
        lines = [
            f"Dataset: {self.name}",
            f"  Description: {self.description}",
            f"  Entries: {len(self.entries)}",
            f"  Positive: {self.n_positive}",
            f"  Negative: {self.n_negative}",
            f"  With residue labels: {self.n_with_residue_labels}",
        ]
        if self.citation:
            lines.append(f"  Citation: {self.citation}")
        return "\n".join(lines)


# =============================================================================
# Database Loaders
# =============================================================================

class WaltzDBLoader:
    """
    Loader for WALTZ-DB hexapeptide aggregation database.
    
    WALTZ-DB contains experimentally validated hexapeptide sequences
    tested for amyloid formation using ThT fluorescence and EM.
    
    The database is particularly useful for:
    - Validating sequence-based predictors
    - Calibrating prediction thresholds
    - Understanding short-range aggregation determinants
    
    Note: WALTZ-DB requires registration for download. This loader
    can parse the downloaded TSV/CSV files.
    """
    
    SOURCE_URL = "https://waltz-db.switchlab.org/"
    CITATION = "Louros et al. (2020) WALTZ-DB 2.0. Nucleic Acids Res."
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize WALTZ-DB loader.
        
        Args:
            data_path: Path to downloaded WALTZ-DB file
        """
        self.data_path = Path(data_path) if data_path else None
    
    def load(self, path: Optional[Path] = None) -> BenchmarkDataset:
        """
        Load WALTZ-DB from file.
        
        Args:
            path: Path to TSV/CSV file (overrides constructor path)
            
        Returns:
            BenchmarkDataset with hexapeptide entries
        """
        path = Path(path) if path else self.data_path
        
        if path is None:
            raise ValueError("No data path provided. Download WALTZ-DB from waltz-db.switchlab.org")
        
        if not path.exists():
            raise FileNotFoundError(f"WALTZ-DB file not found: {path}")
        
        entries = []
        
        # Detect delimiter
        with open(path, 'r') as f:
            first_line = f.readline()
            delimiter = '\t' if '\t' in first_line else ','
        
        with open(path, 'r') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            
            for row in reader:
                entry = self._parse_row(row)
                if entry:
                    entries.append(entry)
        
        return BenchmarkDataset(
            name="WALTZ-DB",
            description="Experimentally validated hexapeptide aggregation",
            entries=entries,
            source_url=self.SOURCE_URL,
            citation=self.CITATION,
        )
    
    def _parse_row(self, row: dict) -> Optional[BenchmarkEntry]:
        """Parse a single WALTZ-DB row."""
        # Handle various column naming conventions
        sequence = row.get('sequence') or row.get('Sequence') or row.get('peptide')
        
        if not sequence or len(sequence) < 3:
            return None
        
        # Parse amyloid status
        status_str = (
            row.get('amyloid') or 
            row.get('Amyloid') or 
            row.get('aggregation') or
            row.get('class') or
            ''
        ).lower()
        
        if status_str in ['yes', 'true', '1', 'positive', 'amyloid', 'aggregating']:
            status = AmyloidStatus.POSITIVE
        elif status_str in ['no', 'false', '0', 'negative', 'non-amyloid', 'non-aggregating']:
            status = AmyloidStatus.NEGATIVE
        else:
            status = AmyloidStatus.UNKNOWN
        
        # Parse experimental method
        method_str = row.get('method', '').lower()
        if 'tht' in method_str or 'thioflavin' in method_str:
            method = ExperimentalMethod.THT_FLUORESCENCE
        elif 'em' in method_str or 'electron' in method_str:
            method = ExperimentalMethod.ELECTRON_MICROSCOPY
        else:
            method = ExperimentalMethod.UNKNOWN
        
        # Build metadata
        metadata = {}
        for key in ['source', 'reference', 'conditions', 'pH', 'temperature']:
            if key in row and row[key]:
                metadata[key] = row[key]
        
        entry_id = row.get('id') or row.get('ID') or f"WALTZ_{sequence}"
        
        return BenchmarkEntry(
            id=entry_id,
            sequence=sequence.upper(),
            amyloid_status=status,
            experimental_method=method,
            source_database="WALTZ-DB",
            metadata=metadata,
        )
    
    @staticmethod
    def create_synthetic_dataset(n_positive: int = 100, n_negative: int = 100) -> BenchmarkDataset:
        """
        Create a synthetic WALTZ-like dataset for testing.
        
        Uses known amyloidogenic motifs for positives and
        charge-rich/proline-containing sequences for negatives.
        """
        import random
        random.seed(42)
        
        entries = []
        
        # Positive examples (amyloid-like hexapeptides)
        positive_templates = [
            "KLVFFA", "VQIVYK", "GNNQQN", "NFGAIL", "GGVVIA",
            "VVIITV", "AIIGLM", "FGAILS", "VVIAIV", "ILMVGG",
        ]
        
        hydrophobic = "VILFAM"
        for i in range(n_positive):
            if i < len(positive_templates):
                seq = positive_templates[i]
            else:
                # Generate random hydrophobic hexapeptide
                seq = ''.join(random.choices(hydrophobic + "GS", k=6))
            
            entries.append(BenchmarkEntry(
                id=f"WALTZ_POS_{i:04d}",
                sequence=seq,
                amyloid_status=AmyloidStatus.POSITIVE,
                experimental_method=ExperimentalMethod.THT_FLUORESCENCE,
                source_database="WALTZ-DB_synthetic",
            ))
        
        # Negative examples (non-aggregating)
        negative_templates = [
            "KKKDDD", "RRREEE", "PPGPPG", "DKDKDK", "EREDRE",
            "PKPKPK", "DGDGDG", "KEKEKE", "RDRDRR", "PPPGGG",
        ]
        
        charged = "KRDEP"
        for i in range(n_negative):
            if i < len(negative_templates):
                seq = negative_templates[i]
            else:
                # Generate charge-rich/proline hexapeptide
                seq = ''.join(random.choices(charged + "G", k=6))
            
            entries.append(BenchmarkEntry(
                id=f"WALTZ_NEG_{i:04d}",
                sequence=seq,
                amyloid_status=AmyloidStatus.NEGATIVE,
                experimental_method=ExperimentalMethod.THT_FLUORESCENCE,
                source_database="WALTZ-DB_synthetic",
            ))
        
        return BenchmarkDataset(
            name="WALTZ-DB_synthetic",
            description="Synthetic hexapeptide dataset based on WALTZ-DB patterns",
            entries=entries,
            source_url="",
            citation="Synthetic dataset for testing",
        )


class CrossBetaDBLoader:
    """
    Loader for Cross-Beta DB structural database.
    
    Cross-Beta DB catalogs experimentally determined amyloid fibril
    structures from cryo-EM and X-ray crystallography, providing:
    - Atomic coordinates (PDB format)
    - Sequence-structure mapping
    - Polymorph classification
    - Resolution and quality metrics
    
    The database is ideal for:
    - Structure-based predictor validation
    - Polymorph classification benchmarking
    - Understanding sequence-structure relationships
    """
    
    SOURCE_URL = "https://crossbeta.org/"
    CITATION = "Sawaya et al. (2021) Amyloid Atlas. Nature Struct. Mol. Biol."
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize Cross-Beta DB loader.
        
        Args:
            data_dir: Directory containing downloaded Cross-Beta DB files
        """
        self.data_dir = Path(data_dir) if data_dir else None
    
    def load(self, path: Optional[Path] = None) -> BenchmarkDataset:
        """
        Load Cross-Beta DB from directory or file.
        
        Supports multiple formats:
        - JSON export from crossbeta.org
        - CSV/TSV summary files
        - Directory of PDB files
        """
        path = Path(path) if path else self.data_dir
        
        if path is None:
            raise ValueError("No data path provided")
        
        if path.is_file():
            if path.suffix == '.json':
                return self._load_json(path)
            elif path.suffix in ['.csv', '.tsv']:
                return self._load_csv(path)
        elif path.is_dir():
            return self._load_directory(path)
        
        raise ValueError(f"Unsupported path type: {path}")
    
    def _load_json(self, path: Path) -> BenchmarkDataset:
        """Load from JSON export."""
        with open(path) as f:
            data = json.load(f)
        
        entries = []
        for item in data.get('entries', data if isinstance(data, list) else []):
            entry = self._parse_json_entry(item)
            if entry:
                entries.append(entry)
        
        return BenchmarkDataset(
            name="Cross-Beta DB",
            description="Experimentally determined amyloid fibril structures",
            entries=entries,
            source_url=self.SOURCE_URL,
            citation=self.CITATION,
        )
    
    def _parse_json_entry(self, item: dict) -> Optional[BenchmarkEntry]:
        """Parse a single Cross-Beta DB JSON entry."""
        sequence = item.get('sequence', '')
        if not sequence:
            return None
        
        # Parse experimental method
        method_str = item.get('method', '').lower()
        if 'cryo' in method_str or 'em' in method_str:
            method = ExperimentalMethod.CRYO_EM
        elif 'xray' in method_str or 'x-ray' in method_str:
            method = ExperimentalMethod.XRAY
        elif 'nmr' in method_str:
            method = ExperimentalMethod.NMR
        else:
            method = ExperimentalMethod.UNKNOWN
        
        # Parse APR regions if available
        regions = []
        if 'amyloid_regions' in item:
            for r in item['amyloid_regions']:
                regions.append(Region(
                    start=r.get('start', 0),
                    end=r.get('end', len(sequence)),
                    sequence=r.get('sequence', ''),
                ))
        
        # Generate residue labels from regions
        residue_labels = None
        if regions:
            residue_labels = [False] * len(sequence)
            for r in regions:
                for i in range(r.start, min(r.end, len(sequence))):
                    residue_labels[i] = True
        
        metadata = {
            'pdb_id': item.get('pdb_id'),
            'resolution': item.get('resolution'),
            'polymorph': item.get('polymorph'),
            'organism': item.get('organism'),
        }
        
        return BenchmarkEntry(
            id=item.get('id', item.get('pdb_id', '')),
            sequence=sequence.upper(),
            amyloid_status=AmyloidStatus.POSITIVE,  # All entries are amyloid
            amyloid_regions=regions,
            residue_labels=residue_labels,
            experimental_method=method,
            source_database="Cross-Beta DB",
            metadata={k: v for k, v in metadata.items() if v is not None},
        )
    
    def _load_csv(self, path: Path) -> BenchmarkDataset:
        """Load from CSV/TSV file."""
        delimiter = '\t' if path.suffix == '.tsv' else ','
        
        entries = []
        with open(path) as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                entry = self._parse_csv_row(row)
                if entry:
                    entries.append(entry)
        
        return BenchmarkDataset(
            name="Cross-Beta DB",
            description="Experimentally determined amyloid structures",
            entries=entries,
            source_url=self.SOURCE_URL,
            citation=self.CITATION,
        )
    
    def _parse_csv_row(self, row: dict) -> Optional[BenchmarkEntry]:
        """Parse CSV row."""
        sequence = row.get('sequence', row.get('Sequence', ''))
        if not sequence:
            return None
        
        return BenchmarkEntry(
            id=row.get('id', row.get('pdb_id', '')),
            sequence=sequence.upper(),
            amyloid_status=AmyloidStatus.POSITIVE,
            source_database="Cross-Beta DB",
            metadata=dict(row),
        )
    
    def _load_directory(self, path: Path) -> BenchmarkDataset:
        """Load from directory of files."""
        entries = []
        
        # Look for JSON files
        for json_file in path.glob("*.json"):
            try:
                ds = self._load_json(json_file)
                entries.extend(ds.entries)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
        
        return BenchmarkDataset(
            name="Cross-Beta DB",
            description="Experimentally determined amyloid structures",
            entries=entries,
            source_url=self.SOURCE_URL,
            citation=self.CITATION,
        )
    
    @staticmethod
    def create_synthetic_dataset() -> BenchmarkDataset:
        """
        Create synthetic Cross-Beta-like dataset with known APR regions.
        
        Based on canonical amyloid sequences with experimentally
        validated APR boundaries.
        """
        entries = []
        
        # Aβ42 with known APRs
        abeta_seq = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"
        entries.append(BenchmarkEntry(
            id="ABETA42",
            sequence=abeta_seq,
            amyloid_status=AmyloidStatus.POSITIVE,
            amyloid_regions=[
                Region(start=16, end=21, sequence="KLVFF"),
                Region(start=29, end=42, sequence="GAIIGLMVGGVVIA"),
            ],
            residue_labels=[
                i in range(16, 21) or i in range(29, 42)
                for i in range(len(abeta_seq))
            ],
            experimental_method=ExperimentalMethod.CRYO_EM,
            source_database="Cross-Beta_synthetic",
            metadata={"pdb_id": "7Q4B", "disease": "Alzheimer's"},
        ))
        
        # α-synuclein NAC region
        asyn_seq = "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA"
        entries.append(BenchmarkEntry(
            id="ASYN",
            sequence=asyn_seq,
            amyloid_status=AmyloidStatus.POSITIVE,
            amyloid_regions=[
                Region(start=61, end=95, sequence=asyn_seq[61:95]),  # NAC region
            ],
            experimental_method=ExperimentalMethod.CRYO_EM,
            source_database="Cross-Beta_synthetic",
            metadata={"pdb_id": "6CU7", "disease": "Parkinson's"},
        ))
        
        # Tau PHF core
        tau_seq = "VQIINKKTTTDTTDSNQGFRSSSTGSIDMVDSPQLATLADEVS"
        entries.append(BenchmarkEntry(
            id="TAU_PHF",
            sequence=tau_seq,
            amyloid_status=AmyloidStatus.POSITIVE,
            amyloid_regions=[
                Region(start=0, end=6, sequence="VQIINK"),
                Region(start=28, end=35, sequence="SPQLATL"),
            ],
            experimental_method=ExperimentalMethod.CRYO_EM,
            source_database="Cross-Beta_synthetic",
            metadata={"pdb_id": "5O3L", "disease": "Alzheimer's"},
        ))
        
        # IAPP (amylin)
        iapp_seq = "KCNTATCATQRLANFLVHSSNNFGAILSSTNVGSNTY"
        entries.append(BenchmarkEntry(
            id="IAPP",
            sequence=iapp_seq,
            amyloid_status=AmyloidStatus.POSITIVE,
            amyloid_regions=[
                Region(start=20, end=29, sequence="NFGAILSST"),
            ],
            experimental_method=ExperimentalMethod.CRYO_EM,
            source_database="Cross-Beta_synthetic",
            metadata={"disease": "Type 2 diabetes"},
        ))
        
        # Sup35 prion domain
        sup35_seq = "GNNQQNYQQYSQNGNQQQGNNRYQGYQAYNAQAQPAGGYYQNYQGYSGYQQGGYQQYNPDAGYQQQYNPQGGYQQYNPQGGYQQQFNPQGGRGNYKNFNYNNNLQGYQAGFQPQSQGMSLNDFQKQQKQ"
        entries.append(BenchmarkEntry(
            id="SUP35_PRION",
            sequence=sup35_seq,
            amyloid_status=AmyloidStatus.POSITIVE,
            amyloid_regions=[
                Region(start=0, end=7, sequence="GNNQQNY"),
            ],
            experimental_method=ExperimentalMethod.XRAY,
            source_database="Cross-Beta_synthetic",
            metadata={"pdb_id": "1YJO", "organism": "S. cerevisiae"},
        ))
        
        return BenchmarkDataset(
            name="Cross-Beta_synthetic",
            description="Canonical amyloid sequences with validated APR regions",
            entries=entries,
            citation="Synthetic dataset based on literature",
        )


class AmyProLoader:
    """
    Loader for AmyPro database of amyloid-forming proteins.
    
    AmyPro catalogs proteins experimentally shown to form amyloid,
    including both pathological and functional amyloids.
    """
    
    SOURCE_URL = "https://amypro.net/"
    CITATION = "Varadi et al. (2018) AmyPro. Nucleic Acids Res."
    
    def load(self, path: Path) -> BenchmarkDataset:
        """Load AmyPro from downloaded file."""
        if path.suffix == '.json':
            return self._load_json(path)
        else:
            return self._load_csv(path)
    
    def _load_json(self, path: Path) -> BenchmarkDataset:
        """Load from JSON."""
        with open(path) as f:
            data = json.load(f)
        
        entries = []
        for item in data:
            seq = item.get('sequence', '')
            if not seq:
                continue
            
            entries.append(BenchmarkEntry(
                id=item.get('uniprot_id', item.get('id', '')),
                sequence=seq.upper(),
                amyloid_status=AmyloidStatus.POSITIVE,
                source_database="AmyPro",
                metadata={
                    'name': item.get('name'),
                    'organism': item.get('organism'),
                    'functional': item.get('functional', False),
                },
            ))
        
        return BenchmarkDataset(
            name="AmyPro",
            description="Database of amyloid-forming proteins",
            entries=entries,
            source_url=self.SOURCE_URL,
            citation=self.CITATION,
        )
    
    def _load_csv(self, path: Path) -> BenchmarkDataset:
        """Load from CSV."""
        entries = []
        
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                seq = row.get('sequence', row.get('Sequence', ''))
                if not seq:
                    continue
                
                entries.append(BenchmarkEntry(
                    id=row.get('id', row.get('uniprot_id', '')),
                    sequence=seq.upper(),
                    amyloid_status=AmyloidStatus.POSITIVE,
                    source_database="AmyPro",
                    metadata=dict(row),
                ))
        
        return BenchmarkDataset(
            name="AmyPro",
            description="Database of amyloid-forming proteins",
            entries=entries,
            source_url=self.SOURCE_URL,
            citation=self.CITATION,
        )


class FASTALoader:
    """
    Generic FASTA loader for custom datasets.
    
    Parses FASTA files with optional label information in headers.
    
    Supported header formats:
    - >ID|POSITIVE or >ID|NEGATIVE
    - >ID label=positive
    - >ID amyloid=yes
    """
    
    def load(
        self,
        path: Path,
        default_status: AmyloidStatus = AmyloidStatus.UNKNOWN,
    ) -> BenchmarkDataset:
        """
        Load sequences from FASTA file.
        
        Args:
            path: Path to FASTA file
            default_status: Status for entries without label in header
        """
        entries = []
        current_id = None
        current_seq = []
        current_status = default_status
        
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # Save previous entry
                    if current_id and current_seq:
                        entries.append(BenchmarkEntry(
                            id=current_id,
                            sequence=''.join(current_seq).upper(),
                            amyloid_status=current_status,
                            source_database="FASTA",
                        ))
                    
                    # Parse header
                    header = line[1:]
                    current_id, current_status = self._parse_header(header, default_status)
                    current_seq = []
                else:
                    current_seq.append(line)
        
        # Save last entry
        if current_id and current_seq:
            entries.append(BenchmarkEntry(
                id=current_id,
                sequence=''.join(current_seq).upper(),
                amyloid_status=current_status,
                source_database="FASTA",
            ))
        
        return BenchmarkDataset(
            name=path.stem,
            description=f"Custom dataset from {path.name}",
            entries=entries,
        )
    
    def _parse_header(
        self,
        header: str,
        default: AmyloidStatus,
    ) -> tuple[str, AmyloidStatus]:
        """Parse FASTA header for ID and status."""
        parts = header.split('|')
        entry_id = parts[0].strip()
        
        status = default
        
        # Check for status in pipe-separated parts
        for part in parts[1:]:
            part_lower = part.lower().strip()
            if part_lower in ['positive', 'amyloid', 'yes', '1']:
                status = AmyloidStatus.POSITIVE
            elif part_lower in ['negative', 'non-amyloid', 'no', '0']:
                status = AmyloidStatus.NEGATIVE
        
        # Check for key=value format
        for match in re.finditer(r'(\w+)=(\w+)', header):
            key, value = match.groups()
            if key.lower() in ['label', 'status', 'amyloid', 'class']:
                if value.lower() in ['positive', 'amyloid', 'yes', '1', 'true']:
                    status = AmyloidStatus.POSITIVE
                elif value.lower() in ['negative', 'non-amyloid', 'no', '0', 'false']:
                    status = AmyloidStatus.NEGATIVE
        
        return entry_id, status


# =============================================================================
# Convenience Functions
# =============================================================================

def load_benchmark_dataset(
    source: str,
    path: Optional[Path] = None,
    **kwargs,
) -> BenchmarkDataset:
    """
    Load a benchmark dataset by name.
    
    Args:
        source: Dataset name ('waltz', 'crossbeta', 'amypro', 'fasta', 'synthetic')
        path: Path to data file/directory
        **kwargs: Additional arguments for specific loaders
        
    Returns:
        BenchmarkDataset
    """
    source_lower = source.lower()
    
    if source_lower in ['waltz', 'waltz-db', 'waltzdb']:
        if path:
            return WaltzDBLoader().load(path)
        else:
            return WaltzDBLoader.create_synthetic_dataset(**kwargs)
    
    elif source_lower in ['crossbeta', 'cross-beta', 'crossbetadb']:
        if path:
            return CrossBetaDBLoader().load(path)
        else:
            return CrossBetaDBLoader.create_synthetic_dataset()
    
    elif source_lower in ['amypro']:
        if path is None:
            raise ValueError("AmyPro requires a data file path")
        return AmyProLoader().load(path)
    
    elif source_lower in ['fasta', 'fa']:
        if path is None:
            raise ValueError("FASTA loader requires a file path")
        return FASTALoader().load(path, **kwargs)
    
    elif source_lower == 'synthetic':
        # Combined synthetic dataset
        waltz = WaltzDBLoader.create_synthetic_dataset(n_positive=50, n_negative=50)
        crossbeta = CrossBetaDBLoader.create_synthetic_dataset()
        
        return BenchmarkDataset(
            name="Synthetic_Combined",
            description="Combined synthetic benchmark dataset",
            entries=waltz.entries + crossbeta.entries,
        )
    
    else:
        raise ValueError(f"Unknown dataset source: {source}")


def get_available_datasets() -> list[dict[str, str]]:
    """Get list of available benchmark datasets."""
    return [
        {
            "name": "WALTZ-DB",
            "description": "Hexapeptide aggregation database",
            "requires_download": True,
            "url": WaltzDBLoader.SOURCE_URL,
        },
        {
            "name": "Cross-Beta DB",
            "description": "Amyloid fibril structures",
            "requires_download": True,
            "url": CrossBetaDBLoader.SOURCE_URL,
        },
        {
            "name": "AmyPro",
            "description": "Amyloid-forming proteins",
            "requires_download": True,
            "url": AmyProLoader.SOURCE_URL,
        },
        {
            "name": "Synthetic",
            "description": "Built-in synthetic dataset for testing",
            "requires_download": False,
            "url": "",
        },
    ]
