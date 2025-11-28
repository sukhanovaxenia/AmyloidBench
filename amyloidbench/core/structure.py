"""
Structure handling utilities for AmyloidBench.

This module provides tools for working with protein 3D structures,
essential for structure-based amyloidogenicity predictors like Aggrescan3D.
Supports PDB and mmCIF formats, with utilities for extracting sequence,
calculating solvent accessibility, and preparing structures for analysis.

Note: Structure-based prediction is particularly valuable because:
1. Amyloidogenic regions may be buried in the native fold
2. Structural context determines which APRs are accessible for aggregation
3. Protein dynamics influence aggregation propensity
"""

from __future__ import annotations

import gzip
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional, TextIO, Union

import numpy as np

from .models import ProteinRecord, Region

logger = logging.getLogger(__name__)

# Try to import structural biology libraries
# These are optional dependencies
try:
    import biotite.structure as struc
    import biotite.structure.io.pdb as pdb
    import biotite.structure.io.pdbx as pdbx
    from biotite.structure import sasa
    BIOTITE_AVAILABLE = True
except ImportError:
    BIOTITE_AVAILABLE = False
    logger.warning("biotite not available. Structure handling will be limited.")


@dataclass
class StructureMetadata:
    """Metadata extracted from a protein structure file."""
    pdb_id: Optional[str] = None
    title: Optional[str] = None
    resolution: Optional[float] = None
    method: Optional[str] = None  # X-RAY DIFFRACTION, CRYO-EM, NMR, etc.
    organism: Optional[str] = None
    n_chains: int = 0
    n_residues: int = 0
    n_atoms: int = 0


@dataclass
class ChainData:
    """Data for a single protein chain."""
    chain_id: str
    sequence: str
    residue_numbers: list[int]
    residue_names: list[str]  # Three-letter codes
    ca_coords: Optional[np.ndarray] = None  # Cα coordinates (N x 3)
    sasa: Optional[np.ndarray] = None  # Per-residue SASA


class StructureError(Exception):
    """Exception for structure-related errors."""
    pass


class StructureHandler:
    """
    Handles protein structure loading, parsing, and analysis.
    
    This class provides a unified interface for working with PDB and mmCIF
    files, extracting sequences and structural features relevant to
    amyloidogenicity prediction.
    """
    
    def __init__(self, structure_path: Union[str, Path]):
        """
        Initialize with a structure file.
        
        Args:
            structure_path: Path to PDB or mmCIF file (can be gzipped)
        
        Raises:
            StructureError: If structure cannot be loaded
        """
        if not BIOTITE_AVAILABLE:
            raise StructureError(
                "Structure handling requires biotite. "
                "Install with: pip install biotite"
            )
        
        self.path = Path(structure_path)
        self._structure = None
        self._metadata = None
        self._chains: dict[str, ChainData] = {}
        
        self._load_structure()
    
    def _load_structure(self):
        """Load and parse the structure file."""
        path_str = str(self.path)
        
        # Handle gzipped files
        is_gzipped = path_str.endswith(".gz")
        
        # Determine format
        if ".pdb" in path_str.lower():
            file_format = "pdb"
        elif ".cif" in path_str.lower():
            file_format = "cif"
        else:
            raise StructureError(f"Unknown structure format: {self.path}")
        
        try:
            if is_gzipped:
                with gzip.open(self.path, "rt") as f:
                    content = f.read()
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=f".{file_format}", delete=False
                ) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                
                if file_format == "pdb":
                    pdb_file = pdb.PDBFile.read(tmp_path)
                    self._structure = pdb_file.get_structure(model=1)
                else:
                    cif_file = pdbx.PDBxFile.read(tmp_path)
                    self._structure = pdbx.get_structure(cif_file, model=1)
                
                Path(tmp_path).unlink()
            else:
                if file_format == "pdb":
                    pdb_file = pdb.PDBFile.read(self.path)
                    self._structure = pdb_file.get_structure(model=1)
                else:
                    cif_file = pdbx.PDBxFile.read(self.path)
                    self._structure = pdbx.get_structure(cif_file, model=1)
            
            # Filter to amino acids only
            self._structure = self._structure[struc.filter_amino_acids(self._structure)]
            
            # Extract chain data
            self._extract_chains()
            
            # Calculate metadata
            self._extract_metadata()
            
        except Exception as e:
            raise StructureError(f"Failed to load structure: {e}") from e
    
    def _extract_chains(self):
        """Extract data for each protein chain."""
        chain_ids = np.unique(self._structure.chain_id)
        
        for chain_id in chain_ids:
            chain_mask = self._structure.chain_id == chain_id
            chain_atoms = self._structure[chain_mask]
            
            # Get CA atoms for sequence extraction
            ca_mask = chain_atoms.atom_name == "CA"
            ca_atoms = chain_atoms[ca_mask]
            
            if len(ca_atoms) == 0:
                continue
            
            # Extract residue information
            residue_numbers = ca_atoms.res_id.tolist()
            residue_names = ca_atoms.res_name.tolist()
            
            # Convert three-letter to one-letter codes
            from .sequence import AA_3TO1
            sequence = ""
            for res_name in residue_names:
                aa = AA_3TO1.get(res_name, "X")
                sequence += aa
            
            # Get CA coordinates
            ca_coords = ca_atoms.coord
            
            self._chains[chain_id] = ChainData(
                chain_id=chain_id,
                sequence=sequence,
                residue_numbers=residue_numbers,
                residue_names=residue_names,
                ca_coords=ca_coords,
            )
    
    def _extract_metadata(self):
        """Extract structure metadata."""
        self._metadata = StructureMetadata(
            n_chains=len(self._chains),
            n_residues=sum(len(c.sequence) for c in self._chains.values()),
            n_atoms=len(self._structure),
        )
    
    @property
    def metadata(self) -> StructureMetadata:
        """Get structure metadata."""
        return self._metadata
    
    @property
    def chain_ids(self) -> list[str]:
        """Get list of chain IDs."""
        return list(self._chains.keys())
    
    def get_chain(self, chain_id: str) -> ChainData:
        """
        Get data for a specific chain.
        
        Args:
            chain_id: Chain identifier
        
        Returns:
            ChainData object
        
        Raises:
            StructureError: If chain not found
        """
        if chain_id not in self._chains:
            raise StructureError(
                f"Chain '{chain_id}' not found. Available: {self.chain_ids}"
            )
        return self._chains[chain_id]
    
    def get_sequence(self, chain_id: Optional[str] = None) -> str:
        """
        Get protein sequence from structure.
        
        Args:
            chain_id: Specific chain (None = first chain)
        
        Returns:
            One-letter amino acid sequence
        """
        if chain_id is None:
            chain_id = self.chain_ids[0]
        return self._chains[chain_id].sequence
    
    def calculate_sasa(
        self,
        chain_id: Optional[str] = None,
        probe_radius: float = 1.4,
    ) -> np.ndarray:
        """
        Calculate Solvent Accessible Surface Area per residue.
        
        SASA is crucial for amyloidogenicity prediction because:
        - Only exposed APRs can initiate aggregation
        - Buried hydrophobic residues are protected in the native fold
        - Aggregation often requires local unfolding to expose APRs
        
        Args:
            chain_id: Chain to analyze (None = first chain)
            probe_radius: Probe radius in Angstroms (default: water)
        
        Returns:
            Array of per-residue SASA values
        """
        if chain_id is None:
            chain_id = self.chain_ids[0]
        
        chain_data = self._chains[chain_id]
        
        # Check if already calculated
        if chain_data.sasa is not None:
            return chain_data.sasa
        
        # Get chain atoms
        chain_mask = self._structure.chain_id == chain_id
        chain_atoms = self._structure[chain_mask]
        
        # Calculate atomic SASA
        atom_sasa = sasa(chain_atoms, probe_radius=probe_radius)
        
        # Aggregate to residue level
        residue_starts = struc.get_residue_starts(chain_atoms)
        n_residues = len(residue_starts)
        
        residue_sasa = np.zeros(n_residues)
        for i, start in enumerate(residue_starts):
            end = residue_starts[i + 1] if i + 1 < n_residues else len(atom_sasa)
            residue_sasa[i] = np.sum(atom_sasa[start:end])
        
        # Cache the result
        chain_data.sasa = residue_sasa
        
        return residue_sasa
    
    def calculate_relative_sasa(
        self,
        chain_id: Optional[str] = None,
    ) -> np.ndarray:
        """
        Calculate relative SASA (fraction of maximum possible SASA).
        
        Relative SASA normalizes by the theoretical maximum SASA for each
        amino acid type, giving values in [0, 1] range.
        
        Args:
            chain_id: Chain to analyze
        
        Returns:
            Array of relative SASA values (0 = fully buried, 1 = fully exposed)
        """
        # Maximum SASA values for each amino acid (Å², Gly-X-Gly tripeptides)
        MAX_SASA = {
            'A': 129, 'R': 274, 'N': 195, 'D': 193, 'C': 167,
            'E': 223, 'Q': 225, 'G': 104, 'H': 224, 'I': 197,
            'L': 201, 'K': 236, 'M': 224, 'F': 240, 'P': 159,
            'S': 155, 'T': 172, 'W': 285, 'Y': 263, 'V': 174,
        }
        
        if chain_id is None:
            chain_id = self.chain_ids[0]
        
        chain_data = self._chains[chain_id]
        absolute_sasa = self.calculate_sasa(chain_id)
        
        relative_sasa = np.zeros(len(chain_data.sequence))
        for i, aa in enumerate(chain_data.sequence):
            max_sasa = MAX_SASA.get(aa, 200)  # Default for unknown AA
            relative_sasa[i] = min(absolute_sasa[i] / max_sasa, 1.0)
        
        return relative_sasa
    
    def find_exposed_regions(
        self,
        chain_id: Optional[str] = None,
        rel_sasa_threshold: float = 0.25,
        min_length: int = 4,
    ) -> list[Region]:
        """
        Identify regions with high surface exposure.
        
        Exposed regions are more likely to be involved in aggregation
        initiation, as they're accessible for intermolecular contacts.
        
        Args:
            chain_id: Chain to analyze
            rel_sasa_threshold: Minimum relative SASA to consider "exposed"
            min_length: Minimum length of exposed region
        
        Returns:
            List of Region objects representing exposed segments
        """
        if chain_id is None:
            chain_id = self.chain_ids[0]
        
        chain_data = self._chains[chain_id]
        rel_sasa = self.calculate_relative_sasa(chain_id)
        
        regions = []
        in_region = False
        start = 0
        
        for i, rsa in enumerate(rel_sasa):
            if rsa >= rel_sasa_threshold and not in_region:
                in_region = True
                start = i
            elif rsa < rel_sasa_threshold and in_region:
                in_region = False
                if i - start >= min_length:
                    regions.append(Region(
                        start=start,
                        end=i,
                        sequence=chain_data.sequence[start:i],
                        score=float(np.mean(rel_sasa[start:i])),
                        annotation="exposed_region",
                    ))
        
        # Handle region at end
        if in_region and len(rel_sasa) - start >= min_length:
            regions.append(Region(
                start=start,
                end=len(rel_sasa),
                sequence=chain_data.sequence[start:],
                score=float(np.mean(rel_sasa[start:])),
                annotation="exposed_region",
            ))
        
        return regions
    
    def calculate_secondary_structure(
        self,
        chain_id: Optional[str] = None,
    ) -> str:
        """
        Assign secondary structure using DSSP-like algorithm.
        
        Secondary structure is relevant to amyloidogenicity:
        - β-strand propensity correlates with amyloid formation
        - α-helices can protect or promote aggregation depending on context
        - Coil regions are often flexible and can adopt β-structure
        
        Args:
            chain_id: Chain to analyze
        
        Returns:
            String with secondary structure assignment:
            H = helix, E = strand, C = coil
        """
        if chain_id is None:
            chain_id = self.chain_ids[0]
        
        chain_mask = self._structure.chain_id == chain_id
        chain_atoms = self._structure[chain_mask]
        
        # Use biotite's secondary structure assignment
        ss_array = struc.annotate_sse(chain_atoms)
        
        # Convert to string
        ss_map = {'a': 'H', 'b': 'E', 'c': 'C'}
        ss_string = "".join(ss_map.get(ss, 'C') for ss in ss_array)
        
        # The SSE is per-residue, but we need to check length matches sequence
        chain_data = self._chains[chain_id]
        if len(ss_string) != len(chain_data.sequence):
            # Pad or truncate to match sequence length
            if len(ss_string) < len(chain_data.sequence):
                ss_string += 'C' * (len(chain_data.sequence) - len(ss_string))
            else:
                ss_string = ss_string[:len(chain_data.sequence)]
        
        return ss_string
    
    def get_beta_strand_content(self, chain_id: Optional[str] = None) -> float:
        """
        Calculate β-strand content as fraction of sequence.
        
        High β-strand content is associated with amyloidogenicity
        as cross-β structure is the hallmark of amyloid fibrils.
        
        Args:
            chain_id: Chain to analyze
        
        Returns:
            Fraction of residues in β-strand conformation
        """
        ss = self.calculate_secondary_structure(chain_id)
        return ss.count('E') / len(ss) if ss else 0.0
    
    def to_protein_record(self, chain_id: Optional[str] = None) -> ProteinRecord:
        """
        Convert structure chain to ProteinRecord.
        
        Args:
            chain_id: Chain to convert (None = first chain)
        
        Returns:
            ProteinRecord with structure path set
        """
        if chain_id is None:
            chain_id = self.chain_ids[0]
        
        chain_data = self._chains[chain_id]
        
        return ProteinRecord(
            id=f"{self.path.stem}_{chain_id}",
            sequence=chain_data.sequence,
            structure_path=self.path,
            structure_source="PDB" if ".pdb" in str(self.path).lower() else "mmCIF",
        )


def load_structure(path: Union[str, Path]) -> StructureHandler:
    """
    Convenience function to load a structure file.
    
    Args:
        path: Path to PDB or mmCIF file
    
    Returns:
        StructureHandler instance
    """
    return StructureHandler(path)


def fetch_structure_from_pdb(
    pdb_id: str,
    output_dir: Optional[Path] = None,
    format: str = "pdb",
) -> Path:
    """
    Download structure from RCSB PDB.
    
    Args:
        pdb_id: 4-character PDB ID
        output_dir: Directory to save file (default: temp directory)
        format: "pdb" or "cif"
    
    Returns:
        Path to downloaded file
    """
    import urllib.request
    
    pdb_id = pdb_id.lower()
    
    if format == "pdb":
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        suffix = ".pdb"
    else:
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        suffix = ".cif"
    
    if output_dir is None:
        output_dir = Path(tempfile.gettempdir())
    
    output_path = output_dir / f"{pdb_id}{suffix}"
    
    urllib.request.urlretrieve(url, output_path)
    
    return output_path


def fetch_alphafold_structure(
    uniprot_id: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Download predicted structure from AlphaFold Protein Structure Database.
    
    AlphaFold structures are valuable for amyloidogenicity prediction
    when experimental structures are not available. However, predictions
    for disordered/aggregation-prone regions should be interpreted cautiously.
    
    Args:
        uniprot_id: UniProt accession (e.g., "P04156" for human prion protein)
        output_dir: Directory to save file
    
    Returns:
        Path to downloaded file
    """
    import urllib.request
    
    url = (
        f"https://alphafold.ebi.ac.uk/files/"
        f"AF-{uniprot_id}-F1-model_v4.pdb"
    )
    
    if output_dir is None:
        output_dir = Path(tempfile.gettempdir())
    
    output_path = output_dir / f"AF-{uniprot_id}.pdb"
    
    urllib.request.urlretrieve(url, output_path)
    
    return output_path
