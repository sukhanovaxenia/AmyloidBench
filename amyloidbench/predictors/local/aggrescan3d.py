"""
Aggrescan3D (A3D) wrapper for structure-based aggregation prediction.

Aggrescan3D is one of the most widely-used structure-based methods for
predicting protein aggregation propensity. It projects the experimentally-
derived AGGRESCAN aggregation scale onto 3D protein structures, accounting
for solvent accessibility to identify Structural Aggregation-Prone Regions
(STAPs) that are exposed on the protein surface.

Key biological principles:
- Only surface-exposed hydrophobic regions can initiate aggregation
- Buried hydrophobic cores are protected in native structures
- Conformational dynamics can transiently expose APRs
- Gatekeeper residues at APR boundaries modulate aggregation

The A3D standalone package supports:
- Static structure analysis
- Dynamic mode (using CABS-flex for flexibility)
- Mutation scanning for solubility design
- Integration with FoldX for stability predictions

References:
    Zambrano et al. (2015) Nucleic Acids Res. 43:W306-W313
    Kuriata et al. (2019) Bioinformatics 35:3834-3835
    Bárcenas et al. (2024) Nucleic Acids Res. 52:W170-W175 (A3D 4.0)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

from amyloidbench.core.models import (
    PerResidueScores,
    PredictionConfidence,
    PredictionResult,
    ProteinRecord,
    Region,
)
from amyloidbench.core.structure import StructureHandler, fetch_alphafold_structure
from ..base import (
    BasePredictor,
    PredictorCapability,
    PredictorConfig,
    PredictorError,
    PredictorType,
    register_predictor,
)

logger = logging.getLogger(__name__)


def check_aggrescan3d_installation() -> tuple[bool, str]:
    """
    Check if Aggrescan3D standalone is installed and accessible.
    
    Returns:
        Tuple of (is_installed, message/version)
    """
    try:
        result = subprocess.run(
            ["aggrescan", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            version = result.stdout.strip() or result.stderr.strip()
            return True, version
        else:
            return False, "aggrescan command failed"
    except FileNotFoundError:
        return False, "aggrescan not found in PATH"
    except subprocess.TimeoutExpired:
        return False, "aggrescan command timed out"
    except Exception as e:
        return False, str(e)


@register_predictor
class Aggrescan3DPredictor(BasePredictor):
    """
    Structure-based aggregation predictor using Aggrescan3D.
    
    A3D identifies Structural Aggregation-Prone Regions (STAPs) by
    combining the AGGRESCAN aggregation propensity scale with solvent
    accessibility calculations from 3D protein structures.
    
    The predictor operates in two modes:
    1. **Static mode**: Analyzes a single structure conformation
    2. **Dynamic mode**: Uses CABS-flex to sample conformational ensemble
    
    Scores interpretation:
    - Positive scores: Aggregation-promoting (exposed hydrophobic)
    - Negative scores: Aggregation-inhibiting (charged/polar)
    - Near zero: Neutral contribution
    
    The A3D score integrates residue aggregation propensity with solvent
    exposure, so buried hydrophobic residues have reduced impact.
    
    Usage:
        predictor = Aggrescan3DPredictor()
        
        # With structure file
        protein = ProteinRecord(id="test", sequence="...", structure_path=Path("1xyz.pdb"))
        result = predictor.predict(protein)
        
        # Using AlphaFold predicted structure
        result = predictor.predict_with_alphafold(protein, uniprot_id="P04156")
    
    Attributes:
        dynamic_mode: Enable CABS-flex flexibility analysis
        distance_threshold: Distance cutoff for A3D calculations (Å)
        chain_id: Specific chain to analyze (None = all chains)
    """
    
    name = "Aggrescan3D"
    version = "2.0"
    predictor_type = PredictorType.STRUCTURE_BASED
    capabilities = {
        PredictorCapability.PER_RESIDUE_SCORES,
        PredictorCapability.REGION_DETECTION,
        PredictorCapability.STRUCTURE_INPUT,
    }
    
    # A3D scoring parameters
    default_threshold = 0.0  # Positive = aggregation-prone
    score_min = -3.0  # Typical range
    score_max = 3.0
    
    # A3D aggregation propensity scale (from AGGRESCAN)
    # Experimentally derived from bacterial inclusion body formation
    A3D_SCALE = {
        'A': -0.036, 'R': -1.509, 'N': -0.881, 'D': -1.185, 'C': 0.326,
        'Q': -0.652, 'E': -0.931, 'G': -0.535, 'H': -0.115, 'I': 1.822,
        'L': 1.380, 'K': -1.503, 'M': 0.724, 'F': 2.007, 'P': -0.577,
        'S': -0.568, 'T': -0.150, 'W': 1.690, 'Y': 1.159, 'V': 1.594,
    }
    
    citation = (
        "Kuriata et al. (2019) Aggrescan3D standalone package for structure-based "
        "prediction of protein aggregation properties. Bioinformatics 35:3834-3835"
    )
    url = "https://bitbucket.org/lcbio/aggrescan3d"
    description = (
        "Structure-based aggregation predictor that projects AGGRESCAN propensity "
        "scale onto 3D structures, accounting for solvent accessibility."
    )
    
    def __init__(
        self,
        config: Optional[PredictorConfig] = None,
        dynamic_mode: bool = False,
        distance_threshold: float = 10.0,
        chain_id: Optional[str] = None,
    ):
        """
        Initialize Aggrescan3D predictor.
        
        Args:
            config: Predictor configuration
            dynamic_mode: Use CABS-flex for flexibility analysis
            distance_threshold: Distance cutoff for calculations (Å)
            chain_id: Specific chain to analyze (None = first chain)
        """
        super().__init__(config)
        self.dynamic_mode = dynamic_mode
        self.distance_threshold = distance_threshold
        self.chain_id = chain_id
        
        # Check installation
        installed, msg = check_aggrescan3d_installation()
        self._standalone_available = installed
        if installed:
            logger.info(f"Aggrescan3D standalone version: {msg}")
        else:
            logger.warning(
                f"Aggrescan3D standalone not available ({msg}). "
                "Using built-in implementation for sequence-based scoring. "
                "For full structure-based analysis, install: pip install Aggrescan3D"
            )
    
    def _calculate_sequence_scores(self, sequence: str) -> np.ndarray:
        """
        Calculate raw AGGRESCAN sequence-based scores.
        
        This is a fallback when structure is not available.
        Uses the original AGGRESCAN aggregation propensity scale
        with a sliding window approach.
        
        Args:
            sequence: Protein sequence
        
        Returns:
            Array of per-residue aggregation scores
        """
        scores = np.array([self.A3D_SCALE.get(aa, 0.0) for aa in sequence])
        
        # Apply sliding window smoothing (AGGRESCAN uses window of 5-7)
        window = self.window_size
        smoothed = np.convolve(scores, np.ones(window) / window, mode='same')
        
        return smoothed
    
    def _calculate_structure_scores(
        self,
        sequence: str,
        structure_path: Path,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Calculate structure-based A3D scores.
        
        Integrates aggregation propensity with solvent accessibility:
        A3D_score = AGGRESCAN_propensity × RSA_weight
        
        Where RSA_weight reduces contribution of buried residues.
        
        Args:
            sequence: Protein sequence
            structure_path: Path to PDB/mmCIF file
        
        Returns:
            Tuple of (scores array, metadata dict)
        """
        try:
            handler = StructureHandler(structure_path)
            
            # Determine chain
            chain = self.chain_id or handler.chain_ids[0]
            struct_seq = handler.get_sequence(chain)
            
            # Verify sequence match
            if struct_seq != sequence:
                logger.warning(
                    f"Sequence mismatch: input ({len(sequence)}) vs "
                    f"structure ({len(struct_seq)}). Using structure sequence."
                )
                sequence = struct_seq
            
            # Get relative solvent accessibility
            rel_sasa = handler.calculate_relative_sasa(chain)
            
            # Calculate base AGGRESCAN scores
            base_scores = np.array([self.A3D_SCALE.get(aa, 0.0) for aa in sequence])
            
            # Apply RSA weighting
            # Buried residues (RSA < 0.25) have reduced contribution
            # Exposed residues (RSA > 0.25) contribute fully
            rsa_weight = np.clip(rel_sasa / 0.25, 0, 1)
            
            # For aggregation-prone residues, weight by exposure
            # For aggregation-inhibiting residues, they protect regardless
            structure_scores = np.where(
                base_scores > 0,
                base_scores * rsa_weight,  # Reduce positive scores for buried
                base_scores  # Keep negative scores (protective)
            )
            
            # Apply smoothing
            window = self.window_size
            smoothed = np.convolve(
                structure_scores, np.ones(window) / window, mode='same'
            )
            
            # Get secondary structure for metadata
            ss = handler.calculate_secondary_structure(chain)
            
            metadata = {
                "chain_id": chain,
                "rel_sasa": rel_sasa.tolist(),
                "secondary_structure": ss,
                "beta_content": ss.count('E') / len(ss),
                "structure_source": str(structure_path),
            }
            
            return smoothed, metadata
            
        except Exception as e:
            logger.error(f"Structure analysis failed: {e}")
            raise PredictorError(f"Failed to analyze structure: {e}") from e
    
    def _run_standalone_a3d(
        self,
        structure_path: Path,
        output_dir: Path,
    ) -> dict[str, Any]:
        """
        Run the official Aggrescan3D standalone tool.
        
        Args:
            structure_path: Path to input PDB file
            output_dir: Directory for output files
        
        Returns:
            Parsed A3D output data
        """
        cmd = [
            "aggrescan",
            "-i", str(structure_path),
            "-o", str(output_dir),
            "-d", str(self.distance_threshold),
        ]
        
        if self.chain_id:
            cmd.extend(["-c", self.chain_id])
        
        if self.dynamic_mode:
            cmd.append("--dynamic")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                cwd=str(output_dir),
            )
            
            if result.returncode != 0:
                raise PredictorError(
                    f"A3D failed with code {result.returncode}: {result.stderr}"
                )
            
            # Parse output files
            output_data = self._parse_a3d_output(output_dir)
            return output_data
            
        except subprocess.TimeoutExpired:
            raise PredictorError(
                f"A3D timed out after {self.config.timeout_seconds}s"
            )
    
    def _parse_a3d_output(self, output_dir: Path) -> dict[str, Any]:
        """
        Parse Aggrescan3D output files.
        
        A3D produces several output files:
        - *_A3D.csv: Per-residue scores
        - *_A3D_hotspots.csv: Identified aggregation-prone regions
        - *.pdb: Structure colored by A3D score (for visualization)
        
        Args:
            output_dir: Directory containing A3D output
        
        Returns:
            Parsed data dictionary
        """
        output_data = {
            "scores": [],
            "hotspots": [],
            "metadata": {},
        }
        
        # Find and parse main output CSV
        csv_files = list(output_dir.glob("*_A3D.csv"))
        if csv_files:
            with open(csv_files[0]) as f:
                lines = f.readlines()
                # Skip header
                for line in lines[1:]:
                    parts = line.strip().split(",")
                    if len(parts) >= 4:
                        output_data["scores"].append({
                            "residue_num": int(parts[0]),
                            "residue_name": parts[1],
                            "chain": parts[2],
                            "score": float(parts[3]),
                        })
        
        # Parse hotspots file
        hotspot_files = list(output_dir.glob("*_hotspots.csv"))
        if hotspot_files:
            with open(hotspot_files[0]) as f:
                lines = f.readlines()
                for line in lines[1:]:
                    parts = line.strip().split(",")
                    if len(parts) >= 3:
                        output_data["hotspots"].append({
                            "start": int(parts[0]),
                            "end": int(parts[1]),
                            "avg_score": float(parts[2]),
                        })
        
        return output_data
    
    def _predict_impl(
        self,
        sequence: str,
        structure_path: Optional[Path] = None,
    ) -> PredictionResult:
        """
        Run Aggrescan3D prediction.
        
        The implementation follows a hierarchical approach:
        1. If A3D standalone is available AND structure provided: Use official tool
        2. If structure provided but no standalone: Use our implementation
        3. If no structure: Fall back to sequence-based AGGRESCAN
        
        Args:
            sequence: Protein sequence
            structure_path: Optional path to structure file
        
        Returns:
            PredictionResult with A3D scores and STAPs
        """
        raw_output = {}
        
        # Strategy 1: Use standalone A3D with structure
        if self._standalone_available and structure_path and structure_path.exists():
            logger.info(f"Running A3D standalone on {structure_path}")
            
            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir)
                
                try:
                    a3d_data = self._run_standalone_a3d(structure_path, output_dir)
                    
                    # Extract scores
                    scores = [r["score"] for r in a3d_data["scores"]]
                    
                    # Ensure length matches sequence
                    if len(scores) != len(sequence):
                        logger.warning(
                            f"Score length mismatch: {len(scores)} vs {len(sequence)}"
                        )
                        # Pad or truncate
                        if len(scores) < len(sequence):
                            scores.extend([0.0] * (len(sequence) - len(scores)))
                        else:
                            scores = scores[:len(sequence)]
                    
                    # Build regions from hotspots
                    regions = []
                    for hs in a3d_data["hotspots"]:
                        regions.append(Region(
                            start=hs["start"] - 1,  # Convert to 0-indexed
                            end=hs["end"],
                            sequence=sequence[hs["start"]-1:hs["end"]],
                            score=hs["avg_score"],
                            annotation="STAP",
                        ))
                    
                    raw_output = a3d_data
                    
                except PredictorError:
                    # Fall back to our implementation
                    logger.warning("A3D standalone failed, using built-in implementation")
                    scores, metadata = self._calculate_structure_scores(
                        sequence, structure_path
                    )
                    scores = scores.tolist()
                    raw_output = {"metadata": metadata}
                    regions = []
        
        # Strategy 2: Our structure-based implementation
        elif structure_path and structure_path.exists():
            logger.info(f"Using built-in structure analysis for {structure_path}")
            scores, metadata = self._calculate_structure_scores(
                sequence, structure_path
            )
            scores = scores.tolist()
            raw_output = {"metadata": metadata}
            regions = []
        
        # Strategy 3: Sequence-only fallback
        else:
            logger.info("No structure available, using sequence-based AGGRESCAN")
            scores = self._calculate_sequence_scores(sequence).tolist()
            raw_output = {"mode": "sequence_only"}
            regions = []
        
        # Create per-residue scores object
        per_residue = PerResidueScores(
            scores=scores,
            sequence=sequence,
            predictor=self.name,
            score_type="raw",
            threshold=self.threshold,
            min_score=self.score_min,
            max_score=self.score_max,
        )
        
        # Identify regions if not already provided by A3D
        if not regions:
            regions = per_residue.to_regions(
                threshold=self.threshold,
                min_length=self.config.min_region_length,
                merge_gap=self.config.merge_gap,
            )
        
        # Determine overall amyloidogenicity
        is_amyloidogenic = len(regions) > 0
        
        # Calculate probability based on maximum regional score
        if regions:
            max_score = max(r.score for r in regions if r.score)
            # Sigmoid transformation for probability
            prob = 1 / (1 + np.exp(-max_score))
        else:
            prob = 0.0
        
        return PredictionResult(
            sequence_id="",  # Will be filled by base class
            sequence=sequence,
            predictor_name=self.name,
            predictor_version=self.version,
            per_residue_scores=per_residue,
            predicted_regions=regions,
            is_amyloidogenic=is_amyloidogenic,
            amyloid_probability=float(prob),
            raw_output=raw_output if self.config.return_raw_output else None,
        )
    
    def predict_with_alphafold(
        self,
        protein: ProteinRecord,
        uniprot_id: str,
        output_dir: Optional[Path] = None,
    ) -> PredictionResult:
        """
        Predict using AlphaFold-predicted structure.
        
        This is useful when experimental structures are unavailable.
        Note that AlphaFold predictions for disordered regions may be
        unreliable, and such regions are often aggregation-prone.
        
        Args:
            protein: ProteinRecord (structure_path will be updated)
            uniprot_id: UniProt accession for AlphaFold lookup
            output_dir: Directory to save downloaded structure
        
        Returns:
            PredictionResult from A3D analysis
        """
        logger.info(f"Fetching AlphaFold structure for {uniprot_id}")
        
        structure_path = fetch_alphafold_structure(uniprot_id, output_dir)
        
        # Update protein record
        protein.structure_path = structure_path
        protein.structure_source = "AlphaFold"
        
        return self.predict(protein, use_structure=True)
    
    def can_handle(self, protein: ProteinRecord) -> bool:
        """
        Check if A3D can handle this protein.
        
        A3D works best with:
        - Structures available (PDB or AlphaFold)
        - Sequences < 1000 residues (for performance)
        - Single domain proteins or defined chains
        """
        if len(protein.sequence) > 2000:
            logger.warning(f"Sequence very long ({len(protein.sequence)}), may be slow")
        
        return True  # Can always fall back to sequence mode


# Convenience function for quick predictions
def predict_with_a3d(
    sequence: str,
    structure_path: Optional[Path] = None,
    dynamic_mode: bool = False,
) -> PredictionResult:
    """
    Quick A3D prediction without creating explicit objects.
    
    Args:
        sequence: Protein sequence
        structure_path: Optional structure file
        dynamic_mode: Use flexibility analysis
    
    Returns:
        PredictionResult
    """
    predictor = Aggrescan3DPredictor(dynamic_mode=dynamic_mode)
    protein = ProteinRecord(
        id="query",
        sequence=sequence,
        structure_path=structure_path,
    )
    return predictor.predict(protein)
