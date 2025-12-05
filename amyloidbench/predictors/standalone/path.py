"""
PATH (Prediction of Amyloidogenicity by THreading) Standalone Launcher.

PATH is a structure-based method that predicts amyloidogenicity by threading
query hexapeptides onto experimentally determined amyloid structures representing
all eight steric zipper classes.

Reference:
Wojciechowski JW, et al. (2020) Scientific Reports 10:7721
DOI: 10.1038/s41598-020-64270-3

Algorithm:
1. Template selection: 8 representative steric zipper structures
2. Threading: Comparative modeling of query on each template
3. Energy calculation: DOPE statistical potential for model quality
4. Classification: Machine learning on energy features

Requirements:
- MODELLER (for comparative modeling)
- PATH source code (github.com/kotulska-lab/PATH)

Note: This is a launcher/wrapper. Users must install PATH dependencies separately.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
import json

import numpy as np

try:
    from ..output_models import (
        PredictorOutput,
        ResidueScore,
        PredictedRegion,
        ClassificationLabel,
        ScoreType,
    )
except ImportError:
    from amyloidbench.predictors.output_models import (
        PredictorOutput,
        ResidueScore,
        PredictedRegion,
        ClassificationLabel,
        ScoreType,
    )


logger = logging.getLogger(__name__)


# PATH steric zipper classes (Sawaya et al., 2007)
STERIC_ZIPPER_CLASSES = {
    1: "Face-to-face, up-up (antiparallel)",
    2: "Face-to-back, up-down (antiparallel)",
    3: "Face-to-face, up-down (antiparallel)",
    4: "Face-to-back, up-up (antiparallel)",
    5: "Face-to-face, up-up (parallel)",
    6: "Face-to-back, up-down (parallel)",
    7: "Face-to-face, up-down (parallel)",
    8: "Face-to-back, up-up (parallel)",
}


class PathPredictor:
    """
    Launcher for PATH amyloid prediction tool.
    
    PATH uses structural threading to predict amyloidogenicity:
    
    1. Input hexapeptide is modeled onto 8 steric zipper templates
    2. DOPE energy score evaluates model quality for each class
    3. Best (lowest) DOPE indicates most compatible structure
    4. Machine learning classifier makes final prediction
    
    The structural approach provides insight into potential fibril
    architecture, distinguishing between parallel/antiparallel and
    different face packing arrangements.
    """
    
    predictor_name = "PATH"
    predictor_version = "1.0"
    score_type = ScoreType.ENERGY  # DOPE energy (lower = better)
    default_threshold = -0.3  # Typical classification threshold
    
    def __init__(
        self,
        path_executable: Optional[str] = None,
        modeller_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        timeout: int = 600,
    ):
        """
        Initialize PATH predictor.
        
        Args:
            path_executable: Path to PATH script or executable
            modeller_path: Path to MODELLER installation
            output_dir: Directory for output files
            timeout: Maximum execution time in seconds
        """
        self.path_executable = path_executable or self._find_path_executable()
        self.modeller_path = modeller_path
        self.output_dir = output_dir
        self.timeout = timeout
        
        # Check installation
        self._verify_installation()
    
    def _find_path_executable(self) -> Optional[str]:
        """Try to find PATH installation."""
        # Common locations
        search_paths = [
            Path.home() / "PATH",
            Path.home() / "software" / "PATH",
            Path("/opt/PATH"),
            Path("/usr/local/PATH"),
        ]
        
        for path in search_paths:
            if (path / "path.py").exists():
                return str(path / "path.py")
            if (path / "PATH.py").exists():
                return str(path / "PATH.py")
        
        # Try PATH environment
        env_path = os.environ.get("PATH_HOME")
        if env_path:
            path_dir = Path(env_path)
            if (path_dir / "path.py").exists():
                return str(path_dir / "path.py")
        
        return None
    
    def _verify_installation(self) -> bool:
        """Verify PATH and dependencies are installed."""
        warnings = []
        
        if not self.path_executable or not Path(self.path_executable).exists():
            warnings.append("PATH executable not found")
        
        # Check for MODELLER
        try:
            result = subprocess.run(
                ["python", "-c", "import modeller"],
                capture_output=True,
                timeout=10,
            )
            if result.returncode != 0:
                warnings.append("MODELLER not available in Python")
        except Exception as e:
            warnings.append(f"Could not check MODELLER: {e}")
        
        if warnings:
            logger.warning(f"PATH installation issues: {'; '.join(warnings)}")
            return False
        
        return True
    
    def predict(
        self,
        sequence: str,
        sequence_id: str = "query",
    ) -> PredictorOutput:
        """
        Run PATH prediction on a sequence.
        
        PATH analyzes hexapeptides, so longer sequences are scanned
        with a sliding window approach.
        
        Args:
            sequence: Protein sequence
            sequence_id: Identifier for the sequence
            
        Returns:
            PredictorOutput with per-residue scores and structural class info
        """
        sequence = self._validate_sequence(sequence)
        
        # Set up output directory
        if self.output_dir:
            output_dir = Path(self.output_dir) / sequence_id
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(tempfile.mkdtemp(prefix="path_"))
        
        # Run PATH on each hexapeptide window
        residue_scores, hexapeptide_results = self._scan_sequence(
            sequence, sequence_id, output_dir
        )
        
        # Identify amyloidogenic regions
        predicted_regions = self._identify_regions(residue_scores, sequence)
        
        # Create output
        return self._create_output(
            sequence=sequence,
            sequence_id=sequence_id,
            residue_scores=residue_scores,
            predicted_regions=predicted_regions,
            hexapeptide_results=hexapeptide_results,
            output_dir=output_dir,
        )
    
    def _validate_sequence(self, sequence: str) -> str:
        """Validate and clean sequence."""
        sequence = "".join(sequence.split()).upper()
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        
        invalid = set(sequence) - valid_aa
        if invalid:
            logger.warning(f"Removing invalid characters: {invalid}")
            sequence = "".join(aa for aa in sequence if aa in valid_aa)
        
        if len(sequence) < 6:
            raise ValueError("Sequence must be at least 6 amino acids")
        
        return sequence
    
    def _scan_sequence(
        self,
        sequence: str,
        sequence_id: str,
        output_dir: Path,
    ) -> tuple[list[ResidueScore], list[dict]]:
        """Scan sequence with hexapeptide windows."""
        n = len(sequence)
        scores = [None] * n
        best_classes = [None] * n
        hexapeptide_results = []
        
        for i in range(n - 5):
            hexapeptide = sequence[i:i + 6]
            
            try:
                result = self._predict_hexapeptide(hexapeptide, i, output_dir)
                hexapeptide_results.append(result)
                
                # Assign scores to positions
                for j in range(i, i + 6):
                    if scores[j] is None or result['best_dope'] < scores[j]:
                        scores[j] = result['best_dope']
                        best_classes[j] = result['best_class']
                        
            except Exception as e:
                logger.warning(f"Failed to predict hexapeptide {i}: {e}")
                hexapeptide_results.append({
                    'position': i + 1,
                    'hexapeptide': hexapeptide,
                    'error': str(e),
                })
        
        # Fill missing scores
        for i in range(n):
            if scores[i] is None:
                scores[i] = 0.0
        
        # Convert to normalized scores (lower DOPE = better = higher normalized)
        min_dope = min(s for s in scores if s is not None)
        max_dope = max(s for s in scores if s is not None)
        
        if max_dope != min_dope:
            normalized = [(max_dope - s) / (max_dope - min_dope) for s in scores]
        else:
            normalized = [0.5] * n
        
        # Create ResidueScore objects
        residue_scores = []
        for i, (aa, raw, norm) in enumerate(zip(sequence, scores, normalized)):
            classification = (
                ClassificationLabel.AMYLOIDOGENIC 
                if norm >= 0.5  # Above median = amyloidogenic
                else ClassificationLabel.NON_AMYLOIDOGENIC
            )
            
            residue_scores.append(ResidueScore(
                position=i + 1,
                residue=aa,
                raw_score=raw if raw is not None else 0.0,
                normalized_score=norm,
                classification=classification,
                confidence=norm,
            ))
        
        return residue_scores, hexapeptide_results
    
    def _predict_hexapeptide(
        self,
        hexapeptide: str,
        position: int,
        output_dir: Path,
    ) -> dict:
        """
        Predict amyloidogenicity of a single hexapeptide.
        
        This method either calls PATH directly or uses a fallback
        approximation if PATH is not available.
        """
        if self.path_executable and Path(self.path_executable).exists():
            return self._run_path(hexapeptide, position, output_dir)
        else:
            return self._fallback_prediction(hexapeptide, position)
    
    def _run_path(
        self,
        hexapeptide: str,
        position: int,
        output_dir: Path,
    ) -> dict:
        """Run actual PATH prediction."""
        # Create input file
        input_file = output_dir / f"hex_{position}.fasta"
        input_file.write_text(f">hex_{position}\n{hexapeptide}\n")
        
        # Run PATH
        result = subprocess.run(
            ["python", self.path_executable, str(input_file)],
            capture_output=True,
            text=True,
            timeout=self.timeout,
            cwd=str(output_dir),
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"PATH failed: {result.stderr}")
        
        # Parse output
        return self._parse_path_output(result.stdout, hexapeptide, position)
    
    def _parse_path_output(
        self,
        output: str,
        hexapeptide: str,
        position: int,
    ) -> dict:
        """Parse PATH command output."""
        # PATH typically outputs:
        # Hexapeptide, Class1_DOPE, Class2_DOPE, ..., Class8_DOPE, Prediction
        
        result = {
            'position': position + 1,
            'hexapeptide': hexapeptide,
            'class_dopes': {},
            'best_class': None,
            'best_dope': 0.0,
            'is_amyloid': False,
        }
        
        lines = output.strip().split('\n')
        for line in lines:
            parts = line.split(',')
            if len(parts) >= 9 and parts[0].strip() == hexapeptide:
                # Parse DOPE scores for each class
                dopes = []
                for i, class_id in enumerate(range(1, 9)):
                    try:
                        dope = float(parts[i + 1].strip())
                        result['class_dopes'][class_id] = dope
                        dopes.append((class_id, dope))
                    except (ValueError, IndexError):
                        continue
                
                if dopes:
                    # Best class has lowest (most negative) DOPE
                    best = min(dopes, key=lambda x: x[1])
                    result['best_class'] = best[0]
                    result['best_dope'] = best[1]
                
                # Check prediction
                if len(parts) > 9:
                    result['is_amyloid'] = parts[9].strip().lower() in ['1', 'true', 'amyloid', 'yes']
                else:
                    result['is_amyloid'] = result['best_dope'] < self.default_threshold
        
        return result
    
    def _fallback_prediction(self, hexapeptide: str, position: int) -> dict:
        """
        Fallback prediction when PATH is not available.
        
        Uses a simplified scoring based on amino acid properties that
        approximate steric zipper compatibility.
        """
        # Steric zipper compatibility scores (simplified)
        # Based on hydrophobicity and β-propensity patterns
        COMPAT = {
            'I': 0.95, 'V': 0.90, 'L': 0.85, 'F': 0.88, 'Y': 0.80,
            'W': 0.75, 'M': 0.70, 'A': 0.60, 'C': 0.65, 'T': 0.55,
            'S': 0.45, 'G': 0.40, 'N': 0.50, 'Q': 0.55, 'H': 0.45,
            'P': 0.15, 'K': 0.25, 'R': 0.25, 'D': 0.20, 'E': 0.20,
        }
        
        # Position weights (central positions more important)
        POSITION_WEIGHTS = [0.8, 1.0, 1.0, 1.0, 1.0, 0.8]
        
        # Calculate weighted score
        score = 0.0
        for i, aa in enumerate(hexapeptide):
            score += COMPAT.get(aa, 0.5) * POSITION_WEIGHTS[i]
        
        score /= sum(POSITION_WEIGHTS)
        
        # Convert to DOPE-like energy (negative = better)
        dope_approx = -score * 2  # Scale to typical DOPE range
        
        return {
            'position': position + 1,
            'hexapeptide': hexapeptide,
            'class_dopes': {i: dope_approx * (0.9 + 0.2 * np.random.random()) 
                          for i in range(1, 9)},
            'best_class': 5 if score > 0.6 else 1,  # Parallel vs antiparallel
            'best_dope': dope_approx,
            'is_amyloid': score > 0.5,
            'method': 'fallback',
        }
    
    def _identify_regions(
        self,
        residue_scores: list[ResidueScore],
        sequence: str,
        min_length: int = 6,
    ) -> list[PredictedRegion]:
        """Identify amyloidogenic regions from scores."""
        regions = []
        in_region = False
        region_start = 0
        region_scores = []
        
        for score in residue_scores:
            if score.classification == ClassificationLabel.AMYLOIDOGENIC:
                if not in_region:
                    region_start = score.position
                    region_scores = [score.normalized_score]
                    in_region = True
                else:
                    region_scores.append(score.normalized_score)
            else:
                if in_region:
                    region_end = score.position - 1
                    if region_end - region_start + 1 >= min_length:
                        regions.append(PredictedRegion(
                            start=region_start,
                            end=region_end,
                            sequence=sequence[region_start - 1:region_end],
                            mean_score=float(np.mean(region_scores)),
                            max_score=float(np.max(region_scores)),
                            mean_normalized=float(np.mean(region_scores)),
                            region_type="steric_zipper",
                        ))
                    in_region = False
        
        if in_region:
            region_end = residue_scores[-1].position
            if region_end - region_start + 1 >= min_length:
                regions.append(PredictedRegion(
                    start=region_start,
                    end=region_end,
                    sequence=sequence[region_start - 1:region_end],
                    mean_score=float(np.mean(region_scores)),
                    max_score=float(np.max(region_scores)),
                    mean_normalized=float(np.mean(region_scores)),
                    region_type="steric_zipper",
                ))
        
        return regions
    
    def _create_output(
        self,
        sequence: str,
        sequence_id: str,
        residue_scores: list[ResidueScore],
        predicted_regions: list[PredictedRegion],
        hexapeptide_results: list[dict],
        output_dir: Path,
    ) -> PredictorOutput:
        """Create standardized PredictorOutput."""
        
        # Aggregate statistics
        valid_results = [r for r in hexapeptide_results if 'error' not in r]
        
        raw_output = {
            'output_dir': str(output_dir),
            'n_hexapeptides': len(hexapeptide_results),
            'n_successful': len(valid_results),
            'hexapeptide_results': hexapeptide_results,
            'steric_zipper_classes': STERIC_ZIPPER_CLASSES,
        }
        
        # Save results to file
        results_file = output_dir / "path_results.json"
        with open(results_file, 'w') as f:
            json.dump(raw_output, f, indent=2, default=str)
        
        is_amyloidogenic = len(predicted_regions) > 0
        raw_values = [r.raw_score for r in residue_scores]
        normalized_values = [r.normalized_score for r in residue_scores]
        
        return PredictorOutput(
            predictor_name=self.predictor_name,
            predictor_version=self.predictor_version,
            sequence_id=sequence_id,
            sequence=sequence,
            residue_scores=residue_scores,
            predicted_regions=predicted_regions,
            overall_classification=(
                ClassificationLabel.AMYLOIDOGENIC if is_amyloidogenic
                else ClassificationLabel.NON_AMYLOIDOGENIC
            ),
            overall_score=float(np.min(raw_values)) if raw_values else 0.0,
            overall_probability=float(np.max(normalized_values)) if normalized_values else 0.0,
            score_type=self.score_type,
            threshold=self.default_threshold,
            source="standalone",
            raw_output=raw_output,
        )


def predict_with_path(
    sequence: str,
    sequence_id: str = "query",
    path_executable: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> PredictorOutput:
    """
    Predict amyloidogenicity using PATH.
    
    Args:
        sequence: Protein sequence
        sequence_id: Identifier for the sequence
        path_executable: Path to PATH installation
        output_dir: Directory for output files
        
    Returns:
        PredictorOutput with per-residue scores and structural class info
    """
    predictor = PathPredictor(
        path_executable=path_executable,
        output_dir=output_dir,
    )
    return predictor.predict(sequence, sequence_id)
