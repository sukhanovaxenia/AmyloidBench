"""
PASTA 2.0 web predictor for amyloidogenicity prediction.

PASTA (Prediction of Amyloid Structure Aggregation) combines two complementary
approaches to identify amyloidogenic regions:

1. **Pairwise Energy Calculation**: Evaluates the propensity of sequence
   segments to form the parallel in-register cross-β structure typical of
   amyloid fibrils. Uses contact potentials derived from native protein
   structures to score potential β-sheet pairings.

2. **Structural Threading**: Threads the query sequence onto known amyloid
   fibril structures from the PDB, assessing compatibility with the steric
   zipper architecture.

Biological Basis:
The cross-β core of amyloid fibrils requires:
- Complementary side chain packing between adjacent β-strands
- Favorable hydrogen bonding along the fibril axis  
- Exclusion of water from the dry steric zipper interface

PASTA captures these requirements by:
- Scoring intra-sheet contacts using knowledge-based potentials
- Evaluating geometric compatibility with cross-β templates
- Penalizing charge clusters and proline residues that disrupt β-structure

Performance (from Walsh et al., 2014):
- Area under ROC: 0.86 on amyloidogenic peptides
- Successfully identifies APRs in Aβ, α-synuclein, tau, and prion proteins
- Particularly strong for predicting parallel in-register amyloid cores

Reference:
    Walsh, I., Seno, F., Tosatto, S.C.E., & Trovato, A. (2014).
    PASTA 2.0: an improved server for protein aggregation prediction.
    Nucleic Acids Research, 42(W1), W301-W307.
    DOI: 10.1093/nar/gku399

Web Server:
    http://protein.bio.unipd.it/pasta2/
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any, Optional

from amyloidbench.core.models import (
    PerResidueScores,
    PredictionResult,
    Region,
)
from amyloidbench.predictors.base import (
    PredictorCapability,
    PredictorConfig,
    PredictorType,
    register_predictor,
)

from .base import WebPredictorBase, WebPredictorConfig, PLAYWRIGHT_AVAILABLE

if PLAYWRIGHT_AVAILABLE:
    from playwright.async_api import Page

logger = logging.getLogger(__name__)


# PASTA energy threshold - more negative = more amyloidogenic
# The default PASTA threshold for significance
PASTA_ENERGY_THRESHOLD = -4.0


@register_predictor
class Pasta2Predictor(WebPredictorBase):
    """
    PASTA 2.0 predictor combining pairwise energy and structural threading.
    
    PASTA identifies amyloidogenic regions by evaluating their compatibility
    with the cross-β fibril architecture. It outputs:
    
    1. **PASTA Energy**: Per-residue aggregation propensity based on
       pairwise interaction energies. More negative = more amyloidogenic.
    
    2. **Best Pairing**: For each position, the optimal pairing partner
       that would form a cross-β structure.
    
    3. **Aggregation Hotspots**: Contiguous regions with favorable
       cross-β interaction energy.
    
    The method is particularly effective for predicting:
    - Parallel in-register β-sheet cores (most disease amyloids)
    - Steric zipper interfaces
    - Aggregation-prone segments in intrinsically disordered proteins
    
    Usage:
        >>> predictor = Pasta2Predictor()
        >>> result = await predictor.predict_async(protein)
        >>> for region in result.predicted_regions:
        ...     print(f"APR: {region.sequence} (energy: {region.score:.2f})")
        
    Attributes:
        base_url: PASTA 2.0 submission endpoint
        energy_threshold: Threshold for APR detection (default: -4.0)
    """
    
    name = "PASTA2"
    version = "2.0-web"
    predictor_type = PredictorType.THREADING
    capabilities = {
        PredictorCapability.PER_RESIDUE_SCORES,
        PredictorCapability.REGION_DETECTION,
        PredictorCapability.ASYNC_EXECUTION,
    }
    
    base_url = "http://protein.bio.unipd.it/pasta2/"
    default_threshold = PASTA_ENERGY_THRESHOLD
    
    # PASTA energies are negative (favorable) for amyloidogenic regions
    score_min = -10.0  # Highly amyloidogenic
    score_max = 0.0    # Not amyloidogenic
    
    citation = (
        "Walsh, I., Seno, F., Tosatto, S.C.E., & Trovato, A. (2014). "
        "PASTA 2.0: an improved server for protein aggregation prediction. "
        "Nucleic Acids Research, 42(W1), W301-W307."
    )
    url = "http://protein.bio.unipd.it/pasta2/"
    description = (
        "Threading-based predictor using pairwise energy calculations to "
        "assess compatibility with cross-β amyloid structure."
    )
    
    def __init__(
        self,
        config: Optional[PredictorConfig] = None,
        web_config: Optional[WebPredictorConfig] = None,
        energy_threshold: float = PASTA_ENERGY_THRESHOLD,
    ):
        """
        Initialize PASTA 2.0 predictor.
        
        Args:
            config: Base predictor configuration
            web_config: Web-specific settings
            energy_threshold: Energy threshold for APR detection
                             (more negative = more stringent)
        """
        if web_config is None:
            web_config = WebPredictorConfig(
                min_request_interval=5.0,  # PASTA can be slow
                result_timeout=180.0,  # May take time for long sequences
                max_retries=2,
            )
        
        super().__init__(config, web_config)
        self.energy_threshold = energy_threshold
    
    async def _submit_and_parse(
        self,
        page: Page,
        sequence: str,
    ) -> dict[str, Any]:
        """
        Submit sequence to PASTA 2.0 and parse results.
        
        PASTA 2.0 workflow:
        1. Submit sequence on main page
        2. Wait for job completion (may show progress)
        3. Parse energy profile and identified regions
        
        Args:
            page: Playwright page instance
            sequence: Protein sequence
            
        Returns:
            Parsed results dictionary
        """
        # Navigate to PASTA
        await page.goto(self.base_url, wait_until="networkidle")
        
        # Find sequence input field
        input_selectors = [
            'textarea[name="sequence"]',
            'textarea#sequence',
            '#seq',
            'textarea',
        ]
        
        textarea = None
        for selector in input_selectors:
            try:
                textarea = await page.wait_for_selector(selector, timeout=5000)
                if textarea:
                    break
            except Exception:
                continue
        
        if not textarea:
            raise Exception("Could not find sequence input")
        
        # Enter sequence
        await textarea.fill(f">query\n{sequence}")
        
        # Find and click submit button
        submit_selectors = [
            'input[type="submit"]',
            'button[type="submit"]',
            'input[value*="Submit"]',
            'button:has-text("Submit")',
            '#submit',
        ]
        
        for selector in submit_selectors:
            try:
                submit_btn = await page.wait_for_selector(selector, timeout=3000)
                if submit_btn:
                    await submit_btn.click()
                    break
            except Exception:
                continue
        
        # Wait for results
        # PASTA may show a waiting page then redirect
        await page.wait_for_load_state("networkidle")
        
        # Check for job queue page and wait
        for _ in range(30):  # Max 5 minutes of waiting
            content = await page.content()
            
            if "queue" in content.lower() or "waiting" in content.lower():
                await asyncio.sleep(10)
                await page.reload()
                await page.wait_for_load_state("networkidle")
            else:
                break
        
        # Get final results
        content = await page.content()
        
        return self._parse_pasta_results(content, sequence)
    
    def _parse_pasta_results(
        self,
        html: str,
        sequence: str,
    ) -> dict[str, Any]:
        """
        Parse PASTA 2.0 results from HTML.
        
        PASTA output includes:
        - Per-residue PASTA energy (aggregation profile)
        - Best pairing energies and partners
        - Identified aggregation hotspots
        
        Args:
            html: HTML content
            sequence: Original sequence
            
        Returns:
            Parsed results
        """
        result = {
            "scores": [],
            "regions": [],
            "is_amyloidogenic": None,
            "pairings": [],
            "raw_html": html if self.config.return_raw_output else None,
        }
        
        # Extract energy profile
        energies = self._extract_pasta_energies(html, len(sequence))
        if energies:
            result["scores"] = energies
        
        # Extract hotspots/regions
        regions = self._extract_pasta_regions(html, sequence)
        result["regions"] = regions
        result["is_amyloidogenic"] = len(regions) > 0
        
        # Extract best pairings if available
        pairings = self._extract_pairings(html)
        if pairings:
            result["pairings"] = pairings
        
        return result
    
    def _extract_pasta_energies(
        self,
        html: str,
        seq_length: int,
    ) -> list[float]:
        """
        Extract per-residue PASTA energies.
        
        PASTA outputs energy values where more negative = more amyloidogenic.
        
        Args:
            html: HTML content
            seq_length: Expected sequence length
            
        Returns:
            List of energy values per residue
        """
        energies = []
        
        # Pattern for PASTA energy table
        # Typically: position, residue, energy
        energy_patterns = [
            # Table format
            r'<tr[^>]*>\s*<td[^>]*>(\d+)</td>\s*<td[^>]*>[A-Z]</td>\s*<td[^>]*>([-+]?\d*\.?\d+)</td>',
            # CSV-like format
            r'(\d+)\s+[A-Z]\s+([-+]?\d*\.?\d+)',
            # Alternative table
            r'<td>(\d+)</td>\s*<td>[A-Z]</td>\s*<td>([-+]?\d+\.?\d*)</td>',
        ]
        
        for pattern in energy_patterns:
            matches = re.findall(pattern, html, re.IGNORECASE | re.DOTALL)
            if matches:
                # Convert to position-indexed energies
                pos_energy = {}
                for match in matches:
                    pos = int(match[0]) - 1  # 0-indexed
                    energy = float(match[1])
                    pos_energy[pos] = energy
                
                if pos_energy:
                    energies = [
                        pos_energy.get(i, 0.0) 
                        for i in range(seq_length)
                    ]
                    break
        
        return energies
    
    def _extract_pasta_regions(
        self,
        html: str,
        sequence: str,
    ) -> list[tuple[int, int, float]]:
        """
        Extract predicted aggregation hotspots from PASTA.
        
        PASTA identifies regions with significant aggregation propensity
        based on sustained negative energy.
        
        Args:
            html: HTML content
            sequence: Original sequence
            
        Returns:
            List of (start, end, energy) tuples
        """
        regions = []
        
        # Pattern for hotspot regions
        # PASTA lists regions like: "Residues 15-25: energy = -6.5"
        region_patterns = [
            r'residues?\s*(\d+)\s*[-–to]+\s*(\d+)\s*[:\s]*energy[:\s=]*([-+]?\d*\.?\d+)',
            r'hotspot\s*[:\s]*(\d+)\s*[-–]+\s*(\d+)\s*[,\s]*([-+]?\d*\.?\d+)',
            r'(\d+)\s*-\s*(\d+)\s*:\s*([-+]?\d+\.?\d*)\s*kcal',
        ]
        
        for pattern in region_patterns:
            for match in re.finditer(pattern, html, re.IGNORECASE):
                start = int(match.group(1)) - 1  # 0-indexed
                end = int(match.group(2))
                energy = float(match.group(3))
                
                if energy <= self.energy_threshold:
                    regions.append((start, end, energy))
        
        # Also extract from highlighted sequence regions
        highlight_pattern = re.compile(
            r'<span[^>]*class="[^"]*hotspot[^"]*"[^>]*>([A-Z]+)</span>',
            re.IGNORECASE
        )
        
        for match in highlight_pattern.finditer(html):
            region_seq = match.group(1)
            pos = sequence.find(region_seq)
            if pos != -1:
                # Check if we already have this region
                if not any(r[0] == pos for r in regions):
                    regions.append((pos, pos + len(region_seq), None))
        
        # Sort by position
        regions.sort(key=lambda x: x[0])
        
        return regions
    
    def _extract_pairings(self, html: str) -> list[dict]:
        """
        Extract best pairing information from PASTA output.
        
        PASTA identifies optimal pairing partners for cross-β formation.
        
        Args:
            html: HTML content
            
        Returns:
            List of pairing dictionaries with pos1, pos2, energy
        """
        pairings = []
        
        # Pattern for pairing data
        pairing_pattern = re.compile(
            r'(\d+)\s*[-–]\s*(\d+)\s*pairs?\s*with\s*(\d+)\s*[-–]\s*(\d+)',
            re.IGNORECASE
        )
        
        for match in pairing_pattern.finditer(html):
            pairings.append({
                "segment1": (int(match.group(1)), int(match.group(2))),
                "segment2": (int(match.group(3)), int(match.group(4))),
            })
        
        return pairings
    
    def can_handle(self, protein) -> bool:
        """
        Check if PASTA can handle this protein.
        
        PASTA 2.0 limits:
        - Minimum length: ~10 residues
        - Maximum length: ~2000 residues (server constraint)
        """
        seq_len = len(protein.sequence)
        if seq_len < 10:
            logger.warning(f"Sequence may be too short for PASTA ({seq_len})")
            return False
        if seq_len > 2000:
            logger.warning(f"Sequence may exceed PASTA length limit ({seq_len})")
        return True


# =============================================================================
# Local PASTA-like approximation
# =============================================================================

# Simplified pairwise potential matrix (subset of real PASTA)
# Values represent favorable (negative) or unfavorable (positive) pairings
# for cross-β sheet formation
PASTA_PAIR_POTENTIAL = {
    # Hydrophobic-hydrophobic: favorable
    ('V', 'V'): -0.8, ('V', 'I'): -0.7, ('V', 'L'): -0.6, ('V', 'F'): -0.9,
    ('I', 'I'): -0.9, ('I', 'L'): -0.7, ('I', 'F'): -1.0,
    ('L', 'L'): -0.6, ('L', 'F'): -0.8,
    ('F', 'F'): -1.2, ('F', 'Y'): -1.0, ('Y', 'Y'): -0.8,
    
    # Aromatic stacking: very favorable
    ('F', 'W'): -1.1, ('Y', 'W'): -0.9, ('W', 'W'): -0.7,
    
    # Charged pairs: unfavorable (same charge)
    ('K', 'K'): 0.8, ('R', 'R'): 0.7, ('K', 'R'): 0.6,
    ('D', 'D'): 0.8, ('E', 'E'): 0.7, ('D', 'E'): 0.6,
    
    # Salt bridges: slightly favorable
    ('K', 'D'): -0.3, ('K', 'E'): -0.2, ('R', 'D'): -0.3, ('R', 'E'): -0.2,
    
    # Proline: very unfavorable (breaks β-sheet)
    ('P', 'P'): 1.5,
    
    # Glycine: mildly unfavorable (too flexible)
    ('G', 'G'): 0.3,
}


def get_pair_energy(aa1: str, aa2: str) -> float:
    """
    Get pairwise interaction energy for two residues.
    
    Uses symmetric lookup in the pair potential matrix.
    
    Args:
        aa1, aa2: Single-letter amino acid codes
        
    Returns:
        Interaction energy (negative = favorable)
    """
    aa1, aa2 = aa1.upper(), aa2.upper()
    
    # Try both orderings
    if (aa1, aa2) in PASTA_PAIR_POTENTIAL:
        return PASTA_PAIR_POTENTIAL[(aa1, aa2)]
    if (aa2, aa1) in PASTA_PAIR_POTENTIAL:
        return PASTA_PAIR_POTENTIAL[(aa2, aa1)]
    
    # Default: based on hydrophobicity similarity
    hydrophobic = set('VILFYWM')
    charged = set('KRDEH')
    
    if aa1 in hydrophobic and aa2 in hydrophobic:
        return -0.3  # Mildly favorable
    if aa1 in charged or aa2 in charged:
        return 0.2  # Mildly unfavorable
    
    return 0.0  # Neutral


def calculate_pasta_like_energy(segment: str) -> float:
    """
    Calculate PASTA-like aggregation energy for a segment.
    
    Simulates the pairwise energy calculation for parallel
    in-register β-sheet formation.
    
    Args:
        segment: Peptide sequence (typically 6-10 residues)
        
    Returns:
        Total interaction energy (more negative = more amyloidogenic)
    """
    if len(segment) < 4:
        return 0.0
    
    energy = 0.0
    
    # Evaluate in-register pairing (residue i pairs with i in adjacent strand)
    for i, aa in enumerate(segment):
        # Self-pairing energy (in-register)
        energy += get_pair_energy(aa, aa) * 0.5
        
        # Cross-strand H-bond pattern (i+2, i-2 interactions in β-sheet)
        if i + 2 < len(segment):
            energy += get_pair_energy(aa, segment[i+2]) * 0.3
    
    return energy


def predict_pasta_local(
    sequence: str,
    window_size: int = 7,
    threshold: float = -2.0,
) -> list[tuple[int, int, float]]:
    """
    Local PASTA-like prediction without web access.
    
    This provides a rough approximation of PASTA scoring
    when the server is unavailable.
    
    Args:
        sequence: Protein sequence
        window_size: Sliding window size
        threshold: Energy threshold for APR detection
        
    Returns:
        List of (start, end, energy) tuples
    """
    if len(sequence) < window_size:
        return []
    
    # Calculate energies for each window
    window_energies = []
    for i in range(len(sequence) - window_size + 1):
        segment = sequence[i:i + window_size]
        energy = calculate_pasta_like_energy(segment)
        window_energies.append((i, energy))
    
    # Find regions below threshold
    regions = []
    in_region = False
    region_start = 0
    region_energies = []
    
    for pos, energy in window_energies:
        if energy <= threshold and not in_region:
            in_region = True
            region_start = pos
            region_energies = [energy]
        elif energy <= threshold and in_region:
            region_energies.append(energy)
        elif energy > threshold and in_region:
            in_region = False
            region_end = pos + window_size
            avg_energy = sum(region_energies) / len(region_energies)
            regions.append((region_start, region_end, avg_energy))
    
    # Handle region at end
    if in_region:
        region_end = len(sequence)
        avg_energy = sum(region_energies) / len(region_energies)
        regions.append((region_start, region_end, avg_energy))
    
    return regions
