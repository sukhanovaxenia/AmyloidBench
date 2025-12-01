"""
WALTZ web predictor for hexapeptide amyloidogenicity.

WALTZ (Workflow for AnaLyzing Turns and Z-segments) predicts amyloidogenic
regions using position-specific scoring matrices (PSSMs) derived from
experimental hexapeptide aggregation data. The algorithm identifies short
segments (≥6 residues) with amino acid patterns characteristic of
amyloid-forming sequences.

Biological Basis:
The WALTZ approach is grounded in the observation that amyloid formation
is largely determined by short sequence segments (4-10 residues) that can
adopt cross-β structure. The method was trained on:
- Positive set: Hexapeptides confirmed to form amyloid-like fibrils by 
  thioflavin T binding and electron microscopy
- Negative set: Hexapeptides that remain soluble under aggregation-promoting
  conditions

The PSSM captures position-specific preferences within the amyloidogenic
hexapeptide:
- Positions 1-2 and 5-6: Often occupied by β-sheet-promoting residues (V, I, L, F)
- Position 3: More permissive, accommodates turn-promoting residues
- Position 4: Often a hydrophobic anchor

Performance (from original publication):
- Sensitivity: 75% on amyloidogenic hexapeptides
- Specificity: 75% on non-aggregating peptides
- Area under ROC: 0.83

Reference:
    Maurer-Stroh, S., et al. (2010). Exploring the sequence determinants
    of amyloid structure using position-specific scoring matrices.
    Nature Methods, 7(3), 237-242. DOI: 10.1038/nmeth.1432

Web Server:
    http://waltz.switchlab.org
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


# WALTZ threshold from original publication
# Scores above this indicate amyloidogenic hexapeptides
WALTZ_THRESHOLD = 0.0  # WALTZ uses 0 as decision boundary


@register_predictor
class WaltzPredictor(WebPredictorBase):
    """
    WALTZ web predictor for hexapeptide-based amyloid prediction.
    
    WALTZ uses position-specific scoring matrices trained on experimental
    hexapeptide aggregation data. It scans the input sequence with a
    sliding window, scoring each hexapeptide for amyloidogenic potential.
    
    The output includes:
    - Per-hexapeptide scores (can be mapped to per-residue)
    - Identified amyloidogenic regions (contiguous high-scoring segments)
    - Visual representation of the aggregation profile
    
    Usage:
        >>> predictor = WaltzPredictor()
        >>> protein = ProteinRecord(id="test", sequence="KLVFFAEDVGSNK")
        >>> result = await predictor.predict_async(protein)
        
    Note:
        This predictor requires internet access and may be slow due to
        server-side computation. Results are cached to avoid redundant
        requests.
    
    Attributes:
        base_url: WALTZ submission endpoint
        threshold: Score threshold for APR detection (default: 0.0)
    """
    
    name = "WALTZ"
    version = "1.0-web"
    predictor_type = PredictorType.SEQUENCE_ML
    capabilities = {
        PredictorCapability.PER_RESIDUE_SCORES,
        PredictorCapability.REGION_DETECTION,
        PredictorCapability.ASYNC_EXECUTION,
    }
    
    base_url = "http://waltz.switchlab.org/"
    default_threshold = WALTZ_THRESHOLD
    
    # WALTZ scores typically range from -2 to +3
    score_min = -2.0
    score_max = 3.0
    
    citation = (
        "Maurer-Stroh, S., et al. (2010). Exploring the sequence determinants "
        "of amyloid structure using position-specific scoring matrices. "
        "Nature Methods, 7(3), 237-242."
    )
    url = "http://waltz.switchlab.org"
    description = (
        "Position-specific scoring matrix approach trained on experimental "
        "hexapeptide aggregation data. Identifies short amyloidogenic segments."
    )
    
    def __init__(
        self,
        config: Optional[PredictorConfig] = None,
        web_config: Optional[WebPredictorConfig] = None,
    ):
        """
        Initialize WALTZ predictor.
        
        Args:
            config: Base predictor configuration
            web_config: Web-specific settings (rate limiting, timeouts)
        """
        # Set conservative defaults for WALTZ server
        if web_config is None:
            web_config = WebPredictorConfig(
                min_request_interval=3.0,  # Be gentle with the server
                result_timeout=120.0,
                max_retries=2,
            )
        
        super().__init__(config, web_config)
    
    async def _submit_and_parse(
        self,
        page: Page,
        sequence: str,
    ) -> dict[str, Any]:
        """
        Submit sequence to WALTZ server and parse results.
        
        The WALTZ workflow:
        1. Navigate to submission page
        2. Enter sequence in the input field
        3. Submit and wait for results
        4. Parse the score table and highlighted regions
        
        Args:
            page: Playwright page instance
            sequence: Protein sequence
            
        Returns:
            Parsed result dictionary
        """
        # Navigate to the submission page
        await page.goto(self.base_url, wait_until="networkidle")
        
        # Find and fill the sequence input
        # WALTZ uses a textarea for sequence input
        textarea_selectors = [
            'textarea[name="sequence"]',
            'textarea#sequence',
            'textarea.sequence-input',
            'textarea',  # Fallback
        ]
        
        textarea = None
        for selector in textarea_selectors:
            try:
                textarea = await page.wait_for_selector(selector, timeout=5000)
                if textarea:
                    break
            except Exception:
                continue
        
        if not textarea:
            raise Exception("Could not find sequence input field")
        
        # Clear and enter sequence
        await textarea.fill(sequence)
        
        # Find and click submit button
        submit_selectors = [
            'input[type="submit"]',
            'button[type="submit"]',
            'button:has-text("Submit")',
            'input[value="Submit"]',
            '#submit',
        ]
        
        submit_btn = None
        for selector in submit_selectors:
            try:
                submit_btn = await page.wait_for_selector(selector, timeout=3000)
                if submit_btn:
                    break
            except Exception:
                continue
        
        if submit_btn:
            await submit_btn.click()
        else:
            # Try submitting form directly
            await page.evaluate("document.forms[0].submit()")
        
        # Wait for results page
        # WALTZ may redirect to a results page or update dynamically
        await page.wait_for_load_state("networkidle", timeout=self.web_config.result_timeout * 1000)
        
        # Additional wait for dynamic content
        await asyncio.sleep(2)
        
        # Get page content
        content = await page.content()
        
        # Parse results
        return self._parse_waltz_results(content, sequence)
    
    def _parse_waltz_results(
        self,
        html: str,
        sequence: str,
    ) -> dict[str, Any]:
        """
        Parse WALTZ results from HTML.
        
        WALTZ output includes:
        - A table of hexapeptide scores
        - Highlighted amyloidogenic regions
        - Summary statistics
        
        Args:
            html: HTML content of results page
            sequence: Original sequence for length reference
            
        Returns:
            Parsed result dictionary
        """
        result = {
            "scores": [],
            "regions": [],
            "is_amyloidogenic": None,
            "raw_html": html if self.config.return_raw_output else None,
        }
        
        # Try to extract hexapeptide scores
        # WALTZ typically outputs scores in a table format
        scores = self._extract_hexapeptide_scores(html, len(sequence))
        if scores:
            result["scores"] = scores
        
        # Extract highlighted regions
        regions = self._extract_waltz_regions(html, sequence)
        if regions:
            result["regions"] = regions
            result["is_amyloidogenic"] = True
        else:
            result["is_amyloidogenic"] = False
        
        return result
    
    def _extract_hexapeptide_scores(
        self,
        html: str,
        seq_length: int,
    ) -> list[float]:
        """
        Extract hexapeptide scores and map to per-residue format.
        
        WALTZ outputs scores for each hexapeptide window. We map these
        to per-residue scores by taking the maximum score of all
        hexapeptides containing each residue.
        
        Args:
            html: HTML content
            seq_length: Sequence length
            
        Returns:
            Per-residue score list
        """
        # Pattern for score extraction - adjust based on actual WALTZ output format
        # This pattern looks for position and score pairs
        score_pattern = re.compile(
            r'(\d+)\s*[-–]\s*(\d+)\s*:\s*([-+]?\d*\.?\d+)',
            re.IGNORECASE
        )
        
        # Alternative: look for table rows with numeric data
        table_pattern = re.compile(
            r'<tr[^>]*>\s*<td[^>]*>(\d+)</td>\s*<td[^>]*>[^<]*</td>\s*<td[^>]*>([-+]?\d*\.?\d+)</td>',
            re.IGNORECASE | re.DOTALL
        )
        
        hexapeptide_scores = {}  # position -> score
        
        # Try score pattern
        for match in score_pattern.finditer(html):
            start = int(match.group(1)) - 1  # Convert to 0-indexed
            score = float(match.group(3))
            hexapeptide_scores[start] = score
        
        # Try table pattern
        for match in table_pattern.finditer(html):
            pos = int(match.group(1)) - 1
            score = float(match.group(2))
            hexapeptide_scores[pos] = score
        
        if not hexapeptide_scores:
            return []
        
        # Map hexapeptide scores to per-residue
        # Each residue gets the max score of hexapeptides containing it
        window_size = 6
        per_residue = [0.0] * seq_length
        
        for start_pos, score in hexapeptide_scores.items():
            for i in range(start_pos, min(start_pos + window_size, seq_length)):
                per_residue[i] = max(per_residue[i], score)
        
        return per_residue
    
    def _extract_waltz_regions(
        self,
        html: str,
        sequence: str,
    ) -> list[tuple[int, int, float]]:
        """
        Extract predicted amyloidogenic regions from WALTZ output.
        
        WALTZ highlights amyloidogenic regions, often in red or with
        specific CSS classes. This function identifies these regions.
        
        Args:
            html: HTML content
            sequence: Original sequence
            
        Returns:
            List of (start, end, score) tuples
        """
        regions = []
        
        # Pattern for highlighted regions
        # WALTZ often uses colored spans or specific classes
        highlight_patterns = [
            r'<span[^>]*style="[^"]*color:\s*red[^"]*"[^>]*>([A-Z]+)</span>',
            r'<span[^>]*class="[^"]*amyloid[^"]*"[^>]*>([A-Z]+)</span>',
            r'<font[^>]*color="red"[^>]*>([A-Z]+)</font>',
            r'<b[^>]*style="[^"]*color:\s*red[^"]*"[^>]*>([A-Z]+)</b>',
        ]
        
        for pattern in highlight_patterns:
            for match in re.finditer(pattern, html, re.IGNORECASE):
                region_seq = match.group(1)
                
                # Find position in original sequence
                pos = sequence.find(region_seq)
                if pos != -1:
                    regions.append((pos, pos + len(region_seq), None))
        
        # Alternative: extract from summary section
        # WALTZ often lists regions in a summary like "Residues 10-16: AMYLOIDOGENIC"
        summary_pattern = re.compile(
            r'residues?\s*(\d+)\s*[-–to]+\s*(\d+)\s*[:\s]*(amyloidogenic|positive)',
            re.IGNORECASE
        )
        
        for match in summary_pattern.finditer(html):
            start = int(match.group(1)) - 1
            end = int(match.group(2))
            regions.append((start, end, None))
        
        # Remove duplicates
        unique_regions = list(set(regions))
        unique_regions.sort(key=lambda x: x[0])
        
        return unique_regions
    
    def can_handle(self, protein) -> bool:
        """
        Check if WALTZ can handle this protein.
        
        WALTZ has practical limitations:
        - Minimum sequence length: 6 (hexapeptide minimum)
        - Maximum sequence length: ~5000 (server limitation)
        """
        seq_len = len(protein.sequence)
        if seq_len < 6:
            logger.warning(f"Sequence too short for WALTZ ({seq_len} < 6)")
            return False
        if seq_len > 5000:
            logger.warning(f"Sequence may be too long for WALTZ ({seq_len} > 5000)")
        return True


# =============================================================================
# Fallback local implementation using WALTZ-like scoring
# =============================================================================

# Simplified WALTZ-like PSSM (position-specific scores)
# These are approximations based on published data
# Real WALTZ uses more sophisticated position-specific matrices
WALTZ_APPROXIMATE_WEIGHTS = {
    # Position 1-2: β-sheet initiators
    'pos1': {'V': 0.8, 'I': 0.9, 'L': 0.7, 'F': 1.0, 'Y': 0.8, 'W': 0.6, 
             'A': 0.3, 'M': 0.5, 'T': 0.2, 'S': 0.1, 'C': 0.4,
             'N': -0.3, 'Q': -0.2, 'G': -0.1, 'P': -0.8,
             'K': -0.6, 'R': -0.5, 'H': -0.1, 'D': -0.5, 'E': -0.4},
    # Position 3-4: Central hydrophobic core
    'core': {'V': 1.0, 'I': 1.1, 'L': 0.9, 'F': 1.2, 'Y': 1.0, 'W': 0.8,
             'A': 0.4, 'M': 0.7, 'T': 0.3, 'S': 0.2, 'C': 0.5,
             'N': -0.2, 'Q': -0.1, 'G': 0.0, 'P': -1.0,
             'K': -0.8, 'R': -0.7, 'H': 0.0, 'D': -0.6, 'E': -0.5},
    # Position 5-6: β-sheet terminators
    'pos6': {'V': 0.7, 'I': 0.8, 'L': 0.6, 'F': 0.9, 'Y': 0.7, 'W': 0.5,
             'A': 0.2, 'M': 0.4, 'T': 0.1, 'S': 0.0, 'C': 0.3,
             'N': -0.4, 'Q': -0.3, 'G': -0.2, 'P': -0.7,
             'K': -0.5, 'R': -0.4, 'H': -0.2, 'D': -0.4, 'E': -0.3},
}


def calculate_waltz_like_score(hexapeptide: str) -> float:
    """
    Calculate a WALTZ-like score for a hexapeptide.
    
    This is a simplified approximation of the WALTZ PSSM scoring.
    For accurate results, use the web server.
    
    Args:
        hexapeptide: 6-residue sequence
        
    Returns:
        Aggregation propensity score
    """
    if len(hexapeptide) != 6:
        raise ValueError("Hexapeptide must be exactly 6 residues")
    
    hexapeptide = hexapeptide.upper()
    score = 0.0
    
    # Position-specific scoring
    positions = [
        (0, 'pos1'), (1, 'pos1'),  # N-terminal
        (2, 'core'), (3, 'core'),  # Central
        (4, 'pos6'), (5, 'pos6'),  # C-terminal
    ]
    
    for pos, weight_key in positions:
        aa = hexapeptide[pos]
        weight_dict = WALTZ_APPROXIMATE_WEIGHTS[weight_key]
        score += weight_dict.get(aa, 0.0)
    
    return score / 6  # Normalize


def predict_waltz_local(
    sequence: str,
    threshold: float = 0.3,
) -> list[tuple[int, int, float]]:
    """
    Local WALTZ-like prediction without web access.
    
    This provides an approximation when the web server is unavailable.
    Results may differ from the official WALTZ server.
    
    Args:
        sequence: Protein sequence
        threshold: Score threshold for positive prediction
        
    Returns:
        List of (start, end, score) tuples for predicted APRs
    """
    if len(sequence) < 6:
        return []
    
    regions = []
    scores = []
    
    # Scan with hexapeptide window
    for i in range(len(sequence) - 5):
        hexapeptide = sequence[i:i+6]
        score = calculate_waltz_like_score(hexapeptide)
        scores.append((i, score))
    
    # Find regions above threshold
    in_region = False
    region_start = 0
    region_scores = []
    
    for pos, score in scores:
        if score >= threshold and not in_region:
            in_region = True
            region_start = pos
            region_scores = [score]
        elif score >= threshold and in_region:
            region_scores.append(score)
        elif score < threshold and in_region:
            in_region = False
            region_end = pos + 6  # Include full hexapeptide
            avg_score = sum(region_scores) / len(region_scores)
            regions.append((region_start, region_end, avg_score))
            region_scores = []
    
    # Handle region at sequence end
    if in_region:
        region_end = len(sequence)
        avg_score = sum(region_scores) / len(region_scores)
        regions.append((region_start, region_end, avg_score))
    
    return regions
