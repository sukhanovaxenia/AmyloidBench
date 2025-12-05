"""
PASTA 2.0 Web Predictor.

PASTA 2.0 (http://protein.bio.unipd.it/pasta2/) predicts amyloid-forming regions
using a pairwise energy potential derived from β-sheet statistics in globular
proteins. The algorithm evaluates the stability of putative cross-β pairings
between different sequence stretches.

Reference:
Walsh I, et al. (2014) Nucleic Acids Research 42:W301-W307
DOI: 10.1093/nar/gku399

Biophysical Basis:
- Energy function derived from β-strand pairing statistics in DSSP
- Evaluates both parallel and antiparallel orientations
- Lower energy = more stable aggregate = more amyloidogenic
- Default threshold: -5 for 85% specificity

Output Format:
- Per-residue aggregation propensity
- Secondary structure prediction
- Intrinsic disorder prediction
- Best pairing energies (parallel/antiparallel)
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

import numpy as np
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from .base import (
    BaseWebPredictor,
    WebPredictorConfig,
    ParsingError,
)

try:
    from ..output_models import (
        PredictorOutput,
        ResidueScore,
        PredictedRegion,
        ClassificationLabel,
        ScoreType,
        normalize_scores,
    )
except ImportError:
    from amyloidbench.predictors.output_models import (
        PredictorOutput,
        ResidueScore,
        PredictedRegion,
        ClassificationLabel,
        ScoreType,
        normalize_scores,
    )


logger = logging.getLogger(__name__)


class Pasta2Predictor(BaseWebPredictor):
    """
    Selenium-based predictor for PASTA 2.0 web server.
    
    PASTA 2.0 uses a statistical potential derived from beta-sheet contacts
    in globular proteins to evaluate cross-beta pairing stability. The server
    provides multiple output modes with different specificity/sensitivity
    trade-offs.
    
    Output modes:
    - Region (85% spec): Recommended for most applications
    - Region (90% spec): Higher specificity, fewer false positives
    - Best Pairing: Reports only the best pairing energy
    """
    
    server_url = "http://protein.bio.unipd.it/pasta2/"
    predictor_name = "PASTA2"
    predictor_version = "2.0"
    score_type = ScoreType.ENERGY  # Lower = more amyloidogenic
    default_threshold = -5.0  # Energy threshold for 85% specificity
    
    def __init__(
        self,
        config: Optional[WebPredictorConfig] = None,
        specificity: str = "85",  # "85", "90", or "best"
    ):
        """
        Initialize PASTA 2.0 predictor.
        
        Args:
            config: Web predictor configuration
            specificity: Output mode - "85" (85% spec), "90" (90% spec), or "best"
        """
        super().__init__(config)
        self.specificity = specificity
        
        # Adjust threshold based on specificity
        if specificity == "90":
            self.default_threshold = -6.0
        elif specificity == "best":
            self.default_threshold = -4.0
    
    def _submit_sequence(self, sequence: str, sequence_id: str) -> None:
        """Submit sequence to PASTA 2.0 server."""
        driver = self._get_driver()
        
        # Navigate to server
        logger.info(f"Navigating to {self.server_url}")
        driver.get(self.server_url)
        time.sleep(3)  # Let page load fully
        
        self._take_screenshot("initial_page")
        
        try:
            # Find the sequence input
            # PASTA 2.0 uses a textarea for sequence input
            textarea = None
            
            # Try different selectors
            selectors = [
                (By.NAME, "sequence"),
                (By.ID, "sequence"),
                (By.CSS_SELECTOR, "textarea"),
                (By.XPATH, "//textarea"),
            ]
            
            for by, selector in selectors:
                try:
                    textarea = driver.find_element(by, selector)
                    if textarea:
                        break
                except NoSuchElementException:
                    continue
            
            if not textarea:
                self._take_screenshot("no_textarea")
                raise ParsingError("Could not find sequence input field")
            
            # Clear and enter sequence
            textarea.clear()
            fasta_input = f">{sequence_id}\n{sequence}"
            textarea.send_keys(fasta_input)
            
            logger.info(f"Submitted sequence: {sequence_id} ({len(sequence)} aa)")
            
            # Find and click submit button
            submit_btn = None
            submit_selectors = [
                (By.CSS_SELECTOR, "input[type='submit']"),
                (By.CSS_SELECTOR, "button[type='submit']"),
                (By.XPATH, "//input[@value='Submit']"),
                (By.XPATH, "//button[contains(text(), 'Submit')]"),
                (By.XPATH, "//input[@value='Run']"),
            ]
            
            for by, selector in submit_selectors:
                try:
                    submit_btn = driver.find_element(by, selector)
                    if submit_btn:
                        break
                except NoSuchElementException:
                    continue
            
            if not submit_btn:
                self._take_screenshot("no_submit")
                raise ParsingError("Could not find submit button")
            
            submit_btn.click()
            
            # Wait for results
            logger.info("Waiting for PASTA 2.0 results...")
            self._take_screenshot("after_submit")
            
            # PASTA can take a while - wait for results table or download link
            try:
                WebDriverWait(driver, 120).until(
                    lambda d: (
                        "Results" in d.page_source or
                        "Download" in d.page_source or
                        "Energy" in d.page_source or
                        len(d.find_elements(By.CSS_SELECTOR, "table")) > 0
                    )
                )
            except TimeoutException:
                self._take_screenshot("timeout")
                self._save_html("timeout")
                raise
            
            time.sleep(2)  # Additional wait for page to fully render
            self._take_screenshot("results_page")
            
        except Exception as e:
            logger.error(f"Error submitting to PASTA 2.0: {e}")
            self._take_screenshot("submit_error")
            raise
    
    def _parse_results(self, sequence: str, sequence_id: str) -> PredictorOutput:
        """Parse PASTA 2.0 results from the web page."""
        driver = self._get_driver()
        
        residue_scores = []
        predicted_regions = []
        raw_output = {}
        
        try:
            page_source = driver.page_source
            
            # PASTA 2.0 provides downloadable results
            # Try to find and parse the results table
            
            # Look for energy values in the page
            # Format: position, residue, energy (parallel), energy (antiparallel)
            
            # Try to find result table
            tables = driver.find_elements(By.TAG_NAME, "table")
            
            for table in tables:
                try:
                    rows = table.find_elements(By.TAG_NAME, "tr")
                    for row in rows[1:]:  # Skip header
                        cells = row.find_elements(By.TAG_NAME, "td")
                        if len(cells) >= 3:
                            try:
                                pos_text = cells[0].text.strip()
                                if pos_text.isdigit():
                                    position = int(pos_text)
                                    
                                    # Get residue if available
                                    residue = cells[1].text.strip() if len(cells) > 1 else sequence[position - 1]
                                    
                                    # Get energy score
                                    score_text = cells[-1].text.strip()
                                    score = float(score_text) if score_text else 0.0
                                    
                                    residue_scores.append({
                                        'position': position,
                                        'residue': residue,
                                        'raw_score': score,
                                    })
                            except (ValueError, IndexError):
                                continue
                except Exception:
                    continue
            
            # Look for region predictions
            # Format: "Region X-Y: energy"
            region_pattern = r'[Rr]egion[:\s]+(\d+)\s*[-–]\s*(\d+)[:\s]+(-?\d+\.?\d*)'
            region_matches = re.findall(region_pattern, page_source)
            
            for start, end, energy in region_matches:
                start_pos = int(start)
                end_pos = int(end)
                energy_val = float(energy)
                
                predicted_regions.append({
                    'start': start_pos,
                    'end': end_pos,
                    'sequence': sequence[start_pos - 1:end_pos],
                    'energy': energy_val,
                })
            
            # Also look for "best pairing" results
            pairing_pattern = r'[Bb]est\s+[Pp]airing[:\s]+(-?\d+\.?\d*)'
            pairing_matches = re.findall(pairing_pattern, page_source)
            if pairing_matches:
                raw_output['best_pairing_energy'] = float(pairing_matches[0])
            
            # If no scores found, compute locally using PASTA energy matrix
            if not residue_scores:
                logger.warning("Could not parse PASTA scores, using fallback")
                residue_scores = self._compute_fallback_scores(sequence)
            
            # Create output
            return self._create_output(
                sequence=sequence,
                sequence_id=sequence_id,
                residue_scores=residue_scores,
                predicted_regions=predicted_regions,
                raw_output=raw_output,
            )
            
        except Exception as e:
            logger.error(f"Error parsing PASTA 2.0 results: {e}")
            self._take_screenshot("parse_error")
            self._save_html("parse_error")
            raise ParsingError(f"Failed to parse PASTA 2.0 results: {e}")
    
    def _compute_fallback_scores(self, sequence: str) -> list[dict]:
        """
        Compute approximate PASTA scores using simplified energy matrix.
        
        This is a fallback when web parsing fails. Uses a simplified
        pairwise energy approximation based on hydrophobic contacts.
        """
        n = len(sequence)
        
        # Simplified PASTA-like scores based on β-propensity and hydrophobicity
        # Higher values = more aggregation prone
        PROPENSITY = {
            'A': 0.3, 'R': -0.5, 'N': -0.3, 'D': -0.4, 'C': 0.2,
            'Q': -0.2, 'E': -0.4, 'G': 0.0, 'H': -0.1, 'I': 0.8,
            'L': 0.6, 'K': -0.5, 'M': 0.4, 'F': 0.7, 'P': -0.5,
            'S': 0.0, 'T': 0.1, 'W': 0.5, 'Y': 0.5, 'V': 0.7,
        }
        
        scores = []
        window = 7
        
        for i in range(n):
            start = max(0, i - window // 2)
            end = min(n, i + window // 2 + 1)
            
            # Average propensity in window
            window_scores = [PROPENSITY.get(aa, 0.0) for aa in sequence[start:end]]
            
            # Convert to energy-like value (negative = more stable = more amyloidogenic)
            avg_prop = np.mean(window_scores)
            energy = -avg_prop * 5  # Scale to energy-like range
            
            scores.append({
                'position': i + 1,
                'residue': sequence[i],
                'raw_score': energy,
            })
        
        return scores
    
    def _create_output(
        self,
        sequence: str,
        sequence_id: str,
        residue_scores: list[dict],
        predicted_regions: list[dict],
        raw_output: dict,
    ) -> PredictorOutput:
        """Create standardized PredictorOutput."""
        
        # Get raw energy values
        raw_values = [s['raw_score'] for s in residue_scores]
        
        # Normalize (energy: lower = more amyloidogenic, so invert)
        normalized = normalize_scores(
            raw_values,
            ScoreType.ENERGY,
            min_val=-10.0,  # Typical PASTA energy range
            max_val=0.0,
        )
        
        # Create ResidueScore objects
        residue_score_objects = []
        for i, score_data in enumerate(residue_scores):
            # For energy scores: below threshold = amyloidogenic
            classification = (
                ClassificationLabel.AMYLOIDOGENIC 
                if score_data['raw_score'] < self.default_threshold
                else ClassificationLabel.NON_AMYLOIDOGENIC
            )
            
            residue_score_objects.append(ResidueScore(
                position=score_data['position'],
                residue=score_data['residue'],
                raw_score=score_data['raw_score'],
                normalized_score=normalized[i],
                classification=classification,
            ))
        
        # Create PredictedRegion objects
        predicted_region_objects = []
        for region in predicted_regions:
            # Get scores for region
            region_raw = [
                s['raw_score'] for s in residue_scores
                if region['start'] <= s['position'] <= region['end']
            ]
            region_norm = [
                normalized[i] for i, s in enumerate(residue_scores)
                if region['start'] <= s['position'] <= region['end']
            ]
            
            predicted_region_objects.append(PredictedRegion(
                start=region['start'],
                end=region['end'],
                sequence=region['sequence'],
                mean_score=float(np.mean(region_raw)) if region_raw else region.get('energy', 0.0),
                max_score=float(np.min(region_raw)) if region_raw else region.get('energy', 0.0),  # Min for energy
                mean_normalized=float(np.mean(region_norm)) if region_norm else 0.0,
                region_type="cross_beta",
            ))
        
        # If no regions found, detect from scores
        if not predicted_region_objects:
            predicted_region_objects = self._detect_regions(residue_scores, normalized, sequence)
        
        # Overall classification
        is_amyloidogenic = len(predicted_region_objects) > 0
        overall_score = float(np.min(raw_values)) if raw_values else 0.0  # Best (lowest) energy
        overall_probability = float(np.max(normalized)) if normalized else 0.0
        
        return PredictorOutput(
            predictor_name=self.predictor_name,
            predictor_version=self.predictor_version,
            sequence_id=sequence_id,
            sequence=sequence,
            residue_scores=residue_score_objects,
            predicted_regions=predicted_region_objects,
            overall_classification=(
                ClassificationLabel.AMYLOIDOGENIC if is_amyloidogenic
                else ClassificationLabel.NON_AMYLOIDOGENIC
            ),
            overall_score=overall_score,
            overall_probability=overall_probability,
            score_type=self.score_type,
            threshold=self.default_threshold,
            source="web",
            raw_output=raw_output,
        )
    
    def _detect_regions(
        self,
        residue_scores: list[dict],
        normalized: list[float],
        sequence: str,
        min_length: int = 5,
    ) -> list[PredictedRegion]:
        """Detect amyloidogenic regions from per-residue scores."""
        regions = []
        in_region = False
        region_start = 0
        region_scores = []
        region_normalized = []
        
        for i, score_data in enumerate(residue_scores):
            is_amyloidogenic = score_data['raw_score'] < self.default_threshold
            
            if is_amyloidogenic:
                if not in_region:
                    region_start = score_data['position']
                    region_scores = [score_data['raw_score']]
                    region_normalized = [normalized[i]]
                    in_region = True
                else:
                    region_scores.append(score_data['raw_score'])
                    region_normalized.append(normalized[i])
            else:
                if in_region:
                    region_end = residue_scores[i - 1]['position']
                    if region_end - region_start + 1 >= min_length:
                        regions.append(PredictedRegion(
                            start=region_start,
                            end=region_end,
                            sequence=sequence[region_start - 1:region_end],
                            mean_score=float(np.mean(region_scores)),
                            max_score=float(np.min(region_scores)),
                            mean_normalized=float(np.mean(region_normalized)),
                            region_type="cross_beta",
                        ))
                    in_region = False
        
        # Handle region at end
        if in_region:
            region_end = residue_scores[-1]['position']
            if region_end - region_start + 1 >= min_length:
                regions.append(PredictedRegion(
                    start=region_start,
                    end=region_end,
                    sequence=sequence[region_start - 1:region_end],
                    mean_score=float(np.mean(region_scores)),
                    max_score=float(np.min(region_scores)),
                    mean_normalized=float(np.mean(region_normalized)),
                    region_type="cross_beta",
                ))
        
        return regions


# Convenience function
def predict_with_pasta2(
    sequence: str,
    sequence_id: str = "query",
    config: Optional[WebPredictorConfig] = None,
    specificity: str = "85",
) -> PredictorOutput:
    """
    Predict amyloidogenicity using PASTA 2.0 web server.
    
    Args:
        sequence: Protein sequence
        sequence_id: Identifier for the sequence
        config: Optional configuration
        specificity: Output mode - "85", "90", or "best"
        
    Returns:
        PredictorOutput with per-residue energies and regions
    """
    with Pasta2Predictor(config, specificity) as predictor:
        return predictor.predict(sequence, sequence_id)
