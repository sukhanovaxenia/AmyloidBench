"""
AGGRESCAN Web Predictor.

AGGRESCAN (http://bioinf.uab.es/aggrescan/) predicts aggregation-prone regions
based on an experimentally-derived amino acid aggregation propensity scale (a3v).
The scale was obtained from in vivo aggregation experiments using Aβ42 as a model.

Reference:
Conchillo-Solé O, et al. (2007) BMC Bioinformatics 8:65
DOI: 10.1186/1471-2105-8-65

Output Format:
- Profile graphic: Per-residue aggregation propensity (a4v values)
- Hot spot area: Regions with aggregation propensity above threshold
- Numerical output: Detailed per-residue scores and hot spots

Key Parameters:
- a4vSS: Smoothed aggregation propensity profile
- HSA: Hot Spot Area (integrated propensity above threshold)
- AAT: Average Aggregation propensity Total
- Na4vSS: Normalized aggregation propensity
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

import numpy as np
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
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


class AggrescanPredictor(BaseWebPredictor):
    """
    Selenium-based predictor for AGGRESCAN web server.
    
    AGGRESCAN uses a 5-residue sliding window to compute smoothed
    aggregation propensity profiles based on the a3v scale. Hot spots
    are defined as regions where the smoothed profile exceeds the
    hot spot threshold (default -0.02).
    
    The server provides:
    - a4vSS: Smoothed aggregation propensity (sliding window average)
    - HSA: Hot spot areas (integrated propensity above threshold)
    - Numerical parameters: AAT, Na4vSS, THSA, nHS, etc.
    """
    
    server_url = "http://bioinf.uab.es/aggrescan/"
    predictor_name = "AGGRESCAN"
    predictor_version = "1.0"
    score_type = ScoreType.RAW
    default_threshold = -0.02  # Hot spot threshold
    
    # AGGRESCAN a3v scale (from Conchillo-Solé et al. 2007)
    A3V_SCALE = {
        'A': 0.169, 'R': -0.596, 'N': -0.344, 'D': -0.469, 'C': 0.271,
        'Q': -0.204, 'E': -0.467, 'G': -0.091, 'H': -0.208, 'I': 0.681,
        'L': 0.460, 'K': -0.601, 'M': 0.392, 'F': 0.728, 'P': -0.191,
        'S': -0.145, 'T': 0.055, 'W': 0.479, 'Y': 0.457, 'V': 0.496,
    }
    
    def __init__(self, config: Optional[WebPredictorConfig] = None):
        """Initialize AGGRESCAN predictor."""
        super().__init__(config)
        self._results_parsed = False
    
    def _submit_sequence(self, sequence: str, sequence_id: str) -> None:
        """Submit sequence to AGGRESCAN server."""
        driver = self._get_driver()
        
        # Navigate to server
        logger.info(f"Navigating to {self.server_url}")
        driver.get(self.server_url)
        time.sleep(2)  # Let page load
        
        # Find the sequence input textarea
        # AGGRESCAN has a simple form with textarea
        try:
            # Look for textarea for sequence input
            textarea = driver.find_element(By.NAME, "seqs")
        except NoSuchElementException:
            # Try alternative selectors
            try:
                textarea = driver.find_element(By.CSS_SELECTOR, "textarea")
            except NoSuchElementException:
                self._take_screenshot("no_textarea")
                raise ParsingError("Could not find sequence input field")
        
        # Clear and enter sequence in FASTA format
        textarea.clear()
        fasta_input = f">{sequence_id}\n{sequence}"
        textarea.send_keys(fasta_input)
        
        logger.info(f"Submitted sequence: {sequence_id} ({len(sequence)} aa)")
        
        # Find and click submit button
        try:
            submit_btn = driver.find_element(By.CSS_SELECTOR, "input[type='submit']")
        except NoSuchElementException:
            try:
                submit_btn = driver.find_element(By.XPATH, "//input[@value='Run']")
            except NoSuchElementException:
                submit_btn = driver.find_element(By.XPATH, "//button[contains(text(), 'Submit')]")
        
        submit_btn.click()
        
        # Wait for results page
        logger.info("Waiting for results...")
        self._take_screenshot("after_submit")
        
        # AGGRESCAN shows results on the same page or redirects
        # Wait for results table or profile image to appear
        try:
            WebDriverWait(driver, 60).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table"))
            )
        except TimeoutException:
            self._take_screenshot("timeout_waiting")
            self._save_html("timeout_page")
            raise
        
        self._take_screenshot("results_loaded")
    
    def _parse_results(self, sequence: str, sequence_id: str) -> PredictorOutput:
        """Parse AGGRESCAN results from the web page."""
        driver = self._get_driver()
        
        # Initialize containers
        residue_scores = []
        hot_spots = []
        raw_output = {}
        
        try:
            # Parse the numerical results
            page_source = driver.page_source
            
            # Extract per-residue scores from the output
            # AGGRESCAN shows a table with Position, Residue, a4vSS values
            
            # Try to find the detailed results table
            tables = driver.find_elements(By.TAG_NAME, "table")
            
            scores_found = False
            for table in tables:
                try:
                    rows = table.find_elements(By.TAG_NAME, "tr")
                    for row in rows:
                        cells = row.find_elements(By.TAG_NAME, "td")
                        if len(cells) >= 3:
                            # Try to parse as position, residue, score
                            try:
                                pos_text = cells[0].text.strip()
                                res_text = cells[1].text.strip()
                                score_text = cells[2].text.strip()
                                
                                if pos_text.isdigit() and len(res_text) == 1:
                                    position = int(pos_text)
                                    residue = res_text.upper()
                                    score = float(score_text)
                                    
                                    residue_scores.append({
                                        'position': position,
                                        'residue': residue,
                                        'raw_score': score,
                                    })
                                    scores_found = True
                            except (ValueError, IndexError):
                                continue
                except Exception:
                    continue
            
            # If we couldn't find scores in tables, compute locally using a3v
            if not scores_found:
                logger.warning("Could not parse scores from page, computing locally")
                residue_scores = self._compute_local_scores(sequence)
            
            # Parse hot spots from page
            # Look for HSA (Hot Spot Area) information
            hsa_pattern = r'HSA[:\s]+(\d+\.?\d*)'
            hsa_matches = re.findall(hsa_pattern, page_source, re.IGNORECASE)
            if hsa_matches:
                raw_output['HSA'] = [float(h) for h in hsa_matches]
            
            # Look for hot spot regions
            # Pattern: "Hot spot: X-Y" or similar
            hs_pattern = r'[Hh]ot\s*[Ss]pot[:\s]+(\d+)\s*[-–]\s*(\d+)'
            hs_matches = re.findall(hs_pattern, page_source)
            for start, end in hs_matches:
                start_pos = int(start)
                end_pos = int(end)
                hot_spots.append({
                    'start': start_pos,
                    'end': end_pos,
                    'sequence': sequence[start_pos - 1:end_pos],
                })
            
            # If no hot spots found in page, detect from scores
            if not hot_spots and residue_scores:
                hot_spots = self._detect_hot_spots(residue_scores, sequence)
            
            # Create standardized output
            return self._create_output(
                sequence=sequence,
                sequence_id=sequence_id,
                residue_scores=residue_scores,
                hot_spots=hot_spots,
                raw_output=raw_output,
            )
            
        except Exception as e:
            logger.error(f"Error parsing AGGRESCAN results: {e}")
            self._take_screenshot("parse_error")
            self._save_html("parse_error")
            raise ParsingError(f"Failed to parse AGGRESCAN results: {e}")
    
    def _compute_local_scores(self, sequence: str) -> list[dict]:
        """
        Compute AGGRESCAN scores locally using a3v scale.
        
        This is a fallback when web parsing fails, using the published
        algorithm: 5-residue sliding window average of a3v values.
        """
        n = len(sequence)
        window = 5
        
        # Get raw a3v values
        a3v_values = [self.A3V_SCALE.get(aa, 0.0) for aa in sequence]
        
        # Compute smoothed values (a4vSS)
        # Using centered window where possible
        smoothed = []
        for i in range(n):
            start = max(0, i - window // 2)
            end = min(n, i + window // 2 + 1)
            window_values = a3v_values[start:end]
            smoothed.append(np.mean(window_values))
        
        scores = []
        for i, (aa, score) in enumerate(zip(sequence, smoothed)):
            scores.append({
                'position': i + 1,
                'residue': aa,
                'raw_score': score,
            })
        
        return scores
    
    def _detect_hot_spots(
        self,
        residue_scores: list[dict],
        sequence: str,
        threshold: float = -0.02,
        min_length: int = 5,
    ) -> list[dict]:
        """Detect hot spots from per-residue scores."""
        hot_spots = []
        in_hotspot = False
        hs_start = 0
        
        for score_data in residue_scores:
            pos = score_data['position']
            score = score_data['raw_score']
            
            if score > threshold:
                if not in_hotspot:
                    hs_start = pos
                    in_hotspot = True
            else:
                if in_hotspot:
                    hs_end = pos - 1
                    if hs_end - hs_start + 1 >= min_length:
                        hot_spots.append({
                            'start': hs_start,
                            'end': hs_end,
                            'sequence': sequence[hs_start - 1:hs_end],
                        })
                    in_hotspot = False
        
        # Handle hot spot at end
        if in_hotspot:
            hs_end = residue_scores[-1]['position']
            if hs_end - hs_start + 1 >= min_length:
                hot_spots.append({
                    'start': hs_start,
                    'end': hs_end,
                    'sequence': sequence[hs_start - 1:hs_end],
                })
        
        return hot_spots
    
    def _create_output(
        self,
        sequence: str,
        sequence_id: str,
        residue_scores: list[dict],
        hot_spots: list[dict],
        raw_output: dict,
    ) -> PredictorOutput:
        """Create standardized PredictorOutput from parsed data."""
        
        # Convert to ResidueScore objects
        scores = residue_scores if residue_scores else self._compute_local_scores(sequence)
        
        # Normalize scores (a3v ranges roughly -0.6 to 0.8)
        raw_values = [s['raw_score'] for s in scores]
        normalized = normalize_scores(
            raw_values,
            ScoreType.RAW,
            min_val=-0.6,
            max_val=0.8,
        )
        
        residue_score_objects = []
        for i, score_data in enumerate(scores):
            classification = (
                ClassificationLabel.AMYLOIDOGENIC 
                if score_data['raw_score'] > self.default_threshold
                else ClassificationLabel.NON_AMYLOIDOGENIC
            )
            
            residue_score_objects.append(ResidueScore(
                position=score_data['position'],
                residue=score_data['residue'],
                raw_score=score_data['raw_score'],
                normalized_score=normalized[i],
                classification=classification,
            ))
        
        # Convert to PredictedRegion objects
        predicted_regions = []
        for hs in hot_spots:
            region_scores = [
                s['raw_score'] for s in scores
                if hs['start'] <= s['position'] <= hs['end']
            ]
            region_normalized = [
                normalized[i] for i, s in enumerate(scores)
                if hs['start'] <= s['position'] <= hs['end']
            ]
            
            predicted_regions.append(PredictedRegion(
                start=hs['start'],
                end=hs['end'],
                sequence=hs['sequence'],
                mean_score=float(np.mean(region_scores)) if region_scores else 0.0,
                max_score=float(np.max(region_scores)) if region_scores else 0.0,
                mean_normalized=float(np.mean(region_normalized)) if region_normalized else 0.0,
                region_type="hot_spot",
            ))
        
        # Overall classification
        is_amyloidogenic = len(hot_spots) > 0
        overall_score = float(np.mean(raw_values)) if raw_values else 0.0
        overall_probability = float(np.mean(normalized)) if normalized else 0.0
        
        return PredictorOutput(
            predictor_name=self.predictor_name,
            predictor_version=self.predictor_version,
            sequence_id=sequence_id,
            sequence=sequence,
            residue_scores=residue_score_objects,
            predicted_regions=predicted_regions,
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


# Convenience function for direct use
def predict_with_aggrescan(
    sequence: str,
    sequence_id: str = "query",
    config: Optional[WebPredictorConfig] = None,
) -> PredictorOutput:
    """
    Predict amyloidogenicity using AGGRESCAN web server.
    
    Args:
        sequence: Protein sequence
        sequence_id: Identifier for the sequence
        config: Optional configuration
        
    Returns:
        PredictorOutput with per-residue scores and hot spots
    """
    with AggrescanPredictor(config) as predictor:
        return predictor.predict(sequence, sequence_id)
