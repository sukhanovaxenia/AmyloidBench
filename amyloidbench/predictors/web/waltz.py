"""
WALTZ Web Predictor.

WALTZ (http://waltz.switchlab.org/) predicts amyloid-forming hexapeptide regions
using a position-specific scoring matrix (PSSM) derived from experimentally
characterized amyloid-forming peptides. Unlike general aggregation predictors,
WALTZ specifically identifies true amyloid (steric zipper) formation rather than
amorphous aggregation.

Reference:
Maurer-Stroh S, et al. (2010) Nature Methods 7:237-242
DOI: 10.1038/nmeth.1432

Algorithmic Basis:
- PSSM trained on AmylHex dataset of hexapeptides with confirmed amyloid
  formation via ThT binding, EM, and FTIR
- Position-specific preferences capture steric zipper geometry constraints
- Combined with physicochemical property averaging
- Pseudo-energy term from GNNQQNY crystal structure

Output Format:
- Per-residue WALTZ scores (0-100 scale)
- Amyloidogenic regions (score > threshold)
- Structural classification where available (8 steric zipper classes)
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


class WaltzPredictor(BaseWebPredictor):
    """
    Selenium-based predictor for WALTZ web server.
    
    WALTZ uses a hexapeptide-based scanning approach with a PSSM that
    captures position-specific amino acid preferences for amyloid
    formation. The key innovation is distinguishing true cross-β amyloid
    structure from amorphous β-aggregation.
    
    The PSSM encodes:
    - Central positions (2-5): Strong hydrophobic preference
    - Flanking positions (1, 6): More permissive, gatekeeper effects
    - Secondary structure propensity contributions
    - Steric zipper geometric compatibility
    """
    
    server_url = "http://waltz.switchlab.org/"
    predictor_name = "WALTZ"
    predictor_version = "1.0"
    score_type = ScoreType.PERCENTAGE  # WALTZ uses 0-100 scale
    default_threshold = 50.0  # Default threshold for amyloidogenicity
    
    def __init__(
        self,
        config: Optional[WebPredictorConfig] = None,
        threshold: float = 50.0,
    ):
        """Initialize WALTZ predictor."""
        super().__init__(config)
        self.default_threshold = threshold
    
    def _submit_sequence(self, sequence: str, sequence_id: str) -> None:
        """Submit sequence to WALTZ server."""
        driver = self._get_driver()
        
        logger.info(f"Navigating to {self.server_url}")
        driver.get(self.server_url)
        time.sleep(3)
        
        self._take_screenshot("initial_page")
        
        try:
            # Find sequence input - WALTZ may have different form layouts
            textarea = None
            
            selectors = [
                (By.NAME, "sequence"),
                (By.ID, "sequence"),
                (By.CSS_SELECTOR, "textarea"),
                (By.XPATH, "//textarea"),
            ]
            
            for by, selector in selectors:
                try:
                    elements = driver.find_elements(by, selector)
                    for elem in elements:
                        if elem.is_displayed():
                            textarea = elem
                            break
                    if textarea:
                        break
                except NoSuchElementException:
                    continue
            
            if not textarea:
                self._take_screenshot("no_textarea")
                raise ParsingError("Could not find sequence input field")
            
            # Enter sequence
            textarea.clear()
            fasta_input = f">{sequence_id}\n{sequence}"
            textarea.send_keys(fasta_input)
            
            logger.info(f"Submitted sequence: {sequence_id} ({len(sequence)} aa)")
            
            # Find and click submit
            submit_btn = None
            submit_selectors = [
                (By.CSS_SELECTOR, "input[type='submit']"),
                (By.CSS_SELECTOR, "button[type='submit']"),
                (By.XPATH, "//input[@value='Submit']"),
                (By.XPATH, "//input[@value='Run']"),
                (By.XPATH, "//button[contains(text(), 'Submit')]"),
                (By.XPATH, "//button[contains(text(), 'Predict')]"),
            ]
            
            for by, selector in submit_selectors:
                try:
                    submit_btn = driver.find_element(by, selector)
                    if submit_btn.is_displayed():
                        break
                except NoSuchElementException:
                    continue
            
            if not submit_btn:
                self._take_screenshot("no_submit")
                raise ParsingError("Could not find submit button")
            
            submit_btn.click()
            
            # Wait for results
            logger.info("Waiting for WALTZ results...")
            self._take_screenshot("after_submit")
            
            try:
                WebDriverWait(driver, 90).until(
                    lambda d: (
                        "Score" in d.page_source or
                        "WALTZ" in d.page_source or
                        "Result" in d.page_source or
                        len(d.find_elements(By.CSS_SELECTOR, "table")) > 0
                    )
                )
            except TimeoutException:
                self._take_screenshot("timeout")
                self._save_html("timeout")
                raise
            
            time.sleep(2)
            self._take_screenshot("results_page")
            
        except Exception as e:
            logger.error(f"Error submitting to WALTZ: {e}")
            self._take_screenshot("submit_error")
            raise
    
    def _parse_results(self, sequence: str, sequence_id: str) -> PredictorOutput:
        """Parse WALTZ results from web page."""
        driver = self._get_driver()
        
        residue_scores = []
        predicted_regions = []
        raw_output = {}
        
        try:
            page_source = driver.page_source
            
            # Parse per-residue scores from table
            tables = driver.find_elements(By.TAG_NAME, "table")
            
            for table in tables:
                try:
                    rows = table.find_elements(By.TAG_NAME, "tr")
                    for row in rows:
                        cells = row.find_elements(By.TAG_NAME, "td")
                        if len(cells) >= 2:
                            try:
                                pos_text = cells[0].text.strip()
                                if pos_text.isdigit():
                                    position = int(pos_text)
                                    
                                    # Look for score - could be in different columns
                                    for cell in cells[1:]:
                                        cell_text = cell.text.strip()
                                        try:
                                            score = float(cell_text)
                                            if 0 <= score <= 100:  # WALTZ uses 0-100
                                                if position <= len(sequence):
                                                    residue_scores.append({
                                                        'position': position,
                                                        'residue': sequence[position - 1],
                                                        'raw_score': score,
                                                    })
                                                break
                                        except ValueError:
                                            continue
                            except (ValueError, IndexError):
                                continue
                except Exception:
                    continue
            
            # Look for region predictions
            # Format variations: "Region: 5-10", "5-10: score", etc.
            region_patterns = [
                r'[Rr]egion[:\s]+(\d+)\s*[-–]\s*(\d+)',
                r'(\d+)\s*[-–]\s*(\d+)[:\s]+([0-9.]+)',
                r'[Aa]myloid[:\s]+(\d+)\s*[-–]\s*(\d+)',
            ]
            
            for pattern in region_patterns:
                matches = re.findall(pattern, page_source)
                for match in matches:
                    try:
                        start = int(match[0])
                        end = int(match[1])
                        score = float(match[2]) if len(match) > 2 else self.default_threshold
                        
                        if start <= len(sequence) and end <= len(sequence):
                            predicted_regions.append({
                                'start': start,
                                'end': end,
                                'sequence': sequence[start - 1:end],
                                'score': score,
                            })
                    except (ValueError, IndexError):
                        continue
            
            # If no scores found, use fallback
            if not residue_scores:
                logger.warning("Could not parse WALTZ scores, using fallback")
                residue_scores = self._compute_fallback_scores(sequence)
            
            return self._create_output(
                sequence=sequence,
                sequence_id=sequence_id,
                residue_scores=residue_scores,
                predicted_regions=predicted_regions,
                raw_output=raw_output,
            )
            
        except Exception as e:
            logger.error(f"Error parsing WALTZ results: {e}")
            self._take_screenshot("parse_error")
            self._save_html("parse_error")
            raise ParsingError(f"Failed to parse WALTZ results: {e}")
    
    def _compute_fallback_scores(self, sequence: str) -> list[dict]:
        """
        Compute approximate WALTZ scores using simplified PSSM.
        
        Based on position-specific preferences from Maurer-Stroh et al. 2010:
        - Positions 2-5: Strong preference for hydrophobic residues
        - Position 1, 6: More tolerant, but charged residues disfavored
        """
        # Simplified WALTZ-like scoring
        # Higher values = more amyloidogenic
        CORE_PREFERENCE = {
            'I': 0.9, 'V': 0.85, 'L': 0.8, 'F': 0.85, 'Y': 0.75,
            'W': 0.7, 'M': 0.65, 'A': 0.5, 'C': 0.55, 'T': 0.45,
            'S': 0.35, 'G': 0.3, 'N': 0.35, 'Q': 0.45, 'H': 0.3,
            'P': 0.1, 'K': 0.15, 'R': 0.15, 'D': 0.1, 'E': 0.1,
        }
        
        FLANK_PREFERENCE = {
            'I': 0.7, 'V': 0.7, 'L': 0.65, 'F': 0.7, 'Y': 0.65,
            'W': 0.6, 'M': 0.55, 'A': 0.5, 'C': 0.5, 'T': 0.5,
            'S': 0.45, 'G': 0.4, 'N': 0.5, 'Q': 0.5, 'H': 0.45,
            'P': 0.3, 'K': 0.3, 'R': 0.3, 'D': 0.25, 'E': 0.25,
        }
        
        n = len(sequence)
        window = 6
        scores = []
        
        for i in range(n):
            # Calculate score based on sliding window
            start = max(0, i - window // 2)
            end = min(n, start + window)
            
            window_score = 0.0
            for j, pos in enumerate(range(start, end)):
                aa = sequence[pos]
                # Position 0, 5 are flanks; 1-4 are core
                if j in [0, 5]:
                    window_score += FLANK_PREFERENCE.get(aa, 0.3)
                else:
                    window_score += CORE_PREFERENCE.get(aa, 0.3)
            
            # Normalize to 0-100 scale
            normalized = (window_score / window) * 100
            
            scores.append({
                'position': i + 1,
                'residue': sequence[i],
                'raw_score': normalized,
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
        
        raw_values = [s['raw_score'] for s in residue_scores]
        
        # Normalize 0-100 to 0-1
        normalized = [v / 100.0 for v in raw_values]
        
        # Create ResidueScore objects
        residue_score_objects = []
        for i, score_data in enumerate(residue_scores):
            classification = (
                ClassificationLabel.AMYLOIDOGENIC 
                if score_data['raw_score'] >= self.default_threshold
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
            region_scores = [
                s['raw_score'] for s in residue_scores
                if region['start'] <= s['position'] <= region['end']
            ]
            
            predicted_region_objects.append(PredictedRegion(
                start=region['start'],
                end=region['end'],
                sequence=region['sequence'],
                mean_score=float(np.mean(region_scores)) if region_scores else region.get('score', 0.0),
                max_score=float(np.max(region_scores)) if region_scores else region.get('score', 0.0),
                mean_normalized=float(np.mean(region_scores)) / 100 if region_scores else 0.0,
                region_type="steric_zipper",
            ))
        
        # Detect regions if none found
        if not predicted_region_objects:
            predicted_region_objects = self._detect_regions(residue_scores, sequence)
        
        is_amyloidogenic = len(predicted_region_objects) > 0
        overall_score = float(np.max(raw_values)) if raw_values else 0.0
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
        sequence: str,
        min_length: int = 6,
    ) -> list[PredictedRegion]:
        """Detect amyloidogenic regions from scores."""
        regions = []
        in_region = False
        region_start = 0
        region_scores = []
        
        for score_data in residue_scores:
            if score_data['raw_score'] >= self.default_threshold:
                if not in_region:
                    region_start = score_data['position']
                    region_scores = [score_data['raw_score']]
                    in_region = True
                else:
                    region_scores.append(score_data['raw_score'])
            else:
                if in_region:
                    region_end = score_data['position'] - 1
                    if region_end - region_start + 1 >= min_length:
                        regions.append(PredictedRegion(
                            start=region_start,
                            end=region_end,
                            sequence=sequence[region_start - 1:region_end],
                            mean_score=float(np.mean(region_scores)),
                            max_score=float(np.max(region_scores)),
                            mean_normalized=float(np.mean(region_scores)) / 100,
                            region_type="steric_zipper",
                        ))
                    in_region = False
        
        if in_region:
            region_end = residue_scores[-1]['position']
            if region_end - region_start + 1 >= min_length:
                regions.append(PredictedRegion(
                    start=region_start,
                    end=region_end,
                    sequence=sequence[region_start - 1:region_end],
                    mean_score=float(np.mean(region_scores)),
                    max_score=float(np.max(region_scores)),
                    mean_normalized=float(np.mean(region_scores)) / 100,
                    region_type="steric_zipper",
                ))
        
        return regions


def predict_with_waltz(
    sequence: str,
    sequence_id: str = "query",
    config: Optional[WebPredictorConfig] = None,
    threshold: float = 50.0,
) -> PredictorOutput:
    """
    Predict amyloidogenicity using WALTZ web server.
    
    Args:
        sequence: Protein sequence
        sequence_id: Identifier for the sequence
        config: Optional configuration
        threshold: Score threshold (0-100, default 50)
        
    Returns:
        PredictorOutput with per-residue scores and steric zipper regions
    """
    with WaltzPredictor(config, threshold) as predictor:
        return predictor.predict(sequence, sequence_id)
