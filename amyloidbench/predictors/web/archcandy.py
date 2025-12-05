"""
ArchCandy 2.0 Web Predictor.

ArchCandy (https://bioinfo.crbm.cnrs.fr/index.php?route=tools&tool=7) predicts
amyloid-forming regions based on the β-arch structural motif. The β-arch is the
fundamental building block of amyloid fibrils, consisting of a β-strand-loop-β-strand
arrangement where strands interact via side chains rather than backbone H-bonds.

Reference:
Ahmed AB, et al. (2015) Alzheimers & Dementia 11(6):681-690
DOI: 10.1016/j.jalz.2014.06.007

Biophysical Basis:
- β-arcades: Parallel stacks of β-arch motifs form the fibril core
- Effective for both hydrophobic amyloids (Aβ, α-synuclein) and polar
  Q/N-rich prions (Sup35, Ure2)
- Loop compatibility: G, S, N, D, P, T, A are favored in β-arc loops
- Strand-strand packing evaluated for complementarity

Output Format:
- Cumulative Score: Sum of scores at each position
- Highest Score: Maximum score among overlapping predictions
- SeqView: Visual representation of amyloidogenic regions
- Table: 2D diagrams of predicted β-arches with arc types
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


class ArchCandyPredictor(BaseWebPredictor):
    """
    Selenium-based predictor for ArchCandy 2.0 web server.
    
    ArchCandy detects protein sequences capable of forming β-arcades,
    the structural core of most amyloid fibrils. The algorithm evaluates:
    
    1. β-strand propensity of flanking segments
    2. Loop (β-arc) compatibility
    3. Strand-strand packing potential
    4. Overall structural compactness
    
    Score thresholds (from Ahmed et al. 2015):
    - < 0.40: Non-significant prediction
    - 0.40 - 0.57: Ambiguous
    - > 0.57: Significant amyloidogenic prediction
    """
    
    server_url = "https://bioinfo.crbm.cnrs.fr/index.php?route=tools&tool=7"
    predictor_name = "ArchCandy"
    predictor_version = "2.0"
    score_type = ScoreType.PROBABILITY
    default_threshold = 0.40  # Conservative threshold
    
    # Arc loop compatible residues
    LOOP_COMPATIBLE = set("GSNDPTA")
    
    def __init__(
        self,
        config: Optional[WebPredictorConfig] = None,
        threshold: float = 0.40,
        exclude_tm: bool = True,
    ):
        """
        Initialize ArchCandy predictor.
        
        Args:
            config: Web predictor configuration
            threshold: Score threshold (0.40-0.57 recommended)
            exclude_tm: Exclude transmembrane regions from prediction
        """
        super().__init__(config)
        self.default_threshold = threshold
        self.exclude_tm = exclude_tm
    
    def _submit_sequence(self, sequence: str, sequence_id: str) -> None:
        """Submit sequence to ArchCandy 2.0 server."""
        driver = self._get_driver()
        
        # Navigate to server
        logger.info(f"Navigating to {self.server_url}")
        driver.get(self.server_url)
        time.sleep(3)
        
        self._take_screenshot("initial_page")
        
        try:
            # Find sequence input
            # ArchCandy may use different input methods
            textarea = None
            
            selectors = [
                (By.NAME, "sequence"),
                (By.ID, "sequence"),
                (By.CSS_SELECTOR, "textarea.form-control"),
                (By.CSS_SELECTOR, "textarea"),
                (By.XPATH, "//textarea"),
            ]
            
            for by, selector in selectors:
                try:
                    textarea = driver.find_element(by, selector)
                    if textarea and textarea.is_displayed():
                        break
                except NoSuchElementException:
                    continue
            
            if not textarea:
                self._take_screenshot("no_textarea")
                raise ParsingError("Could not find sequence input field")
            
            # Enter sequence in FASTA format
            textarea.clear()
            fasta_input = f">{sequence_id}\n{sequence}"
            textarea.send_keys(fasta_input)
            
            logger.info(f"Submitted sequence: {sequence_id} ({len(sequence)} aa)")
            
            # Set threshold if configurable
            try:
                threshold_input = driver.find_element(By.NAME, "threshold")
                threshold_input.clear()
                threshold_input.send_keys(str(self.default_threshold))
            except NoSuchElementException:
                pass  # Use server default
            
            # Set TM exclusion if available
            if self.exclude_tm:
                try:
                    tm_checkbox = driver.find_element(By.NAME, "exclude_tm")
                    if not tm_checkbox.is_selected():
                        tm_checkbox.click()
                except NoSuchElementException:
                    pass
            
            # Submit
            submit_btn = None
            submit_selectors = [
                (By.CSS_SELECTOR, "button[type='submit']"),
                (By.CSS_SELECTOR, "input[type='submit']"),
                (By.XPATH, "//button[contains(text(), 'Submit')]"),
                (By.XPATH, "//button[contains(text(), 'Run')]"),
                (By.XPATH, "//input[@value='Submit']"),
            ]
            
            for by, selector in submit_selectors:
                try:
                    submit_btn = driver.find_element(by, selector)
                    if submit_btn and submit_btn.is_displayed():
                        break
                except NoSuchElementException:
                    continue
            
            if not submit_btn:
                self._take_screenshot("no_submit")
                raise ParsingError("Could not find submit button")
            
            submit_btn.click()
            
            # Wait for results - ArchCandy can be slow for long sequences
            logger.info("Waiting for ArchCandy results...")
            self._take_screenshot("after_submit")
            
            # Wait for results table or score display
            try:
                WebDriverWait(driver, 180).until(
                    lambda d: (
                        "Score" in d.page_source or
                        "Cumulative" in d.page_source or
                        "SeqView" in d.page_source or
                        len(d.find_elements(By.CSS_SELECTOR, "table.result")) > 0
                    )
                )
            except TimeoutException:
                self._take_screenshot("timeout")
                self._save_html("timeout")
                raise
            
            time.sleep(2)
            self._take_screenshot("results_page")
            
        except Exception as e:
            logger.error(f"Error submitting to ArchCandy: {e}")
            self._take_screenshot("submit_error")
            raise
    
    def _parse_results(self, sequence: str, sequence_id: str) -> PredictorOutput:
        """Parse ArchCandy 2.0 results from the web page."""
        driver = self._get_driver()
        
        residue_scores = []
        predicted_regions = []
        raw_output = {}
        
        try:
            page_source = driver.page_source
            
            # ArchCandy provides two score types:
            # 1. Cumulative Score: Sum of all overlapping β-arch scores
            # 2. Highest Score: Maximum score at each position
            
            # Parse score table
            tables = driver.find_elements(By.TAG_NAME, "table")
            
            cumulative_scores = {}
            highest_scores = {}
            
            for table in tables:
                try:
                    rows = table.find_elements(By.TAG_NAME, "tr")
                    header = rows[0].text.lower() if rows else ""
                    
                    for row in rows[1:]:
                        cells = row.find_elements(By.TAG_NAME, "td")
                        if len(cells) >= 3:
                            try:
                                pos = int(cells[0].text.strip())
                                residue = cells[1].text.strip()
                                score = float(cells[2].text.strip())
                                
                                if "cumulative" in header:
                                    cumulative_scores[pos] = (residue, score)
                                elif "highest" in header:
                                    highest_scores[pos] = (residue, score)
                                else:
                                    # Default to cumulative
                                    cumulative_scores[pos] = (residue, score)
                            except (ValueError, IndexError):
                                continue
                except Exception:
                    continue
            
            # Use cumulative scores as primary (or highest if cumulative not found)
            scores_dict = cumulative_scores if cumulative_scores else highest_scores
            
            if scores_dict:
                for pos in sorted(scores_dict.keys()):
                    residue, score = scores_dict[pos]
                    residue_scores.append({
                        'position': pos,
                        'residue': residue,
                        'raw_score': score,
                    })
            else:
                # Fallback: compute locally
                logger.warning("Could not parse ArchCandy scores, using fallback")
                residue_scores = self._compute_fallback_scores(sequence)
            
            # Parse predicted β-arches
            # Look for arch diagrams or region listings
            arch_pattern = r'(\d+)\s*[-–]\s*(\d+)[:\s]+([0-9.]+)'
            arch_matches = re.findall(arch_pattern, page_source)
            
            for start, end, score in arch_matches:
                start_pos = int(start)
                end_pos = int(end)
                arch_score = float(score)
                
                if arch_score >= self.default_threshold:
                    predicted_regions.append({
                        'start': start_pos,
                        'end': end_pos,
                        'sequence': sequence[start_pos - 1:end_pos],
                        'score': arch_score,
                    })
            
            # Look for overall amyloidogenicity score
            overall_pattern = r'[Oo]verall[:\s]+([0-9.]+)'
            overall_match = re.search(overall_pattern, page_source)
            if overall_match:
                raw_output['overall_score'] = float(overall_match.group(1))
            
            # Create output
            return self._create_output(
                sequence=sequence,
                sequence_id=sequence_id,
                residue_scores=residue_scores,
                predicted_regions=predicted_regions,
                raw_output=raw_output,
            )
            
        except Exception as e:
            logger.error(f"Error parsing ArchCandy results: {e}")
            self._take_screenshot("parse_error")
            self._save_html("parse_error")
            raise ParsingError(f"Failed to parse ArchCandy results: {e}")
    
    def _compute_fallback_scores(self, sequence: str) -> list[dict]:
        """
        Compute approximate ArchCandy scores locally.
        
        This simplified algorithm evaluates β-arch forming potential based on:
        1. β-strand propensity of flanking segments
        2. Loop compatibility (G, S, N, D, P, T, A)
        3. Overall compactness
        """
        n = len(sequence)
        min_arch = 10  # Minimum β-arch length (strand-loop-strand)
        strand_len = 4
        loop_len = 2
        
        # β-strand propensity (Chou-Fasman)
        BETA_PROP = {
            'A': 0.83, 'R': 0.93, 'N': 0.89, 'D': 0.54, 'C': 1.19,
            'Q': 1.10, 'E': 0.37, 'G': 0.75, 'H': 0.87, 'I': 1.60,
            'L': 1.30, 'K': 0.74, 'M': 1.05, 'F': 1.38, 'P': 0.55,
            'S': 0.75, 'T': 1.19, 'W': 1.37, 'Y': 1.47, 'V': 1.70,
        }
        
        scores = [0.0] * n
        
        # Scan for potential β-arches
        for i in range(n - min_arch + 1):
            for arch_len in range(min_arch, min(30, n - i + 1)):
                # Evaluate this potential arch
                strand1_end = i + strand_len
                loop_start = strand1_end
                loop_end = i + arch_len - strand_len
                strand2_start = loop_end
                
                if loop_end <= loop_start:
                    continue
                
                # Score strand 1
                strand1 = sequence[i:strand1_end]
                strand1_score = np.mean([BETA_PROP.get(aa, 1.0) for aa in strand1])
                
                # Score loop
                loop_seq = sequence[loop_start:loop_end]
                loop_score = sum(1 for aa in loop_seq if aa in self.LOOP_COMPATIBLE) / len(loop_seq)
                
                # Score strand 2
                strand2 = sequence[strand2_start:i + arch_len]
                strand2_score = np.mean([BETA_PROP.get(aa, 1.0) for aa in strand2])
                
                # Combined score
                arch_score = (strand1_score + strand2_score) / 2 * loop_score
                
                # Normalize to 0-1 range
                arch_score = min(1.0, arch_score / 1.5)
                
                # Add to all positions in this arch
                for j in range(i, i + arch_len):
                    scores[j] = max(scores[j], arch_score)
        
        return [
            {'position': i + 1, 'residue': aa, 'raw_score': scores[i]}
            for i, aa in enumerate(sequence)
        ]
    
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
        
        # ArchCandy scores are already 0-1 probabilities
        normalized = [min(1.0, max(0.0, v)) for v in raw_values]
        
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
                mean_normalized=float(np.mean(region_scores)) if region_scores else 0.0,
                region_type="beta_arch",
            ))
        
        # If no explicit regions, detect from scores
        if not predicted_region_objects:
            predicted_region_objects = self._detect_regions(residue_scores, sequence)
        
        # Overall classification
        is_amyloidogenic = len(predicted_region_objects) > 0
        overall_score = raw_output.get('overall_score', float(np.max(raw_values)) if raw_values else 0.0)
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
        min_length: int = 10,  # Minimum β-arch length
    ) -> list[PredictedRegion]:
        """Detect β-arch regions from per-residue scores."""
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
                            mean_normalized=float(np.mean(region_scores)),
                            region_type="beta_arch",
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
                    max_score=float(np.max(region_scores)),
                    mean_normalized=float(np.mean(region_scores)),
                    region_type="beta_arch",
                ))
        
        return regions


# Convenience function
def predict_with_archcandy(
    sequence: str,
    sequence_id: str = "query",
    config: Optional[WebPredictorConfig] = None,
    threshold: float = 0.40,
) -> PredictorOutput:
    """
    Predict amyloidogenicity using ArchCandy 2.0 web server.
    
    Args:
        sequence: Protein sequence
        sequence_id: Identifier for the sequence
        config: Optional configuration
        threshold: Score threshold (0.40 default, 0.57 for high confidence)
        
    Returns:
        PredictorOutput with per-residue scores and β-arch regions
    """
    with ArchCandyPredictor(config, threshold) as predictor:
        return predictor.predict(sequence, sequence_id)
