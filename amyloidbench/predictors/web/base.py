"""
Base class for Selenium-based web predictors.

This module provides the foundation for automated interaction with amyloid
prediction web servers. Using Selenium ensures we capture the exact behavior
of the original tools, including any server-side preprocessing, trained model
weights, and algorithm implementations that may differ from published
descriptions.

Design Philosophy
-----------------
Web automation is methodologically superior to local reimplementation because:
1. **Algorithm fidelity**: Exact behavior of published tools
2. **Model weights**: Neural networks use trained parameters
3. **Reproducibility**: Results match manual server submission
4. **Validation**: Easy to verify wrapper correctness

However, web automation has limitations:
- Server availability and rate limits
- Network latency
- Output format changes require wrapper updates
- Some servers may block automated access
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    WebDriverException,
)

# Try to import output models from parent
try:
    from ..output_models import (
        PredictorOutput,
        ResidueScore,
        PredictedRegion,
        ClassificationLabel,
        ScoreType,
        normalize_scores,
        classify_residues,
    )
except ImportError:
    # Fallback for standalone testing
    PredictorOutput = None
    ResidueScore = None


logger = logging.getLogger(__name__)


@dataclass
class WebPredictorConfig:
    """Configuration for web-based predictors."""
    
    # Browser settings
    headless: bool = True
    browser: str = "chrome"  # 'chrome', 'firefox'
    implicit_wait: float = 10.0
    page_load_timeout: float = 60.0
    script_timeout: float = 30.0
    
    # Request settings
    max_retries: int = 3
    retry_delay: float = 5.0
    request_delay: float = 1.0  # Delay between requests (rate limiting)
    
    # Output settings
    save_screenshots: bool = False
    screenshot_dir: Optional[str] = None
    save_raw_html: bool = False
    html_output_dir: Optional[str] = None
    
    # Caching
    use_cache: bool = True
    cache_dir: Optional[str] = None
    cache_ttl_hours: float = 24.0
    
    # User agent
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 "
        "AmyloidBench/1.0 (Research; contact@example.com)"
    )


class WebPredictorError(Exception):
    """Base exception for web predictor errors."""
    pass


class ServerUnavailableError(WebPredictorError):
    """Server is not responding or unavailable."""
    pass


class ParsingError(WebPredictorError):
    """Failed to parse server output."""
    pass


class RateLimitError(WebPredictorError):
    """Server rate limit exceeded."""
    pass


class BaseWebPredictor(ABC):
    """
    Abstract base class for Selenium-based web predictors.
    
    Subclasses must implement:
    - server_url: The URL of the prediction server
    - predictor_name: Name of the predictor
    - _submit_sequence(): Logic to submit sequence to server
    - _parse_results(): Logic to extract results from page
    
    The base class handles:
    - Browser initialization and cleanup
    - Retry logic and error handling
    - Rate limiting
    - Result caching
    - Screenshot capture for debugging
    - Standardized output format
    """
    
    # To be defined by subclasses
    server_url: str = ""
    predictor_name: str = ""
    predictor_version: str = "unknown"
    score_type: ScoreType = ScoreType.RAW
    default_threshold: float = 0.5
    
    def __init__(self, config: Optional[WebPredictorConfig] = None):
        """Initialize the web predictor."""
        self.config = config or WebPredictorConfig()
        self._driver: Optional[WebDriver] = None
        self._cache: dict[str, Any] = {}
        
        # Set up directories
        if self.config.screenshot_dir:
            Path(self.config.screenshot_dir).mkdir(parents=True, exist_ok=True)
        if self.config.cache_dir:
            Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def _get_driver(self) -> WebDriver:
        """Get or create a WebDriver instance."""
        if self._driver is None:
            self._driver = self._create_driver()
        return self._driver
    
    def _create_driver(self) -> WebDriver:
        """Create a new WebDriver instance."""
        if self.config.browser == "chrome":
            options = ChromeOptions()
            
            if self.config.headless:
                options.add_argument("--headless=new")
            
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
            options.add_argument(f"--user-agent={self.config.user_agent}")
            
            # Disable images for faster loading
            prefs = {
                "profile.managed_default_content_settings.images": 2,
            }
            options.add_experimental_option("prefs", prefs)
            
            driver = webdriver.Chrome(options=options)
            
        elif self.config.browser == "firefox":
            from selenium.webdriver.firefox.options import Options as FirefoxOptions
            
            options = FirefoxOptions()
            if self.config.headless:
                options.add_argument("--headless")
            
            driver = webdriver.Firefox(options=options)
        else:
            raise ValueError(f"Unsupported browser: {self.config.browser}")
        
        # Set timeouts
        driver.implicitly_wait(self.config.implicit_wait)
        driver.set_page_load_timeout(self.config.page_load_timeout)
        driver.set_script_timeout(self.config.script_timeout)
        
        return driver
    
    def close(self) -> None:
        """Close the WebDriver and clean up resources."""
        if self._driver is not None:
            try:
                self._driver.quit()
            except Exception as e:
                logger.warning(f"Error closing driver: {e}")
            finally:
                self._driver = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure driver is closed."""
        self.close()
        return False
    
    def _take_screenshot(self, name: str) -> Optional[str]:
        """Take a screenshot for debugging."""
        if not self.config.save_screenshots or not self.config.screenshot_dir:
            return None
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.predictor_name}_{name}_{timestamp}.png"
            filepath = Path(self.config.screenshot_dir) / filename
            self._driver.save_screenshot(str(filepath))
            logger.debug(f"Screenshot saved: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.warning(f"Failed to take screenshot: {e}")
            return None
    
    def _save_html(self, name: str) -> Optional[str]:
        """Save page HTML for debugging."""
        if not self.config.save_raw_html or not self.config.html_output_dir:
            return None
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.predictor_name}_{name}_{timestamp}.html"
            filepath = Path(self.config.html_output_dir) / filename
            filepath.write_text(self._driver.page_source)
            logger.debug(f"HTML saved: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.warning(f"Failed to save HTML: {e}")
            return None
    
    def _get_cache_key(self, sequence: str) -> str:
        """Generate cache key for a sequence."""
        import hashlib
        seq_hash = hashlib.md5(sequence.encode()).hexdigest()
        return f"{self.predictor_name}_{seq_hash}"
    
    def _check_cache(self, sequence: str) -> Optional[PredictorOutput]:
        """Check if result is cached."""
        if not self.config.use_cache:
            return None
        
        cache_key = self._get_cache_key(sequence)
        
        # Check memory cache
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            # Check TTL
            cache_time = datetime.fromisoformat(cached.get('timestamp', '1970-01-01'))
            age_hours = (datetime.now() - cache_time).total_seconds() / 3600
            if age_hours < self.config.cache_ttl_hours:
                logger.debug(f"Cache hit for {cache_key}")
                return cached.get('result')
        
        return None
    
    def _save_to_cache(self, sequence: str, result: PredictorOutput) -> None:
        """Save result to cache."""
        if not self.config.use_cache:
            return
        
        cache_key = self._get_cache_key(sequence)
        self._cache[cache_key] = {
            'timestamp': datetime.now().isoformat(),
            'result': result,
        }
    
    @abstractmethod
    def _submit_sequence(self, sequence: str, sequence_id: str) -> None:
        """
        Submit a sequence to the web server.
        
        This method should:
        1. Navigate to the server URL if needed
        2. Fill in the sequence input form
        3. Submit the form
        4. Wait for results to load
        
        Args:
            sequence: The protein sequence to analyze
            sequence_id: Identifier for the sequence
        """
        pass
    
    @abstractmethod
    def _parse_results(self, sequence: str, sequence_id: str) -> PredictorOutput:
        """
        Parse the results from the web page.
        
        This method should:
        1. Extract per-residue scores
        2. Identify predicted regions
        3. Determine overall classification
        4. Return standardized PredictorOutput
        
        Args:
            sequence: The original sequence
            sequence_id: Identifier for the sequence
            
        Returns:
            Standardized PredictorOutput object
        """
        pass
    
    def _wait_for_results(
        self,
        locator: tuple[str, str],
        timeout: float = 60.0,
        poll_frequency: float = 1.0,
    ) -> Any:
        """
        Wait for results to appear on the page.
        
        Args:
            locator: Tuple of (By.XXX, "selector")
            timeout: Maximum wait time in seconds
            poll_frequency: How often to check (seconds)
            
        Returns:
            The found element
            
        Raises:
            TimeoutException: If results don't appear within timeout
        """
        wait = WebDriverWait(
            self._driver,
            timeout,
            poll_frequency=poll_frequency,
        )
        return wait.until(EC.presence_of_element_located(locator))
    
    def _wait_for_element_clickable(
        self,
        locator: tuple[str, str],
        timeout: float = 10.0,
    ) -> Any:
        """Wait for an element to be clickable."""
        wait = WebDriverWait(self._driver, timeout)
        return wait.until(EC.element_to_be_clickable(locator))
    
    def predict(
        self,
        sequence: str,
        sequence_id: str = "query",
    ) -> PredictorOutput:
        """
        Submit a sequence for prediction and return results.
        
        Args:
            sequence: Protein sequence (single-letter amino acid codes)
            sequence_id: Optional identifier for the sequence
            
        Returns:
            PredictorOutput with per-residue scores and regions
        """
        # Validate sequence
        sequence = self._validate_sequence(sequence)
        
        # Check cache
        cached = self._check_cache(sequence)
        if cached is not None:
            return cached
        
        # Perform prediction with retries
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                driver = self._get_driver()
                
                # Submit and wait for results
                self._submit_sequence(sequence, sequence_id)
                
                # Parse results
                result = self._parse_results(sequence, sequence_id)
                result.execution_time = time.time() - start_time
                
                # Cache result
                self._save_to_cache(sequence, result)
                
                # Rate limiting delay
                time.sleep(self.config.request_delay)
                
                return result
                
            except TimeoutException as e:
                last_error = e
                logger.warning(
                    f"{self.predictor_name}: Timeout on attempt {attempt + 1}, "
                    f"retrying in {self.config.retry_delay}s..."
                )
                self._take_screenshot(f"timeout_attempt{attempt + 1}")
                time.sleep(self.config.retry_delay)
                
            except WebDriverException as e:
                last_error = e
                logger.warning(
                    f"{self.predictor_name}: WebDriver error on attempt {attempt + 1}: {e}"
                )
                # Recreate driver
                self.close()
                time.sleep(self.config.retry_delay)
                
            except Exception as e:
                last_error = e
                logger.error(f"{self.predictor_name}: Error: {e}")
                self._take_screenshot(f"error_attempt{attempt + 1}")
                self._save_html(f"error_attempt{attempt + 1}")
                raise
        
        # All retries failed
        raise ServerUnavailableError(
            f"{self.predictor_name}: Failed after {self.config.max_retries} attempts. "
            f"Last error: {last_error}"
        )
    
    def _validate_sequence(self, sequence: str) -> str:
        """Validate and clean protein sequence."""
        # Remove whitespace and convert to uppercase
        sequence = "".join(sequence.split()).upper()
        
        # Valid amino acid codes
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        
        # Check for invalid characters
        invalid = set(sequence) - valid_aa
        if invalid:
            logger.warning(
                f"Sequence contains non-standard amino acids: {invalid}. "
                "These will be removed."
            )
            sequence = "".join(aa for aa in sequence if aa in valid_aa)
        
        if len(sequence) < 6:
            raise ValueError(
                f"Sequence too short ({len(sequence)} aa). "
                "Most predictors require at least 6 residues."
            )
        
        return sequence
    
    def predict_batch(
        self,
        sequences: list[tuple[str, str]],  # List of (id, sequence) tuples
        progress_callback: Optional[callable] = None,
    ) -> list[PredictorOutput]:
        """
        Predict amyloidogenicity for multiple sequences.
        
        Args:
            sequences: List of (sequence_id, sequence) tuples
            progress_callback: Optional callback(current, total, result)
            
        Returns:
            List of PredictorOutput objects
        """
        results = []
        total = len(sequences)
        
        for i, (seq_id, sequence) in enumerate(sequences):
            try:
                result = self.predict(sequence, seq_id)
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, total, result)
                    
            except Exception as e:
                logger.error(f"Failed to predict {seq_id}: {e}")
                # Create error result
                if PredictorOutput is not None:
                    error_result = PredictorOutput(
                        predictor_name=self.predictor_name,
                        predictor_version=self.predictor_version,
                        sequence_id=seq_id,
                        sequence=sequence,
                        residue_scores=[],
                        predicted_regions=[],
                        overall_classification=ClassificationLabel.NOT_ANALYZED,
                        overall_score=0.0,
                        overall_probability=0.0,
                        score_type=self.score_type,
                        threshold=self.default_threshold,
                        source="web",
                        error=str(e),
                    )
                    results.append(error_result)
        
        return results
    
    def check_server_status(self) -> bool:
        """Check if the server is available."""
        try:
            driver = self._get_driver()
            driver.get(self.server_url)
            time.sleep(2)
            return True
        except Exception as e:
            logger.warning(f"{self.predictor_name} server check failed: {e}")
            return False


# Utility functions for parsing common output formats

def parse_score_table(
    table_element,
    position_col: int = 0,
    residue_col: int = 1,
    score_col: int = 2,
    skip_header: bool = True,
) -> list[tuple[int, str, float]]:
    """
    Parse a simple HTML table with position, residue, score columns.
    
    Returns:
        List of (position, residue, score) tuples
    """
    rows = table_element.find_elements(By.TAG_NAME, "tr")
    results = []
    
    for i, row in enumerate(rows):
        if skip_header and i == 0:
            continue
        
        cells = row.find_elements(By.TAG_NAME, "td")
        if len(cells) > max(position_col, residue_col, score_col):
            try:
                position = int(cells[position_col].text.strip())
                residue = cells[residue_col].text.strip()
                score = float(cells[score_col].text.strip())
                results.append((position, residue, score))
            except (ValueError, IndexError):
                continue
    
    return results


def extract_regions_from_scores(
    positions: list[int],
    scores: list[float],
    sequence: str,
    threshold: float,
    min_length: int = 4,
) -> list[PredictedRegion]:
    """
    Extract continuous regions above threshold from per-residue scores.
    
    Args:
        positions: 1-indexed positions
        scores: Corresponding scores
        sequence: Full protein sequence
        threshold: Score threshold for region detection
        min_length: Minimum region length
        
    Returns:
        List of PredictedRegion objects
    """
    if PredictedRegion is None:
        return []
    
    regions = []
    in_region = False
    region_start = 0
    region_scores = []
    
    for pos, score in zip(positions, scores):
        if score >= threshold:
            if not in_region:
                in_region = True
                region_start = pos
                region_scores = [score]
            else:
                region_scores.append(score)
        else:
            if in_region:
                # End of region
                region_end = pos - 1
                if region_end - region_start + 1 >= min_length:
                    import numpy as np
                    regions.append(PredictedRegion(
                        start=region_start,
                        end=region_end,
                        sequence=sequence[region_start - 1:region_end],
                        mean_score=float(np.mean(region_scores)),
                        max_score=float(np.max(region_scores)),
                        mean_normalized=float(np.mean(region_scores)),  # Adjust if needed
                    ))
                in_region = False
                region_scores = []
    
    # Handle region at end of sequence
    if in_region:
        region_end = positions[-1]
        if region_end - region_start + 1 >= min_length:
            import numpy as np
            regions.append(PredictedRegion(
                start=region_start,
                end=region_end,
                sequence=sequence[region_start - 1:region_end],
                mean_score=float(np.mean(region_scores)),
                max_score=float(np.max(region_scores)),
                mean_normalized=float(np.mean(region_scores)),
            ))
    
    return regions
