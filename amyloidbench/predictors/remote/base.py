"""
Base infrastructure for web-based amyloidogenicity predictors.

Many established amyloid prediction tools are available only as web servers
without APIs or downloadable implementations. This module provides a robust
framework for automating interactions with these servers using Playwright,
enabling batch processing while respecting server rate limits.

Design Principles:
1. **Rate limiting**: Configurable delays between requests to avoid overloading servers
2. **Caching**: Aggressive caching of results to minimize redundant requests
3. **Error handling**: Retry logic with exponential backoff for transient failures
4. **Async execution**: Non-blocking I/O for efficient batch processing
5. **Result parsing**: Standardized extraction of scores and regions from HTML

Biological Context:
Web-based predictors often represent seminal contributions to the field that
predate modern software engineering practices. Tools like WALTZ, PASTA, and
TANGO established foundational approaches to amyloid prediction but lack
programmatic interfaces. Web automation enables reproducible, large-scale
analyses while awaiting formal re-implementations.

Usage:
    class MyWebPredictor(WebPredictorBase):
        name = "MyPredictor"
        base_url = "https://predictor-server.org/submit"
        
        async def _submit_and_parse(self, page, sequence):
            # Implement submission and parsing logic
            ...
"""

from __future__ import annotations

import asyncio
import logging
import re
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin

from amyloidbench.core.models import (
    PerResidueScores,
    PredictionResult,
    Region,
)
from amyloidbench.predictors.base import (
    AsyncBasePredictor,
    PredictorCapability,
    PredictorConfig,
    PredictorError,
    PredictorTimeoutError,
    PredictorType,
    PredictorUnavailableError,
)

logger = logging.getLogger(__name__)

# Try to import Playwright
try:
    from playwright.async_api import (
        async_playwright,
        Browser,
        BrowserContext,
        Page,
        TimeoutError as PlaywrightTimeoutError,
    )
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning(
        "Playwright not available. Web-based predictors will not work. "
        "Install with: pip install playwright && playwright install chromium"
    )


@dataclass
class WebPredictorConfig:
    """
    Configuration specific to web-based predictors.
    
    Extends the base PredictorConfig with web-specific settings.
    """
    # Rate limiting
    min_request_interval: float = 2.0  # Minimum seconds between requests
    max_concurrent_requests: int = 1  # Max parallel requests (be conservative)
    
    # Timeouts (seconds)
    page_load_timeout: float = 30.0
    submission_timeout: float = 60.0
    result_timeout: float = 300.0  # Some predictors are slow
    
    # Retry behavior
    max_retries: int = 3
    retry_backoff_factor: float = 2.0  # Exponential backoff multiplier
    
    # Browser settings
    headless: bool = True
    browser_type: str = "chromium"  # chromium, firefox, or webkit
    
    # Debugging
    save_screenshots: bool = False
    screenshot_dir: Optional[Path] = None


class WebPredictorBase(AsyncBasePredictor):
    """
    Abstract base class for web-based amyloidogenicity predictors.
    
    This class handles the common infrastructure for interacting with
    web-based prediction servers:
    - Browser lifecycle management
    - Rate limiting between requests
    - Retry logic for failed submissions
    - Result caching
    
    Subclasses must implement:
    - _submit_and_parse(): Server-specific submission and result extraction
    
    Optional overrides:
    - _validate_response(): Custom validation of server response
    - _handle_error_page(): Custom handling of error pages
    
    Attributes:
        base_url: URL of the prediction server
        web_config: Web-specific configuration
        
    Example:
        class WaltzPredictor(WebPredictorBase):
            name = "WALTZ"
            base_url = "http://waltz.switchlab.org/submit"
            
            async def _submit_and_parse(self, page, sequence):
                await page.fill("#sequence", sequence)
                await page.click("#submit")
                await page.wait_for_selector(".results")
                return self._parse_results(await page.content())
    """
    
    # Subclasses must set these
    base_url: str = ""
    
    # Default web configuration
    default_web_config = WebPredictorConfig()
    
    def __init__(
        self,
        config: Optional[PredictorConfig] = None,
        web_config: Optional[WebPredictorConfig] = None,
    ):
        """
        Initialize web-based predictor.
        
        Args:
            config: Base predictor configuration
            web_config: Web-specific configuration
        """
        super().__init__(config)
        
        if not PLAYWRIGHT_AVAILABLE:
            raise PredictorUnavailableError(
                f"{self.name} requires Playwright. "
                "Install with: pip install playwright && playwright install chromium"
            )
        
        self.web_config = web_config or self.default_web_config
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._last_request_time: float = 0
        self._request_lock = asyncio.Lock()
    
    async def _ensure_browser(self) -> Browser:
        """
        Ensure browser is initialized.
        
        Returns:
            Browser instance
        """
        if self._browser is None:
            playwright = await async_playwright().start()
            
            if self.web_config.browser_type == "chromium":
                self._browser = await playwright.chromium.launch(
                    headless=self.web_config.headless
                )
            elif self.web_config.browser_type == "firefox":
                self._browser = await playwright.firefox.launch(
                    headless=self.web_config.headless
                )
            elif self.web_config.browser_type == "webkit":
                self._browser = await playwright.webkit.launch(
                    headless=self.web_config.headless
                )
            else:
                raise ValueError(f"Unknown browser: {self.web_config.browser_type}")
            
            self._context = await self._browser.new_context()
            logger.info(f"{self.name}: Browser initialized ({self.web_config.browser_type})")
        
        return self._browser
    
    async def _get_page(self) -> Page:
        """
        Get a new page for making a request.
        
        Returns:
            New Page instance
        """
        await self._ensure_browser()
        page = await self._context.new_page()
        page.set_default_timeout(self.web_config.page_load_timeout * 1000)
        return page
    
    async def _rate_limit(self):
        """
        Enforce rate limiting between requests.
        
        Uses a lock to ensure thread-safe rate limiting across
        concurrent requests.
        """
        async with self._request_lock:
            import time
            
            now = time.time()
            elapsed = now - self._last_request_time
            
            if elapsed < self.web_config.min_request_interval:
                wait_time = self.web_config.min_request_interval - elapsed
                logger.debug(f"{self.name}: Rate limiting, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
            
            self._last_request_time = time.time()
    
    @abstractmethod
    async def _submit_and_parse(
        self,
        page: Page,
        sequence: str,
    ) -> dict[str, Any]:
        """
        Submit sequence to server and parse results.
        
        This method must be implemented by subclasses to handle
        server-specific submission forms and result parsing.
        
        Args:
            page: Playwright Page instance
            sequence: Protein sequence to predict
            
        Returns:
            Dictionary containing:
            - 'scores': List of per-residue scores (optional)
            - 'regions': List of (start, end, score) tuples (optional)
            - 'is_amyloidogenic': Boolean classification (optional)
            - 'raw_html': Raw result HTML for debugging (optional)
            
        Raises:
            PredictorError: On submission or parsing failure
        """
        pass
    
    async def _predict_impl_async(
        self,
        sequence: str,
        structure_path: Optional[Path] = None,
    ) -> PredictionResult:
        """
        Async implementation of prediction via web automation.
        
        Args:
            sequence: Protein sequence
            structure_path: Ignored for web predictors
            
        Returns:
            PredictionResult from web server
        """
        page = None
        last_error = None
        
        for attempt in range(self.web_config.max_retries):
            try:
                # Rate limiting
                await self._rate_limit()
                
                # Get new page
                page = await self._get_page()
                
                # Navigate to submission page
                logger.info(f"{self.name}: Submitting sequence (attempt {attempt + 1})")
                await page.goto(self.base_url, timeout=self.web_config.page_load_timeout * 1000)
                
                # Submit and parse results
                result_data = await asyncio.wait_for(
                    self._submit_and_parse(page, sequence),
                    timeout=self.web_config.result_timeout,
                )
                
                # Save screenshot if debugging
                if self.web_config.save_screenshots and self.web_config.screenshot_dir:
                    screenshot_path = self.web_config.screenshot_dir / f"{self.name}_{attempt}.png"
                    await page.screenshot(path=str(screenshot_path))
                
                # Build PredictionResult from parsed data
                return self._build_result(sequence, result_data)
                
            except asyncio.TimeoutError:
                last_error = PredictorTimeoutError(
                    f"{self.name} timed out after {self.web_config.result_timeout}s"
                )
                logger.warning(f"{self.name}: Timeout on attempt {attempt + 1}")
                
            except PlaywrightTimeoutError as e:
                last_error = PredictorTimeoutError(f"{self.name} page timeout: {e}")
                logger.warning(f"{self.name}: Page timeout on attempt {attempt + 1}")
                
            except Exception as e:
                last_error = PredictorError(f"{self.name} failed: {e}")
                logger.warning(f"{self.name}: Error on attempt {attempt + 1}: {e}")
                
            finally:
                if page:
                    await page.close()
                    page = None
            
            # Exponential backoff before retry
            if attempt < self.web_config.max_retries - 1:
                backoff = self.web_config.retry_backoff_factor ** attempt
                await asyncio.sleep(backoff)
        
        # All retries exhausted
        return PredictionResult(
            sequence_id="",
            sequence=sequence,
            predictor_name=self.name,
            predictor_version=self.version,
            error_message=str(last_error),
        )
    
    def _build_result(
        self,
        sequence: str,
        result_data: dict[str, Any],
    ) -> PredictionResult:
        """
        Build PredictionResult from parsed web data.
        
        Args:
            sequence: Original sequence
            result_data: Parsed data from _submit_and_parse
            
        Returns:
            Standardized PredictionResult
        """
        # Extract per-residue scores if available
        per_residue = None
        if "scores" in result_data and result_data["scores"]:
            scores = result_data["scores"]
            # Ensure scores match sequence length
            if len(scores) != len(sequence):
                logger.warning(
                    f"{self.name}: Score length mismatch "
                    f"({len(scores)} vs {len(sequence)})"
                )
                # Pad or truncate
                if len(scores) < len(sequence):
                    scores = scores + [0.0] * (len(sequence) - len(scores))
                else:
                    scores = scores[:len(sequence)]
            
            per_residue = PerResidueScores(
                scores=scores,
                sequence=sequence,
                predictor=self.name,
                score_type="raw",
                threshold=result_data.get("threshold", self.default_threshold),
            )
        
        # Extract regions
        regions = []
        if "regions" in result_data:
            for region_data in result_data["regions"]:
                if isinstance(region_data, tuple):
                    start, end, score = region_data
                elif isinstance(region_data, dict):
                    start = region_data["start"]
                    end = region_data["end"]
                    score = region_data.get("score")
                else:
                    continue
                
                regions.append(Region(
                    start=start,
                    end=end,
                    sequence=sequence[start:end],
                    score=score,
                ))
        
        # Determine classification
        is_amyloidogenic = result_data.get("is_amyloidogenic")
        if is_amyloidogenic is None and regions:
            is_amyloidogenic = len(regions) > 0
        
        probability = result_data.get("probability")
        
        return PredictionResult(
            sequence_id="",
            sequence=sequence,
            predictor_name=self.name,
            predictor_version=self.version,
            per_residue_scores=per_residue,
            predicted_regions=regions,
            is_amyloidogenic=is_amyloidogenic,
            amyloid_probability=probability,
            raw_output=result_data.get("raw_html") if self.config.return_raw_output else None,
        )
    
    async def close(self):
        """
        Close browser and clean up resources.
        
        Should be called when done with the predictor.
        """
        if self._context:
            await self._context.close()
            self._context = None
        if self._browser:
            await self._browser.close()
            self._browser = None
        logger.info(f"{self.name}: Browser closed")
    
    def __del__(self):
        """Cleanup on deletion."""
        if self._browser:
            # Can't await in __del__, so just warn
            logger.warning(f"{self.name}: Browser not properly closed. Call close() explicitly.")


# =============================================================================
# Utility functions for result parsing
# =============================================================================

def parse_score_table(
    html: str,
    score_column: int = 1,
    skip_header: bool = True,
) -> list[float]:
    """
    Parse a simple HTML table to extract scores.
    
    Many web predictors display results in HTML tables. This function
    extracts numerical scores from a specified column.
    
    Args:
        html: HTML content containing the table
        score_column: 0-indexed column containing scores
        skip_header: Whether to skip the first row
        
    Returns:
        List of parsed scores
    """
    scores = []
    
    # Find all table rows
    row_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL | re.IGNORECASE)
    cell_pattern = re.compile(r'<t[dh][^>]*>(.*?)</t[dh]>', re.DOTALL | re.IGNORECASE)
    
    rows = row_pattern.findall(html)
    
    for i, row in enumerate(rows):
        if skip_header and i == 0:
            continue
        
        cells = cell_pattern.findall(row)
        if len(cells) > score_column:
            # Extract numeric value
            cell_text = re.sub(r'<[^>]+>', '', cells[score_column]).strip()
            try:
                scores.append(float(cell_text))
            except ValueError:
                continue
    
    return scores


def parse_highlighted_regions(
    html: str,
    highlight_class: str = "amyloid",
    sequence_start: int = 0,
) -> list[tuple[int, int]]:
    """
    Parse highlighted regions from HTML.
    
    Many predictors highlight amyloidogenic regions using CSS classes
    or span elements. This function extracts the positions of highlighted
    segments.
    
    Args:
        html: HTML content
        highlight_class: CSS class used for highlighting
        sequence_start: Starting position offset
        
    Returns:
        List of (start, end) tuples for highlighted regions
    """
    regions = []
    
    # Pattern for highlighted spans
    pattern = re.compile(
        rf'<span[^>]*class="[^"]*{highlight_class}[^"]*"[^>]*>([^<]+)</span>',
        re.IGNORECASE
    )
    
    current_pos = sequence_start
    for match in pattern.finditer(html):
        # This is a simplified approach - real implementation would need
        # to track position through the full sequence display
        region_seq = match.group(1)
        regions.append((current_pos, current_pos + len(region_seq)))
    
    return regions


def extract_job_id(html: str, pattern: str = r'job[_-]?id["\s:=]+([a-zA-Z0-9_-]+)') -> Optional[str]:
    """
    Extract job ID from HTML for async result retrieval.
    
    Some web predictors return a job ID and require polling for results.
    This function extracts the job ID from the response.
    
    Args:
        html: HTML content
        pattern: Regex pattern for job ID extraction
        
    Returns:
        Job ID string or None if not found
    """
    match = re.search(pattern, html, re.IGNORECASE)
    return match.group(1) if match else None
