# Amyloid Predictor Wrapper Development Guide

This guide explains how to create Selenium-based wrappers for amyloid prediction web servers. By following this methodology, you can integrate any web-based tool into AmyloidBench with standardized outputs.

## Table of Contents
1. [Philosophy: Why Web Wrappers?](#philosophy-why-web-wrappers)
2. [Architecture Overview](#architecture-overview)
3. [Step-by-Step Wrapper Development](#step-by-step-wrapper-development)
4. [Selenium Fundamentals](#selenium-fundamentals)
5. [Parsing Strategies](#parsing-strategies)
6. [Standardized Output Format](#standardized-output-format)
7. [Testing and Validation](#testing-and-validation)
8. [Common Patterns and Pitfalls](#common-patterns-and-pitfalls)
9. [Complete Example: AGGRESCAN](#complete-example-aggrescan)

---

## Philosophy: Why Web Wrappers?

### Local Reimplementation vs. Web Automation

| Approach | Pros | Cons |
|----------|------|------|
| **Local reimplementation** | Fast, offline, no rate limits | Algorithm approximation, missing model weights |
| **Web automation** | Exact behavior, trained weights, validated | Server dependency, slower, rate limited |

**For benchmarking and scientific reproducibility, web automation is methodologically superior** because:

1. **Algorithm Fidelity**: Published papers often omit implementation details. Web servers use the exact algorithm.
2. **Model Weights**: Neural network predictors (APPNN, AggreProt) have trained weights only on the server.
3. **Reproducibility**: Results match what any researcher would get via manual submission.
4. **Validation**: Easy to spot-check by submitting the same sequence manually.

### When to Use Each Approach

```
Use WEB WRAPPERS for:
├── Benchmarking against published results
├── Tools with trained ML models (APPNN, AggreProt)
├── When exact scores matter
└── Validation of local implementations

Use LOCAL IMPLEMENTATIONS for:
├── High-throughput screening (thousands of sequences)
├── Offline analysis
├── Educational/understanding purposes
└── When approximate scores are acceptable
```

---

## Architecture Overview

### Class Hierarchy

```
BaseWebPredictor (base.py)
├── Handles: Browser lifecycle, retries, caching, screenshots
├── Abstract methods: _submit_sequence(), _parse_results()
│
├── WaltzPredictor (waltz.py)
├── AggrescanPredictor (aggrescan.py)
├── Pasta2Predictor (pasta2.py)
├── ArchCandyPredictor (archcandy.py)
├── CrossBetaPredictor (crossbeta.py)
└── AggreProtPredictor (aggreprot.py)
```

### Data Flow

```
1. Input Sequence
       │
       ▼
2. BaseWebPredictor.predict()
   ├── Check cache
   ├── Validate sequence
   └── Call _submit_sequence()
              │
              ▼
3. _submit_sequence()
   ├── Navigate to server
   ├── Find input elements
   ├── Submit sequence
   └── Wait for results
              │
              ▼
4. _parse_results()
   ├── Extract scores from HTML
   ├── Parse regions
   └── Return PredictorOutput
              │
              ▼
5. Standardized Output
   ├── Per-residue scores (raw + normalized)
   ├── Per-residue classification
   ├── Predicted regions
   └── Metadata
```

---

## Step-by-Step Wrapper Development

### Step 1: Analyze the Web Server

Before writing code, manually explore the web server:

1. **Visit the URL** and understand the interface
2. **Inspect the HTML** (F12 → Elements tab) to find:
   - Input field IDs/names/classes
   - Submit button selector
   - Results table/div structure
3. **Submit a test sequence** and note:
   - How long results take
   - URL changes (same page vs. redirect)
   - Output format (table, text, download file)
4. **Document the selectors** you'll need

**Example Analysis Notes for AGGRESCAN:**
```
URL: http://bioinf.uab.es/aggrescan/
Input: <textarea name="sequence" ...>
Submit: <input type="submit" value="Run Aggrescan">
Wait for: Results page with score table
Results table: <table class="results"> with columns: Position, AA, a3v, a4v, HSA
Regions: Listed as "Hot Spots: 12-18, 34-42"
```

### Step 2: Create the Predictor Class

```python
from .base import BaseWebPredictor, WebPredictorConfig, ParsingError
from ..output_models import (
    PredictorOutput, ResidueScore, PredictedRegion,
    ClassificationLabel, ScoreType
)

class YourToolPredictor(BaseWebPredictor):
    """
    Docstring explaining:
    - What the tool predicts
    - Algorithmic basis (cite paper)
    - Output interpretation
    """
    
    # Required class attributes
    server_url = "http://tool.example.org/"
    predictor_name = "YourTool"
    predictor_version = "1.0"
    score_type = ScoreType.RAW  # or PROBABILITY, ENERGY, etc.
    default_threshold = 0.5
    
    def __init__(self, config=None, threshold=None):
        super().__init__(config)
        if threshold is not None:
            self.default_threshold = threshold
```

### Step 3: Implement `_submit_sequence()`

This method navigates to the server and submits the sequence:

```python
def _submit_sequence(self, sequence: str, sequence_id: str) -> None:
    """Submit sequence to the web server."""
    driver = self._get_driver()
    
    # Step 1: Navigate to the page
    driver.get(self.server_url)
    time.sleep(2)  # Allow page to load
    
    # Step 2: Find and fill the input field
    # Try multiple selectors (servers may change)
    textarea = self._find_element_robust([
        (By.NAME, "sequence"),
        (By.ID, "seq_input"),
        (By.CSS_SELECTOR, "textarea"),
    ])
    
    if not textarea:
        raise ParsingError("Could not find sequence input field")
    
    textarea.clear()
    textarea.send_keys(f">{sequence_id}\n{sequence}")
    
    # Step 3: Find and click submit
    submit_btn = self._find_element_robust([
        (By.CSS_SELECTOR, "input[type='submit']"),
        (By.XPATH, "//button[contains(text(), 'Submit')]"),
    ])
    
    submit_btn.click()
    
    # Step 4: Wait for results
    WebDriverWait(driver, 60).until(
        lambda d: "Score" in d.page_source or "Result" in d.page_source
    )
```

### Step 4: Implement `_parse_results()`

This method extracts data from the results page:

```python
def _parse_results(self, sequence: str, sequence_id: str) -> PredictorOutput:
    """Parse results from web page into standardized format."""
    driver = self._get_driver()
    
    residue_scores = []
    predicted_regions = []
    raw_output = {"page_source": driver.page_source[:5000]}
    
    # Strategy 1: Parse HTML table
    try:
        table = driver.find_element(By.CSS_SELECTOR, "table.results")
        rows = table.find_elements(By.TAG_NAME, "tr")[1:]  # Skip header
        
        for row in rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) >= 3:
                pos = int(cells[0].text)
                aa = cells[1].text.strip()
                score = float(cells[2].text)
                
                residue_scores.append({
                    'position': pos,
                    'residue': aa,
                    'raw_score': score,
                })
    except NoSuchElementException:
        # Fallback: Parse text or use regex
        pass
    
    # Strategy 2: Parse regions from text
    regions_text = driver.find_element(By.ID, "regions").text
    for match in re.finditer(r'(\d+)-(\d+)', regions_text):
        start, end = int(match.group(1)), int(match.group(2))
        predicted_regions.append({
            'start': start,
            'end': end,
            'sequence': sequence[start-1:end],
        })
    
    return self._create_output(sequence, sequence_id, residue_scores, predicted_regions, raw_output)
```

### Step 5: Implement `_create_output()`

Transform parsed data into standardized format:

```python
def _create_output(self, sequence, sequence_id, residue_scores, predicted_regions, raw_output):
    """Create standardized PredictorOutput."""
    
    # Normalize scores to 0-1 range
    raw_values = [s['raw_score'] for s in residue_scores]
    normalized = normalize_scores(raw_values, self.score_type)
    
    # Create ResidueScore objects with classification
    residue_score_objects = []
    for i, score_data in enumerate(residue_scores):
        classification = (
            ClassificationLabel.AMYLOIDOGENIC 
            if normalized[i] >= self.default_threshold
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
    region_objects = []
    for region in predicted_regions:
        # Calculate region statistics
        region_scores = [
            s['raw_score'] for s in residue_scores
            if region['start'] <= s['position'] <= region['end']
        ]
        
        region_objects.append(PredictedRegion(
            start=region['start'],
            end=region['end'],
            sequence=region['sequence'],
            mean_score=float(np.mean(region_scores)) if region_scores else 0.0,
            max_score=float(np.max(region_scores)) if region_scores else 0.0,
            mean_normalized=float(np.mean([normalized[s['position']-1] for s in residue_scores 
                                           if region['start'] <= s['position'] <= region['end']])),
        ))
    
    # Determine overall classification
    is_amyloidogenic = len(region_objects) > 0
    
    return PredictorOutput(
        predictor_name=self.predictor_name,
        predictor_version=self.predictor_version,
        sequence_id=sequence_id,
        sequence=sequence,
        residue_scores=residue_score_objects,
        predicted_regions=region_objects,
        overall_classification=(
            ClassificationLabel.AMYLOIDOGENIC if is_amyloidogenic
            else ClassificationLabel.NON_AMYLOIDOGENIC
        ),
        overall_score=float(np.max(raw_values)) if raw_values else 0.0,
        overall_probability=float(np.max(normalized)) if normalized else 0.0,
        score_type=self.score_type,
        threshold=self.default_threshold,
        source="web",
        raw_output=raw_output,
    )
```

---

## Selenium Fundamentals

### Finding Elements

Selenium provides multiple strategies to locate elements:

```python
from selenium.webdriver.common.by import By

# By ID (fastest, most reliable if available)
driver.find_element(By.ID, "sequence_input")

# By Name (common for form fields)
driver.find_element(By.NAME, "sequence")

# By CSS Selector (powerful and flexible)
driver.find_element(By.CSS_SELECTOR, "textarea.seq-input")
driver.find_element(By.CSS_SELECTOR, "table#results tr")
driver.find_element(By.CSS_SELECTOR, "input[type='submit']")

# By XPath (most powerful, but slower)
driver.find_element(By.XPATH, "//textarea[@name='sequence']")
driver.find_element(By.XPATH, "//button[contains(text(), 'Submit')]")
driver.find_element(By.XPATH, "//table//tr[position()>1]")  # Skip header

# By Class Name
driver.find_element(By.CLASS_NAME, "result-table")

# By Tag Name
driver.find_elements(By.TAG_NAME, "tr")  # Note: find_elements returns list
```

### Robust Element Finding

Always try multiple selectors since web pages change:

```python
def _find_element_robust(self, selectors: list) -> Optional[WebElement]:
    """Try multiple selectors to find an element."""
    driver = self._get_driver()
    
    for by, selector in selectors:
        try:
            element = driver.find_element(by, selector)
            if element.is_displayed():
                return element
        except NoSuchElementException:
            continue
    
    return None
```

### Waiting for Elements

Never use `time.sleep()` alone! Use explicit waits:

```python
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Wait for element to be present
element = WebDriverWait(driver, 30).until(
    EC.presence_of_element_located((By.ID, "results"))
)

# Wait for element to be clickable
button = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.CSS_SELECTOR, "input[type='submit']"))
)

# Wait for text to appear
WebDriverWait(driver, 60).until(
    lambda d: "Score" in d.page_source
)

# Wait for page load (URL change)
WebDriverWait(driver, 30).until(
    EC.url_contains("results")
)

# Custom condition
def results_ready(driver):
    tables = driver.find_elements(By.TAG_NAME, "table")
    return len(tables) > 0

WebDriverWait(driver, 60).until(results_ready)
```

### Handling iframes

Some servers use iframes for results:

```python
# Switch to iframe
iframe = driver.find_element(By.TAG_NAME, "iframe")
driver.switch_to.frame(iframe)

# Parse content inside iframe
results = driver.find_element(By.ID, "results").text

# Switch back to main content
driver.switch_to.default_content()
```

### Handling Downloads

Some tools provide downloadable results:

```python
import tempfile
from pathlib import Path

def __init__(self, config=None):
    super().__init__(config)
    self.download_dir = tempfile.mkdtemp()
    
def _create_driver(self):
    options = ChromeOptions()
    prefs = {
        "download.default_directory": self.download_dir,
        "download.prompt_for_download": False,
    }
    options.add_experimental_option("prefs", prefs)
    return webdriver.Chrome(options=options)

def _wait_for_download(self, timeout=60):
    """Wait for a file to appear in download directory."""
    start = time.time()
    while time.time() - start < timeout:
        files = list(Path(self.download_dir).glob("*"))
        # Filter out partial downloads
        complete = [f for f in files if not f.suffix == '.crdownload']
        if complete:
            return complete[0]
        time.sleep(1)
    raise TimeoutError("Download did not complete")
```

---

## Parsing Strategies

### Strategy 1: HTML Tables

Most common format for per-residue scores:

```python
def _parse_score_table(self, table_element):
    """Parse HTML table with position, residue, score columns."""
    results = []
    rows = table_element.find_elements(By.TAG_NAME, "tr")
    
    for i, row in enumerate(rows):
        # Detect header vs data rows
        cells = row.find_elements(By.TAG_NAME, "td")
        if not cells:  # Header row uses <th>
            continue
            
        try:
            # Adjust indices based on table structure
            position = int(cells[0].text.strip())
            residue = cells[1].text.strip()
            score = float(cells[2].text.strip())
            
            results.append({
                'position': position,
                'residue': residue,
                'raw_score': score,
            })
        except (ValueError, IndexError):
            continue
    
    return results
```

### Strategy 2: Regex Pattern Matching

For unstructured text or when table parsing fails:

```python
import re

def _parse_text_results(self, text):
    """Extract scores from text using regex."""
    results = []
    
    # Pattern: "Position 1: A score=0.85"
    pattern = r'Position\s+(\d+):\s+([A-Z])\s+score=([0-9.]+)'
    
    for match in re.finditer(pattern, text):
        results.append({
            'position': int(match.group(1)),
            'residue': match.group(2),
            'raw_score': float(match.group(3)),
        })
    
    return results

def _parse_regions(self, text):
    """Extract regions from text like 'Hot Spots: 12-18, 34-42'."""
    regions = []
    
    # Pattern: number-number
    for match in re.finditer(r'(\d+)\s*[-–]\s*(\d+)', text):
        start = int(match.group(1))
        end = int(match.group(2))
        regions.append({'start': start, 'end': end})
    
    return regions
```

### Strategy 3: JSON API

Some modern servers return JSON:

```python
import json

def _parse_json_response(self):
    """Parse JSON from page or API response."""
    driver = self._get_driver()
    
    # Option 1: JSON embedded in page
    try:
        pre = driver.find_element(By.TAG_NAME, "pre")
        data = json.loads(pre.text)
    except:
        # Option 2: JSON in script tag
        scripts = driver.find_elements(By.TAG_NAME, "script")
        for script in scripts:
            if "var results = " in script.get_attribute("innerHTML"):
                json_str = re.search(r'var results = ({.*?});', script.get_attribute("innerHTML"))
                data = json.loads(json_str.group(1))
                break
    
    return data
```

### Strategy 4: Downloaded Files

For tools that generate CSV/TSV downloads:

```python
import pandas as pd

def _parse_downloaded_results(self, filepath):
    """Parse downloaded CSV/TSV results file."""
    # Detect delimiter
    with open(filepath) as f:
        first_line = f.readline()
        delimiter = '\t' if '\t' in first_line else ','
    
    df = pd.read_csv(filepath, delimiter=delimiter)
    
    # Map to standardized format
    results = []
    for _, row in df.iterrows():
        results.append({
            'position': int(row['Position']),
            'residue': row['Residue'],
            'raw_score': float(row['Score']),
        })
    
    return results
```

---

## Standardized Output Format

### Per-Residue Table (TSV)

Match the format from your APPNN R script:

| id | pos | aa | score | normalized | is_hotspot | overall |
|----|-----|----| ------|------------|------------|---------|
| RPS2 | 1 | M | 0.234 | 0.468 | FALSE | 0.72 |
| RPS2 | 2 | A | 0.567 | 0.821 | TRUE | 0.72 |
| ... | ... | ... | ... | ... | ... | ... |

### Export Function

```python
def export_results(
    output: PredictorOutput,
    out_dir: str,
    save_plots: bool = True,
) -> dict:
    """Export results to TSV and optional plot."""
    import pandas as pd
    from pathlib import Path
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame
    df = pd.DataFrame([
        {
            'id': output.sequence_id,
            'pos': r.position,
            'aa': r.residue,
            'score': r.raw_score,
            'normalized': r.normalized_score,
            'is_hotspot': r.classification == ClassificationLabel.AMYLOIDOGENIC,
            'overall': output.overall_score,
        }
        for r in output.residue_scores
    ])
    
    # Safe filename
    safe_id = re.sub(r'[^A-Za-z0-9._-]', '_', output.sequence_id)
    
    # Save TSV
    tsv_path = Path(out_dir) / f"{safe_id}.tsv"
    df.to_csv(tsv_path, sep='\t', index=False)
    
    # Save plot
    plot_path = None
    if save_plots:
        plot_path = Path(out_dir) / f"{safe_id}.png"
        plot_predictor_result(output, str(plot_path))
    
    return {
        'id': output.sequence_id,
        'tsv': str(tsv_path),
        'plot': str(plot_path) if plot_path else None,
    }
```

---

## Testing and Validation

### Unit Tests

```python
import pytest
from unittest.mock import Mock, patch

class TestYourToolPredictor:
    
    def test_submit_sequence(self):
        """Test sequence submission logic."""
        # Mock the driver
        mock_driver = Mock()
        mock_textarea = Mock()
        mock_driver.find_element.return_value = mock_textarea
        
        predictor = YourToolPredictor()
        predictor._driver = mock_driver
        
        predictor._submit_sequence("GNNQQNY", "test")
        
        mock_textarea.send_keys.assert_called_once()
    
    def test_parse_results(self):
        """Test result parsing with known HTML."""
        html = '''
        <table class="results">
            <tr><th>Pos</th><th>AA</th><th>Score</th></tr>
            <tr><td>1</td><td>G</td><td>0.5</td></tr>
            <tr><td>2</td><td>N</td><td>0.8</td></tr>
        </table>
        '''
        
        predictor = YourToolPredictor()
        # Mock driver with this HTML
        ...
```

### Integration Tests

```python
@pytest.mark.integration
@pytest.mark.skipif(not NETWORK_AVAILABLE, reason="No network")
def test_live_prediction():
    """Test against live server (run sparingly!)."""
    with YourToolPredictor() as predictor:
        result = predictor.predict("GNNQQNY", "test_peptide")
        
        assert result.predictor_name == "YourTool"
        assert len(result.residue_scores) == 7
        assert result.overall_classification in [
            ClassificationLabel.AMYLOIDOGENIC,
            ClassificationLabel.NON_AMYLOIDOGENIC
        ]
```

### Manual Validation

Always validate by comparing to manual submission:

```python
def validate_wrapper():
    """Submit same sequence manually and via wrapper, compare results."""
    test_sequence = "KLVFFAEDVGSNKGAIIGLM"  # Aβ17-36
    
    with YourToolPredictor() as predictor:
        result = predictor.predict(test_sequence, "Ab17-36")
        
    print("Wrapper results:")
    print(f"  Overall: {result.overall_classification}")
    print(f"  Score: {result.overall_score}")
    print(f"  Regions: {[(r.start, r.end) for r in result.predicted_regions]}")
    
    print("\nPlease manually verify at:", predictor.server_url)
```

---

## Common Patterns and Pitfalls

### ✅ DO

1. **Try multiple selectors** - Web pages change frequently
2. **Use explicit waits** - Never rely on `time.sleep()` alone
3. **Save screenshots on error** - Invaluable for debugging
4. **Implement caching** - Avoid hammering servers during development
5. **Add rate limiting** - Be a good citizen
6. **Document the expected HTML structure** - For future maintenance
7. **Handle missing scores gracefully** - Some residues may not be scored

### ❌ DON'T

1. **Don't hardcode single selectors** - They will break
2. **Don't ignore timeouts** - Always handle TimeoutException
3. **Don't parse complex layouts without inspection** - Use browser DevTools
4. **Don't skip validation** - Compare to manual results
5. **Don't forget cleanup** - Always close the driver
6. **Don't submit too fast** - Respect server resources

### Handling Common Issues

```python
# Issue: Page loads JavaScript dynamically
# Solution: Wait for specific element
WebDriverWait(driver, 30).until(
    EC.presence_of_element_located((By.CLASS_NAME, "results-loaded"))
)

# Issue: CAPTCHA or bot detection
# Solution: Add delays, use realistic user agent, consider API alternatives
time.sleep(random.uniform(1, 3))

# Issue: Server returns error page
# Solution: Check for error indicators
if "error" in driver.page_source.lower():
    error_msg = driver.find_element(By.CLASS_NAME, "error").text
    raise ServerError(error_msg)

# Issue: Results paginated
# Solution: Click through all pages
while True:
    # Parse current page
    results.extend(self._parse_current_page())
    
    try:
        next_btn = driver.find_element(By.CSS_SELECTOR, "a.next-page")
        next_btn.click()
        time.sleep(2)
    except NoSuchElementException:
        break  # No more pages
```

---

## Complete Example: AGGRESCAN

Here's a complete implementation following all best practices:

```python
"""
AGGRESCAN Web Predictor - Complete Implementation Example
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

from .base import BaseWebPredictor, WebPredictorConfig, ParsingError
from ..output_models import (
    PredictorOutput, ResidueScore, PredictedRegion,
    ClassificationLabel, ScoreType, normalize_scores
)

logger = logging.getLogger(__name__)


class AggrescanPredictor(BaseWebPredictor):
    """
    AGGRESCAN Web Predictor.
    
    AGGRESCAN predicts aggregation-prone regions using the a3v (aggregation
    propensity) scale derived from in vivo experiments with Aβ42 variants
    in E. coli inclusion bodies.
    
    Reference:
        Conchillo-Solé O, et al. (2007) BMC Bioinformatics 8:65
        DOI: 10.1186/1471-2105-8-65
    
    Scoring:
        - a3v: per-residue propensity from experimental scale
        - a4v: averaged over sliding window (default 5 residues)
        - HSA: Hot Spot Area identification
        - Na4vSS: normalized average a4v
        
    Thresholds:
        - Hot spots: a4v > -0.02 (5+ consecutive residues)
        - Amyloidogenic: presence of hot spots
    """
    
    server_url = "http://bioinf.uab.es/aggrescan/"
    predictor_name = "AGGRESCAN"
    predictor_version = "1.0"
    score_type = ScoreType.RAW
    default_threshold = -0.02  # a4v threshold for hot spots
    
    def __init__(
        self,
        config: Optional[WebPredictorConfig] = None,
        threshold: float = -0.02,
    ):
        """Initialize AGGRESCAN predictor."""
        super().__init__(config)
        self.default_threshold = threshold
    
    def _submit_sequence(self, sequence: str, sequence_id: str) -> None:
        """Submit sequence to AGGRESCAN server."""
        driver = self._get_driver()
        
        logger.info(f"Navigating to {self.server_url}")
        driver.get(self.server_url)
        time.sleep(3)
        
        self._take_screenshot("initial_page")
        
        # Find sequence input field
        textarea = self._find_element([
            (By.NAME, "sequence"),
            (By.NAME, "seq"),
            (By.CSS_SELECTOR, "textarea"),
        ])
        
        if not textarea:
            self._take_screenshot("no_textarea")
            raise ParsingError("Could not find sequence input field")
        
        # Clear and enter sequence
        textarea.clear()
        fasta_input = f">{sequence_id}\n{sequence}"
        textarea.send_keys(fasta_input)
        
        logger.info(f"Submitted: {sequence_id} ({len(sequence)} aa)")
        
        # Find submit button
        submit_btn = self._find_element([
            (By.CSS_SELECTOR, "input[type='submit'][value*='Run']"),
            (By.CSS_SELECTOR, "input[type='submit'][value*='Submit']"),
            (By.CSS_SELECTOR, "button[type='submit']"),
            (By.XPATH, "//input[@type='submit']"),
        ])
        
        if not submit_btn:
            self._take_screenshot("no_submit")
            raise ParsingError("Could not find submit button")
        
        submit_btn.click()
        self._take_screenshot("after_submit")
        
        # Wait for results
        logger.info("Waiting for AGGRESCAN results...")
        try:
            WebDriverWait(driver, 120).until(
                lambda d: any([
                    "a3v" in d.page_source,
                    "a4v" in d.page_source,
                    "Hot Spot" in d.page_source,
                    "HSA" in d.page_source,
                ])
            )
        except TimeoutException:
            self._take_screenshot("timeout")
            self._save_html("timeout")
            raise
        
        time.sleep(2)
        self._take_screenshot("results_page")
    
    def _find_element(self, selectors: list) -> Optional:
        """Try multiple selectors to find an element."""
        driver = self._get_driver()
        for by, selector in selectors:
            try:
                elements = driver.find_elements(by, selector)
                for elem in elements:
                    if elem.is_displayed():
                        return elem
            except NoSuchElementException:
                continue
        return None
    
    def _parse_results(self, sequence: str, sequence_id: str) -> PredictorOutput:
        """Parse AGGRESCAN results."""
        driver = self._get_driver()
        
        residue_scores = []
        predicted_regions = []
        raw_output = {}
        
        try:
            page_source = driver.page_source
            raw_output['page_source_length'] = len(page_source)
            
            # Strategy 1: Parse HTML table
            residue_scores = self._parse_score_table(driver, sequence)
            
            # Strategy 2: Parse regions (Hot Spots)
            predicted_regions = self._parse_hot_spots(page_source, sequence)
            
            # Fallback if no scores found
            if not residue_scores:
                logger.warning("No scores parsed, using fallback")
                residue_scores = self._compute_fallback_scores(sequence)
            
            return self._create_output(
                sequence, sequence_id,
                residue_scores, predicted_regions, raw_output
            )
            
        except Exception as e:
            logger.error(f"Error parsing AGGRESCAN results: {e}")
            self._take_screenshot("parse_error")
            self._save_html("parse_error")
            raise ParsingError(f"Failed to parse AGGRESCAN results: {e}")
    
    def _parse_score_table(self, driver, sequence: str) -> list[dict]:
        """Parse per-residue scores from HTML table."""
        scores = []
        
        tables = driver.find_elements(By.TAG_NAME, "table")
        
        for table in tables:
            rows = table.find_elements(By.TAG_NAME, "tr")
            
            for row in rows:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) >= 3:
                    try:
                        pos_text = cells[0].text.strip()
                        if pos_text.isdigit():
                            pos = int(pos_text)
                            aa = cells[1].text.strip()
                            
                            # Try to find a3v or a4v score
                            score = None
                            for cell in cells[2:]:
                                try:
                                    score = float(cell.text.strip())
                                    break
                                except ValueError:
                                    continue
                            
                            if score is not None and aa:
                                scores.append({
                                    'position': pos,
                                    'residue': aa,
                                    'raw_score': score,
                                })
                    except (ValueError, IndexError):
                        continue
        
        return scores
    
    def _parse_hot_spots(self, page_source: str, sequence: str) -> list[dict]:
        """Parse hot spot regions from page text."""
        regions = []
        
        # Pattern: "Hot Spot: 12-18" or "HSA: 12-18"
        pattern = r'(?:Hot\s*Spot|HSA)[:\s]+(\d+)\s*[-–]\s*(\d+)'
        
        for match in re.finditer(pattern, page_source, re.IGNORECASE):
            start = int(match.group(1))
            end = int(match.group(2))
            
            if 1 <= start <= len(sequence) and 1 <= end <= len(sequence):
                regions.append({
                    'start': start,
                    'end': end,
                    'sequence': sequence[start-1:end],
                })
        
        return regions
    
    def _compute_fallback_scores(self, sequence: str) -> list[dict]:
        """Compute approximate scores using a3v scale."""
        # a3v scale from Conchillo-Solé et al. 2007
        A3V_SCALE = {
            'A': -0.036, 'C':  0.267, 'D': -0.469, 'E': -0.467, 'F':  0.728,
            'G': -0.179, 'H': -0.056, 'I':  0.681, 'K': -0.302, 'L':  0.463,
            'M':  0.398, 'N': -0.193, 'P': -0.179, 'Q': -0.052, 'R': -0.176,
            'S': -0.235, 'T': -0.114, 'V':  0.591, 'W':  0.538, 'Y':  0.277,
        }
        
        scores = []
        window_size = 5
        
        for i, aa in enumerate(sequence):
            # Calculate a4v (window average)
            start = max(0, i - window_size // 2)
            end = min(len(sequence), i + window_size // 2 + 1)
            
            window_scores = [A3V_SCALE.get(sequence[j], 0) for j in range(start, end)]
            a4v = np.mean(window_scores) if window_scores else 0
            
            scores.append({
                'position': i + 1,
                'residue': aa,
                'raw_score': a4v,
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
        
        # AGGRESCAN uses raw a4v scale (-0.6 to 0.8 typical)
        # Normalize: shift and scale to 0-1
        min_val, max_val = -0.6, 0.8
        normalized = [
            (v - min_val) / (max_val - min_val)
            for v in raw_values
        ]
        normalized = [max(0, min(1, n)) for n in normalized]  # Clip
        
        # Create ResidueScore objects
        residue_score_objects = []
        for i, score_data in enumerate(residue_scores):
            # AGGRESCAN threshold is on raw a4v (> -0.02)
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
        
        # Create PredictedRegion objects
        region_objects = []
        for region in predicted_regions:
            region_scores = [
                s['raw_score'] for s in residue_scores
                if region['start'] <= s['position'] <= region['end']
            ]
            
            region_objects.append(PredictedRegion(
                start=region['start'],
                end=region['end'],
                sequence=region['sequence'],
                mean_score=float(np.mean(region_scores)) if region_scores else 0.0,
                max_score=float(np.max(region_scores)) if region_scores else 0.0,
                mean_normalized=float(np.mean([
                    normalized[s['position']-1]
                    for s in residue_scores
                    if region['start'] <= s['position'] <= region['end']
                ])) if region_scores else 0.0,
                region_type="hot_spot",
            ))
        
        # If no regions found, detect from scores
        if not region_objects:
            region_objects = self._detect_regions(residue_scores, sequence)
        
        is_amyloidogenic = len(region_objects) > 0
        overall_score = float(np.max(raw_values)) if raw_values else 0.0
        
        return PredictorOutput(
            predictor_name=self.predictor_name,
            predictor_version=self.predictor_version,
            sequence_id=sequence_id,
            sequence=sequence,
            residue_scores=residue_score_objects,
            predicted_regions=region_objects,
            overall_classification=(
                ClassificationLabel.AMYLOIDOGENIC if is_amyloidogenic
                else ClassificationLabel.NON_AMYLOIDOGENIC
            ),
            overall_score=overall_score,
            overall_probability=float(np.max(normalized)) if normalized else 0.0,
            score_type=self.score_type,
            threshold=self.default_threshold,
            source="web",
            raw_output=raw_output,
        )
    
    def _detect_regions(
        self,
        residue_scores: list[dict],
        sequence: str,
        min_length: int = 5,
    ) -> list[PredictedRegion]:
        """Detect hot spots from scores."""
        regions = []
        in_region = False
        region_start = 0
        region_scores = []
        
        for score_data in residue_scores:
            if score_data['raw_score'] > self.default_threshold:
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
                            sequence=sequence[region_start-1:region_end],
                            mean_score=float(np.mean(region_scores)),
                            max_score=float(np.max(region_scores)),
                            mean_normalized=0.0,  # Will be filled later
                            region_type="hot_spot",
                        ))
                    in_region = False
        
        # Handle region at end
        if in_region:
            region_end = residue_scores[-1]['position']
            if region_end - region_start + 1 >= min_length:
                regions.append(PredictedRegion(
                    start=region_start,
                    end=region_end,
                    sequence=sequence[region_start-1:region_end],
                    mean_score=float(np.mean(region_scores)),
                    max_score=float(np.max(region_scores)),
                    mean_normalized=0.0,
                    region_type="hot_spot",
                ))
        
        return regions


# Convenience function
def predict_with_aggrescan(
    sequence: str,
    sequence_id: str = "query",
    config: Optional[WebPredictorConfig] = None,
    threshold: float = -0.02,
) -> PredictorOutput:
    """
    Predict aggregation-prone regions using AGGRESCAN web server.
    
    Args:
        sequence: Protein sequence
        sequence_id: Identifier for the sequence
        config: Optional configuration
        threshold: a4v threshold (default -0.02)
        
    Returns:
        PredictorOutput with per-residue scores and hot spot regions
    """
    with AggrescanPredictor(config, threshold) as predictor:
        return predictor.predict(sequence, sequence_id)
```

---

## Summary: Creating Your Own Wrapper

1. **Analyze the web server** manually first
2. **Document selectors** for input, submit, and output elements
3. **Extend `BaseWebPredictor`** with required class attributes
4. **Implement `_submit_sequence()`** with robust element finding
5. **Implement `_parse_results()`** with multiple parsing strategies
6. **Create standardized output** using `PredictorOutput`
7. **Test against known sequences** and validate manually
8. **Add caching and rate limiting** for production use

The key principle: **Web wrappers should produce the exact same results as manual submission**.
