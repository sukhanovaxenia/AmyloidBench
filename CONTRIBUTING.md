# Contributing to AmyloidBench

Thank you for your interest in contributing to AmyloidBench! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/sukhanovaxenia/AmyloidBench.git
cd AmyloidBench

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Install Playwright browsers (for web-based predictors)
playwright install chromium
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=amyloidbench --cov-report=html

# Run specific test file
pytest tests/test_predictors.py -v
```

## Adding a New Predictor

Predictors are the core components of AmyloidBench. Here's how to add a new one:

### 1. Create the Predictor Class

```python
# amyloidbench/predictors/local/my_predictor.py

from amyloidbench.core.models import (
    PerResidueScores,
    PredictionResult,
)
from amyloidbench.predictors.base import (
    BasePredictor,
    PredictorCapability,
    PredictorType,
    register_predictor,
)

@register_predictor
class MyPredictor(BasePredictor):
    """
    Docstring explaining the biological basis and methodology.
    
    Include reference to original publication.
    """
    
    name = "MyPredictor"
    version = "1.0"
    predictor_type = PredictorType.SEQUENCE_HEURISTIC
    capabilities = {
        PredictorCapability.PER_RESIDUE_SCORES,
        PredictorCapability.REGION_DETECTION,
    }
    
    default_threshold = 0.5
    score_min = 0.0
    score_max = 1.0
    
    citation = "Author et al. (Year) Title. Journal DOI"
    url = "https://predictor-website.com"
    description = "Brief description of methodology"
    
    def _predict_impl(self, sequence, structure_path=None):
        # Your implementation here
        scores = self._calculate_scores(sequence)
        
        return PredictionResult(
            sequence_id="",
            sequence=sequence,
            predictor_name=self.name,
            per_residue_scores=PerResidueScores(
                scores=scores,
                sequence=sequence,
                predictor=self.name,
                threshold=self.threshold,
            ),
        )
```

### 2. Register the Predictor

Add the import to `amyloidbench/predictors/local/__init__.py`:

```python
from .my_predictor import MyPredictor
```

### 3. Add Tests

Create `tests/test_my_predictor.py` with:
- Basic functionality tests
- Biological validation tests (known amyloidogenic sequences)
- Edge case handling

### 4. Update Documentation

- Add predictor to the table in `README.md`
- Include citation information

## Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all public functions/classes (Google style)
- Keep lines under 100 characters
- Use `black` for formatting, `ruff` for linting

```bash
# Format code
black amyloidbench tests

# Lint
ruff check amyloidbench tests
```

## Commit Messages

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Adding tests
- `refactor`: Code refactoring
- `chore`: Maintenance tasks

Examples:
```
feat(predictors): add FoldAmyloid re-implementation
fix(core): correct region overlap detection
docs(readme): update predictor table
test(a3d): add biological validation tests
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with appropriate tests
3. Ensure all tests pass: `pytest`
4. Update documentation if needed
5. Submit PR with clear description

## Biological Validation

When adding predictors or modifying scoring, include validation against:

1. **Positive controls**: Known amyloidogenic sequences
   - Aβ42: `DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA`
   - α-Synuclein NAC: `EQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFV`
   - PrP(106-126): `KTNMKHMAGAAAAGAVVGGLG`

2. **Negative controls**: Soluble, non-aggregating proteins
   - Ubiquitin: `MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG`

3. **Expected patterns**:
   - KLVFF region in Aβ42 should score highly
   - Gatekeepers (P, K, R, E, D) should reduce local scores
   - Overall score ordering: Aβ42 > α-Syn NAC > Ubiquitin

## Questions?

Open an issue on GitHub or contact: xenia@arriam.ru
