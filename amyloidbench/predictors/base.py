"""
Abstract base classes for amyloidogenicity predictors.

This module defines the interface that all predictors must implement,
enabling a unified API for consensus prediction and benchmarking.
The design follows the Strategy pattern, allowing predictors to be
interchangeable while maintaining consistent behavior.

Key design principles:
1. All predictors expose the same interface for predictions
2. Both per-residue scores and region-level predictions are supported
3. Caching is built-in to avoid redundant computations
4. Asynchronous execution is supported for web-based tools
5. Error handling is standardized with graceful degradation
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, TypeVar

import numpy as np
from diskcache import Cache

from ..core.models import (
    AmyloidPolymorph,
    PerResidueScores,
    PredictionConfidence,
    PredictionResult,
    ProteinRecord,
    Region,
)
from ..core.sequence import sequence_hash

logger = logging.getLogger(__name__)

# Type variable for predictor subclasses
P = TypeVar("P", bound="BasePredictor")


class PredictorType(str, Enum):
    """Classification of predictor types by methodology."""
    
    # Sequence-based methods
    SEQUENCE_HEURISTIC = "sequence_heuristic"  # AGGRESCAN, FoldAmyloid
    SEQUENCE_ML = "sequence_ml"  # APPNN, FISH Amyloid
    
    # Structure-based methods
    STRUCTURE_BASED = "structure_based"  # Aggrescan3D, Zyggregator3D
    
    # Threading/template-based
    THREADING = "threading"  # PATH, PASTA 2.0
    
    # Consensus/meta-predictors
    CONSENSUS = "consensus"  # AmylPred2, MetAmyl
    
    # Our fallback predictor
    FALLBACK = "fallback"


class PredictorCapability(Enum):
    """Capabilities that predictors may support."""
    
    PER_RESIDUE_SCORES = auto()  # Outputs scores for each residue
    REGION_DETECTION = auto()  # Identifies discrete APRs
    BINARY_CLASSIFICATION = auto()  # Amyloid vs non-amyloid
    POLYMORPH_PREDICTION = auto()  # Structural type prediction
    BATCH_PROCESSING = auto()  # Can handle multiple sequences
    ASYNC_EXECUTION = auto()  # Supports async/await
    STRUCTURE_INPUT = auto()  # Can use 3D structure
    TRAINABLE = auto()  # Can be retrained on new data


@dataclass
class PredictorConfig:
    """
    Configuration for predictor behavior.
    
    Allows customization of thresholds, caching, and runtime parameters
    without modifying predictor code.
    """
    # Prediction parameters
    threshold: Optional[float] = None  # Override default threshold
    window_size: Optional[int] = None  # Override default window size
    min_region_length: int = 5  # Minimum APR length to report
    merge_gap: int = 2  # Merge regions separated by ≤ this many residues
    
    # Caching
    use_cache: bool = True
    cache_dir: Optional[Path] = None
    cache_ttl: int = 86400 * 30  # 30 days default
    
    # Runtime
    timeout_seconds: float = 300.0  # 5 minutes default for web tools
    max_retries: int = 3
    retry_delay: float = 2.0
    
    # Output
    return_raw_output: bool = False  # Include raw predictor output
    normalize_scores: bool = True  # Normalize to [0, 1] range
    
    def __post_init__(self):
        if self.cache_dir is None:
            self.cache_dir = Path.home() / ".cache" / "amyloidbench"
        self.cache_dir.mkdir(parents=True, exist_ok=True)


class PredictorError(Exception):
    """Base exception for predictor errors."""
    pass


class PredictorTimeoutError(PredictorError):
    """Raised when prediction exceeds timeout."""
    pass


class PredictorUnavailableError(PredictorError):
    """Raised when a predictor service is unavailable."""
    pass


class BasePredictor(ABC):
    """
    Abstract base class for all amyloidogenicity predictors.
    
    This class defines the interface that all predictors must implement,
    ensuring consistent behavior across different prediction methods.
    Subclasses implement the actual prediction logic while this base
    class handles caching, error handling, and standardization.
    
    Implementation guide for new predictors:
    1. Inherit from BasePredictor
    2. Set class attributes (name, version, type, capabilities)
    3. Implement _predict_impl() with the actual prediction logic
    4. Optionally override other methods for customization
    
    Example:
        class MyPredictor(BasePredictor):
            name = "MyPredictor"
            version = "1.0"
            predictor_type = PredictorType.SEQUENCE_ML
            capabilities = {PredictorCapability.PER_RESIDUE_SCORES}
            default_threshold = 0.5
            
            def _predict_impl(self, sequence: str) -> PredictionResult:
                # Your implementation here
                scores = self._calculate_scores(sequence)
                return PredictionResult(...)
    """
    
    # Class attributes - must be set by subclasses
    name: str = "BasePredictor"
    version: str = "0.0"
    predictor_type: PredictorType = PredictorType.SEQUENCE_HEURISTIC
    capabilities: set[PredictorCapability] = set()
    
    # Default prediction parameters
    default_threshold: float = 0.5
    default_window_size: int = 6
    
    # Score range (for normalization)
    score_min: Optional[float] = None
    score_max: Optional[float] = None
    
    # Documentation
    citation: Optional[str] = None
    url: Optional[str] = None
    description: str = ""
    
    def __init__(self, config: Optional[PredictorConfig] = None):
        """
        Initialize predictor with configuration.
        
        Args:
            config: Predictor configuration (uses defaults if None)
        """
        self.config = config or PredictorConfig()
        self._cache: Optional[Cache] = None
        
        if self.config.use_cache:
            cache_path = self.config.cache_dir / self.name.lower().replace(" ", "_")
            self._cache = Cache(str(cache_path))
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, version={self.version!r})"
    
    @property
    def threshold(self) -> float:
        """Get effective threshold (config override or default)."""
        return self.config.threshold or self.default_threshold
    
    @property
    def window_size(self) -> int:
        """Get effective window size."""
        return self.config.window_size or self.default_window_size
    
    def _get_cache_key(self, sequence: str) -> str:
        """Generate cache key for a sequence."""
        seq_hash = sequence_hash(sequence)
        config_hash = hashlib.md5(
            f"{self.threshold}:{self.window_size}".encode()
        ).hexdigest()[:8]
        return f"{self.name}:{self.version}:{seq_hash}:{config_hash}"
    
    def _check_cache(self, sequence: str) -> Optional[PredictionResult]:
        """Check if result is cached."""
        if not self._cache:
            return None
        
        key = self._get_cache_key(sequence)
        return self._cache.get(key)
    
    def _store_cache(self, sequence: str, result: PredictionResult):
        """Store result in cache."""
        if not self._cache:
            return
        
        key = self._get_cache_key(sequence)
        self._cache.set(key, result, expire=self.config.cache_ttl)
    
    @abstractmethod
    def _predict_impl(
        self,
        sequence: str,
        structure_path: Optional[Path] = None,
    ) -> PredictionResult:
        """
        Internal prediction implementation.
        
        This method must be implemented by all subclasses to perform
        the actual prediction. The base class handles caching,
        validation, and standardization.
        
        Args:
            sequence: Protein sequence (validated, uppercase)
            structure_path: Optional path to structure file
        
        Returns:
            PredictionResult with at minimum per_residue_scores or
            predicted_regions populated
        
        Raises:
            PredictorError: On prediction failure
        """
        pass
    
    def predict(
        self,
        protein: ProteinRecord,
        use_structure: bool = True,
    ) -> PredictionResult:
        """
        Run prediction on a protein.
        
        This is the main public interface for predictions. It handles:
        - Input validation
        - Cache lookup
        - Error handling with retries
        - Score normalization
        - Result standardization
        
        Args:
            protein: ProteinRecord with sequence and optional structure
            use_structure: Whether to use structure if available
        
        Returns:
            PredictionResult with prediction data
        """
        sequence = protein.sequence.upper()
        
        # Check cache
        cached = self._check_cache(sequence)
        if cached is not None:
            logger.debug(f"{self.name}: Using cached result for {protein.id}")
            return cached
        
        # Determine if structure should be used
        structure_path = None
        if (
            use_structure
            and protein.structure_path
            and PredictorCapability.STRUCTURE_INPUT in self.capabilities
        ):
            structure_path = protein.structure_path
        
        # Run prediction with retry logic
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                result = self._predict_impl(sequence, structure_path)
                break
            except PredictorError as e:
                last_error = e
                logger.warning(
                    f"{self.name}: Attempt {attempt + 1} failed: {e}"
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
        else:
            # All retries exhausted
            result = PredictionResult(
                sequence_id=protein.id,
                sequence=sequence,
                predictor_name=self.name,
                predictor_version=self.version,
                error_message=str(last_error),
            )
            return result
        
        # Enrich result
        result.sequence_id = protein.id
        result.sequence = sequence
        result.predictor_name = self.name
        result.predictor_version = self.version
        result.runtime_seconds = time.time() - start_time
        
        # Normalize scores if requested
        if (
            self.config.normalize_scores
            and result.per_residue_scores
            and result.per_residue_scores.score_type == "raw"
        ):
            result.per_residue_scores = result.per_residue_scores.normalize()
        
        # Ensure regions are extracted
        if (
            not result.predicted_regions
            and result.per_residue_scores
        ):
            result.predicted_regions = result.per_residue_scores.to_regions(
                threshold=self.threshold,
                min_length=self.config.min_region_length,
                merge_gap=self.config.merge_gap,
            )
        
        # Determine binary classification if not set
        if result.is_amyloidogenic is None and result.predicted_regions:
            result.is_amyloidogenic = len(result.predicted_regions) > 0
        
        # Store in cache
        self._store_cache(sequence, result)
        
        return result
    
    def predict_batch(
        self,
        proteins: Sequence[ProteinRecord],
        use_structure: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[PredictionResult]:
        """
        Run predictions on multiple proteins.
        
        Args:
            proteins: Sequence of ProteinRecord objects
            use_structure: Whether to use structures if available
            progress_callback: Optional callback(current, total) for progress
        
        Returns:
            List of PredictionResult objects
        """
        results = []
        total = len(proteins)
        
        for i, protein in enumerate(proteins):
            result = self.predict(protein, use_structure)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return results
    
    def can_handle(self, protein: ProteinRecord) -> bool:
        """
        Check if this predictor can handle the given protein.
        
        Override this method to add predictor-specific constraints
        (e.g., maximum sequence length for web tools).
        
        Args:
            protein: Protein to check
        
        Returns:
            True if predictor can handle this protein
        """
        return True
    
    def get_info(self) -> dict[str, Any]:
        """
        Get predictor information for documentation/logging.
        
        Returns:
            Dictionary with predictor metadata
        """
        return {
            "name": self.name,
            "version": self.version,
            "type": self.predictor_type.value,
            "capabilities": [c.name for c in self.capabilities],
            "threshold": self.threshold,
            "window_size": self.window_size,
            "score_range": (self.score_min, self.score_max),
            "citation": self.citation,
            "url": self.url,
            "description": self.description,
        }
    
    def clear_cache(self):
        """Clear the prediction cache for this predictor."""
        if self._cache:
            self._cache.clear()


class AsyncBasePredictor(BasePredictor):
    """
    Base class for asynchronous predictors.
    
    Use this for web-based predictors that benefit from async I/O.
    Implements both sync and async interfaces.
    """
    
    capabilities: set[PredictorCapability] = {PredictorCapability.ASYNC_EXECUTION}
    
    @abstractmethod
    async def _predict_impl_async(
        self,
        sequence: str,
        structure_path: Optional[Path] = None,
    ) -> PredictionResult:
        """
        Async implementation of prediction.
        
        Args:
            sequence: Protein sequence
            structure_path: Optional structure file path
        
        Returns:
            PredictionResult
        """
        pass
    
    def _predict_impl(
        self,
        sequence: str,
        structure_path: Optional[Path] = None,
    ) -> PredictionResult:
        """Sync wrapper around async implementation."""
        return asyncio.run(self._predict_impl_async(sequence, structure_path))
    
    async def predict_async(
        self,
        protein: ProteinRecord,
        use_structure: bool = True,
    ) -> PredictionResult:
        """
        Async prediction interface.
        
        Args:
            protein: ProteinRecord to predict
            use_structure: Whether to use structure if available
        
        Returns:
            PredictionResult
        """
        sequence = protein.sequence.upper()
        
        # Check cache
        cached = self._check_cache(sequence)
        if cached is not None:
            return cached
        
        structure_path = None
        if (
            use_structure
            and protein.structure_path
            and PredictorCapability.STRUCTURE_INPUT in self.capabilities
        ):
            structure_path = protein.structure_path
        
        start_time = time.time()
        result = await self._predict_impl_async(sequence, structure_path)
        
        # Enrich result (same as sync version)
        result.sequence_id = protein.id
        result.sequence = sequence
        result.predictor_name = self.name
        result.predictor_version = self.version
        result.runtime_seconds = time.time() - start_time
        
        if (
            self.config.normalize_scores
            and result.per_residue_scores
            and result.per_residue_scores.score_type == "raw"
        ):
            result.per_residue_scores = result.per_residue_scores.normalize()
        
        if not result.predicted_regions and result.per_residue_scores:
            result.predicted_regions = result.per_residue_scores.to_regions(
                threshold=self.threshold,
                min_length=self.config.min_region_length,
                merge_gap=self.config.merge_gap,
            )
        
        if result.is_amyloidogenic is None and result.predicted_regions:
            result.is_amyloidogenic = len(result.predicted_regions) > 0
        
        self._store_cache(sequence, result)
        
        return result
    
    async def predict_batch_async(
        self,
        proteins: Sequence[ProteinRecord],
        use_structure: bool = True,
        max_concurrent: int = 5,
    ) -> list[PredictionResult]:
        """
        Run async predictions with concurrency control.
        
        Args:
            proteins: Proteins to predict
            use_structure: Whether to use structures
            max_concurrent: Maximum concurrent predictions
        
        Returns:
            List of PredictionResult objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def predict_with_semaphore(protein: ProteinRecord) -> PredictionResult:
            async with semaphore:
                return await self.predict_async(protein, use_structure)
        
        tasks = [predict_with_semaphore(p) for p in proteins]
        return await asyncio.gather(*tasks)


class TrainablePredictor(BasePredictor):
    """
    Base class for predictors that can be trained/fine-tuned.
    
    Use this for ML-based predictors like the fallback predictor
    that can be trained on new data.
    """
    
    capabilities: set[PredictorCapability] = {PredictorCapability.TRAINABLE}
    
    @abstractmethod
    def fit(
        self,
        proteins: Sequence[ProteinRecord],
        labels: Optional[Sequence[bool]] = None,
    ) -> "TrainablePredictor":
        """
        Train the predictor on labeled data.
        
        Args:
            proteins: Training proteins (with known_amyloid_regions set)
            labels: Optional binary labels (uses protein annotations if None)
        
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def save(self, path: Path):
        """Save trained model to disk."""
        pass
    
    @abstractmethod
    def load(self, path: Path) -> "TrainablePredictor":
        """Load trained model from disk."""
        pass


# Registry for available predictors
_PREDICTOR_REGISTRY: dict[str, type[BasePredictor]] = {}


def register_predictor(predictor_class: type[BasePredictor]) -> type[BasePredictor]:
    """
    Decorator to register a predictor class.
    
    Usage:
        @register_predictor
        class MyPredictor(BasePredictor):
            name = "MyPredictor"
            ...
    """
    _PREDICTOR_REGISTRY[predictor_class.name] = predictor_class
    return predictor_class


def get_predictor(name: str, config: Optional[PredictorConfig] = None) -> BasePredictor:
    """
    Get a predictor instance by name.
    
    Args:
        name: Predictor name
        config: Optional configuration
    
    Returns:
        Predictor instance
    
    Raises:
        KeyError: If predictor not found
    """
    if name not in _PREDICTOR_REGISTRY:
        available = ", ".join(_PREDICTOR_REGISTRY.keys())
        raise KeyError(f"Predictor '{name}' not found. Available: {available}")
    
    return _PREDICTOR_REGISTRY[name](config)


def list_predictors() -> list[dict[str, Any]]:
    """
    List all registered predictors with their info.
    
    Returns:
        List of predictor info dictionaries
    """
    return [
        cls(PredictorConfig(use_cache=False)).get_info()
        for cls in _PREDICTOR_REGISTRY.values()
    ]
