"""
Consensus engine for combining multiple amyloidogenicity predictors.

The consensus approach addresses a fundamental challenge in amyloid prediction:
no single predictor captures all aspects of amyloid formation. Different methods
emphasize different biophysical properties and use different training data,
leading to complementary strengths and weaknesses.

Biological Rationale
--------------------
Amyloid formation is a complex process involving:
1. Primary sequence determinants (captured by sequence-based methods)
2. Structural compatibility (captured by threading and structure-based methods)
3. Thermodynamic stability (captured by energy-based calculations)
4. Kinetic accessibility (influenced by disorder and context)

By combining predictions from methods that emphasize different aspects,
we can achieve:
- Higher sensitivity: APRs missed by one method may be caught by another
- Higher specificity: False positives are less likely to be confirmed by
  multiple methods using different principles
- More robust predictions: Less sensitive to method-specific biases

Consensus Methods
-----------------
1. **Majority Voting**: Simple democracy—a region is predicted if a
   majority of predictors agree. Robust but ignores predictor confidence.

2. **Weighted Average**: Combines per-residue scores using method-specific
   weights derived from validation studies. More nuanced than voting.

3. **Intersection**: Conservative approach—only reports regions identified
   by all (or N-of-M) predictors. Maximizes specificity.

4. **Union**: Liberal approach—reports regions identified by any predictor.
   Maximizes sensitivity at the cost of specificity.

5. **Stacked Ensemble**: ML model trained on predictor outputs to learn
   optimal combination. Highest potential accuracy with sufficient training.

Implementation
--------------
The ConsensusEngine class provides:
- Flexible predictor registration and management
- Multiple consensus methods with configurable parameters
- Per-residue and region-level aggregation
- Confidence scoring based on predictor agreement
- Interpretable output showing which predictors contributed to each call

References
----------
- AmylPred2 (Tsolis et al., 2013) - 11-predictor consensus
- MetAmyl (Emily et al., 2013) - Meta-prediction approach
- Gasior & Bhanu (2017) - Ensemble methods for aggregation prediction
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np

from amyloidbench.core.models import (
    ConsensusResult,
    PerResidueScores,
    PredictionResult,
    ProteinRecord,
    Region,
)
from amyloidbench.predictors.base import (
    BasePredictor,
    PredictorConfig,
    get_predictor,
    list_predictors,
)

logger = logging.getLogger(__name__)


class ConsensusMethod(str, Enum):
    """Available consensus aggregation methods."""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    INTERSECTION = "intersection"
    UNION = "union"
    STACKED = "stacked"
    METAMYL = "metamyl"  # Logistic regression on predictor outputs


@dataclass
class PredictorWeight:
    """
    Weight and configuration for a predictor in the ensemble.
    
    Attributes:
        predictor_name: Name of the predictor
        weight: Relative importance (0-1, will be normalized)
        enabled: Whether to include in consensus
        threshold_override: Custom threshold for this predictor
    """
    predictor_name: str
    weight: float = 1.0
    enabled: bool = True
    threshold_override: Optional[float] = None


@dataclass 
class ConsensusConfig:
    """
    Configuration for consensus prediction.
    
    Attributes:
        method: Consensus aggregation method
        min_agreement: Minimum fraction of predictors that must agree
        predictor_weights: Optional custom weights for predictors
        region_threshold: Threshold for region detection in consensus scores
        min_region_length: Minimum APR length
        merge_gap: Maximum gap to merge nearby regions
        require_n_predictors: Minimum number of predictors for valid consensus
    """
    method: ConsensusMethod = ConsensusMethod.WEIGHTED_AVERAGE
    min_agreement: float = 0.5
    predictor_weights: dict[str, float] = field(default_factory=dict)
    region_threshold: float = 0.5
    min_region_length: int = 5
    merge_gap: int = 2
    require_n_predictors: int = 2


class ConsensusEngine:
    """
    Engine for combining multiple amyloidogenicity predictors.
    
    The ConsensusEngine manages a collection of predictors and combines
    their outputs using various aggregation strategies. It provides:
    
    1. **Predictor management**: Add, remove, configure predictors
    2. **Batch prediction**: Run all predictors on input sequences
    3. **Score aggregation**: Combine per-residue scores
    4. **Region consensus**: Identify regions supported by multiple predictors
    5. **Confidence estimation**: Quantify prediction reliability
    
    Usage:
        >>> engine = ConsensusEngine()
        >>> engine.add_predictor("Aggrescan3D")
        >>> engine.add_predictor("FoldAmyloid")
        >>> engine.add_predictor("FallbackPredictor")
        >>> 
        >>> result = engine.predict(protein)
        >>> print(f"Consensus: {result.consensus_is_amyloidogenic}")
        >>> print(f"Agreement: {result.agreement_ratio():.1%}")
    
    Attributes:
        predictors: Dictionary of active predictors
        config: Consensus configuration
        results_cache: Cache of individual predictor results
    """
    
    def __init__(
        self,
        config: Optional[ConsensusConfig] = None,
        auto_register: bool = False,
    ):
        """
        Initialize consensus engine.
        
        Args:
            config: Consensus configuration
            auto_register: If True, automatically add all available predictors
        """
        self.config = config or ConsensusConfig()
        self.predictors: dict[str, BasePredictor] = {}
        self.predictor_weights: dict[str, PredictorWeight] = {}
        self._stacked_model = None
        
        if auto_register:
            self._auto_register_predictors()
    
    def _auto_register_predictors(self):
        """Register all available local predictors."""
        for pred_info in list_predictors():
            try:
                self.add_predictor(pred_info["name"])
            except Exception as e:
                logger.warning(f"Could not auto-register {pred_info['name']}: {e}")
    
    def add_predictor(
        self,
        name: str,
        config: Optional[PredictorConfig] = None,
        weight: float = 1.0,
        threshold_override: Optional[float] = None,
    ):
        """
        Add a predictor to the consensus.
        
        Args:
            name: Predictor name (must be registered)
            config: Optional predictor configuration
            weight: Relative weight for weighted consensus
            threshold_override: Custom threshold for this predictor
        """
        predictor = get_predictor(name, config)
        self.predictors[name] = predictor
        self.predictor_weights[name] = PredictorWeight(
            predictor_name=name,
            weight=weight,
            threshold_override=threshold_override,
        )
        logger.info(f"Added predictor: {name} (weight={weight})")
    
    def remove_predictor(self, name: str):
        """Remove a predictor from the consensus."""
        if name in self.predictors:
            del self.predictors[name]
            del self.predictor_weights[name]
            logger.info(f"Removed predictor: {name}")
    
    def set_weight(self, name: str, weight: float):
        """Set the weight for a predictor."""
        if name in self.predictor_weights:
            self.predictor_weights[name].weight = weight
    
    def enable_predictor(self, name: str, enabled: bool = True):
        """Enable or disable a predictor."""
        if name in self.predictor_weights:
            self.predictor_weights[name].enabled = enabled
    
    def get_active_predictors(self) -> list[str]:
        """Get list of currently enabled predictors."""
        return [
            name for name, pw in self.predictor_weights.items()
            if pw.enabled
        ]
    
    def predict(
        self,
        protein: ProteinRecord,
        method: Optional[ConsensusMethod] = None,
    ) -> ConsensusResult:
        """
        Generate consensus prediction for a protein.
        
        Runs all enabled predictors and combines their results using
        the specified (or configured) consensus method.
        
        Args:
            protein: Input protein
            method: Optional override for consensus method
            
        Returns:
            ConsensusResult with aggregated predictions
        """
        method = method or self.config.method
        active_predictors = self.get_active_predictors()
        
        if len(active_predictors) < self.config.require_n_predictors:
            raise ValueError(
                f"Need at least {self.config.require_n_predictors} predictors, "
                f"but only {len(active_predictors)} are active"
            )
        
        # Run individual predictors
        individual_results: dict[str, PredictionResult] = {}
        
        for name in active_predictors:
            predictor = self.predictors[name]
            try:
                result = predictor.predict(protein)
                individual_results[name] = result
                logger.debug(f"{name}: amyloidogenic={result.is_amyloidogenic}")
            except Exception as e:
                logger.warning(f"Predictor {name} failed: {e}")
        
        if not individual_results:
            raise RuntimeError("All predictors failed")
        
        # Combine results
        if method == ConsensusMethod.MAJORITY_VOTE:
            return self._majority_vote(protein, individual_results)
        elif method == ConsensusMethod.WEIGHTED_AVERAGE:
            return self._weighted_average(protein, individual_results)
        elif method == ConsensusMethod.INTERSECTION:
            return self._intersection(protein, individual_results)
        elif method == ConsensusMethod.UNION:
            return self._union(protein, individual_results)
        elif method == ConsensusMethod.STACKED:
            return self._stacked_ensemble(protein, individual_results)
        elif method == ConsensusMethod.METAMYL:
            return self._metamyl(protein, individual_results)
        else:
            raise ValueError(f"Unknown consensus method: {method}")
    
    def _majority_vote(
        self,
        protein: ProteinRecord,
        results: dict[str, PredictionResult],
    ) -> ConsensusResult:
        """
        Combine predictions using simple majority voting.
        
        For binary classification: majority of predictors must agree.
        For per-residue: score is fraction of predictors calling residue positive.
        """
        n = len(protein.sequence)
        n_predictors = len(results)
        
        # Per-residue voting
        votes = np.zeros(n)
        
        for name, result in results.items():
            if result.per_residue_scores:
                # Use predictor's threshold
                pw = self.predictor_weights[name]
                threshold = pw.threshold_override or result.per_residue_scores.threshold or 0.5
                
                for i, score in enumerate(result.per_residue_scores.scores):
                    if score >= threshold:
                        votes[i] += 1
        
        # Convert votes to fraction
        consensus_scores = votes / n_predictors
        
        # Binary classification vote
        n_positive = sum(1 for r in results.values() if r.is_amyloidogenic)
        is_amyloidogenic = n_positive >= n_predictors * self.config.min_agreement
        
        # Create per-residue scores
        per_residue = PerResidueScores(
            scores=consensus_scores.tolist(),
            sequence=protein.sequence,
            predictor="Consensus",
            score_type="vote_fraction",
            threshold=self.config.min_agreement,
        )
        
        # Extract consensus regions
        regions = per_residue.to_regions(
            threshold=self.config.min_agreement,
            min_length=self.config.min_region_length,
            merge_gap=self.config.merge_gap,
        )
        
        return ConsensusResult(
            sequence_id=protein.id,
            sequence=protein.sequence,
            individual_results=results,
            consensus_per_residue=per_residue,
            consensus_regions=regions,
            consensus_is_amyloidogenic=is_amyloidogenic,
            consensus_probability=float(np.mean(consensus_scores)),
            n_predictors_agree_positive=n_positive,
            n_predictors_agree_negative=n_predictors - n_positive,
            n_predictors_total=n_predictors,
            consensus_method=ConsensusMethod.MAJORITY_VOTE.value,
        )
    
    def _weighted_average(
        self,
        protein: ProteinRecord,
        results: dict[str, PredictionResult],
    ) -> ConsensusResult:
        """
        Combine predictions using weighted average of scores.
        
        Weights can be set based on predictor validation performance
        or domain knowledge about which methods are most relevant.
        """
        n = len(protein.sequence)
        
        # Normalize weights
        total_weight = sum(
            self.predictor_weights[name].weight
            for name in results.keys()
        )
        
        # Weighted per-residue scores
        weighted_scores = np.zeros(n)
        
        for name, result in results.items():
            weight = self.predictor_weights[name].weight / total_weight
            
            if result.per_residue_scores:
                scores = np.array(result.per_residue_scores.scores)
                
                # Normalize scores to [0, 1] if needed
                if result.per_residue_scores.min_score is not None:
                    min_s = result.per_residue_scores.min_score
                    max_s = result.per_residue_scores.max_score
                    if max_s > min_s:
                        scores = (scores - min_s) / (max_s - min_s)
                
                weighted_scores += weight * scores
        
        # Weighted probability
        total_weight = sum(self.predictor_weights[name].weight for name in results.keys())
        weighted_prob = sum(
            self.predictor_weights[name].weight * (results[name].amyloid_probability or 0)
            for name in results.keys()
        ) / total_weight
        
        # Binary classification
        is_amyloidogenic = weighted_prob >= self.config.region_threshold
        
        # Per-residue scores
        per_residue = PerResidueScores(
            scores=weighted_scores.tolist(),
            sequence=protein.sequence,
            predictor="Consensus",
            score_type="weighted_average",
            threshold=self.config.region_threshold,
        )
        
        # Extract regions
        regions = per_residue.to_regions(
            threshold=self.config.region_threshold,
            min_length=self.config.min_region_length,
            merge_gap=self.config.merge_gap,
        )
        
        # Count agreements
        n_positive = sum(1 for r in results.values() if r.is_amyloidogenic)
        
        return ConsensusResult(
            sequence_id=protein.id,
            sequence=protein.sequence,
            individual_results=results,
            consensus_per_residue=per_residue,
            consensus_regions=regions,
            consensus_is_amyloidogenic=is_amyloidogenic,
            consensus_probability=float(weighted_prob),
            n_predictors_agree_positive=n_positive,
            n_predictors_agree_negative=len(results) - n_positive,
            n_predictors_total=len(results),
            consensus_method=ConsensusMethod.WEIGHTED_AVERAGE.value,
            predictor_weights={
                name: self.predictor_weights[name].weight
                for name in results.keys()
            },
        )
    
    def _intersection(
        self,
        protein: ProteinRecord,
        results: dict[str, PredictionResult],
    ) -> ConsensusResult:
        """
        Report only regions identified by ALL predictors.
        
        This is the most conservative approach, maximizing specificity.
        """
        n = len(protein.sequence)
        n_predictors = len(results)
        
        # Count how many predictors call each residue positive
        residue_counts = np.zeros(n)
        
        for name, result in results.items():
            for region in result.predicted_regions:
                residue_counts[region.start:region.end] += 1
        
        # Intersection: all predictors must agree
        intersection_mask = residue_counts == n_predictors
        
        # Convert to scores (1 if all agree, 0 otherwise)
        scores = intersection_mask.astype(float)
        
        # Extract regions from intersection
        per_residue = PerResidueScores(
            scores=scores.tolist(),
            sequence=protein.sequence,
            predictor="Consensus",
            score_type="intersection",
            threshold=0.5,
        )
        
        regions = per_residue.to_regions(
            threshold=0.5,
            min_length=self.config.min_region_length,
            merge_gap=self.config.merge_gap,
        )
        
        # Binary: all must agree
        all_positive = all(r.is_amyloidogenic for r in results.values())
        
        return ConsensusResult(
            sequence_id=protein.id,
            sequence=protein.sequence,
            individual_results=results,
            consensus_per_residue=per_residue,
            consensus_regions=regions,
            consensus_is_amyloidogenic=all_positive and len(regions) > 0,
            consensus_probability=float(np.mean(residue_counts / n_predictors)),
            n_predictors_agree_positive=sum(1 for r in results.values() if r.is_amyloidogenic),
            n_predictors_agree_negative=sum(1 for r in results.values() if not r.is_amyloidogenic),
            n_predictors_total=n_predictors,
            consensus_method=ConsensusMethod.INTERSECTION.value,
        )
    
    def _union(
        self,
        protein: ProteinRecord,
        results: dict[str, PredictionResult],
    ) -> ConsensusResult:
        """
        Report regions identified by ANY predictor.
        
        This is the most liberal approach, maximizing sensitivity.
        """
        n = len(protein.sequence)
        n_predictors = len(results)
        
        # Union: any predictor positive
        union_mask = np.zeros(n, dtype=bool)
        residue_counts = np.zeros(n)
        
        for name, result in results.items():
            for region in result.predicted_regions:
                union_mask[region.start:region.end] = True
                residue_counts[region.start:region.end] += 1
        
        # Convert to scores (fraction of predictors, but 1 if any)
        scores = np.where(union_mask, residue_counts / n_predictors, 0.0)
        
        per_residue = PerResidueScores(
            scores=scores.tolist(),
            sequence=protein.sequence,
            predictor="Consensus",
            score_type="union",
            threshold=0.0,  # Any positive
        )
        
        regions = per_residue.to_regions(
            threshold=0.01,  # Very low threshold for union
            min_length=self.config.min_region_length,
            merge_gap=self.config.merge_gap,
        )
        
        # Binary: any positive
        any_positive = any(r.is_amyloidogenic for r in results.values())
        
        return ConsensusResult(
            sequence_id=protein.id,
            sequence=protein.sequence,
            individual_results=results,
            consensus_per_residue=per_residue,
            consensus_regions=regions,
            consensus_is_amyloidogenic=any_positive,
            consensus_probability=float(np.mean(scores[scores > 0]) if np.any(scores > 0) else 0),
            n_predictors_agree_positive=sum(1 for r in results.values() if r.is_amyloidogenic),
            n_predictors_agree_negative=sum(1 for r in results.values() if not r.is_amyloidogenic),
            n_predictors_total=n_predictors,
            consensus_method=ConsensusMethod.UNION.value,
        )
    
    def _stacked_ensemble(
        self,
        protein: ProteinRecord,
        results: dict[str, PredictionResult],
    ) -> ConsensusResult:
        """
        Use a trained ML model to combine predictor outputs.
        
        This requires prior training on labeled data using `train_stacked_model()`.
        Falls back to weighted average if no model is trained.
        """
        if self._stacked_model is None:
            logger.warning("Stacked model not trained, falling back to weighted average")
            return self._weighted_average(protein, results)
        
        n = len(protein.sequence)
        
        # Build feature matrix: one row per residue, one column per predictor
        n_predictors = len(results)
        X = np.zeros((n, n_predictors))
        
        for j, (name, result) in enumerate(results.items()):
            if result.per_residue_scores:
                X[:, j] = result.per_residue_scores.scores
        
        # Predict
        if hasattr(self._stacked_model, 'predict_proba'):
            scores = self._stacked_model.predict_proba(X)[:, 1]
        else:
            scores = self._stacked_model.predict(X)
        
        per_residue = PerResidueScores(
            scores=scores.tolist(),
            sequence=protein.sequence,
            predictor="Consensus",
            score_type="stacked_ensemble",
            threshold=self.config.region_threshold,
        )
        
        regions = per_residue.to_regions(
            threshold=self.config.region_threshold,
            min_length=self.config.min_region_length,
            merge_gap=self.config.merge_gap,
        )
        
        is_amyloidogenic = len(regions) > 0
        
        return ConsensusResult(
            sequence_id=protein.id,
            sequence=protein.sequence,
            individual_results=results,
            consensus_per_residue=per_residue,
            consensus_regions=regions,
            consensus_is_amyloidogenic=is_amyloidogenic,
            consensus_probability=float(np.mean(scores)),
            n_predictors_agree_positive=sum(1 for r in results.values() if r.is_amyloidogenic),
            n_predictors_agree_negative=sum(1 for r in results.values() if not r.is_amyloidogenic),
            n_predictors_total=len(results),
            consensus_method=ConsensusMethod.STACKED.value,
        )
    
    def _metamyl(
        self,
        protein: ProteinRecord,
        results: dict[str, PredictionResult],
    ) -> ConsensusResult:
        """
        MetAmyl-style logistic regression combination.
        
        Uses a simple logistic combination of predictor binary outputs
        following the approach of Emily et al. (2013).
        """
        # For now, implement as weighted average with equal weights
        # Full MetAmyl would require training the logistic coefficients
        return self._weighted_average(protein, results)
    
    def train_stacked_model(
        self,
        proteins: list[ProteinRecord],
        labels: list[list[bool]],  # Per-residue labels
        model_type: str = "logistic",
    ):
        """
        Train the stacked ensemble model.
        
        Args:
            proteins: List of training proteins
            labels: Per-residue binary labels for each protein
            model_type: Type of stacking model ('logistic', 'rf', 'xgboost')
        """
        from sklearn.linear_model import LogisticRegression
        
        # Collect training data
        X_all = []
        y_all = []
        
        active_predictors = self.get_active_predictors()
        
        for protein, protein_labels in zip(proteins, labels):
            # Get predictions from all predictors
            results = {}
            for name in active_predictors:
                try:
                    results[name] = self.predictors[name].predict(protein)
                except:
                    continue
            
            if len(results) < len(active_predictors):
                continue
            
            # Build feature matrix
            n = len(protein.sequence)
            X = np.zeros((n, len(results)))
            
            for j, (name, result) in enumerate(results.items()):
                if result.per_residue_scores:
                    X[:, j] = result.per_residue_scores.scores
            
            X_all.append(X)
            y_all.extend(protein_labels[:n])
        
        X_train = np.vstack(X_all)
        y_train = np.array(y_all, dtype=int)
        
        # Train model
        if model_type == "logistic":
            self._stacked_model = LogisticRegression(max_iter=1000)
        elif model_type == "rf":
            from sklearn.ensemble import RandomForestClassifier
            self._stacked_model = RandomForestClassifier(n_estimators=100)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self._stacked_model.fit(X_train, y_train)
        logger.info(f"Stacked model trained on {len(X_train)} residues")
    
    def predict_batch(
        self,
        proteins: list[ProteinRecord],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[ConsensusResult]:
        """
        Generate consensus predictions for multiple proteins.
        
        Args:
            proteins: List of proteins
            progress_callback: Optional callback(current, total)
            
        Returns:
            List of ConsensusResult objects
        """
        results = []
        
        for i, protein in enumerate(proteins):
            try:
                result = self.predict(protein)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to predict {protein.id}: {e}")
                results.append(None)
            
            if progress_callback:
                progress_callback(i + 1, len(proteins))
        
        return results
    
    def get_predictor_agreement_matrix(
        self,
        results: ConsensusResult,
    ) -> np.ndarray:
        """
        Calculate pairwise agreement between predictors.
        
        Returns a matrix where entry [i,j] is the fraction of residues
        where predictors i and j agree on classification.
        """
        predictor_names = list(results.individual_results.keys())
        n_predictors = len(predictor_names)
        n = len(results.sequence)
        
        # Get per-residue predictions
        predictions = {}
        for name, result in results.individual_results.items():
            if result.per_residue_scores:
                threshold = result.per_residue_scores.threshold or 0.5
                pred = np.array(result.per_residue_scores.scores) >= threshold
            else:
                pred = np.zeros(n, dtype=bool)
            predictions[name] = pred
        
        # Calculate agreement matrix
        agreement = np.zeros((n_predictors, n_predictors))
        
        for i, name1 in enumerate(predictor_names):
            for j, name2 in enumerate(predictor_names):
                agree = predictions[name1] == predictions[name2]
                agreement[i, j] = np.mean(agree)
        
        return agreement


# =============================================================================
# Convenience functions
# =============================================================================

def quick_consensus(
    protein: ProteinRecord,
    predictor_names: Optional[list[str]] = None,
    method: ConsensusMethod = ConsensusMethod.WEIGHTED_AVERAGE,
) -> ConsensusResult:
    """
    Quick consensus prediction without explicit engine setup.
    
    Args:
        protein: Input protein
        predictor_names: Optional list of predictors to use (None = all local)
        method: Consensus method
        
    Returns:
        ConsensusResult
    """
    engine = ConsensusEngine()
    
    if predictor_names is None:
        # Use all available local predictors
        predictor_names = [p["name"] for p in list_predictors()]
    
    for name in predictor_names:
        try:
            engine.add_predictor(name)
        except Exception as e:
            logger.warning(f"Could not add {name}: {e}")
    
    return engine.predict(protein, method=method)
