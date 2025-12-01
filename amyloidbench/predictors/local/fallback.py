"""
Fallback predictor using biophysical features and machine learning.

The FallbackPredictor provides a robust baseline for amyloidogenicity prediction
using a combination of biophysical scoring and optional machine learning. It is
designed to:

1. Work when external predictors are unavailable
2. Provide interpretable predictions based on fundamental biophysical principles
3. Serve as a baseline for benchmarking other methods
4. Integrate diverse features into a unified scoring framework

Biological Rationale
--------------------
Amyloid formation is driven by well-characterized biophysical principles:

1. **Hydrophobicity**: The aggregation process is fundamentally driven by the
   hydrophobic effect—the tendency to minimize water-exposed hydrophobic surface.
   Sequences with clustered hydrophobic residues have higher aggregation propensity.

2. **β-sheet propensity**: Amyloid fibrils adopt cross-β structure where
   individual peptides form β-strands. Residues with high β-sheet propensity
   (V, I, F, Y, W, L, T) favor this conformation.

3. **Gatekeepers**: Charged residues (K, R, D, E) and proline are evolutionary
   safeguards against aberrant aggregation. They introduce electrostatic repulsion
   (charged) or conformational constraints (proline) that disrupt β-sheet formation.

4. **Sequence context**: Amyloidogenic segments typically span 5-15 residues.
   Single high-propensity residues are insufficient; contiguous stretches are
   required to nucleate aggregation.

The fallback predictor quantifies these principles through feature extraction
and combines them using either:
- Rule-based scoring (interpretable, no training required)
- Trained ML model (higher accuracy when training data available)

Algorithm
---------
Rule-based mode (default):
1. Extract per-residue biophysical features (hydrophobicity, β-propensity, etc.)
2. Calculate sliding window scores using weighted combination
3. Apply gatekeeper penalty to windows containing protective residues
4. Identify regions exceeding threshold as aggregation-prone
5. Merge nearby regions and filter by minimum length

ML mode (when model trained):
1. Extract comprehensive feature set (composition, patterns, context)
2. Apply trained classifier (LogisticRegression/XGBoost)
3. Generate per-residue predictions using sliding window
4. Post-process to identify contiguous APRs

References
----------
- Fernandez-Escamilla et al. (2004) - TANGO biophysical model
- Tartaglia et al. (2008) - Zyggregator feature analysis
- Rousseau et al. (2006) - Gatekeeper concept
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from amyloidbench.core.models import (
    PerResidueScores,
    PredictionResult,
    Region,
)
from amyloidbench.features.extraction import (
    FeatureExtractor,
    SequenceFeatures,
    HYDROPHOBICITY_KD,
    BETA_PROPENSITY_CF,
    AGGREGATION_PROPENSITY,
    CHARGE_PH7,
)
from amyloidbench.predictors.base import (
    BasePredictor,
    PredictorCapability,
    PredictorConfig,
    PredictorType,
    TrainablePredictor,
    register_predictor,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Scoring Weights (empirically derived from aggregation studies)
# =============================================================================

@dataclass
class ScoringWeights:
    """
    Weights for combining biophysical features into aggregation score.
    
    Default values are derived from analysis of experimentally validated
    amyloidogenic sequences (Aβ, α-synuclein, tau, prion, etc.)
    """
    hydrophobicity: float = 0.30
    beta_propensity: float = 0.25
    aggregation_scale: float = 0.25
    gatekeeper_penalty: float = -0.20
    disorder_bonus: float = 0.10  # Disordered regions often contain APRs
    charge_penalty: float = -0.10
    aromatic_bonus: float = 0.05  # π-stacking contribution
    
    # Context weights
    flanking_gatekeeper_weight: float = -0.15
    stretch_bonus: float = 0.05  # Bonus for extended hydrophobic stretches


DEFAULT_WEIGHTS = ScoringWeights()


# =============================================================================
# Fallback Predictor
# =============================================================================

@register_predictor
class FallbackPredictor(TrainablePredictor):
    """
    Biophysics-based fallback predictor with optional ML enhancement.
    
    This predictor combines fundamental biophysical principles with optional
    machine learning to provide robust amyloidogenicity predictions. It can
    operate in two modes:
    
    1. **Rule-based** (default): Uses weighted combination of biophysical
       features. No training required, fully interpretable, works immediately.
    
    2. **ML-enhanced**: After training on labeled data, uses a classifier
       to weight features optimally. Higher accuracy but requires training.
    
    The predictor is designed to:
    - Serve as a reliable baseline for benchmarking
    - Work when web-based predictors are unavailable
    - Provide interpretable predictions linked to biophysical properties
    - Allow customization of scoring weights based on domain knowledge
    
    Usage:
        >>> predictor = FallbackPredictor()
        >>> result = predictor.predict(protein)
        >>> 
        >>> # For higher accuracy, train on labeled data:
        >>> predictor.fit(training_proteins, labels)
        >>> result = predictor.predict(protein)
    
    Attributes:
        weights: ScoringWeights for feature combination
        feature_extractor: FeatureExtractor instance
        model: Optional trained ML model
    """
    
    name = "FallbackPredictor"
    version = "1.0"
    predictor_type = PredictorType.FALLBACK
    capabilities = {
        PredictorCapability.PER_RESIDUE_SCORES,
        PredictorCapability.REGION_DETECTION,
        PredictorCapability.BINARY_CLASSIFICATION,
        PredictorCapability.TRAINABLE,
    }
    
    default_threshold = 0.5
    default_window_size = 7
    
    score_min = 0.0
    score_max = 1.0
    
    citation = "AmyloidBench Fallback Predictor (internal)"
    url = "https://github.com/sukhanovaxenia/AmyloidBench"
    description = (
        "Biophysics-based predictor combining hydrophobicity, β-propensity, "
        "and gatekeeper analysis. Provides interpretable predictions grounded "
        "in fundamental principles of protein aggregation."
    )
    
    def __init__(
        self,
        config: Optional[PredictorConfig] = None,
        weights: Optional[ScoringWeights] = None,
        use_ml: bool = False,
    ):
        """
        Initialize fallback predictor.
        
        Args:
            config: Predictor configuration
            weights: Custom scoring weights (None = use defaults)
            use_ml: Whether to use ML mode (requires training)
        """
        super().__init__(config)
        
        self.weights = weights or DEFAULT_WEIGHTS
        self.feature_extractor = FeatureExtractor(window_size=self.window_size)
        self.use_ml = use_ml
        self.model = None
        self._feature_names = None
    
    def _calculate_residue_score(self, aa: str, context: str = "") -> float:
        """
        Calculate aggregation propensity score for a single residue.
        
        Args:
            aa: Single amino acid
            context: Surrounding sequence (for context-aware scoring)
            
        Returns:
            Normalized score between 0 and 1
        """
        # Base properties
        hydro = HYDROPHOBICITY_KD.get(aa, 0.0)
        beta = BETA_PROPENSITY_CF.get(aa, 1.0)
        agg = AGGREGATION_PROPENSITY.get(aa, 0.0)
        charge = abs(CHARGE_PH7.get(aa, 0))
        
        # Is this a gatekeeper?
        is_gatekeeper = aa in 'PKRED'
        
        # Is this aromatic?
        is_aromatic = aa in 'FYW'
        
        # Normalize hydrophobicity to [0, 1]
        hydro_norm = (hydro + 4.5) / 9.0  # KD scale: -4.5 to 4.5
        
        # Normalize β-propensity (values ~0.5 to 1.7)
        beta_norm = (beta - 0.5) / 1.2
        
        # Normalize aggregation propensity
        agg_norm = (agg + 1.5) / 3.0  # Scale: -1.5 to 1.5
        
        # Calculate weighted score
        score = (
            self.weights.hydrophobicity * hydro_norm +
            self.weights.beta_propensity * beta_norm +
            self.weights.aggregation_scale * agg_norm +
            self.weights.charge_penalty * charge
        )
        
        if is_gatekeeper:
            score += self.weights.gatekeeper_penalty
        
        if is_aromatic:
            score += self.weights.aromatic_bonus
        
        return max(0, min(1, score))  # Clamp to [0, 1]
    
    def _calculate_window_score(
        self,
        sequence: str,
        position: int,
        features: SequenceFeatures,
    ) -> float:
        """
        Calculate aggregation score for a sequence window.
        
        The window score considers:
        1. Mean residue scores within the window
        2. Presence of gatekeepers (penalty)
        3. Hydrophobic stretch length (bonus)
        4. Charge distribution
        
        Args:
            sequence: Full sequence
            position: Center position of window
            features: Pre-extracted features
            
        Returns:
            Window aggregation score
        """
        n = len(sequence)
        half_w = self.window_size // 2
        
        start = max(0, position - half_w)
        end = min(n, position + half_w + 1)
        
        window_seq = sequence[start:end]
        
        # Calculate mean residue score
        residue_scores = [self._calculate_residue_score(aa) for aa in window_seq]
        mean_score = np.mean(residue_scores)
        
        # Count gatekeepers in window
        n_gatekeepers = sum(1 for aa in window_seq if aa in 'PKRED')
        gatekeeper_penalty = n_gatekeepers * 0.05
        
        # Check for hydrophobic stretch
        hydrophobic = set('VILFYWM')
        max_hydro_stretch = 0
        current_stretch = 0
        for aa in window_seq:
            if aa in hydrophobic:
                current_stretch += 1
                max_hydro_stretch = max(max_hydro_stretch, current_stretch)
            else:
                current_stretch = 0
        
        stretch_bonus = min(0.2, max_hydro_stretch * self.weights.stretch_bonus)
        
        # Net charge in window
        window_charge = sum(CHARGE_PH7.get(aa, 0) for aa in window_seq)
        charge_penalty = abs(window_charge) * 0.02
        
        # Combine
        final_score = mean_score + stretch_bonus - gatekeeper_penalty - charge_penalty
        
        return max(0, min(1, final_score))
    
    def _predict_impl(
        self,
        sequence: str,
        structure_path: Optional[Path] = None,
    ) -> PredictionResult:
        """
        Generate prediction using biophysical scoring or trained model.
        
        Args:
            sequence: Protein sequence
            structure_path: Ignored (sequence-only method)
            
        Returns:
            PredictionResult with per-residue scores and APRs
        """
        # Extract features
        features = self.feature_extractor.extract(sequence)
        
        if self.use_ml and self.model is not None:
            # Use trained ML model
            scores = self._predict_with_model(sequence, features)
        else:
            # Use rule-based scoring
            scores = self._predict_rule_based(sequence, features)
        
        # Create per-residue scores object
        per_residue = PerResidueScores(
            scores=scores.tolist(),
            sequence=sequence,
            predictor=self.name,
            score_type="normalized",
            threshold=self.threshold,
            min_score=self.score_min,
            max_score=self.score_max,
        )
        
        # Extract regions above threshold
        regions = per_residue.to_regions(
            threshold=self.threshold,
            min_length=self.config.min_region_length,
            merge_gap=self.config.merge_gap,
        )
        
        # Calculate probability
        if regions:
            max_region_score = max(r.score for r in regions if r.score)
            probability = 0.3 + 0.7 * max_region_score
        else:
            probability = np.mean(scores)
        
        is_amyloidogenic = len(regions) > 0
        
        return PredictionResult(
            sequence_id="",
            sequence=sequence,
            predictor_name=self.name,
            predictor_version=self.version,
            per_residue_scores=per_residue,
            predicted_regions=regions,
            is_amyloidogenic=is_amyloidogenic,
            amyloid_probability=float(probability),
            raw_output={
                "mode": "ml" if (self.use_ml and self.model) else "rule_based",
                "weights": self.weights.__dict__,
                "features_used": list(features.global_features.keys())[:10],
            } if self.config.return_raw_output else None,
        )
    
    def _predict_rule_based(
        self,
        sequence: str,
        features: SequenceFeatures,
    ) -> np.ndarray:
        """
        Generate predictions using rule-based scoring.
        
        This method combines multiple biophysical properties using
        empirically-derived weights.
        """
        n = len(sequence)
        scores = np.zeros(n)
        
        for i in range(n):
            scores[i] = self._calculate_window_score(sequence, i, features)
        
        return scores
    
    def _predict_with_model(
        self,
        sequence: str,
        features: SequenceFeatures,
    ) -> np.ndarray:
        """
        Generate predictions using trained ML model.
        
        Uses sliding window feature extraction and model prediction.
        """
        n = len(sequence)
        scores = np.zeros(n)
        
        # For each position, extract local features and predict
        half_w = self.window_size // 2
        
        for i in range(n):
            start = max(0, i - half_w)
            end = min(n, i + half_w + 1)
            
            # Extract features for window
            window_seq = sequence[start:end]
            window_features = self.feature_extractor.extract(window_seq)
            
            # Get feature vector
            X = window_features.to_vector(self._feature_names)
            
            # Predict
            if hasattr(self.model, 'predict_proba'):
                prob = self.model.predict_proba(X.reshape(1, -1))[0, 1]
            else:
                prob = self.model.predict(X.reshape(1, -1))[0]
            
            scores[i] = prob
        
        return scores
    
    def fit(
        self,
        proteins: list,
        labels: list[bool],
        model_type: str = "logistic",
    ):
        """
        Train the ML model on labeled data.
        
        This enables higher-accuracy predictions by learning optimal
        feature weights from experimental data.
        
        Args:
            proteins: List of ProteinRecord objects
            labels: List of boolean labels (True = amyloidogenic)
            model_type: Type of model ('logistic', 'xgboost', 'rf')
        """
        logger.info(f"Training {self.name} with {len(proteins)} samples")
        
        # Extract features for all proteins
        X_list = []
        for protein in proteins:
            features = self.feature_extractor.extract(protein.sequence)
            X_list.append(features.to_vector())
        
        # Store feature names for later use
        sample_features = self.feature_extractor.extract(proteins[0].sequence)
        self._feature_names = sorted(sample_features.global_features.keys())
        
        X = np.array(X_list)
        y = np.array(labels, dtype=int)
        
        # Select model
        if model_type == "logistic":
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
                self.model = XGBClassifier(n_estimators=100, max_depth=5, random_state=42)
            except ImportError:
                logger.warning("XGBoost not available, falling back to LogisticRegression")
                from sklearn.linear_model import LogisticRegression
                self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == "rf":
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train
        self.model.fit(X, y)
        self.use_ml = True
        
        logger.info(f"Model trained successfully")
    
    def save(self, path: Union[str, Path]):
        """Save trained model to file."""
        import joblib
        
        path = Path(path)
        
        model_data = {
            "model": self.model,
            "feature_names": self._feature_names,
            "weights": self.weights,
            "version": self.version,
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Union[str, Path]):
        """Load trained model from file."""
        import joblib
        
        path = Path(path)
        model_data = joblib.load(path)
        
        self.model = model_data["model"]
        self._feature_names = model_data["feature_names"]
        self.weights = model_data.get("weights", DEFAULT_WEIGHTS)
        self.use_ml = True
        
        logger.info(f"Model loaded from {path}")
    
    def get_feature_importance(self) -> dict[str, float]:
        """
        Get feature importance from trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None or self._feature_names is None:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # Linear models
            importances = np.abs(self.model.coef_[0])
        else:
            return {}
        
        return dict(zip(self._feature_names, importances))


# =============================================================================
# TANGO-like scoring function
# =============================================================================

def calculate_tango_like_score(
    sequence: str,
    window_size: int = 7,
    beta_weight: float = 0.4,
    hydro_weight: float = 0.4,
    gatekeeper_weight: float = -0.3,
) -> tuple[np.ndarray, list[Region]]:
    """
    Calculate TANGO-like aggregation propensity scores.
    
    This provides a simplified approximation of the TANGO algorithm
    (Fernandez-Escamilla et al., 2004) based on the published principles:
    
    1. β-sheet propensity of residues in the aggregating conformation
    2. Hydrophobicity (desolvation penalty for exposing residues)
    3. Electrostatic contribution (charge repulsion)
    4. Gatekeeper effects (proline breaks, charged flanks)
    
    The full TANGO algorithm uses thermodynamic calculations that are
    not publicly available. This approximation captures the key features.
    
    Args:
        sequence: Protein sequence
        window_size: Sliding window size
        beta_weight: Weight for β-sheet propensity
        hydro_weight: Weight for hydrophobicity
        gatekeeper_weight: Weight for gatekeeper penalty
        
    Returns:
        Tuple of (per-residue scores, identified regions)
    """
    n = len(sequence)
    scores = np.zeros(n)
    
    half_w = window_size // 2
    
    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        window = sequence[start:end]
        
        # β-sheet propensity
        beta_score = np.mean([BETA_PROPENSITY_CF.get(aa, 1.0) for aa in window])
        beta_contrib = (beta_score - 0.8) / 0.9  # Normalize
        
        # Hydrophobicity
        hydro_score = np.mean([HYDROPHOBICITY_KD.get(aa, 0) for aa in window])
        hydro_contrib = (hydro_score + 4.5) / 9.0  # Normalize to [0, 1]
        
        # Gatekeeper penalty
        n_gatekeepers = sum(1 for aa in window if aa in 'PKRED')
        gatekeeper_contrib = n_gatekeepers / len(window)
        
        # Combine
        score = (
            beta_weight * beta_contrib +
            hydro_weight * hydro_contrib +
            gatekeeper_weight * gatekeeper_contrib
        )
        
        scores[i] = max(0, min(1, score))
    
    # Identify regions
    regions = []
    threshold = 0.5
    in_region = False
    region_start = 0
    region_scores = []
    
    for i, score in enumerate(scores):
        if score >= threshold and not in_region:
            in_region = True
            region_start = i
            region_scores = [score]
        elif score >= threshold and in_region:
            region_scores.append(score)
        elif score < threshold and in_region:
            in_region = False
            if len(region_scores) >= 5:  # Minimum length
                regions.append(Region(
                    start=region_start,
                    end=i,
                    sequence=sequence[region_start:i],
                    score=np.mean(region_scores),
                ))
            region_scores = []
    
    # Handle region at end
    if in_region and len(region_scores) >= 5:
        regions.append(Region(
            start=region_start,
            end=n,
            sequence=sequence[region_start:n],
            score=np.mean(region_scores),
        ))
    
    return scores, regions
