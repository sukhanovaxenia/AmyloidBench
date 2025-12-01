"""
Polymorph-aware benchmarking for amyloidogenicity prediction.

This module extends the benchmarking framework to evaluate predictors
in the context of structural polymorph classification. Different amyloid
polymorphs have distinct:

1. **Structural architectures**: Steric zipper vs β-solenoid vs β-arcade
2. **Toxicity profiles**: Not all polymorphs are equally pathogenic
3. **Seeding efficiency**: Template-directed assembly varies by structure
4. **Therapeutic targets**: Structure-specific inhibitors require polymorph ID

Biological Rationale
--------------------
Cryo-EM studies have revealed remarkable structural diversity among
amyloid fibrils. A single protein can form multiple polymorphs with
different biological consequences:

- **α-Synuclein**: Different polymorphs in PD vs MSA patients
- **Tau**: AD paired helical filaments differ from Pick's bodies
- **Aβ**: Patient-to-patient structural variation correlates with
  disease progression

Predictor performance may vary significantly across polymorph types.
For example, predictors trained on short peptide zipper data may
perform differently on full-length proteins forming β-arcade structures.

Evaluation Metrics
------------------

**Per-Polymorph Sensitivity/Specificity**
How well does the predictor identify APRs in different structural classes?

**Polymorph Classification Accuracy**
If the predictor also classifies polymorph type, how accurate is it?

**Concordance with Structural Data**
How well do predicted APR boundaries match structurally-defined core regions?

**Cross-Polymorph Generalization**
Performance when training on one polymorph type, testing on another.

References
----------
- Fitzpatrick et al. (2017) - Cryo-EM of tau filaments from AD brain
- Guerrero-Ferreira et al. (2018) - α-synuclein polymorphs
- Sawaya et al. (2007) - Steric zipper classification
- Eisenberg & Sawaya (2017) - Amyloid fibril structural diversity
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from amyloidbench.classification.polymorph import (
    AmyloidFold,
    CrossBetaGeometry,
    PolymorphClassifier,
    PolymorphPrediction,
    StericZipperClass,
)
from amyloidbench.core.models import Region

from .datasets import AmyloidStatus, BenchmarkDataset, BenchmarkEntry
from .metrics import (
    BenchmarkResult,
    ClassificationMetrics,
    PerResidueMetrics,
    calculate_classification_metrics,
    calculate_mcc,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PolymorphBenchmarkEntry:
    """
    Extended benchmark entry with polymorph annotation.
    
    Attributes:
        base_entry: Underlying BenchmarkEntry
        known_fold: Experimentally determined fold type
        known_geometry: Experimentally determined β-strand geometry
        known_zipper_class: Steric zipper class (for short peptides)
        structure_source: PDB ID or other structural reference
        confidence: Confidence in polymorph assignment
    """
    base_entry: BenchmarkEntry
    known_fold: Optional[AmyloidFold] = None
    known_geometry: Optional[CrossBetaGeometry] = None
    known_zipper_class: Optional[StericZipperClass] = None
    structure_source: str = ""
    confidence: float = 1.0
    
    @property
    def sequence(self) -> str:
        return self.base_entry.sequence
    
    @property
    def id(self) -> str:
        return self.base_entry.id


@dataclass
class PolymorphBenchmarkResult:
    """
    Benchmark results stratified by polymorph type.
    
    Attributes:
        predictor_name: Evaluated predictor
        overall_metrics: Aggregate classification metrics
        per_fold_metrics: Metrics for each fold type
        per_geometry_metrics: Metrics for each geometry type
        fold_classification_accuracy: How well polymorph was predicted
        confusion_matrix: Polymorph classification confusion matrix
    """
    predictor_name: str
    overall_metrics: ClassificationMetrics
    per_fold_metrics: dict[str, ClassificationMetrics]
    per_geometry_metrics: dict[str, ClassificationMetrics]
    fold_classification_accuracy: float = 0.0
    geometry_classification_accuracy: float = 0.0
    n_samples: int = 0
    per_fold_counts: dict[str, int] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Generate summary report."""
        lines = [
            f"Polymorph-Aware Benchmark: {self.predictor_name}",
            "=" * 60,
            f"Total samples: {self.n_samples}",
            "",
            "Overall Classification:",
            f"  Sensitivity: {self.overall_metrics.sensitivity:.3f}",
            f"  Specificity: {self.overall_metrics.specificity:.3f}",
            f"  MCC: {self.overall_metrics.mcc:.3f}",
            "",
            "Per-Fold Performance:",
        ]
        
        for fold_name, metrics in sorted(self.per_fold_metrics.items()):
            count = self.per_fold_counts.get(fold_name, 0)
            lines.append(f"  {fold_name} (n={count}):")
            lines.append(f"    Sens: {metrics.sensitivity:.3f}, "
                        f"Spec: {metrics.specificity:.3f}, "
                        f"MCC: {metrics.mcc:.3f}")
        
        if self.fold_classification_accuracy > 0:
            lines.extend([
                "",
                f"Polymorph Classification Accuracy: {self.fold_classification_accuracy:.1%}",
            ])
        
        return "\n".join(lines)


# =============================================================================
# Polymorph-Annotated Dataset Builder
# =============================================================================

def annotate_dataset_with_polymorphs(
    dataset: BenchmarkDataset,
) -> list[PolymorphBenchmarkEntry]:
    """
    Add polymorph annotations to a benchmark dataset.
    
    Uses sequence-based heuristics and metadata to assign
    likely polymorph classifications.
    
    Args:
        dataset: Input benchmark dataset
        
    Returns:
        List of PolymorphBenchmarkEntry with annotations
    """
    classifier = PolymorphClassifier()
    annotated = []
    
    for entry in dataset.entries:
        # Check for known polymorph in metadata
        known_fold = None
        known_geometry = None
        known_zipper = None
        
        if entry.metadata:
            # Check for zipper class annotation
            if "zipper_class" in entry.metadata:
                zc = entry.metadata["zipper_class"]
                if zc is not None:
                    try:
                        known_zipper = StericZipperClass(f"class_{zc}")
                        known_fold = AmyloidFold.STERIC_ZIPPER
                        known_geometry = _infer_geometry_from_zipper(zc)
                    except ValueError:
                        pass
            
            # Check for explicit fold annotation
            if "fold_type" in entry.metadata:
                ft = entry.metadata["fold_type"].lower()
                if "solenoid" in ft:
                    known_fold = AmyloidFold.BETA_SOLENOID
                elif "arcade" in ft:
                    known_fold = AmyloidFold.BETA_ARCADE
                elif "zipper" in ft:
                    known_fold = AmyloidFold.STERIC_ZIPPER
        
        # Predict if not annotated
        if known_fold is None and entry.is_positive:
            prediction = classifier.predict(entry.sequence)
            known_fold = prediction.predicted_fold
            known_geometry = prediction.predicted_geometry
            
            if known_fold == AmyloidFold.STERIC_ZIPPER:
                known_zipper = prediction.steric_zipper_class
        
        # Get structure source
        structure_source = ""
        if entry.metadata:
            pdb_ids = entry.metadata.get("pdb_ids", [])
            if pdb_ids:
                structure_source = pdb_ids[0] if isinstance(pdb_ids, list) else pdb_ids
        
        annotated.append(PolymorphBenchmarkEntry(
            base_entry=entry,
            known_fold=known_fold,
            known_geometry=known_geometry,
            known_zipper_class=known_zipper,
            structure_source=structure_source,
        ))
    
    return annotated


def _infer_geometry_from_zipper(zipper_class: int) -> CrossBetaGeometry:
    """Infer β-strand geometry from steric zipper class."""
    if zipper_class in [1, 2, 3, 4]:
        return CrossBetaGeometry.PARALLEL_IN_REGISTER
    elif zipper_class in [5, 6, 7, 8]:
        return CrossBetaGeometry.ANTIPARALLEL
    return CrossBetaGeometry.UNKNOWN


# =============================================================================
# Polymorph-Aware Benchmark Runner
# =============================================================================

class PolymorphBenchmarkRunner:
    """
    Benchmark runner with polymorph stratification.
    
    Evaluates predictors separately for each polymorph type,
    enabling identification of structure-specific performance gaps.
    """
    
    def __init__(self):
        self.classifier = PolymorphClassifier()
    
    def run(
        self,
        predictor,
        annotated_entries: list[PolymorphBenchmarkEntry],
    ) -> PolymorphBenchmarkResult:
        """
        Run polymorph-stratified benchmark.
        
        Args:
            predictor: Predictor instance with predict() method
            annotated_entries: Polymorph-annotated benchmark entries
            
        Returns:
            PolymorphBenchmarkResult with stratified metrics
        """
        # Group entries by fold type
        by_fold: dict[str, list] = {}
        by_geometry: dict[str, list] = {}
        
        all_true = []
        all_pred = []
        all_scores = []
        
        fold_true: dict[str, list] = {}
        fold_pred: dict[str, list] = {}
        fold_scores: dict[str, list] = {}
        
        for entry in annotated_entries:
            # Get prediction - handle different input types
            try:
                # Predictors may expect sequence string or ProteinRecord
                sequence = entry.sequence
                result = predictor.predict(sequence)
                pred_positive = result.is_amyloidogenic
                score = result.amyloid_probability
            except AttributeError:
                # If predictor returns different format, try to adapt
                try:
                    from amyloidbench.core.models import ProteinRecord
                    protein = ProteinRecord(id=entry.id, sequence=entry.sequence)
                    result = predictor.predict(protein)
                    pred_positive = result.is_amyloidogenic
                    score = result.amyloid_probability
                except Exception as e:
                    logger.warning(f"Prediction failed for {entry.id}: {e}")
                    continue
            except Exception as e:
                logger.warning(f"Prediction failed for {entry.id}: {e}")
                continue
            
            true_positive = entry.base_entry.is_positive
            
            # Overall
            all_true.append(true_positive)
            all_pred.append(pred_positive)
            all_scores.append(score)
            
            # By fold
            if entry.known_fold:
                fold_name = entry.known_fold.value
                
                if fold_name not in by_fold:
                    by_fold[fold_name] = []
                    fold_true[fold_name] = []
                    fold_pred[fold_name] = []
                    fold_scores[fold_name] = []
                
                by_fold[fold_name].append(entry)
                fold_true[fold_name].append(true_positive)
                fold_pred[fold_name].append(pred_positive)
                fold_scores[fold_name].append(score)
            
            # By geometry
            if entry.known_geometry:
                geom_name = entry.known_geometry.value
                
                if geom_name not in by_geometry:
                    by_geometry[geom_name] = []
                
                by_geometry[geom_name].append((true_positive, pred_positive, score))
        
        # Calculate overall metrics
        if all_true:
            overall_metrics = calculate_classification_metrics(
                all_true, all_pred, all_scores
            )
        else:
            overall_metrics = ClassificationMetrics()
        
        # Calculate per-fold metrics
        per_fold_metrics = {}
        per_fold_counts = {}
        
        for fold_name in fold_true:
            if fold_true[fold_name]:
                per_fold_metrics[fold_name] = calculate_classification_metrics(
                    fold_true[fold_name],
                    fold_pred[fold_name],
                    fold_scores[fold_name] if fold_scores[fold_name] else None,
                )
                per_fold_counts[fold_name] = len(fold_true[fold_name])
        
        # Calculate per-geometry metrics
        per_geometry_metrics = {}
        
        for geom_name, data in by_geometry.items():
            if data:
                g_true = [d[0] for d in data]
                g_pred = [d[1] for d in data]
                g_scores = [d[2] for d in data]
                
                per_geometry_metrics[geom_name] = calculate_classification_metrics(
                    g_true, g_pred, g_scores
                )
        
        # Calculate polymorph classification accuracy
        fold_accuracy = self._calculate_fold_classification_accuracy(
            predictor, annotated_entries
        )
        
        return PolymorphBenchmarkResult(
            predictor_name=getattr(predictor, 'name', 'Unknown'),
            overall_metrics=overall_metrics,
            per_fold_metrics=per_fold_metrics,
            per_geometry_metrics=per_geometry_metrics,
            fold_classification_accuracy=fold_accuracy,
            n_samples=len(all_true),
            per_fold_counts=per_fold_counts,
        )
    
    def _calculate_fold_classification_accuracy(
        self,
        predictor,
        annotated_entries: list[PolymorphBenchmarkEntry],
    ) -> float:
        """Calculate how well the predictor classifies polymorph types."""
        # Only for entries with known folds
        correct = 0
        total = 0
        
        for entry in annotated_entries:
            if entry.known_fold is None:
                continue
            
            # Get polymorph prediction
            poly_pred = self.classifier.predict(entry.sequence)
            
            if poly_pred.predicted_fold == entry.known_fold:
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0.0


# =============================================================================
# Per-Region Polymorph Analysis
# =============================================================================

@dataclass
class RegionPolymorphAnalysis:
    """Analysis of predicted APRs with polymorph context."""
    region: Region
    sequence: str
    predicted_fold: AmyloidFold
    fold_probability: float
    predicted_geometry: CrossBetaGeometry
    zipper_class: Optional[StericZipperClass]
    confidence: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "region": f"{self.region.start}-{self.region.end}",
            "sequence": self.sequence,
            "fold": self.predicted_fold.value,
            "fold_probability": self.fold_probability,
            "geometry": self.predicted_geometry.value,
            "zipper_class": self.zipper_class.value if self.zipper_class else None,
            "confidence": self.confidence,
        }


def analyze_regions_with_polymorphs(
    sequence: str,
    predicted_regions: list[Region],
) -> list[RegionPolymorphAnalysis]:
    """
    Analyze predicted APR regions with polymorph classification.
    
    For each predicted region, determines likely polymorph type
    based on sequence characteristics.
    
    Args:
        sequence: Full protein sequence
        predicted_regions: APR regions from a predictor
        
    Returns:
        List of RegionPolymorphAnalysis objects
    """
    classifier = PolymorphClassifier()
    analyses = []
    
    for region in predicted_regions:
        # Extract region sequence
        # Adjust for 0/1 indexing
        start = max(0, region.start - 1 if region.start > 0 else region.start)
        end = min(len(sequence), region.end)
        region_seq = sequence[start:end]
        
        if len(region_seq) < 3:
            continue
        
        # Classify the region
        prediction = classifier.predict(region_seq)
        
        analyses.append(RegionPolymorphAnalysis(
            region=region,
            sequence=region_seq,
            predicted_fold=prediction.predicted_fold,
            fold_probability=prediction.fold_probabilities.get(
                prediction.predicted_fold.value, 0.0
            ),
            predicted_geometry=prediction.predicted_geometry,
            zipper_class=prediction.steric_zipper_class,
            confidence=prediction.confidence,
        ))
    
    return analyses


# =============================================================================
# Cross-Polymorph Generalization Test
# =============================================================================

def test_cross_polymorph_generalization(
    predictor,
    annotated_entries: list[PolymorphBenchmarkEntry],
) -> dict[str, dict[str, float]]:
    """
    Test predictor generalization across polymorph types.
    
    Evaluates how well a predictor trained on data from one
    polymorph type performs on other types.
    
    Args:
        predictor: Predictor to evaluate
        annotated_entries: Polymorph-annotated benchmark entries
        
    Returns:
        Dict mapping fold type to performance metrics
    """
    # Group by fold
    by_fold: dict[str, list[PolymorphBenchmarkEntry]] = {}
    
    for entry in annotated_entries:
        if entry.known_fold:
            fold_name = entry.known_fold.value
            if fold_name not in by_fold:
                by_fold[fold_name] = []
            by_fold[fold_name].append(entry)
    
    # Evaluate on each fold
    results = {}
    
    for fold_name, entries in by_fold.items():
        if len(entries) < 3:
            continue
        
        true_labels = []
        pred_labels = []
        scores = []
        
        for entry in entries:
            try:
                result = predictor.predict(entry.sequence)
                pred_labels.append(result.is_amyloidogenic)
                scores.append(result.amyloid_probability)
                true_labels.append(entry.base_entry.is_positive)
            except Exception:
                continue
        
        if true_labels:
            metrics = calculate_classification_metrics(true_labels, pred_labels, scores)
            results[fold_name] = {
                "n_samples": len(true_labels),
                "sensitivity": metrics.sensitivity,
                "specificity": metrics.specificity,
                "mcc": metrics.mcc,
                "auc_roc": metrics.auc_roc or 0.0,
            }
    
    return results


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_polymorph_benchmark(
    predictor,
    dataset: Optional[BenchmarkDataset] = None,
) -> PolymorphBenchmarkResult:
    """
    Quick polymorph-aware benchmark.
    
    Args:
        predictor: Predictor to evaluate
        dataset: Optional dataset (uses reference dataset if None)
        
    Returns:
        PolymorphBenchmarkResult
    """
    from .reference_datasets import create_comprehensive_dataset
    
    if dataset is None:
        dataset = create_comprehensive_dataset()
    
    annotated = annotate_dataset_with_polymorphs(dataset)
    runner = PolymorphBenchmarkRunner()
    
    return runner.run(predictor, annotated)


def get_polymorph_specific_performance(
    results: list[PolymorphBenchmarkResult],
) -> dict[str, dict[str, float]]:
    """
    Compare predictor performance across polymorphs.
    
    Args:
        results: List of benchmark results
        
    Returns:
        Dict mapping predictor -> fold -> MCC
    """
    comparison = {}
    
    for result in results:
        comparison[result.predictor_name] = {}
        
        for fold_name, metrics in result.per_fold_metrics.items():
            comparison[result.predictor_name][fold_name] = metrics.mcc
    
    return comparison
