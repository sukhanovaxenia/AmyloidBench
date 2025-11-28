"""
Benchmarking framework for amyloidogenicity predictor evaluation.

Rigorous benchmarking against experimentally-validated datasets is essential
for establishing predictor reliability and comparing methodologies. This
module provides:

**Curated database integration**

*WALTZ-DB* (Louros et al., 2020)
Hexapeptide-resolution data on amyloid-forming and non-forming sequences.
Contains experimentally validated positive examples from thioflavin T
binding assays and negative examples verified as non-aggregating.
Advantages: High-quality experimental validation, defined sequence length.
Limitations: Hexapeptide context may not reflect full-length protein behavior.

*Cross-Beta DB* (Gonay et al., 2024)
Curated from PDB structures with confirmed cross-β amyloid architecture.
Contains amyloidogenic regions from naturally-occurring amyloids (both
functional and disease-associated) with experimentally determined structures.
Advantages: Structure-validated, represents physiological amyloids.
Limitations: Biased toward well-studied amyloids with solved structures.

*AmyPro* (Varadi et al.)
Database of amyloid-forming proteins with annotated regions from literature
curation. Broader coverage than structure-based databases.

**Evaluation metrics**

For per-residue predictions:
- Sensitivity (TPR): Fraction of APR residues correctly identified
- Specificity (TNR): Fraction of non-APR residues correctly rejected
- Precision (PPV): Fraction of predicted APR residues that are true APRs
- F1 score: Harmonic mean of precision and sensitivity
- Matthews Correlation Coefficient (MCC): Balanced measure for imbalanced data

For protein-level classification:
- Area Under ROC Curve (AUC): Overall discriminative ability
- Precision-Recall curve: Appropriate for imbalanced datasets

**Methodological considerations**

1. *Data leakage*: Many predictors were trained on subsets of benchmark
   databases. We implement sequence-similarity filtering to identify
   potential overlaps between training and test sets.

2. *Redundancy reduction*: CD-HIT clustering at various thresholds (30%,
   50%, 90% identity) to assess performance on diverse vs. similar sequences.

3. *Stratified cross-validation*: Maintain class proportions across folds
   and ensure sequence families don't span train/test splits.

4. *Statistical testing*: DeLong test for AUC comparison, bootstrap
   confidence intervals for all metrics.

Submodules:
    datasets: Database loaders and parsers
    metrics: Performance metric calculations
    crossval: Cross-validation infrastructure
"""

# Benchmark module will be implemented in Phase 6
__all__ = []
