"""
Statistical comparison methods for amyloidogenicity predictor evaluation.

This module provides rigorous statistical tests for comparing predictor
performance, essential for valid claims about superiority or equivalence
between algorithms.

Statistical Tests
-----------------

**McNemar's Test** (paired nominal comparison)
Used when comparing two classifiers on the same samples.
Tests whether error patterns differ significantly.
Appropriate for: Binary classification comparison on matched samples.

**Wilcoxon Signed-Rank Test** (paired continuous comparison)
Non-parametric test for paired continuous measurements (e.g., AUC scores
across cross-validation folds). More robust than paired t-test.

**Friedman Test** (multiple classifier comparison)
Non-parametric omnibus test for comparing multiple classifiers.
Use as preliminary test before post-hoc comparisons.

**Nemenyi Post-hoc Test**
Follow-up test after significant Friedman test to identify
which pairs of classifiers differ significantly.

**DeLong Test** (AUC comparison)
Specifically designed for comparing AUC-ROC values from
the same dataset. Accounts for correlation between predictions.

Correction Methods
------------------
- Bonferroni: Conservative, controls family-wise error rate
- Holm-Bonferroni: Step-down procedure, less conservative
- Benjamini-Hochberg: Controls false discovery rate (FDR)

References
----------
- McNemar (1947) - Note on the sampling error of differences
- Dietterich (1998) - Approximate statistical tests for comparing
  supervised classification learning algorithms
- Demšar (2006) - Statistical comparisons of classifiers over
  multiple data sets
- DeLong et al. (1988) - Comparing the areas under two or more
  correlated ROC curves
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np
from scipy import stats

from .metrics import BenchmarkResult

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class StatisticalComparison:
    """
    Results of statistical comparison between predictors.
    
    Attributes:
        predictor_a: First predictor name
        predictor_b: Second predictor name  
        test_name: Statistical test used
        statistic: Test statistic value
        p_value: P-value (two-tailed)
        p_value_corrected: P-value after multiple testing correction
        significant: Whether difference is significant at alpha level
        effect_size: Effect size measure (if applicable)
        confidence_interval: CI for effect (if applicable)
        notes: Additional information
    """
    predictor_a: str
    predictor_b: str
    test_name: str
    statistic: float
    p_value: float
    p_value_corrected: Optional[float] = None
    significant: bool = False
    effect_size: Optional[float] = None
    confidence_interval: Optional[tuple[float, float]] = None
    alpha: float = 0.05
    notes: list[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        sig_str = "SIGNIFICANT" if self.significant else "not significant"
        lines = [
            f"Statistical Comparison: {self.predictor_a} vs {self.predictor_b}",
            f"Test: {self.test_name}",
            f"Statistic: {self.statistic:.4f}",
            f"P-value: {self.p_value:.4e}",
        ]
        
        if self.p_value_corrected is not None:
            lines.append(f"P-value (corrected): {self.p_value_corrected:.4e}")
        
        lines.append(f"Result: {sig_str} at α={self.alpha}")
        
        if self.effect_size is not None:
            lines.append(f"Effect size: {self.effect_size:.4f}")
        
        if self.confidence_interval:
            lines.append(f"95% CI: [{self.confidence_interval[0]:.4f}, {self.confidence_interval[1]:.4f}]")
        
        return "\n".join(lines)


@dataclass
class MultipleComparisonResult:
    """
    Results from comparing multiple predictors.
    
    Attributes:
        test_name: Overall test name (e.g., "Friedman")
        overall_statistic: Omnibus test statistic
        overall_p_value: Omnibus p-value
        pairwise_comparisons: List of pairwise comparisons
        rankings: Predictor rankings with mean ranks
        critical_difference: CD for post-hoc tests (if applicable)
    """
    test_name: str
    overall_statistic: float
    overall_p_value: float
    overall_significant: bool
    pairwise_comparisons: list[StatisticalComparison]
    rankings: list[tuple[str, float]]  # (predictor_name, mean_rank)
    critical_difference: Optional[float] = None
    alpha: float = 0.05
    
    def summary(self) -> str:
        """Generate summary."""
        lines = [
            f"Multiple Predictor Comparison: {self.test_name}",
            "=" * 50,
            f"Overall statistic: {self.overall_statistic:.4f}",
            f"Overall p-value: {self.overall_p_value:.4e}",
            f"Overall significant: {self.overall_significant}",
            "",
            "Rankings (lower is better):",
        ]
        
        for i, (name, rank) in enumerate(self.rankings, 1):
            lines.append(f"  {i}. {name}: {rank:.2f}")
        
        if self.critical_difference:
            lines.append(f"\nCritical Difference (α={self.alpha}): {self.critical_difference:.3f}")
        
        if self.pairwise_comparisons:
            lines.append("\nSignificant pairwise differences:")
            sig_pairs = [c for c in self.pairwise_comparisons if c.significant]
            if sig_pairs:
                for comp in sig_pairs:
                    lines.append(f"  {comp.predictor_a} vs {comp.predictor_b} "
                               f"(p={comp.p_value_corrected or comp.p_value:.4e})")
            else:
                lines.append("  None")
        
        return "\n".join(lines)


# =============================================================================
# Pairwise Comparison Tests
# =============================================================================

def mcnemar_test(
    errors_a: Sequence[bool],
    errors_b: Sequence[bool],
    alpha: float = 0.05,
    continuity_correction: bool = True,
) -> StatisticalComparison:
    """
    McNemar's test for comparing two classifiers.
    
    Compares the error patterns of two classifiers on the same test set.
    Tests H0: Both classifiers have the same error rate.
    
    Args:
        errors_a: Boolean array where True = classifier A made an error
        errors_b: Boolean array where True = classifier B made an error
        alpha: Significance level
        continuity_correction: Apply Edwards' continuity correction
        
    Returns:
        StatisticalComparison with test results
        
    Example:
        >>> # A correct when B wrong: b01
        >>> # A wrong when B correct: b10
        >>> errors_a = [False, True, False, True, False]
        >>> errors_b = [True, True, False, False, False]
        >>> result = mcnemar_test(errors_a, errors_b)
    """
    errors_a = np.asarray(errors_a, dtype=bool)
    errors_b = np.asarray(errors_b, dtype=bool)
    
    if len(errors_a) != len(errors_b):
        raise ValueError("Error arrays must have same length")
    
    # Build contingency table
    # b01: A correct (False), B wrong (True)
    # b10: A wrong (True), B correct (False)
    b01 = np.sum(~errors_a & errors_b)
    b10 = np.sum(errors_a & ~errors_b)
    
    # Handle small sample sizes
    n_discordant = b01 + b10
    
    notes = []
    
    if n_discordant < 25:
        # Use exact binomial test for small samples
        # Under H0, b01 ~ Binomial(n_discordant, 0.5)
        if n_discordant > 0:
            binom_result = stats.binomtest(b01, n_discordant, 0.5)
            p_value = binom_result.pvalue
        else:
            p_value = 1.0
        statistic = float(b01 - b10)
        notes.append("Used exact binomial test (n_discordant < 25)")
    else:
        # Chi-square approximation
        if continuity_correction:
            # Edwards' correction
            statistic = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
            notes.append("Applied continuity correction")
        else:
            statistic = (b01 - b10) ** 2 / (b01 + b10)
        
        p_value = 1 - stats.chi2.cdf(statistic, df=1)
    
    # Effect size: odds ratio
    if b10 > 0:
        odds_ratio = b01 / b10
        effect_size = np.log(odds_ratio) if odds_ratio > 0 else None
    else:
        effect_size = None
    
    return StatisticalComparison(
        predictor_a="Predictor_A",
        predictor_b="Predictor_B",
        test_name="McNemar's Test",
        statistic=float(statistic),
        p_value=float(p_value),
        significant=p_value < alpha,
        effect_size=effect_size,
        alpha=alpha,
        notes=notes,
    )


def wilcoxon_signed_rank_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> StatisticalComparison:
    """
    Wilcoxon signed-rank test for paired samples.
    
    Non-parametric alternative to paired t-test. Tests whether
    the median difference between pairs is zero.
    
    Args:
        scores_a: Performance scores from predictor A
        scores_b: Performance scores from predictor B
        alpha: Significance level
        alternative: "two-sided", "greater", or "less"
        
    Returns:
        StatisticalComparison with test results
        
    Example:
        >>> # AUC scores across 10 CV folds
        >>> auc_a = [0.85, 0.82, 0.88, 0.81, 0.86, 0.83, 0.87, 0.84, 0.85, 0.86]
        >>> auc_b = [0.80, 0.78, 0.82, 0.79, 0.81, 0.77, 0.83, 0.80, 0.79, 0.81]
        >>> result = wilcoxon_signed_rank_test(auc_a, auc_b)
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    
    if len(scores_a) != len(scores_b):
        raise ValueError("Score arrays must have same length")
    
    if len(scores_a) < 6:
        logger.warning("Wilcoxon test may be unreliable with n < 6")
    
    # Calculate differences
    differences = scores_a - scores_b
    
    # Perform test
    try:
        stat_result = stats.wilcoxon(
            scores_a, scores_b,
            alternative=alternative,
            zero_method='wilcox',  # Exclude zeros
        )
        statistic = stat_result.statistic
        p_value = stat_result.pvalue
    except ValueError:
        # All differences are zero
        statistic = 0.0
        p_value = 1.0
    
    # Effect size: matched-pairs rank-biserial correlation
    n = len(differences)
    r = 1 - (2 * statistic) / (n * (n + 1) / 2) if n > 0 else 0
    
    # Confidence interval for median difference
    median_diff = np.median(differences)
    
    return StatisticalComparison(
        predictor_a="Predictor_A",
        predictor_b="Predictor_B",
        test_name="Wilcoxon Signed-Rank Test",
        statistic=float(statistic),
        p_value=float(p_value),
        significant=p_value < alpha,
        effect_size=float(r),
        alpha=alpha,
        notes=[f"Median difference: {median_diff:.4f}"],
    )


def paired_t_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> StatisticalComparison:
    """
    Paired t-test for comparing predictor scores.
    
    Assumes differences are normally distributed. Consider
    Wilcoxon test for non-normal data or small samples.
    
    Args:
        scores_a: Performance scores from predictor A
        scores_b: Performance scores from predictor B
        alpha: Significance level
        alternative: "two-sided", "greater", or "less"
        
    Returns:
        StatisticalComparison with test results
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    
    if len(scores_a) != len(scores_b):
        raise ValueError("Score arrays must have same length")
    
    # Perform paired t-test
    stat_result = stats.ttest_rel(scores_a, scores_b, alternative=alternative)
    
    # Effect size: Cohen's d for paired samples
    differences = scores_a - scores_b
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0
    
    # Confidence interval
    n = len(differences)
    se = std_diff / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    ci = (mean_diff - t_crit * se, mean_diff + t_crit * se)
    
    notes = []
    
    # Check normality assumption
    if n >= 8:
        _, norm_p = stats.shapiro(differences)
        if norm_p < 0.05:
            notes.append("Warning: Differences may not be normally distributed "
                        f"(Shapiro-Wilk p={norm_p:.4f}). Consider Wilcoxon test.")
    
    return StatisticalComparison(
        predictor_a="Predictor_A",
        predictor_b="Predictor_B",
        test_name="Paired t-test",
        statistic=float(stat_result.statistic),
        p_value=float(stat_result.pvalue),
        significant=stat_result.pvalue < alpha,
        effect_size=float(cohens_d),
        confidence_interval=ci,
        alpha=alpha,
        notes=notes,
    )


def delong_test(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
) -> StatisticalComparison:
    """
    DeLong test for comparing two correlated AUC values.
    
    Tests whether two classifiers have significantly different
    AUC-ROC values when evaluated on the same test set.
    
    Based on DeLong et al. (1988) "Comparing the areas under two
    or more correlated receiver operating characteristic curves:
    a nonparametric approach."
    
    Args:
        y_true: True binary labels
        scores_a: Prediction scores from classifier A
        scores_b: Prediction scores from classifier B
        alpha: Significance level
        
    Returns:
        StatisticalComparison with test results
    """
    y_true = np.asarray(y_true, dtype=bool)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    
    # Separate positive and negative samples
    pos_indices = np.where(y_true)[0]
    neg_indices = np.where(~y_true)[0]
    
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)
    
    if n_pos == 0 or n_neg == 0:
        return StatisticalComparison(
            predictor_a="Predictor_A",
            predictor_b="Predictor_B",
            test_name="DeLong Test",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            notes=["Cannot perform test: need both positive and negative samples"],
        )
    
    # Calculate placement values (Mann-Whitney U statistic basis)
    def compute_auc_and_placements(scores):
        pos_scores = scores[pos_indices]
        neg_scores = scores[neg_indices]
        
        # Placements for positive samples
        v10 = np.zeros(n_pos)
        for i, pos_score in enumerate(pos_scores):
            v10[i] = np.mean((pos_score > neg_scores).astype(float) + 
                            0.5 * (pos_score == neg_scores).astype(float))
        
        # Placements for negative samples
        v01 = np.zeros(n_neg)
        for j, neg_score in enumerate(neg_scores):
            v01[j] = np.mean((neg_score < pos_scores).astype(float) + 
                            0.5 * (neg_score == pos_scores).astype(float))
        
        auc = np.mean(v10)
        return auc, v10, v01
    
    auc_a, v10_a, v01_a = compute_auc_and_placements(scores_a)
    auc_b, v10_b, v01_b = compute_auc_and_placements(scores_b)
    
    # Covariance matrix
    s10_a = np.var(v10_a, ddof=1) if n_pos > 1 else 0
    s01_a = np.var(v01_a, ddof=1) if n_neg > 1 else 0
    s10_b = np.var(v10_b, ddof=1) if n_pos > 1 else 0
    s01_b = np.var(v01_b, ddof=1) if n_neg > 1 else 0
    s10_ab = np.cov(v10_a, v10_b, ddof=1)[0, 1] if n_pos > 1 else 0
    s01_ab = np.cov(v01_a, v01_b, ddof=1)[0, 1] if n_neg > 1 else 0
    
    var_a = s10_a / n_pos + s01_a / n_neg
    var_b = s10_b / n_pos + s01_b / n_neg
    cov_ab = s10_ab / n_pos + s01_ab / n_neg
    
    var_diff = var_a + var_b - 2 * cov_ab
    
    if var_diff <= 0:
        return StatisticalComparison(
            predictor_a="Predictor_A",
            predictor_b="Predictor_B",
            test_name="DeLong Test",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            notes=["Variance estimate is non-positive"],
        )
    
    # Z-statistic
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    # Confidence interval for AUC difference
    se = np.sqrt(var_diff)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    auc_diff = auc_a - auc_b
    ci = (auc_diff - z_crit * se, auc_diff + z_crit * se)
    
    return StatisticalComparison(
        predictor_a="Predictor_A",
        predictor_b="Predictor_B",
        test_name="DeLong Test",
        statistic=float(z),
        p_value=float(p_value),
        significant=p_value < alpha,
        effect_size=float(auc_diff),
        confidence_interval=ci,
        alpha=alpha,
        notes=[f"AUC_A: {auc_a:.4f}, AUC_B: {auc_b:.4f}"],
    )


# =============================================================================
# Multiple Comparison Tests
# =============================================================================

def friedman_test(
    scores_matrix: np.ndarray,
    predictor_names: list[str],
    alpha: float = 0.05,
) -> MultipleComparisonResult:
    """
    Friedman test for comparing multiple classifiers.
    
    Non-parametric omnibus test for comparing k classifiers across
    n datasets/folds. Use as preliminary test before post-hoc comparisons.
    
    Args:
        scores_matrix: Shape (n_datasets, n_predictors) performance scores
        predictor_names: Names of predictors (columns)
        alpha: Significance level
        
    Returns:
        MultipleComparisonResult with rankings and pairwise comparisons
    """
    scores_matrix = np.asarray(scores_matrix, dtype=float)
    n_datasets, n_predictors = scores_matrix.shape
    
    if len(predictor_names) != n_predictors:
        raise ValueError("Number of names must match number of predictors")
    
    if n_datasets < 2:
        raise ValueError("Need at least 2 datasets for Friedman test")
    
    # Rank predictors within each dataset (1 = best)
    ranks = np.zeros_like(scores_matrix)
    for i in range(n_datasets):
        # Descending ranks (higher score = rank 1)
        ranks[i] = stats.rankdata(-scores_matrix[i])
    
    # Mean ranks
    mean_ranks = np.mean(ranks, axis=0)
    
    # Friedman statistic
    stat_result = stats.friedmanchisquare(*[scores_matrix[:, j] for j in range(n_predictors)])
    
    # Rankings sorted by mean rank
    ranking_indices = np.argsort(mean_ranks)
    rankings = [(predictor_names[i], mean_ranks[i]) for i in ranking_indices]
    
    # Post-hoc Nemenyi test if significant
    pairwise = []
    critical_difference = None
    
    if stat_result.pvalue < alpha:
        # Nemenyi critical difference
        q_alpha = _nemenyi_critical_value(n_predictors, alpha)
        critical_difference = q_alpha * np.sqrt(n_predictors * (n_predictors + 1) / (6 * n_datasets))
        
        # Pairwise comparisons
        for i in range(n_predictors):
            for j in range(i + 1, n_predictors):
                rank_diff = abs(mean_ranks[i] - mean_ranks[j])
                sig = rank_diff > critical_difference
                
                pairwise.append(StatisticalComparison(
                    predictor_a=predictor_names[i],
                    predictor_b=predictor_names[j],
                    test_name="Nemenyi Post-hoc",
                    statistic=float(rank_diff),
                    p_value=_nemenyi_pvalue(rank_diff, n_predictors, n_datasets),
                    significant=sig,
                    alpha=alpha,
                ))
    
    return MultipleComparisonResult(
        test_name="Friedman Test with Nemenyi Post-hoc",
        overall_statistic=float(stat_result.statistic),
        overall_p_value=float(stat_result.pvalue),
        overall_significant=stat_result.pvalue < alpha,
        pairwise_comparisons=pairwise,
        rankings=rankings,
        critical_difference=critical_difference,
        alpha=alpha,
    )


def _nemenyi_critical_value(k: int, alpha: float = 0.05) -> float:
    """Get Nemenyi critical value q_α for k classifiers."""
    # Approximate values from studentized range distribution
    # More accurate: use scipy.stats.studentized_range
    q_values_005 = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
    }
    q_values_010 = {
        2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459,
        6: 2.589, 7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920,
    }
    
    table = q_values_005 if alpha <= 0.05 else q_values_010
    return table.get(k, 3.0 + 0.1 * (k - 10))  # Approximate for k > 10


def _nemenyi_pvalue(rank_diff: float, k: int, n: int) -> float:
    """Approximate p-value for Nemenyi test."""
    # Standard error of rank difference
    se = np.sqrt(k * (k + 1) / (6 * n))
    z = rank_diff / se
    
    # Two-tailed p-value
    return 2 * (1 - stats.norm.cdf(abs(z)))


# =============================================================================
# Multiple Testing Correction
# =============================================================================

def correct_pvalues(
    p_values: Sequence[float],
    method: str = "holm",
) -> tuple[np.ndarray, float]:
    """
    Apply multiple testing correction to p-values.
    
    Args:
        p_values: Raw p-values
        method: "bonferroni", "holm", "fdr_bh" (Benjamini-Hochberg)
        
    Returns:
        Tuple of (corrected p-values, corrected alpha threshold)
    """
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    
    if method == "bonferroni":
        corrected = np.minimum(p_values * n, 1.0)
        alpha_corrected = 0.05 / n
        
    elif method == "holm":
        # Holm-Bonferroni step-down
        order = np.argsort(p_values)
        corrected = np.zeros(n)
        
        for i, idx in enumerate(order):
            corrected[idx] = p_values[idx] * (n - i)
        
        # Ensure monotonicity
        cummax = np.maximum.accumulate(corrected[order])
        corrected[order] = np.minimum(cummax, 1.0)
        alpha_corrected = 0.05 / n  # Most conservative step
        
    elif method == "fdr_bh":
        # Benjamini-Hochberg FDR
        order = np.argsort(p_values)
        corrected = np.zeros(n)
        
        for i, idx in enumerate(order):
            corrected[idx] = p_values[idx] * n / (i + 1)
        
        # Ensure monotonicity (cumulative minimum from end)
        cummin = np.minimum.accumulate(corrected[order[::-1]])[::-1]
        corrected[order] = np.minimum(cummin, 1.0)
        alpha_corrected = 0.05  # FDR controls expected proportion, not FWER
        
    else:
        raise ValueError(f"Unknown correction method: {method}")
    
    return corrected, alpha_corrected


# =============================================================================
# Convenience Functions
# =============================================================================

def compare_benchmark_results(
    results: list[BenchmarkResult],
    metric: str = "mcc",
    alpha: float = 0.05,
) -> MultipleComparisonResult:
    """
    Compare multiple predictors from benchmark results.
    
    Uses Friedman test with Nemenyi post-hoc when per-sample
    results are available and >= 3 predictors, otherwise
    uses simple ranking or pairwise comparison.
    
    Args:
        results: List of BenchmarkResult objects
        metric: Metric to compare ("mcc", "sensitivity", "auc_roc", etc.)
        alpha: Significance level
        
    Returns:
        MultipleComparisonResult
    """
    predictor_names = [r.predictor_name for r in results]
    n_predictors = len(results)
    
    # Extract metric values
    metric_values = []
    for r in results:
        m = r.classification_metrics
        if metric == "mcc":
            metric_values.append(m.mcc)
        elif metric == "sensitivity":
            metric_values.append(m.sensitivity)
        elif metric == "specificity":
            metric_values.append(m.specificity)
        elif metric == "f1_score":
            metric_values.append(m.f1_score)
        elif metric == "auc_roc":
            metric_values.append(m.auc_roc or 0)
        elif metric == "accuracy":
            metric_values.append(m.accuracy)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    # For 2 predictors, use pairwise comparison
    if n_predictors == 2:
        metric_values = np.array(metric_values)
        ranking_indices = np.argsort(-metric_values)
        rankings = [(predictor_names[i], float(metric_values[i])) for i in ranking_indices]
        
        # Pairwise comparison if possible
        pairwise = []
        if (len(results[0].per_sample_results) > 0 and 
            len(results[1].per_sample_results) > 0):
            try:
                comp = pairwise_compare_predictors(results[0], results[1], test="mcnemar", alpha=alpha)
                pairwise.append(comp)
            except Exception:
                pass
        
        return MultipleComparisonResult(
            test_name="Pairwise Comparison (2 predictors)",
            overall_statistic=0.0,
            overall_p_value=1.0,
            overall_significant=len(pairwise) > 0 and pairwise[0].significant,
            pairwise_comparisons=pairwise,
            rankings=rankings,
            alpha=alpha,
        )
    
    # Check for per-sample results for proper statistical testing
    has_per_sample = all(len(r.per_sample_results) > 0 for r in results)
    
    if has_per_sample and len(results[0].per_sample_results) >= 10 and n_predictors >= 3:
        # Build score matrix from per-sample predictions
        # Group by sample ID for paired comparison
        sample_ids = [s['id'] for s in results[0].per_sample_results]
        n_samples = len(sample_ids)
        
        scores_matrix = np.zeros((n_samples, n_predictors))
        
        for j, r in enumerate(results):
            sample_scores = {s['id']: s.get('score', 0.5) for s in r.per_sample_results}
            for i, sid in enumerate(sample_ids):
                scores_matrix[i, j] = sample_scores.get(sid, 0.5)
        
        return friedman_test(scores_matrix, predictor_names, alpha)
    
    # Fallback: simple ranking based on aggregate metrics
    metric_values = np.array(metric_values)
    ranking_indices = np.argsort(-metric_values)  # Descending
    rankings = [(predictor_names[i], float(metric_values[i])) for i in ranking_indices]
    
    return MultipleComparisonResult(
        test_name="Aggregate Metric Comparison",
        overall_statistic=0.0,
        overall_p_value=1.0,
        overall_significant=False,
        pairwise_comparisons=[],
        rankings=rankings,
        alpha=alpha,
    )


def pairwise_compare_predictors(
    result_a: BenchmarkResult,
    result_b: BenchmarkResult,
    test: str = "mcnemar",
    alpha: float = 0.05,
) -> StatisticalComparison:
    """
    Pairwise comparison of two predictors.
    
    Args:
        result_a: Results from predictor A
        result_b: Results from predictor B
        test: Test to use ("mcnemar", "wilcoxon", "delong")
        alpha: Significance level
        
    Returns:
        StatisticalComparison
    """
    if test == "mcnemar":
        # Need per-sample predictions
        if not result_a.per_sample_results or not result_b.per_sample_results:
            raise ValueError("McNemar test requires per-sample results")
        
        # Build error arrays
        samples_a = {s['id']: s['predicted'] for s in result_a.per_sample_results}
        samples_b = {s['id']: s['predicted'] for s in result_b.per_sample_results}
        
        common_ids = set(samples_a.keys()) & set(samples_b.keys())
        
        errors_a = []
        errors_b = []
        
        for r in result_a.per_sample_results:
            if r['id'] in common_ids:
                errors_a.append(r['predicted'] != r['true_label'])
                
        for r in result_b.per_sample_results:
            if r['id'] in common_ids:
                errors_b.append(r['predicted'] != r['true_label'])
        
        comparison = mcnemar_test(errors_a, errors_b, alpha)
        
    elif test == "wilcoxon":
        # Compare score distributions
        scores_a = [s.get('score', 0.5) for s in result_a.per_sample_results]
        scores_b = [s.get('score', 0.5) for s in result_b.per_sample_results]
        
        comparison = wilcoxon_signed_rank_test(scores_a, scores_b, alpha)
        
    else:
        raise ValueError(f"Unknown test: {test}")
    
    comparison.predictor_a = result_a.predictor_name
    comparison.predictor_b = result_b.predictor_name
    
    return comparison
