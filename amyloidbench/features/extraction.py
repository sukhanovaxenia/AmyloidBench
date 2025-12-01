"""
Feature extraction for amyloidogenicity prediction.

This module provides comprehensive feature extraction from protein sequences,
capturing the biophysical and contextual properties that determine aggregation
propensity. Features are organized into functional categories reflecting the
underlying biology of amyloid formation.

Biological Framework
--------------------
Amyloid formation requires sequences that can:
1. Adopt extended β-strand conformations
2. Form stable intermolecular hydrogen bonds
3. Pack hydrophobic side chains in steric zipper interfaces
4. Avoid disruption by charged residues or β-breakers

The features extracted here quantify these properties at multiple scales:
- **Residue-level**: Individual amino acid propensities
- **Window-level**: Local sequence context (5-15 residues)
- **Global**: Full-sequence composition and patterns

Feature Categories
------------------
1. **Hydrophobicity**: Core driver of aggregation through hydrophobic effect
2. **Secondary structure propensity**: β-sheet vs α-helix vs coil tendency
3. **Charge and polarity**: Electrostatic effects on aggregation
4. **Amino acid composition**: Global and local sequence makeup
5. **Sequence patterns**: Motifs, repeats, low complexity regions
6. **Disorder propensity**: Intrinsically disordered regions often harbor APRs
7. **Gatekeeper analysis**: Protective charged/proline residues

References
----------
- Fernandez-Escamilla et al. (2004) - TANGO algorithm
- Tartaglia et al. (2008) - Zyggregator features
- Goldschmidt et al. (2010) - 3D profile method
- Familia et al. (2015) - Feature analysis for aggregation
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Amino Acid Property Scales
# =============================================================================

# Kyte-Doolittle hydrophobicity scale
# Positive = hydrophobic, Negative = hydrophilic
HYDROPHOBICITY_KD = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}

# Eisenberg consensus hydrophobicity
HYDROPHOBICITY_EISENBERG = {
    'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
    'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
    'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
    'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08,
}

# Wimley-White octanol scale (membrane insertion)
HYDROPHOBICITY_WW = {
    'A': 0.50, 'R': 1.81, 'N': 0.85, 'D': 3.64, 'C': -0.02,
    'Q': 0.77, 'E': 3.63, 'G': 1.15, 'H': 2.33, 'I': -1.12,
    'L': -1.25, 'K': 2.80, 'M': -0.67, 'F': -1.71, 'P': 0.14,
    'S': 0.46, 'T': 0.25, 'W': -2.09, 'Y': -0.71, 'V': -0.46,
}

# Chou-Fasman β-sheet propensity
# Values > 1.0 indicate β-sheet preference
BETA_PROPENSITY_CF = {
    'A': 0.83, 'R': 0.93, 'N': 0.89, 'D': 0.54, 'C': 1.19,
    'Q': 1.10, 'E': 0.37, 'G': 0.75, 'H': 0.87, 'I': 1.60,
    'L': 1.30, 'K': 0.74, 'M': 1.05, 'F': 1.38, 'P': 0.55,
    'S': 0.75, 'T': 1.19, 'W': 1.37, 'Y': 1.47, 'V': 1.70,
}

# Levitt β-sheet propensity (alternative scale)
BETA_PROPENSITY_LEVITT = {
    'A': 0.90, 'R': 0.99, 'N': 0.76, 'D': 0.72, 'C': 0.94,
    'Q': 0.80, 'E': 0.75, 'G': 0.92, 'H': 1.08, 'I': 1.45,
    'L': 1.02, 'K': 0.77, 'M': 0.97, 'F': 1.28, 'P': 0.64,
    'S': 0.95, 'T': 1.21, 'W': 1.14, 'Y': 1.29, 'V': 1.49,
}

# α-helix propensity (Chou-Fasman)
ALPHA_PROPENSITY_CF = {
    'A': 1.42, 'R': 0.98, 'N': 0.67, 'D': 1.01, 'C': 0.70,
    'Q': 1.11, 'E': 1.51, 'G': 0.57, 'H': 1.00, 'I': 1.08,
    'L': 1.21, 'K': 1.16, 'M': 1.45, 'F': 1.13, 'P': 0.57,
    'S': 0.77, 'T': 0.83, 'W': 1.08, 'Y': 0.69, 'V': 1.06,
}

# Turn propensity (Chou-Fasman)
TURN_PROPENSITY = {
    'A': 0.66, 'R': 0.95, 'N': 1.56, 'D': 1.46, 'C': 1.19,
    'Q': 0.98, 'E': 0.74, 'G': 1.56, 'H': 0.95, 'I': 0.47,
    'L': 0.59, 'K': 1.01, 'M': 0.60, 'F': 0.60, 'P': 1.52,
    'S': 1.43, 'T': 0.96, 'W': 0.96, 'Y': 1.14, 'V': 0.50,
}

# Side chain volume (Å³) - important for steric zipper packing
SIDE_CHAIN_VOLUME = {
    'A': 88.6, 'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5,
    'Q': 143.8, 'E': 138.4, 'G': 60.1, 'H': 153.2, 'I': 166.7,
    'L': 166.7, 'K': 168.6, 'M': 162.9, 'F': 189.9, 'P': 112.7,
    'S': 89.0, 'T': 116.1, 'W': 227.8, 'Y': 193.6, 'V': 140.0,
}

# Disorder propensity (from IUPred-like analysis)
# Positive = disorder promoting, Negative = order promoting
DISORDER_PROPENSITY = {
    'A': 0.06, 'R': -0.18, 'N': 0.01, 'D': 0.19, 'C': -0.20,
    'Q': 0.32, 'E': 0.30, 'G': 0.17, 'H': -0.08, 'I': -0.49,
    'L': -0.34, 'K': 0.26, 'M': -0.19, 'F': -0.42, 'P': 0.41,
    'S': 0.34, 'T': 0.01, 'W': -0.49, 'Y': -0.25, 'V': -0.38,
}

# Aggregation propensity (TANGO-derived approximation)
# Higher = more aggregation-prone
AGGREGATION_PROPENSITY = {
    'A': 0.3, 'R': -1.0, 'N': -0.5, 'D': -0.8, 'C': 0.2,
    'Q': -0.3, 'E': -0.7, 'G': -0.2, 'H': -0.4, 'I': 1.4,
    'L': 1.1, 'K': -1.0, 'M': 0.8, 'F': 1.5, 'P': -1.5,
    'S': -0.1, 'T': 0.1, 'W': 1.2, 'Y': 1.0, 'V': 1.3,
}

# Charge at pH 7
CHARGE_PH7 = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
    'Q': 0, 'E': -1, 'G': 0, 'H': 0.1, 'I': 0,
    'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
    'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0,
}


# =============================================================================
# Feature Extraction Classes
# =============================================================================

@dataclass
class SequenceFeatures:
    """
    Container for extracted sequence features.
    
    Organizes features by category for interpretability and allows
    selective feature usage in downstream models.
    
    Attributes:
        sequence: Original sequence
        length: Sequence length
        
        hydrophobicity: Hydrophobicity-related features
        secondary_structure: Secondary structure propensity features
        charge: Charge and polarity features
        composition: Amino acid composition features
        patterns: Sequence pattern features
        disorder: Disorder propensity features
        aggregation: Direct aggregation propensity features
        
        per_residue: Dict of per-residue feature arrays
        global_features: Dict of single-value global features
        window_features: Dict of window-based feature arrays
    """
    sequence: str
    length: int
    
    # Feature dictionaries
    per_residue: dict = field(default_factory=dict)
    global_features: dict = field(default_factory=dict)
    window_features: dict = field(default_factory=dict)
    
    def to_vector(self, feature_names: Optional[list[str]] = None) -> np.ndarray:
        """
        Convert global features to a feature vector.
        
        Args:
            feature_names: Optional list of feature names to include.
                          If None, includes all global features.
                          
        Returns:
            1D numpy array of feature values
        """
        if feature_names is None:
            feature_names = sorted(self.global_features.keys())
        
        return np.array([self.global_features.get(name, 0.0) for name in feature_names])
    
    def get_per_residue_matrix(self, feature_names: Optional[list[str]] = None) -> np.ndarray:
        """
        Get per-residue features as a matrix.
        
        Args:
            feature_names: Optional list of feature names
            
        Returns:
            2D array of shape (length, n_features)
        """
        if feature_names is None:
            feature_names = sorted(self.per_residue.keys())
        
        matrix = np.zeros((self.length, len(feature_names)))
        for i, name in enumerate(feature_names):
            if name in self.per_residue:
                matrix[:, i] = self.per_residue[name]
        
        return matrix


class FeatureExtractor:
    """
    Comprehensive feature extraction for amyloidogenicity prediction.
    
    Extracts features at multiple scales:
    - Per-residue: Position-specific property values
    - Window-based: Local sequence context (sliding window statistics)
    - Global: Whole-sequence composition and properties
    
    The extracted features capture the key determinants of amyloid formation:
    1. Hydrophobicity drives the aggregation process
    2. β-sheet propensity enables cross-β structure
    3. Charge patterns (gatekeepers) modulate aggregation
    4. Sequence composition influences overall solubility
    5. Disorder propensity correlates with aggregation hotspots
    
    Usage:
        >>> extractor = FeatureExtractor()
        >>> features = extractor.extract(sequence)
        >>> print(features.global_features['mean_hydrophobicity'])
    """
    
    def __init__(
        self,
        window_size: int = 7,
        hydrophobicity_scale: str = "kyte_doolittle",
        include_dipeptides: bool = True,
    ):
        """
        Initialize feature extractor.
        
        Args:
            window_size: Size of sliding window for local features
            hydrophobicity_scale: Which hydrophobicity scale to use
                                 ('kyte_doolittle', 'eisenberg', 'wimley_white')
            include_dipeptides: Whether to include dipeptide frequencies
        """
        self.window_size = window_size
        self.include_dipeptides = include_dipeptides
        
        # Select hydrophobicity scale
        if hydrophobicity_scale == "kyte_doolittle":
            self.hydro_scale = HYDROPHOBICITY_KD
        elif hydrophobicity_scale == "eisenberg":
            self.hydro_scale = HYDROPHOBICITY_EISENBERG
        elif hydrophobicity_scale == "wimley_white":
            self.hydro_scale = HYDROPHOBICITY_WW
        else:
            raise ValueError(f"Unknown hydrophobicity scale: {hydrophobicity_scale}")
    
    def extract(self, sequence: str) -> SequenceFeatures:
        """
        Extract all features from a sequence.
        
        Args:
            sequence: Protein sequence (single-letter codes)
            
        Returns:
            SequenceFeatures object containing all extracted features
        """
        sequence = sequence.upper()
        features = SequenceFeatures(sequence=sequence, length=len(sequence))
        
        # Per-residue features
        self._extract_per_residue(sequence, features)
        
        # Window-based features
        self._extract_window_features(sequence, features)
        
        # Global features
        self._extract_global_features(sequence, features)
        
        # Pattern features
        self._extract_pattern_features(sequence, features)
        
        return features
    
    def _extract_per_residue(self, sequence: str, features: SequenceFeatures):
        """Extract per-residue property values."""
        n = len(sequence)
        
        # Initialize arrays
        hydro = np.zeros(n)
        beta_prop = np.zeros(n)
        alpha_prop = np.zeros(n)
        turn_prop = np.zeros(n)
        charge = np.zeros(n)
        volume = np.zeros(n)
        disorder = np.zeros(n)
        aggregation = np.zeros(n)
        
        for i, aa in enumerate(sequence):
            hydro[i] = self.hydro_scale.get(aa, 0.0)
            beta_prop[i] = BETA_PROPENSITY_CF.get(aa, 1.0)
            alpha_prop[i] = ALPHA_PROPENSITY_CF.get(aa, 1.0)
            turn_prop[i] = TURN_PROPENSITY.get(aa, 1.0)
            charge[i] = CHARGE_PH7.get(aa, 0)
            volume[i] = SIDE_CHAIN_VOLUME.get(aa, 100.0)
            disorder[i] = DISORDER_PROPENSITY.get(aa, 0.0)
            aggregation[i] = AGGREGATION_PROPENSITY.get(aa, 0.0)
        
        features.per_residue = {
            'hydrophobicity': hydro,
            'beta_propensity': beta_prop,
            'alpha_propensity': alpha_prop,
            'turn_propensity': turn_prop,
            'charge': charge,
            'volume': volume,
            'disorder': disorder,
            'aggregation': aggregation,
        }
    
    def _extract_window_features(self, sequence: str, features: SequenceFeatures):
        """Extract sliding window statistics."""
        n = len(sequence)
        w = self.window_size
        half_w = w // 2
        
        if n < w:
            # Sequence too short for window analysis
            return
        
        # Window-averaged values
        hydro_window = np.zeros(n)
        beta_window = np.zeros(n)
        charge_window = np.zeros(n)
        aggregation_window = np.zeros(n)
        
        # Window statistics
        hydro_std = np.zeros(n)
        
        for i in range(n):
            start = max(0, i - half_w)
            end = min(n, i + half_w + 1)
            
            window_seq = sequence[start:end]
            
            # Window hydrophobicity
            window_hydro = [self.hydro_scale.get(aa, 0.0) for aa in window_seq]
            hydro_window[i] = np.mean(window_hydro)
            hydro_std[i] = np.std(window_hydro) if len(window_hydro) > 1 else 0
            
            # Window β-propensity
            window_beta = [BETA_PROPENSITY_CF.get(aa, 1.0) for aa in window_seq]
            beta_window[i] = np.mean(window_beta)
            
            # Window charge
            window_charge = [CHARGE_PH7.get(aa, 0) for aa in window_seq]
            charge_window[i] = np.sum(window_charge)
            
            # Window aggregation
            window_agg = [AGGREGATION_PROPENSITY.get(aa, 0.0) for aa in window_seq]
            aggregation_window[i] = np.mean(window_agg)
        
        features.window_features = {
            'hydrophobicity_window': hydro_window,
            'hydrophobicity_std': hydro_std,
            'beta_propensity_window': beta_window,
            'net_charge_window': charge_window,
            'aggregation_window': aggregation_window,
        }
    
    def _extract_global_features(self, sequence: str, features: SequenceFeatures):
        """Extract global sequence features."""
        n = len(sequence)
        
        # Amino acid composition
        aa_counts = {}
        for aa in sequence:
            aa_counts[aa] = aa_counts.get(aa, 0) + 1
        
        # Standard amino acids
        standard_aa = 'ACDEFGHIKLMNPQRSTVWY'
        composition = {f'frac_{aa}': aa_counts.get(aa, 0) / n for aa in standard_aa}
        
        # Class-based composition
        hydrophobic = sum(aa_counts.get(aa, 0) for aa in 'VILFYWM') / n
        charged = sum(aa_counts.get(aa, 0) for aa in 'KRDE') / n
        polar = sum(aa_counts.get(aa, 0) for aa in 'STNQ') / n
        aromatic = sum(aa_counts.get(aa, 0) for aa in 'FYW') / n
        aliphatic = sum(aa_counts.get(aa, 0) for aa in 'AVIL') / n
        
        # Gatekeeper analysis
        gatekeepers = set('PKRED')
        gatekeeper_count = sum(1 for aa in sequence if aa in gatekeepers)
        gatekeeper_frac = gatekeeper_count / n
        
        # Mean properties
        mean_hydro = np.mean([self.hydro_scale.get(aa, 0) for aa in sequence])
        mean_beta = np.mean([BETA_PROPENSITY_CF.get(aa, 1.0) for aa in sequence])
        mean_alpha = np.mean([ALPHA_PROPENSITY_CF.get(aa, 1.0) for aa in sequence])
        mean_disorder = np.mean([DISORDER_PROPENSITY.get(aa, 0) for aa in sequence])
        mean_aggregation = np.mean([AGGREGATION_PROPENSITY.get(aa, 0) for aa in sequence])
        
        # Net charge
        net_charge = sum(CHARGE_PH7.get(aa, 0) for aa in sequence)
        charge_density = net_charge / n
        
        # Charge asymmetry (distribution of positive vs negative)
        pos_charges = [i for i, aa in enumerate(sequence) if CHARGE_PH7.get(aa, 0) > 0]
        neg_charges = [i for i, aa in enumerate(sequence) if CHARGE_PH7.get(aa, 0) < 0]
        
        if pos_charges and neg_charges:
            charge_asymmetry = abs(np.mean(pos_charges) - np.mean(neg_charges)) / n
        else:
            charge_asymmetry = 0.0
        
        # Isoelectric point approximation
        # pI ≈ 6 + (positive - negative) * 0.5 (rough approximation)
        n_pos = sum(1 for aa in sequence if aa in 'KRH')
        n_neg = sum(1 for aa in sequence if aa in 'DE')
        approx_pI = 6.0 + (n_pos - n_neg) * 0.3
        
        # β/α ratio (higher = more β-sheet prone)
        beta_alpha_ratio = mean_beta / mean_alpha if mean_alpha > 0 else mean_beta
        
        # Hydrophobic moment (amphipathicity)
        # Simplified calculation - real calculation requires structural context
        hydro_values = [self.hydro_scale.get(aa, 0) for aa in sequence]
        if len(hydro_values) >= 7:
            # Calculate for 7-residue windows assuming ideal helix (100° per residue)
            angles = np.arange(len(hydro_values)) * (100 * np.pi / 180)
            hm_x = np.sum(np.array(hydro_values) * np.cos(angles))
            hm_y = np.sum(np.array(hydro_values) * np.sin(angles))
            hydrophobic_moment = np.sqrt(hm_x**2 + hm_y**2) / n
        else:
            hydrophobic_moment = 0.0
        
        # Compile global features
        global_feats = {
            # Composition
            'length': n,
            'frac_hydrophobic': hydrophobic,
            'frac_charged': charged,
            'frac_polar': polar,
            'frac_aromatic': aromatic,
            'frac_aliphatic': aliphatic,
            'frac_gatekeeper': gatekeeper_frac,
            
            # Mean properties
            'mean_hydrophobicity': mean_hydro,
            'mean_beta_propensity': mean_beta,
            'mean_alpha_propensity': mean_alpha,
            'mean_disorder': mean_disorder,
            'mean_aggregation': mean_aggregation,
            
            # Charge features
            'net_charge': net_charge,
            'charge_density': charge_density,
            'charge_asymmetry': charge_asymmetry,
            'approx_pI': approx_pI,
            
            # Derived features
            'beta_alpha_ratio': beta_alpha_ratio,
            'hydrophobic_moment': hydrophobic_moment,
        }
        
        # Add individual AA fractions
        global_feats.update(composition)
        
        # Dipeptide frequencies
        if self.include_dipeptides and n >= 2:
            dipeptides = {}
            for i in range(n - 1):
                dp = sequence[i:i+2]
                dipeptides[dp] = dipeptides.get(dp, 0) + 1
            
            # Normalize by (n-1) to get frequencies
            for dp, count in dipeptides.items():
                global_feats[f'dp_{dp}'] = count / (n - 1)
        
        features.global_features = global_feats
    
    def _extract_pattern_features(self, sequence: str, features: SequenceFeatures):
        """Extract sequence pattern features."""
        n = len(sequence)
        
        # Hydrophobic stretches (≥4 consecutive hydrophobic residues)
        hydrophobic = set('VILFYWM')
        hydro_stretches = self._find_stretches(sequence, hydrophobic, min_length=4)
        
        # Longest hydrophobic stretch
        max_hydro_stretch = max((len(s) for s in hydro_stretches), default=0)
        total_hydro_stretch = sum(len(s) for s in hydro_stretches)
        
        # β-sheet prone stretches (V, I, F, Y, W, L, T, C)
        beta_residues = set('VIFYWLTC')
        beta_stretches = self._find_stretches(sequence, beta_residues, min_length=4)
        max_beta_stretch = max((len(s) for s in beta_stretches), default=0)
        
        # Charged stretches (potential aggregation disruptors)
        charged = set('KRDE')
        charged_stretches = self._find_stretches(sequence, charged, min_length=3)
        max_charged_stretch = max((len(s) for s in charged_stretches), default=0)
        
        # Proline content (β-breaker)
        proline_count = sequence.count('P')
        
        # Glycine content (flexibility)
        glycine_count = sequence.count('G')
        
        # QN-rich regions (polyglutamine/asparagine - associated with some amyloids)
        qn_count = sequence.count('Q') + sequence.count('N')
        qn_fraction = qn_count / n
        
        # Aromatic clusters (important for π-stacking in amyloid)
        aromatic = set('FYW')
        aromatic_stretches = self._find_stretches(sequence, aromatic, min_length=2)
        
        # Low complexity regions (Shannon entropy)
        entropy = self._calculate_entropy(sequence)
        
        # Sequence repetitiveness
        repeat_score = self._calculate_repeat_score(sequence)
        
        # Add to global features
        features.global_features.update({
            'max_hydrophobic_stretch': max_hydro_stretch,
            'total_hydrophobic_stretch': total_hydro_stretch,
            'frac_in_hydro_stretch': total_hydro_stretch / n,
            'max_beta_stretch': max_beta_stretch,
            'max_charged_stretch': max_charged_stretch,
            'proline_count': proline_count,
            'glycine_count': glycine_count,
            'qn_fraction': qn_fraction,
            'n_aromatic_clusters': len(aromatic_stretches),
            'sequence_entropy': entropy,
            'repeat_score': repeat_score,
        })
    
    def _find_stretches(
        self, 
        sequence: str, 
        residue_set: set, 
        min_length: int = 3
    ) -> list[str]:
        """Find stretches of consecutive residues from a given set."""
        stretches = []
        current_stretch = ""
        
        for aa in sequence:
            if aa in residue_set:
                current_stretch += aa
            else:
                if len(current_stretch) >= min_length:
                    stretches.append(current_stretch)
                current_stretch = ""
        
        if len(current_stretch) >= min_length:
            stretches.append(current_stretch)
        
        return stretches
    
    def _calculate_entropy(self, sequence: str, window_size: int = 12) -> float:
        """
        Calculate Shannon entropy of sequence.
        
        Lower entropy indicates more repetitive/low-complexity sequence.
        """
        if len(sequence) < 2:
            return 0.0
        
        # Count amino acids
        aa_counts = {}
        for aa in sequence:
            aa_counts[aa] = aa_counts.get(aa, 0) + 1
        
        # Calculate entropy
        entropy = 0.0
        n = len(sequence)
        for count in aa_counts.values():
            if count > 0:
                p = count / n
                entropy -= p * np.log2(p)
        
        # Normalize by maximum possible entropy (log2(20))
        max_entropy = np.log2(min(len(aa_counts), 20))
        if max_entropy > 0:
            entropy /= max_entropy
        
        return entropy
    
    def _calculate_repeat_score(self, sequence: str) -> float:
        """
        Calculate a simple repeat/pattern score.
        
        Looks for di-, tri-, and tetrapeptide repeats.
        """
        n = len(sequence)
        if n < 4:
            return 0.0
        
        repeat_score = 0.0
        
        # Check for dipeptide repeats (e.g., VGVGVG)
        for i in range(n - 3):
            dipep = sequence[i:i+2]
            if sequence[i+2:i+4] == dipep:
                repeat_score += 1
        
        # Check for tripeptide repeats
        for i in range(n - 5):
            tripep = sequence[i:i+3]
            if sequence[i+3:i+6] == tripep:
                repeat_score += 1.5
        
        # Normalize by sequence length
        return repeat_score / n


# =============================================================================
# Convenience functions
# =============================================================================

def extract_features(sequence: str, window_size: int = 7) -> SequenceFeatures:
    """
    Quick feature extraction without creating explicit extractor object.
    
    Args:
        sequence: Protein sequence
        window_size: Window size for local features
        
    Returns:
        SequenceFeatures object
    """
    extractor = FeatureExtractor(window_size=window_size)
    return extractor.extract(sequence)


def get_feature_names(include_dipeptides: bool = False) -> list[str]:
    """
    Get list of standard global feature names.
    
    Args:
        include_dipeptides: Whether to include dipeptide features
        
    Returns:
        List of feature names
    """
    basic_features = [
        'length',
        'frac_hydrophobic', 'frac_charged', 'frac_polar',
        'frac_aromatic', 'frac_aliphatic', 'frac_gatekeeper',
        'mean_hydrophobicity', 'mean_beta_propensity', 'mean_alpha_propensity',
        'mean_disorder', 'mean_aggregation',
        'net_charge', 'charge_density', 'charge_asymmetry', 'approx_pI',
        'beta_alpha_ratio', 'hydrophobic_moment',
        'max_hydrophobic_stretch', 'total_hydrophobic_stretch',
        'frac_in_hydro_stretch', 'max_beta_stretch', 'max_charged_stretch',
        'proline_count', 'glycine_count', 'qn_fraction',
        'n_aromatic_clusters', 'sequence_entropy', 'repeat_score',
    ]
    
    # Add individual AA fractions
    for aa in 'ACDEFGHIKLMNPQRSTVWY':
        basic_features.append(f'frac_{aa}')
    
    return basic_features


def calculate_aggregation_score(
    sequence: str,
    weights: Optional[dict[str, float]] = None,
) -> float:
    """
    Calculate a simple aggregation propensity score.
    
    This combines multiple features into a single score using
    empirically-derived weights. Higher scores indicate greater
    aggregation propensity.
    
    Args:
        sequence: Protein sequence
        weights: Optional custom feature weights
        
    Returns:
        Aggregation propensity score (typically 0-1)
    """
    features = extract_features(sequence)
    
    # Default weights based on literature
    if weights is None:
        weights = {
            'mean_hydrophobicity': 0.25,
            'mean_beta_propensity': 0.20,
            'frac_hydrophobic': 0.15,
            'frac_gatekeeper': -0.20,  # Negative - gatekeepers reduce aggregation
            'mean_aggregation': 0.15,
            'charge_density': -0.05,   # Charge reduces aggregation
        }
    
    score = 0.0
    for feature_name, weight in weights.items():
        value = features.global_features.get(feature_name, 0)
        score += weight * value
    
    # Normalize to 0-1 range using sigmoid
    score = 1 / (1 + np.exp(-score))
    
    return score
