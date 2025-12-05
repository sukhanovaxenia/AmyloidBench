# Amyloid Predictor Implementation Strategy

## Executive Summary

This document analyzes ten amyloid prediction tools and provides implementation strategies for AmyloidBench. The goal is to create faithful algorithmic reimplementations that don't rely on external web services, ensuring reproducibility and offline capability.

## Tool Analysis Matrix

| Tool | Algorithm Type | Implementation Feasibility | Strategy |
|------|---------------|---------------------------|----------|
| WALTZ | PSSM + physicochemical | ✅ High | Reimplementation |
| PASTA 2.0 | Pairwise energy | ✅ High | Reimplementation |
| PATH | Threading + ML | ⚠️ Medium | Simplified reimplementation |
| FoldAmyloid | Packing density scale | ✅ Complete | Already implemented |
| ArchCandy | β-arch scoring | ✅ High | Reimplementation |
| Cross-Beta | ML classifier | ⚠️ Medium | Feature-based approximation |
| AggreProt | Deep neural network | ❌ Low | Web API wrapper only |
| APPNN | R neural network | ✅ High | R wrapper + Python port |
| TANGO | Statistical mechanics | ✅ High | Reimplementation |
| ReRF-Pred | Random Forest | ⚠️ Medium | Feature-based reimplementation |

---

## Detailed Tool Analysis

### 1. WALTZ (Maurer-Stroh et al., 2010)

**Algorithm Basis:**
- Position-specific scoring matrix (PSSM) for hexapeptides
- Three scoring components:
  1. PSSM derived from AmylHex dataset
  2. Physicochemical properties (hydrophobicity, β-propensity)
  3. Position-specific pseudoenergy from GNNQQNY structure

**Key Parameters:**
- Window size: 6 amino acids
- Trained on ~280 hexapeptides (original), expanded to 1400+ in WALTZ-DB 2.0
- Threshold: Position-dependent based on score distribution

**Implementation Strategy:**
```
1. Derive PSSM from WALTZ-DB amino acid frequencies
2. Add physicochemical property terms (hydrophobicity, β-propensity)
3. Sliding window scoring with position weighting
```

**Expected Performance:** Sensitivity ~72%, Specificity ~79% (original publication)

---

### 2. PASTA 2.0 (Walsh et al., 2014)

**Algorithm Basis:**
- Pairwise β-strand pairing energy
- Derived from statistics of β-bridges in globular proteins (DSSP)
- Evaluates cross-β pairing stability

**Key Parameters:**
- Energy threshold: -5 kcal/mol (default for 85% specificity)
- Window size: Variable (typically 7 residues)
- Orientations: Parallel and antiparallel

**Algorithm Details:**
```
For each segment pair (i, j):
  E_pairing = Σ E(aa_i, aa_j) for residue pairs
  
Energy matrix from β-sheet statistics:
- Favorable: hydrophobic-hydrophobic pairs (V-V, I-I, F-F)
- Unfavorable: charged pairs (K-K, E-E)
```

**Implementation Strategy:**
- Derive energy matrix from published β-sheet contact statistics
- Implement parallel and antiparallel scoring
- Include secondary structure reinforcement

---

### 3. PATH (Kotulska et al., 2020)

**Algorithm Basis:**
- Structure-based threading
- Machine learning on FoldX energies from steric zipper templates
- Random Forest classifier

**Key Components:**
1. Template library of known amyloid structures (8 steric zipper classes)
2. FoldX threading to evaluate compatibility
3. Feature extraction: stability, geometry, volume
4. Random Forest classification

**Implementation Strategy (Simplified):**
- Use pseudo-energy scoring based on zipper class compatibility
- Incorporate volume/packing considerations
- Feature-based classification without full FoldX

---

### 4. FoldAmyloid (Garbuzynskiy et al., 2010) ✅ IMPLEMENTED

**Status:** Complete implementation in `reimplemented/foldamyloid.py`

**Algorithm:**
- Expected packing density scale
- Sliding window averaging
- Threshold: 21.4 for amyloidogenicity

---

### 5. ArchCandy (Ahmed et al., 2015)

**Algorithm Basis:**
- β-arch detection (β-strand-loop-β-strand motif)
- Based on the observation that most amyloid fibrils form β-arcade structures
- Scores multiple factors for arch formation

**Scoring Components:**
1. **Strand propensity** - β-sheet forming potential
2. **Loop compatibility** - Length and composition of connecting loop
3. **Packing score** - Side chain complementarity
4. **Compactness score** - Overall structural compatibility

**Key Parameters:**
- Minimum arch length: 8-15 residues
- Score threshold: 0.40 (ambiguous), 0.57 (significant)
- Loop length: 2-5 residues typical

**Implementation Strategy:**
```python
def score_beta_arch(segment):
    strand1 = segment[:strand_len]
    loop = segment[strand_len:strand_len+loop_len]
    strand2 = segment[strand_len+loop_len:]
    
    score = (
        w1 * beta_propensity(strand1, strand2) +
        w2 * loop_score(loop) +
        w3 * packing_complementarity(strand1, strand2) +
        w4 * hydrophobic_moment(segment)
    )
    return score
```

---

### 6. Cross-Beta Predictor

**Algorithm Basis:**
- Machine learning on Cross-Beta DB
- Features: physicochemical properties, composition, predicted features
- Classifier: Various (SVM, Random Forest, etc.)

**Implementation Strategy:**
- Feature extraction based on published descriptors
- Train simple classifier on available data
- Or use rule-based approximation from key features

---

### 7. AggreProt (Planas-Iglesias et al., 2024)

**Algorithm Basis:**
- Deep neural network ensemble
- Two models: Sequential (36 atomic features) and Static (18 WaltzDB features)
- Trained on WaltzDB hexapeptides

**Challenge:** Model weights not publicly available

**Implementation Strategy:**
- Web API wrapper for online predictions
- Feature-based approximation using the 18 static features
- Cannot fully reimplment without model weights

---

### 8. APPNN (Família et al., 2015)

**Algorithm Basis:**
- Feed-forward neural network
- Recursive feature selection from AAindex properties
- 14 key physicochemical features identified

**Key Features (from publication):**
1. α-helical propensity (Chou-Fasman)
2. β-sheet propensity
3. Hydrophobicity (multiple scales)
4. Solvent accessibility
5. Molecular weight/volume

**R Package Structure:**
```R
library(appnn)
result <- appnn(sequence)
# Returns: overall score, per-AA scores, hotspots
```

**Implementation Strategy:**
- R wrapper via rpy2 (if R available)
- Pure Python port using identified features + simple NN

---

### 9. TANGO (Fernandez-Escamilla et al., 2004)

**Algorithm Basis:**
- Statistical mechanical model for β-aggregation
- Partition function approach considering multiple conformational states
- Key assumption: aggregating core is fully buried

**Energy Terms:**
1. **β-sheet propensity** (ΔG_β)
2. **Hydrophobic burial** (ΔG_burial)
3. **Electrostatic term** (ΔG_charge)
4. **Solvation penalty** (ΔG_solvation)

**Key Equations:**
```
ΔG_agg = ΔG_β + ΔG_burial + ΔG_charge + ΔG_solvation

Aggregation % = 100 × exp(-ΔG_agg / RT) / Z
where Z = partition function
```

**Parameters:**
- Temperature: 298.15 K (default)
- pH: 7.0 (affects charge states)
- Window: 5-7 residues

---

### 10. ReRF-Pred / RFAmyloid

**Algorithm Basis:**
- Random Forest classifier
- Features: SVMProt 188-D + pse-in-one features
- Composition and physicochemical properties

**Feature Categories:**
1. Amino acid composition (20D)
2. Dipeptide composition (400D)
3. Physicochemical property composition
4. Autocorrelation features
5. Pseudo amino acid composition

**Implementation Strategy:**
- Extract key features (simplified set)
- Train lightweight Random Forest
- Or use ensemble voting from other predictors

---

## Implementation Priority

### Phase 1: High-Value Reimplementations
1. **WALTZ** - Most cited, well-documented PSSM
2. **TANGO** - Widely used, clear statistical mechanics model
3. **PASTA** - Pairwise energy approach complements others
4. **ArchCandy** - Unique β-arch perspective

### Phase 2: Wrappers and Approximations
5. **APPNN** - R wrapper + Python feature approximation
6. **PATH** - Simplified pseudo-threading
7. **Cross-Beta** - Feature-based approximation

### Phase 3: External Integration
8. **AggreProt** - Web API wrapper only
9. **ReRF-Pred** - Ensemble approach

---

## Feature Extraction Requirements

### Common Amino Acid Scales

All predictors use overlapping physicochemical scales:

| Property | Scale Source | Used By |
|----------|-------------|---------|
| Hydrophobicity | Kyte-Doolittle | All |
| β-propensity | Chou-Fasman | TANGO, ArchCandy, APPNN |
| α-propensity | Chou-Fasman | TANGO, APPNN |
| Volume | Zamyatnin | FoldAmyloid, PATH |
| Polarity | Grantham | WALTZ, TANGO |
| Charge | Henderson-Hasselbalch | TANGO, PASTA |
| Aromaticity | Count(FYW) | ArchCandy, PASTA |

### Derived Features

| Feature | Calculation | Used By |
|---------|-------------|---------|
| Hydrophobic moment | μH = |Σ H_i × exp(iδ)| | ArchCandy, Zyggregator |
| Net charge | n_pos - n_neg | TANGO, PASTA |
| Gatekeeper density | count(PKRED) / length | All |
| Aromatic clustering | count adjacent FYW | PASTA, ArchCandy |

---

## Validation Strategy

### Benchmark Datasets
1. **WALTZ-DB 2.0** - 1416 hexapeptides (515 amyloid, 901 non-amyloid)
2. **AmyPro** - Full-length proteins with APR annotations
3. **Cross-Beta DB** - Structurally validated amyloids

### Metrics
- Sensitivity (TPR)
- Specificity (TNR)
- MCC (Matthew's Correlation Coefficient)
- AUC-ROC
- Per-residue overlap with known APRs

### Cross-Validation
- Leave-one-out for hexapeptides
- 5-fold CV for proteins
- External validation on held-out set

---

## Consensus Integration

The final AmyloidBench meta-predictor will combine all available predictors:

```python
def consensus_prediction(sequence, predictors):
    results = {p.name: p.predict(sequence) for p in predictors}
    
    # Weighted voting
    weights = {
        'WALTZ': 1.0,      # High specificity for true amyloids
        'TANGO': 1.0,      # Good general performance
        'PASTA': 0.9,      # Strong for parallel arrangements
        'FoldAmyloid': 0.8, # Good for packing-driven aggregation
        'ArchCandy': 0.9,  # Unique β-arch perspective
        'APPNN': 0.8,      # Neural network complement
    }
    
    # Score aggregation
    consensus_score = weighted_average(results, weights)
    
    # Region consensus
    consensus_regions = intersect_regions(results, min_overlap=3)
    
    return ConsensusResult(score, regions)
```

---

## Next Steps

1. Implement WALTZ predictor with PSSM
2. Implement TANGO statistical mechanics model
3. Implement ArchCandy β-arch detector
4. Add APPNN R wrapper and Python approximation
5. Update fallback predictor with multi-scale consensus
6. Comprehensive benchmarking against reference datasets
