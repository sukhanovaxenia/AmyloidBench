# AmyloidBench

**Comprehensive consensus meta-predictor and benchmark suite for protein amyloidogenicity**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

AmyloidBench provides a unified framework for predicting amyloidogenic regions in proteins by integrating multiple established prediction algorithms with proper benchmarking against curated experimental databases.

### Biological Background

Amyloid fibrils are characterized by the cross-β architecture—extended β-sheets running perpendicular to the fibril axis with hydrogen bonds parallel to it. This structure can be adopted by a remarkably diverse set of protein sequences, leading to both:

- **Pathological amyloids**: Aggregates implicated in Alzheimer's disease (Aβ, tau), Parkinson's disease (α-synuclein), prion diseases (PrP), and systemic amyloidoses (transthyretin, immunoglobulin light chains)
- **Functional amyloids**: Physiological structures serving roles in biofilm formation (curli, chaplins), hormone storage (PMEL17), and epigenetic inheritance (yeast prions)

The sequence determinants of amyloidogenicity include:
- **High β-sheet propensity**: Enrichment of V, I, L, F, Y residues
- **Hydrophobic character**: Favorable for steric zipper formation
- **Absence of gatekeepers**: P, K, R, E, D residues inhibit aggregation
- **Aromatic interactions**: F, Y, W contribute to fibril stability

### Key Features

- **Consensus prediction**: Meta-scoring from 11+ established tools
- **Per-residue profiles**: Detailed aggregation propensity across sequences
- **Structural classification**: Cross-β parallel/antiparallel, β-solenoid, cross-α polymorphs
- **Rigorous benchmarking**: Validated against WALTZ-DB, Cross-Beta DB, AmyPro
- **Extensible architecture**: Easy integration of new predictors

## Installation

### From source (development)

```bash
git clone https://github.com/arriam/amyloidbench.git
cd amyloidbench
pip install -e ".[dev]"
```

### Dependencies

Core dependencies are installed automatically. For full functionality:

```bash
# Structure-based prediction (optional)
pip install -e ".[structural]"

# R integration for original AGGRESCAN (optional)
pip install -e ".[r-integration]"

# Web automation for remote predictors
playwright install chromium
```

## Quick Start

### Python API

```python
from amyloidbench import predict, ProteinRecord

# Create a protein record
protein = ProteinRecord(
    id="PrP_human",
    sequence="MANLGCWMLVLFVATWSDLGLCKKRPKPGGWNTGGSRYPGQGSPGGNRYPPQGGGGWGQPHGGGWGQPHGGGWGQPHGGGWGQPHGGGWGQGGGTHSQWNKPSKPKTNMKHMAGAAAAGAVVGGLGGYMLGSAMSRPIIHFGSDYEDRYYRENMHRYPNQVYYRPMDEYSNQNNFVHDCVNITIKQHTVTTTTKGENFTETDVKMMERVVEQMCITQYERESQAYYQRGSSMVLFSSPPVILLISFLIFLIVG",
    organism="Homo sapiens"
)

# Run prediction with all available predictors
result = predict(protein)

print(f"Amyloidogenic: {result.consensus_is_amyloidogenic}")
print(f"Agreement: {result.n_predictors_agree_positive}/{result.n_predictors_total}")

# Examine individual predictor results
for name, pred_result in result.individual_results.items():
    print(f"\n{name}:")
    for region in pred_result.predicted_regions:
        print(f"  APR {region.start}-{region.end}: {region.sequence} (score: {region.score:.2f})")
```

### Using specific predictors

```python
from amyloidbench.predictors.local import Aggrescan3DPredictor
from amyloidbench import ProteinRecord
from pathlib import Path

# Structure-based prediction with Aggrescan3D
predictor = Aggrescan3DPredictor()

protein = ProteinRecord(
    id="lysozyme",
    sequence="KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
    structure_path=Path("1lyz.pdb")  # Optional PDB structure
)

result = predictor.predict(protein)

# Access per-residue scores
scores = result.per_residue_scores
print(f"Score range: {min(scores.scores):.2f} to {max(scores.scores):.2f}")

# Identified aggregation-prone regions
for apr in result.predicted_regions:
    print(f"APR: {apr.sequence} at positions {apr.start}-{apr.end}")
```

### Command Line Interface

```bash
# Predict amyloidogenicity for sequences in a FASTA file
amyloidbench predict proteins.fasta -o results/

# Use specific predictor with structure
amyloidbench predict query.fasta -p Aggrescan3D -s structure.pdb

# List available predictors
amyloidbench list-predictors --detailed

# Get information about a specific predictor
amyloidbench info Aggrescan3D

# Validate sequences before prediction
amyloidbench validate-sequence -f proteins.fasta
```

## Architecture

```
amyloidbench/
├── core/                    # Fundamental data structures
│   ├── models.py           # Pydantic models for sequences, regions, results
│   ├── sequence.py         # Sequence parsing, validation, composition
│   └── structure.py        # PDB/mmCIF handling, SASA calculation
├── predictors/
│   ├── base.py             # Abstract predictor interface
│   ├── local/              # Standalone predictors
│   │   ├── aggrescan3d.py  # Aggrescan3D wrapper
│   │   └── reimplemented/  # Our implementations of published methods
│   └── remote/             # Web-based predictors (Playwright)
├── features/               # Feature extraction for ML predictor
├── classification/         # Binary and polymorph classification
├── benchmark/              # Database loaders.and metrics
├── visualization/          # Score profiles and comparison plots
└── cli/                    # Command-line interface
```

## Supported Predictors

| Predictor | Type | Methodology | Status |
|-----------|------|-------------|--------|
| Aggrescan3D | Structure-based | AGGRESCAN scale + solvent accessibility | ✅ Implemented |
| FoldAmyloid | Sequence | Expected packing density | ✅ Re-implemented |
| WALTZ | Sequence | Hexapeptide PSSM | ✅ Web + local approx |
| PASTA 2.0 | Threading | Pairwise energy + structural templates | ✅ Web + local approx |
| APPNN | ML | Neural network on sequence features | 🔄 Phase 3 |
| AmylPred2 | Consensus | 11-predictor meta-method | 🔄 Phase 3 |
| TAPASS | Sequence | Aggregation-prone motifs | 🔄 Phase 3 |
| PATH | Threading | Structural compatibility scoring | 📋 Planned |
| AGGRESCAN | Sequence | Experimental aggregation scale | 📋 Planned |
| Fallback | ML+Biophysical | Our contextual predictor | 🔄 Phase 3 |

## Consensus Methods

AmyloidBench implements multiple consensus strategies:

### 1. Majority Voting (AmylPred2-style)
Simple threshold-based voting where a residue is predicted amyloidogenic if ≥k/n predictors agree.

### 2. Weighted Logistic Regression (MetAmyl-style)
Statistical weighting via likelihood maximization, automatically selecting complementary predictors and outputting calibrated probabilities.

### 3. Stacked Ensemble (AMYPred-FRL-style)
Multi-algorithm ML approach combining predictor outputs with additional sequence-derived features.

## Benchmarking

```python
from amyloidbench.benchmark import load_database, evaluate_predictor

# Load benchmark database
waltz_db = load_database("waltz-db")

# Evaluate predictor performance
metrics = evaluate_predictor(
    predictor=Aggrescan3DPredictor(),
    dataset=waltz_db,
    cv_folds=5
)

print(f"Sensitivity: {metrics['sensitivity']:.3f} ± {metrics['sensitivity_std']:.3f}")
print(f"Specificity: {metrics['specificity']:.3f} ± {metrics['specificity_std']:.3f}")
print(f"MCC: {metrics['mcc']:.3f} ± {metrics['mcc_std']:.3f}")
print(f"AUC: {metrics['auc']:.3f} ± {metrics['auc_std']:.3f}")
```

## Development Roadmap

- [x] **Phase 1**: Core foundation (sequence/structure handling, predictor interface)
- [x] **Phase 2**: FoldAmyloid re-implementation + web predictor infrastructure
- [ ] **Phase 3**: Fallback predictor with biophysical and contextual features
- [ ] **Phase 4**: Consensus engine with multiple aggregation strategies
- [ ] **Phase 5**: Structural classification (polymorph prediction)
- [ ] **Phase 6**: Benchmarking framework and visualization

## Scientific Validation

AmyloidBench validation strategy:

1. **Cross-predictor agreement**: Regions identified by multiple methods are higher confidence
2. **Database validation**: Performance metrics against experimentally-validated datasets
3. **Literature comparison**: Reproduce published benchmark results
4. **Case studies**: Known amyloidogenic proteins (Aβ, PrP, α-syn) as positive controls

## Citation

If you use AmyloidBench in your research, please cite:

```bibtex
@software{amyloidbench,
  title = {AmyloidBench: Consensus Meta-Predictor for Protein Amyloidogenicity},
  author = {Xenia},
  year = {2024},
  institution = {ARRIAM},
  url = {https://github.com/arriam/amyloidbench}
}
```

Please also cite the original publications for individual predictors used.

## Key References

- **Aggrescan3D**: Kuriata et al. (2019) Bioinformatics 35:3834-3835
- **FoldAmyloid**: Garbuzynskiy et al. (2010) Bioinformatics 26:326-332
- **WALTZ**: Maurer-Stroh et al. (2010) Nat Methods 7:237-242
- **PASTA 2.0**: Walsh et al. (2014) Nucleic Acids Res 42:W301-W307
- **AmylPred2**: Tsolis et al. (2013) PLoS ONE 8:e54تت175
- **MetAmyl**: Emily et al. (2013) BMC Bioinformatics 14:81
- **Cross-Beta DB**: Gonay et al. (2024) Nucleic Acids Res (database issue)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

For questions or collaborations, contact: xenia@arriam.ru
