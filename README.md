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
- **Polymorph classification**: Steric zipper classes, cross-β geometry, fold type
- **Structural classification**: Cross-β parallel/antiparallel, β-solenoid, cross-α polymorphs
- **Rigorous benchmarking**: Validated against WALTZ-DB, Cross-Beta DB, AmyPro
- **Extensible architecture**: Easy integration of new predictors

## Polymorph Classification

AmyloidBench predicts the structural class of amyloidogenic sequences:

```python
from amyloidbench.classification import predict_polymorph

# Aβ42 peptide
result = predict_polymorph("DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA")

print(f"Fold type: {result.predicted_fold.value}")           # steric_zipper, beta_arcade, etc.
print(f"Geometry: {result.predicted_geometry.value}")        # parallel_in_register, antiparallel
print(f"Steric zipper: {result.steric_zipper_class.value}")  # class_1 through class_8
print(f"Confidence: {result.confidence:.0%}")
```

Supported classifications:

| Classification | Classes | Basis |
|----------------|---------|-------|
| **Steric Zipper** | 8 classes | Strand/sheet orientation, face packing (Eisenberg system) |
| **Cross-β Geometry** | Parallel in-register, antiparallel, out-of-register | Strand arrangement |
| **Fold Type** | Steric zipper, β-arcade, β-solenoid, Greek key | Higher-order topology |

## Benchmarking Framework

AmyloidBench includes comprehensive benchmarking against experimental databases:

```python
from amyloidbench.benchmark import BenchmarkRunner, load_benchmark_dataset

# Load synthetic or experimental datasets
dataset = load_benchmark_dataset("synthetic")  # Built-in test data
# dataset = load_benchmark_dataset("waltz", path="waltz_data.csv")  # Real data

# Create benchmark runner
runner = BenchmarkRunner()
runner.add_predictor("Aggrescan3D")
runner.add_predictor("FoldAmyloid")
runner.add_predictor("FallbackPredictor")

# Run evaluation
results = runner.run(dataset)

# Generate comparison report
report = runner.generate_report(results)
print(report.summary())
```

### Supported Databases

| Database | Description | Data Type |
|----------|-------------|-----------|
| **WALTZ-DB** | Hexapeptide aggregation (ThT, EM validated) | Binary labels |
| **Cross-Beta DB** | Structural amyloid atlas (cryo-EM, X-ray) | APR regions |
| **AmyPro** | Amyloid-forming proteins | Full-length sequences |
| **Reference** | Curated gold-standard sequences | APR + polymorph |
| **Synthetic** | Built-in canonical sequences | Testing/development |

### Evaluation Metrics

- **Binary**: Sensitivity, Specificity, Precision, F1, MCC, AUC-ROC
- **Per-residue**: SOV score, Jaccard index, boundary accuracy
- **Statistical**: Bootstrap confidence intervals, optimal threshold

### Reference Datasets

AmyloidBench includes curated gold-standard sequences for rigorous benchmarking:

```python
from amyloidbench.benchmark import (
    create_comprehensive_dataset,
    create_canonical_peptide_dataset,
    get_canonical_peptides,
    get_sequence_by_name,
)

# Full reference dataset (canonical + disease + functional + negative)
dataset = create_comprehensive_dataset()
print(f"Total: {len(dataset)} sequences ({dataset.n_positive} positive, {dataset.n_negative} negative)")

# Access canonical peptides with full annotations
peptides = get_canonical_peptides()
for p in peptides[:3]:
    print(f"{p.name}: {p.sequence} (Class {p.zipper_class}, PDB: {p.pdb_ids})")

# Quick sequence lookup
gnnqqny = get_sequence_by_name("GNNQQNY")  # Returns "GNNQQNY"
```

**Reference Dataset Categories:**
- **Canonical Peptides** (12 sequences): GNNQQNY, KLVFFA, VQIVYK, NFGAIL with PDB structures
- **Disease Proteins** (6 sequences): Aβ42, α-synuclein, tau, huntingtin with mapped APRs
- **Functional Amyloids** (4 sequences): Curli, HET-s, Pmel17 with fold annotations
- **Negative Controls** (5 sequences): Ubiquitin, lysozyme, GFP as non-amyloid examples

### Statistical Comparison

Rigorous statistical tests for predictor comparison:

```python
from amyloidbench.benchmark import (
    compare_benchmark_results,
    mcnemar_test,
    wilcoxon_signed_rank_test,
    friedman_test,
    correct_pvalues,
)

# Compare multiple predictors
comparison = compare_benchmark_results(results, metric="mcc")
print(comparison.summary())

# Pairwise McNemar test (paired classification comparison)
errors_a = [pred != true for pred, true in zip(preds_a, labels)]
errors_b = [pred != true for pred, true in zip(preds_b, labels)]
result = mcnemar_test(errors_a, errors_b)
print(f"McNemar p-value: {result.p_value:.4f}")

# Multiple testing correction
p_values = [0.01, 0.04, 0.06, 0.10]
corrected, alpha_corrected = correct_pvalues(p_values, method="holm")
```

**Available Statistical Tests:**
- **McNemar's Test**: Paired classification comparison
- **Wilcoxon Signed-Rank**: Paired score comparison (non-parametric)
- **DeLong Test**: AUC-ROC comparison on same dataset
- **Friedman Test**: Multiple classifier comparison with Nemenyi post-hoc
- **Correction Methods**: Bonferroni, Holm-Bonferroni, Benjamini-Hochberg FDR

### Polymorph-Aware Benchmarking

Evaluate predictor performance stratified by structural polymorph type:

```python
from amyloidbench.benchmark import (
    quick_polymorph_benchmark,
    annotate_dataset_with_polymorphs,
    analyze_regions_with_polymorphs,
)

# Quick polymorph-stratified benchmark
result = quick_polymorph_benchmark(predictor, dataset)
print(result.summary())

# Per-fold performance analysis
for fold_name, metrics in result.per_fold_metrics.items():
    print(f"{fold_name}: Sens={metrics.sensitivity:.3f}, MCC={metrics.mcc:.3f}")

# Analyze APR regions with polymorph context
regions = predictor.predict(sequence).aggregation_prone_regions
analyses = analyze_regions_with_polymorphs(sequence, regions)
for analysis in analyses:
    print(f"Region {analysis.region.start}-{analysis.region.end}: "
          f"{analysis.predicted_fold.value} ({analysis.confidence:.0%})")
```

## Visualization and Reporting

AmyloidBench generates publication-quality figures and interactive HTML reports:

```python
from amyloidbench.visualization import (
    plot_score_profile,
    plot_multi_predictor_profile,
    generate_sequence_report,
)

# Plot per-residue scores
fig = plot_score_profile(
    result.per_residue_scores.scores,
    sequence=protein.sequence,
    threshold=0.5,
    predictor_name="Aggrescan3D"
)
fig.savefig("score_profile.png", dpi=300)

# Generate comprehensive HTML report
html = generate_sequence_report(
    sequence=protein.sequence,
    sequence_id=protein.id,
    prediction_results={"Aggrescan3D": result},
    consensus_result=consensus_data,
    polymorph_result=polymorph_data
)
```

### Visualization Types

| Type | Function | Description |
|------|----------|-------------|
| **Score Profile** | `plot_score_profile()` | Per-residue scores with APR highlighting |
| **Multi-Predictor** | `plot_multi_predictor_profile()` | Overlay comparison of predictors |
| **Agreement Heatmap** | `plot_agreement_heatmap()` | Pairwise predictor correlation |
| **Benchmark Comparison** | `plot_benchmark_comparison()` | Performance metric bar charts |
| **Polymorph Distribution** | `plot_polymorph_probabilities()` | Fold type probability bars |
| **Region Diagram** | `plot_region_diagram()` | APR region schematic |
| **HTML Report** | `generate_sequence_report()` | Self-contained analysis report |

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
| FallbackPredictor | Biophysics+ML | Hydrophobicity, β-propensity, gatekeepers | ✅ Implemented |
| WALTZ | Sequence | Hexapeptide PSSM | ✅ Web + local approx |
| PASTA 2.0 | Threading | Pairwise energy + structural templates | ✅ Web + local approx |
| APPNN | ML | Neural network on sequence features | 📋 Planned |
| AmylPred2 | Consensus | 11-predictor meta-method | 📋 Planned |
| TAPASS | Sequence | Aggregation-prone motifs | 📋 Planned |

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
- [x] **Phase 3**: Feature extraction, Fallback predictor, Consensus engine
- [x] **Phase 4**: Structural classification (polymorph prediction)
- [x] **Phase 5**: Benchmarking framework (datasets, metrics, runner)
- [x] **Phase 6**: Visualization and reporting

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
