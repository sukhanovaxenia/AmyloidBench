# Conda Installation Guide for AmyloidBench

AmyloidBench can be installed via conda, which is the recommended approach for bioinformatics workflows due to superior dependency resolution and environment isolation.

## Quick Installation

### Option 1: From Bioconda (Recommended for Production)

Once the package is accepted to Bioconda:

```bash
# Add required channels (order matters!)
conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge
conda config --set channel_priority strict

# Create dedicated environment and install
conda create -n amyloidbench python=3.11 amyloidbench
conda activate amyloidbench

# Verify installation
amyloidbench --version
```

### Option 2: Local Build (Development)

For development or before Bioconda submission:

```bash
# Clone the repository
git clone https://github.com/sukhanovaxenia/AmyloidBench.git
cd AmyloidBench

# Build conda package locally
conda build conda-recipe

# Install from local build
conda install --use-local amyloidbench

# Or create environment directly
conda create -n amyloidbench-dev python=3.11
conda activate amyloidbench-dev
pip install -e .
```

### Option 3: Environment File

Create a reproducible environment:

```bash
# Using the provided environment file
conda env create -f environment.yml
conda activate amyloidbench
```

## Environment File

Save as `environment.yml`:

```yaml
name: amyloidbench
channels:
  - conda-forge
  - bioconda
  - defaults
dependencies:
  - python>=3.10
  - numpy>=1.24
  - pandas>=2.0
  - scipy>=1.10
  - biopython>=1.81
  - scikit-learn>=1.3
  - xgboost>=2.0
  - biotite>=0.38
  - matplotlib>=3.7
  - seaborn>=0.12
  - click>=8.1
  - rich>=13.0
  - pydantic>=2.5
  - tqdm>=4.66
  - joblib>=1.3
  - pip
  - pip:
    - amyloidbench
```

## Submitting to Bioconda

### Prerequisites

1. Fork the [bioconda-recipes](https://github.com/bioconda/bioconda-recipes) repository
2. Ensure your package is on PyPI or has a stable GitHub release

### Submission Steps

```bash
# 1. Clone your fork
git clone https://github.com/YOUR_USERNAME/bioconda-recipes.git
cd bioconda-recipes

# 2. Create recipe branch
git checkout -b amyloidbench

# 3. Create recipe directory
mkdir -p recipes/amyloidbench

# 4. Copy and adapt the recipe
cp /path/to/AmyloidBench/conda-recipe/meta.yaml recipes/amyloidbench/

# 5. Update source to use GitHub release
# Edit recipes/amyloidbench/meta.yaml:
#   source:
#     url: https://github.com/sukhanovaxenia/AmyloidBench/archive/v{{ version }}.tar.gz
#     sha256: <calculate with: curl -sL <url> | sha256sum>

# 6. Test locally
conda build recipes/amyloidbench

# 7. Commit and push
git add recipes/amyloidbench
git commit -m "Add amyloidbench v0.1.0"
git push origin amyloidbench

# 8. Open Pull Request to bioconda/bioconda-recipes
```

### Bioconda Requirements Checklist

- [ ] Package builds successfully on Linux and macOS
- [ ] All tests pass (`conda build --test`)
- [ ] License file is included
- [ ] Recipe follows [Bioconda guidelines](https://bioconda.github.io/contributor/guidelines.html)
- [ ] No proprietary dependencies
- [ ] Version pinning follows conda-forge practices

## Integration with Workflow Managers

### Snakemake

```yaml
# Snakefile
rule predict_amyloid:
    input:
        fasta = "proteins.fasta"
    output:
        results = "results/predictions.json"
    conda:
        "envs/amyloidbench.yaml"
    shell:
        "amyloidbench predict {input.fasta} -o results/ --format json"
```

### Nextflow

```groovy
// nextflow.config
process {
    withName: 'AMYLOID_PREDICT' {
        conda = 'bioconda::amyloidbench=0.1.0'
    }
}

// main.nf
process AMYLOID_PREDICT {
    input:
    path fasta

    output:
    path "predictions.json"

    script:
    """
    amyloidbench predict ${fasta} -o . --format json
    """
}
```

### Galaxy Tool Wrapper

AmyloidBench can be wrapped for Galaxy. See `galaxy/` directory for tool XML definitions.

## Troubleshooting

### Common Issues

**1. Channel conflicts**

```bash
# Reset channels and use strict priority
conda config --remove-key channels
conda config --add channels conda-forge
conda config --add channels bioconda
conda config --set channel_priority strict
```

**2. Slow solving**

```bash
# Use mamba for faster dependency resolution
conda install -n base -c conda-forge mamba
mamba create -n amyloidbench amyloidbench
```

**3. Missing libstdc++**

```bash
# On some Linux systems
conda install -c conda-forge libstdcxx-ng
```

### Minimal Installation

For systems with limited resources:

```bash
conda create -n amyloidbench-minimal python=3.11 \
    numpy scipy biopython scikit-learn click rich pydantic

conda activate amyloidbench-minimal
pip install amyloidbench --no-deps
```

## Version Compatibility Matrix

| AmyloidBench | Python | NumPy | SciPy | Biopython |
|--------------|--------|-------|-------|-----------|
| 0.1.x        | ≥3.10  | ≥1.24 | ≥1.10 | ≥1.81     |

## Citation

If you use AmyloidBench installed via conda/Bioconda, please cite both the software and Bioconda:

```bibtex
@software{amyloidbench,
  title = {AmyloidBench: Consensus Meta-Predictor for Protein Amyloidogenicity},
  author = {Sukhanova, Xenia},
  year = {2024},
  url = {https://github.com/sukhanovaxenia/AmyloidBench}
}

@article{bioconda,
  title = {Bioconda: sustainable and comprehensive software distribution for the life sciences},
  author = {Grüning, Björn and others},
  journal = {Nature Methods},
  volume = {15},
  pages = {475--476},
  year = {2018},
  doi = {10.1038/s41592-018-0046-7}
}
```
