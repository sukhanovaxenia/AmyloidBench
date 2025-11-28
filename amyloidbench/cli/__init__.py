"""
Command-line interface for AmyloidBench.

The CLI provides accessible entry points for all major functionalities
without requiring Python programming knowledge. It's designed for:

1. **Batch prediction**: Process multiple sequences from FASTA files
2. **Benchmarking**: Evaluate predictors against reference datasets
3. **Exploration**: Query predictor capabilities and parameters

Usage patterns:
    amyloidbench predict sequences.fasta -o results/
    amyloidbench benchmark --database waltz-db
    amyloidbench list-predictors --detailed
"""

from .main import cli, main

__all__ = ["cli", "main"]
