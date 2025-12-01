"""
Re-implementations of published amyloidogenicity algorithms.

This submodule contains our implementations of established prediction
methods where standalone versions are unavailable. Each implementation
follows the original publication methodology to ensure scientific validity
and comparability with published benchmarks.

Re-implementation strategy:
1. Identify the core algorithm from the original publication
2. Extract any published propensity scales or trained parameters
3. Implement the scoring function with appropriate windowing
4. Validate against the original web server outputs
5. Document any deviations or ambiguities in the original method

Planned re-implementations:

**FoldAmyloid** (Garbuzynskiy et al., 2010)
Based on the observation that amyloidogenic regions have increased
expected contact density. The algorithm uses average packing density
and contact number scales derived from globular protein structures.

**AGGRESCAN sequence** (Conchillo-Solé et al., 2007)
Uses an experimentally-derived amino acid aggregation propensity scale
from bacterial inclusion body formation studies. Identifies hot spots
where consecutive residues exceed an aggregation threshold.

**TANGO** (Fernandez-Escamilla et al., 2004)
Statistical mechanics approach to β-aggregation based on secondary
structure propensity, hydrophobicity, and electrostatic interactions.
More complex to re-implement due to proprietary energy function.

References:
    Garbuzynskiy et al. (2010) Bioinformatics 26:326-332
    Conchillo-Solé et al. (2007) BMC Bioinformatics 8:65
    Fernandez-Escamilla et al. (2004) Nat Biotechnol 22:1302-1306
"""

from .foldamyloid import (
    FoldAmyloidPredictor,
    predict_with_foldamyloid,
    get_packing_density_profile,
    PACKING_DENSITY,
    EXPECTED_CONTACTS,
)

__all__ = [
    "FoldAmyloidPredictor",
    "predict_with_foldamyloid",
    "get_packing_density_profile",
    "PACKING_DENSITY",
    "EXPECTED_CONTACTS",
]
