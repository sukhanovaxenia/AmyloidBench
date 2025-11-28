#!/usr/bin/env python3
"""
AmyloidBench Example: Analyzing Amyloidogenic Proteins

This script demonstrates the core functionality of AmyloidBench by
analyzing well-characterized amyloidogenic proteins from human disease.
The examples are chosen to illustrate different aspects of amyloid biology.

Run with: python examples/basic_usage.py
"""

from pathlib import Path

# Import AmyloidBench components
from amyloidbench import (
    ProteinRecord,
    predict,
    get_predictor,
    list_predictors,
)
from amyloidbench.core.sequence import (
    calculate_class_composition,
    find_motifs,
    extract_region_context,
)
from amyloidbench.core.models import AmyloidPolymorph, Region


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def analyze_amyloid_beta():
    """
    Analyze Amyloid-β (1-42) - the archetypal pathological amyloid.
    
    Aβ42 is derived from amyloid precursor protein (APP) and forms the
    senile plaques characteristic of Alzheimer's disease. Its aggregation
    propensity is driven by:
    - Central hydrophobic cluster (KLVFF, residues 17-21)
    - C-terminal hydrophobic tail (residues 29-42)
    - Lack of protective charged residues in critical regions
    """
    print_header("Amyloid-β (1-42) Analysis")
    
    # The sequence from human APP
    sequence = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"
    
    protein = ProteinRecord(
        id="Abeta42",
        name="Amyloid-beta peptide (1-42)",
        sequence=sequence,
        organism="Homo sapiens",
        is_known_amyloid=True,
        known_polymorph=AmyloidPolymorph.CROSS_BETA_PARALLEL,
        known_amyloid_regions=[
            Region(start=16, end=21, sequence="KLVFF", annotation="Central hydrophobic"),
            Region(start=29, end=42, sequence="GAIIGLMVGGVVIA", annotation="C-terminal core"),
        ],
        source_database="PDB (multiple structures)",
    )
    
    print(f"\nProtein: {protein.name}")
    print(f"Sequence: {protein.sequence}")
    print(f"Length: {protein.sequence_length} residues")
    
    # Analyze amino acid composition
    composition = calculate_class_composition(protein.sequence)
    print("\nAmino acid class composition:")
    print(f"  Amyloidogenic (V,I,L,F,Y,W,M): {composition['amyloidogenic']*100:.1f}%")
    print(f"  Gatekeepers (P,K,R,E,D): {composition['gatekeepers']*100:.1f}%")
    print(f"  Hydrophobic aromatic (F,Y,W): {composition['hydrophobic_aromatic']*100:.1f}%")
    
    # Find hydrophobic stretches
    hydrophobic_regions = find_motifs(protein.sequence, r"[VILFYWM]{4,}")
    print(f"\nHydrophobic stretches (≥4 residues): {len(hydrophobic_regions)}")
    for region in hydrophobic_regions:
        print(f"  {region.start}-{region.end}: {region.sequence}")
    
    # Run prediction
    print("\nRunning prediction...")
    predictor = get_predictor("Aggrescan3D")
    result = predictor.predict(protein)
    
    if result.success:
        print(f"\nPrediction successful (runtime: {result.runtime_seconds:.2f}s)")
        print(f"Is amyloidogenic: {result.is_amyloidogenic}")
        
        if result.predicted_regions:
            print(f"\nPredicted APRs ({len(result.predicted_regions)}):")
            for apr in result.predicted_regions:
                print(f"  {apr.start}-{apr.end}: {apr.sequence} (score: {apr.score:.3f})")
                
                # Analyze context around APR
                context = extract_region_context(protein.sequence, apr, flank_size=5)
                print(f"    Context: ...{context['upstream']}[{context['region']}]{context['downstream']}...")
                print(f"    Flanking gatekeepers: {context['upstream_gatekeepers']} upstream, "
                      f"{context['downstream_gatekeepers']} downstream")
        
        if result.per_residue_scores:
            scores = result.per_residue_scores.scores
            print(f"\nScore profile statistics:")
            print(f"  Min: {min(scores):.3f}")
            print(f"  Max: {max(scores):.3f}")
            print(f"  Mean: {sum(scores)/len(scores):.3f}")
    else:
        print(f"Prediction failed: {result.error_message}")


def analyze_alpha_synuclein():
    """
    Analyze α-synuclein NAC region - core of Parkinson's disease aggregates.
    
    α-Synuclein is a 140-residue protein whose aggregation is central to
    Parkinson's disease and other synucleinopathies. The NAC (Non-Amyloid-β
    Component) region (61-95) is necessary and sufficient for fibrillization.
    """
    print_header("α-Synuclein NAC Region Analysis")
    
    # Full α-synuclein with NAC region highlighted
    nac_region = "EQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFV"
    
    protein = ProteinRecord(
        id="aSyn_NAC",
        name="Alpha-synuclein NAC region (61-95)",
        sequence=nac_region,
        organism="Homo sapiens",
        is_known_amyloid=True,
        known_polymorph=AmyloidPolymorph.CROSS_BETA_PARALLEL,
    )
    
    print(f"\nProtein: {protein.name}")
    print(f"Sequence: {protein.sequence}")
    print(f"Length: {protein.sequence_length} residues")
    
    # Composition analysis
    composition = calculate_class_composition(protein.sequence)
    print("\nAmino acid class composition:")
    print(f"  Amyloidogenic: {composition['amyloidogenic']*100:.1f}%")
    print(f"  Gatekeepers: {composition['gatekeepers']*100:.1f}%")
    
    # Note the VVG, VT repeats characteristic of NAC
    print("\nCharacteristic motifs:")
    for motif in ["VV", "VT", "GG", "AA"]:
        count = protein.sequence.count(motif)
        if count > 0:
            print(f"  {motif}: {count} occurrence(s)")
    
    # Run prediction
    predictor = get_predictor("Aggrescan3D")
    result = predictor.predict(protein)
    
    if result.success and result.predicted_regions:
        print(f"\nPredicted APRs:")
        for apr in result.predicted_regions:
            print(f"  {apr.start}-{apr.end}: {apr.sequence}")


def analyze_prion_protein():
    """
    Analyze human prion protein PrP(106-126) - neurotoxic peptide.
    
    This peptide from the human prion protein is both neurotoxic and
    capable of forming amyloid fibrils independently. It demonstrates
    how a short peptide fragment can retain amyloidogenic properties.
    """
    print_header("Prion Protein PrP(106-126) Analysis")
    
    sequence = "KTNMKHMAGAAAAGAVVGGLG"
    
    protein = ProteinRecord(
        id="PrP_106_126",
        name="Prion protein fragment (106-126)",
        sequence=sequence,
        organism="Homo sapiens",
        is_known_amyloid=True,
    )
    
    print(f"\nProtein: {protein.name}")
    print(f"Sequence: {protein.sequence}")
    print(f"Length: {protein.sequence_length} residues")
    
    # This peptide contains the AGAAAAGA palindrome
    print("\nNotable features:")
    print("  - Contains AGAAAAGA palindromic motif")
    print("  - Gly-rich region may confer flexibility")
    print("  - Ala repeats promote β-sheet formation")
    
    # Composition
    composition = calculate_class_composition(protein.sequence)
    print(f"\nAmyloidogenic content: {composition['amyloidogenic']*100:.1f}%")
    print(f"Gatekeeper content: {composition['gatekeepers']*100:.1f}%")
    
    # Predict
    predictor = get_predictor("Aggrescan3D")
    result = predictor.predict(protein)
    
    if result.success:
        print(f"\nAmyloidogenic prediction: {result.is_amyloidogenic}")
        if result.predicted_regions:
            for apr in result.predicted_regions:
                print(f"  APR: {apr.sequence} at {apr.start}-{apr.end}")


def compare_functional_vs_pathological():
    """
    Compare functional amyloid (CsgA) with pathological amyloid (Aβ42).
    
    This comparison illustrates that amyloidogenicity per se is not
    pathological—evolution has co-opted the cross-β fold for beneficial
    functions in bacterial biofilms, mammalian pigmentation, and more.
    """
    print_header("Functional vs Pathological Amyloid Comparison")
    
    # E. coli CsgA R1 repeat unit (functional amyloid in curli)
    csga_r1 = "SELNIYQYGGGNSALALQTDARN"
    
    # Aβ42 (pathological)
    abeta42 = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"
    
    proteins = [
        ProteinRecord(
            id="CsgA_R1",
            name="CsgA R1 repeat (functional)",
            sequence=csga_r1,
            organism="Escherichia coli",
        ),
        ProteinRecord(
            id="Abeta42",
            name="Aβ42 (pathological)",
            sequence=abeta42,
            organism="Homo sapiens",
        ),
    ]
    
    predictor = get_predictor("Aggrescan3D")
    
    print("\nComparative analysis:")
    print("-" * 60)
    
    for protein in proteins:
        result = predictor.predict(protein)
        composition = calculate_class_composition(protein.sequence)
        
        print(f"\n{protein.name}:")
        print(f"  Length: {protein.sequence_length}")
        print(f"  Amyloidogenic AA: {composition['amyloidogenic']*100:.1f}%")
        print(f"  Gatekeepers: {composition['gatekeepers']*100:.1f}%")
        
        if result.success and result.per_residue_scores:
            scores = result.per_residue_scores.scores
            print(f"  Mean A3D score: {sum(scores)/len(scores):.3f}")
            print(f"  Max A3D score: {max(scores):.3f}")
            print(f"  APRs found: {len(result.predicted_regions)}")
    
    print("\n" + "-" * 60)
    print("Note: Both sequences are amyloidogenic, but CsgA R1 forms")
    print("functional biofilm fibers while Aβ42 forms toxic aggregates.")
    print("The difference lies in cellular context, not intrinsic propensity.")


def show_available_predictors():
    """Display information about available predictors."""
    print_header("Available Predictors")
    
    predictors = list_predictors()
    
    for pred in predictors:
        print(f"\n{pred['name']} (v{pred['version']})")
        print(f"  Type: {pred['type']}")
        print(f"  Threshold: {pred['threshold']}")
        print(f"  Capabilities: {', '.join(pred['capabilities'][:3])}")
        if pred.get('citation'):
            print(f"  Citation: {pred['citation'][:60]}...")


def main():
    """Run all example analyses."""
    print("\n" + "=" * 70)
    print("  AmyloidBench Example: Analyzing Amyloidogenic Proteins")
    print("=" * 70)
    
    # Show available predictors
    show_available_predictors()
    
    # Analyze individual proteins
    analyze_amyloid_beta()
    analyze_alpha_synuclein()
    analyze_prion_protein()
    
    # Comparative analysis
    compare_functional_vs_pathological()
    
    print("\n" + "=" * 70)
    print("  Example complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Try prediction on your own sequences")
    print("  2. Use structure files for more accurate predictions")
    print("  3. Compare multiple predictors using consensus mode")
    print("  4. Run benchmarking against validation databases")


if __name__ == "__main__":
    main()
