"""
Curated reference datasets for amyloidogenicity benchmarking.

This module provides gold-standard datasets of experimentally validated
amyloid-forming sequences with known aggregation-prone regions (APRs).
These sequences serve as ground truth for benchmarking predictor accuracy.

Dataset Categories
------------------

**Canonical Amyloid Peptides**
Well-characterized short peptides from landmark structural studies:
- GNNQQNY (Sup35 prion core, PDB: 1YJO) - Class 1 steric zipper
- KLVFFA (Aβ17-22 core) - Parallel β-sheet
- VQIVYK (Tau PHF6, PDB: 2ON9) - Steric zipper
- NFGAIL (IAPP amyloid core) - β-sheet forming
- SSTSAA (β2-microglobulin, PDB: 3LÖW) - Class 1 zipper

**Disease-Associated Proteins**
Full-length sequences with mapped APR regions:
- Aβ42 (Alzheimer's disease) - APRs: 17-21 (LVFFA), 31-42 (C-terminus)
- α-Synuclein (Parkinson's) - NAC region (61-95)
- Tau (Tauopathies) - PHF6/PHF6* regions
- Huntingtin exon 1 (Huntington's) - PolyQ expansion
- Prion protein (TSEs) - Region 106-126

**Functional Amyloids**
Non-pathological amyloid-forming proteins:
- Curli (E. coli biofilms) - CsgA subunit
- HET-s (fungal prion) - β-solenoid
- Pmel17 (melanosomes) - Repeat domain
- Sup35 NM (yeast prion) - N-terminal domain

**Negative Controls**
Well-folded, non-aggregating proteins:
- Ubiquitin - Stable globular fold
- Lysozyme (native) - Unless destabilized
- Green fluorescent protein - β-barrel
- Bovine serum albumin - Helical, soluble

Biological Basis
----------------
Amyloid formation requires:
1. Sequence determinants: Hydrophobic stretches, β-sheet propensity
2. Environmental triggers: pH, concentration, metal ions
3. Conformational change: Native → partially unfolded intermediate
4. Nucleation: Rate-limiting oligomer formation
5. Elongation: Templated addition to fibril ends

APR annotations are based on:
- X-ray crystallography of amyloid microcrystals
- Cryo-EM of full-length fibrils
- Solid-state NMR structural constraints
- Mutagenesis/aggregation kinetics correlation
- Hydrogen-deuterium exchange (HDX) protection

References
----------
- Eisenberg & Sawaya (2017) - Structural studies of amyloid proteins
- Fitzpatrick et al. (2017) - Cryo-EM structures of tau filaments
- Guerrero-Ferreira et al. (2018) - α-synuclein fibril structures
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from amyloidbench.core.models import Region
from .datasets import (
    AmyloidStatus,
    BenchmarkDataset,
    BenchmarkEntry,
    ExperimentalMethod,
)


# =============================================================================
# Canonical Peptide Sequences
# =============================================================================

@dataclass
class CanonicalPeptide:
    """
    Canonical amyloid peptide with complete annotation.
    
    Attributes:
        name: Common name
        sequence: Amino acid sequence
        source_protein: Parent protein
        residue_range: Position in parent protein
        pdb_ids: Associated PDB structures
        zipper_class: Steric zipper classification (if known)
        experimental_method: Primary validation method
        kinetics: Aggregation kinetics data (if available)
        notes: Additional biological context
    """
    name: str
    sequence: str
    source_protein: str
    residue_range: Optional[tuple[int, int]] = None
    pdb_ids: list[str] = field(default_factory=list)
    zipper_class: Optional[int] = None  # 1-8
    experimental_method: ExperimentalMethod = ExperimentalMethod.XRAY
    notes: str = ""


# Canonical hexapeptides from Sawaya et al. crystallography
CANONICAL_PEPTIDES = [
    # Yeast prion Sup35 - the most studied steric zipper
    CanonicalPeptide(
        name="GNNQQNY",
        sequence="GNNQQNY",
        source_protein="Sup35 (yeast prion)",
        residue_range=(7, 13),
        pdb_ids=["1YJO", "2OMM", "2OLX"],
        zipper_class=1,
        experimental_method=ExperimentalMethod.XRAY,
        notes="Class 1 steric zipper. Parallel strands, parallel sheets, face-to-face. "
              "Resolution: 1.8 Å. First high-resolution amyloid crystal structure.",
    ),
    CanonicalPeptide(
        name="NNQQNY",
        sequence="NNQQNY",
        source_protein="Sup35 (yeast prion)",
        residue_range=(8, 13),
        pdb_ids=["2OMP"],
        zipper_class=1,
        experimental_method=ExperimentalMethod.XRAY,
        notes="Variant of GNNQQNY. Also forms Class 1 zipper.",
    ),
    
    # Aβ amyloid core
    CanonicalPeptide(
        name="KLVFFA",
        sequence="KLVFFA",
        source_protein="Amyloid β (Aβ)",
        residue_range=(16, 21),
        pdb_ids=["2Y3J", "3OW9"],
        zipper_class=1,
        experimental_method=ExperimentalMethod.XRAY,
        notes="Core hydrophobic segment of Aβ. Essential for aggregation. "
              "L17, V18, F19, F20 form dry interface in steric zipper.",
    ),
    CanonicalPeptide(
        name="LVFFAE",
        sequence="LVFFAE",
        source_protein="Amyloid β (Aβ)",
        residue_range=(17, 22),
        pdb_ids=["2Y3K"],
        zipper_class=1,
        experimental_method=ExperimentalMethod.XRAY,
        notes="Extended Aβ core. E22 (Iowa mutation site) at C-terminus.",
    ),
    
    # Tau PHF core
    CanonicalPeptide(
        name="VQIVYK",
        sequence="VQIVYK",
        source_protein="Tau (microtubule-associated)",
        residue_range=(306, 311),
        pdb_ids=["2ON9"],
        zipper_class=1,
        experimental_method=ExperimentalMethod.XRAY,
        notes="PHF6 motif. Primary aggregation hotspot in tau. "
              "Mutations here affect tauopathy progression.",
    ),
    CanonicalPeptide(
        name="VQIINK",
        sequence="VQIINK",
        source_protein="Tau (microtubule-associated)",
        residue_range=(275, 280),
        pdb_ids=["3OVL"],
        zipper_class=2,
        experimental_method=ExperimentalMethod.XRAY,
        notes="PHF6* motif. Second aggregation hotspot in tau repeat domain.",
    ),
    
    # IAPP (amylin) - Type 2 diabetes
    CanonicalPeptide(
        name="NFGAIL",
        sequence="NFGAIL",
        source_protein="IAPP (islet amyloid polypeptide)",
        residue_range=(22, 27),
        pdb_ids=["3DG1"],
        zipper_class=1,
        experimental_method=ExperimentalMethod.XRAY,
        notes="Core amyloidogenic segment of IAPP. Forms deposits in "
              "pancreatic islets in Type 2 diabetes.",
    ),
    CanonicalPeptide(
        name="SSTNVG",
        sequence="SSTNVG",
        source_protein="IAPP (islet amyloid polypeptide)",
        residue_range=(28, 33),
        pdb_ids=["3DGJ"],
        zipper_class=8,
        experimental_method=ExperimentalMethod.XRAY,
        notes="Class 8 zipper. Antiparallel strands, antiparallel sheets, face-to-back.",
    ),
    
    # β2-microglobulin - Dialysis-related amyloidosis
    CanonicalPeptide(
        name="SSTSAA",
        sequence="SSTSAA",
        source_protein="β2-microglobulin",
        pdb_ids=["3LOW"],
        zipper_class=1,
        experimental_method=ExperimentalMethod.XRAY,
        notes="Forms amyloid in dialysis patients. Serine-rich sequence.",
    ),
    
    # Prion protein (PrP)
    CanonicalPeptide(
        name="SNQNNF",
        sequence="SNQNNF",
        source_protein="Prion protein",
        residue_range=(170, 175),
        pdb_ids=["3NVE"],
        zipper_class=1,
        experimental_method=ExperimentalMethod.XRAY,
        notes="From human PrP amyloidogenic region.",
    ),
    
    # Insulin - Pharmaceutical amyloid
    CanonicalPeptide(
        name="VEALYL",
        sequence="VEALYL",
        source_protein="Insulin B-chain",
        residue_range=(12, 17),
        pdb_ids=["3HYD"],
        zipper_class=5,
        experimental_method=ExperimentalMethod.XRAY,
        notes="Class 5 zipper. Important for insulin aggregation during storage.",
    ),
    
    # Transthyretin - Familial amyloidosis
    CanonicalPeptide(
        name="YTIAALLSPYS",
        sequence="YTIAALLSPYS",
        source_protein="Transthyretin",
        residue_range=(105, 115),
        pdb_ids=["4TLT"],
        zipper_class=None,
        experimental_method=ExperimentalMethod.CRYO_EM,
        notes="From TTR amyloid fibril core. Longer segment.",
    ),
]


# =============================================================================
# Disease-Associated Full Proteins
# =============================================================================

@dataclass
class DiseaseProtein:
    """
    Full-length disease-associated amyloid protein.
    
    Attributes:
        name: Common name
        uniprot_id: UniProt accession
        sequence: Full amino acid sequence
        apr_regions: Experimentally validated APR positions
        disease: Associated disease(s)
        pdb_ids: Structural data
        polymorph: Known polymorphs (if characterized)
        notes: Biological context
    """
    name: str
    uniprot_id: str
    sequence: str
    apr_regions: list[tuple[int, int, str]]  # (start, end, name)
    disease: str
    pdb_ids: list[str] = field(default_factory=list)
    polymorph: Optional[str] = None
    notes: str = ""


DISEASE_PROTEINS = [
    DiseaseProtein(
        name="Amyloid-β 42",
        uniprot_id="P05067",  # APP precursor
        sequence="DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA",
        apr_regions=[
            (17, 21, "Central hydrophobic core (CHC)"),
            (30, 40, "C-terminal APR"),
        ],
        disease="Alzheimer's disease",
        pdb_ids=["5OQV", "5KK3", "6SHS"],  # Cryo-EM structures
        polymorph="Multiple polymorphs identified in AD patients",
        notes="Primary component of senile plaques. Multiple cryo-EM structures "
              "reveal patient-to-patient structural variation.",
    ),
    DiseaseProtein(
        name="α-Synuclein",
        uniprot_id="P37840",
        sequence=(
            "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVA"
            "EKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGI"
            "LEDMPVDPDNEAYEMPSEEGYQDYEPEA"
        ),
        apr_regions=[
            (61, 95, "NAC region (non-Aβ component)"),
            (71, 82, "NAC core"),
        ],
        disease="Parkinson's disease, Lewy body dementia, MSA",
        pdb_ids=["6A6B", "6CU7", "6XYO"],  # Cryo-EM structures
        polymorph="Disease-specific polymorphs (PD vs MSA)",
        notes="140 residues. NAC region is minimally required for aggregation. "
              "N-terminus binds membranes, C-terminus is disordered.",
    ),
    DiseaseProtein(
        name="Tau (0N4R)",
        uniprot_id="P10636",
        sequence=(
            "MAEPRQEFEVMEDHAGTYGLGDRKDQGGYTMHQDQEGDTDAGLKESPLQTPTEDGS"
            "EEPGSETSDAKSTPTAEAEEAGIGDTPSLEDEAAGHVTQARMVSKSKDGTGSDDK"
            "KAKGADGKTKIATPRGAAPPGQKGQANATRIPAKTPPAPKTPPSSGEPPKSGDRS"
            "GYSSSPGSPGTPGSRSRTPSLPTPPTREPKKVAVVRTPPKSPSSAKSRLQTAPVP"
            "MPDLKNVKSKIGSTENLKHQPGGGKVQIINKKLDLSNVQSKCGSKDNIKHVPGGG"
            "SVQIVYKPVDLSKVTSKCGSLGNIHHKPGGGQVEVKSEKLDFKDRVQSKIGSLDNI"
            "THVPGGGN"
        ),
        apr_regions=[
            (275, 280, "PHF6* (VQIINK)"),
            (306, 311, "PHF6 (VQIVYK)"),
            (337, 343, "Third repeat APR"),
        ],
        disease="Alzheimer's disease, frontotemporal dementia, CTE",
        pdb_ids=["5O3T", "5O3L", "6QJH"],  # Cryo-EM AD, Pick's, CTE
        polymorph="Disease-specific folds: AD pair vs Pick's fold",
        notes="Alternatively spliced (0N-2N, 3R-4R). PHF6/PHF6* hexapeptides "
              "drive aggregation. Disease-specific structural polymorphs.",
    ),
    DiseaseProtein(
        name="Huntingtin exon 1 (Q23)",
        uniprot_id="P42858",
        sequence=(
            "MATLEKLMKAFESLKSFQQQQQQQQQQQQQQQQQQQQQQQ"
            "PPPPPPPPPPPQLPQPPPQAQPLLPQPQPPPPPPPPPPGPAVAEEPLHRP"
        ),
        apr_regions=[
            (18, 40, "PolyQ tract"),
        ],
        disease="Huntington's disease",
        pdb_ids=["6EZ8"],  # Cryo-EM
        polymorph="Length-dependent: Q>35 causes disease",
        notes="Shown with 23 glutamines (normal). Disease threshold Q>35. "
              "PolyQ forms β-hairpin structure in fibrils.",
    ),
    DiseaseProtein(
        name="Prion protein (human)",
        uniprot_id="P04156",
        sequence=(
            "MANLGCWMLVLFVATWSDLGLCKKRPKPGGWNTGGSRYPGQGSPGGNRYPPQGGGG"
            "WGQPHGGGWGQPHGGGWGQPHGGGWGQPHGGGWGQGGGTHSQWNKPSKPKTNMKHM"
            "AGAAAAGAVVGGLGGYMLGSAMSRPIIHFGSDYEDRYYRENMHRYPNQVYYRPMDE"
            "YSNQNNFVHDCVNITIKQHTVTTTTKGENFTETDVKMMERVVEQMCITQYERESQA"
            "YYQRGS"
        ),
        apr_regions=[
            (106, 126, "Neurotoxic peptide region"),
            (127, 147, "Hydrophobic core"),
            (170, 175, "SNQNNF palindrome"),
        ],
        disease="Creutzfeldt-Jakob disease, kuru, FFI, GSS",
        pdb_ids=["6LNI"],  # Cryo-EM of brain-derived
        polymorph="Strain-specific polymorphs (Type 1 vs Type 2)",
        notes="253 residues with signal peptide. GPI-anchored. "
              "Octapeptide repeats bind copper. Misfolding causes TSEs.",
    ),
    DiseaseProtein(
        name="Serum amyloid A",
        uniprot_id="P0DJI8",
        sequence=(
            "MKLLTGLVFCSLVLGVSSRSFFSFLGEAFDGARDMWRAYSDMREANYIGSDKYFHA"
            "RGNYDAAKRGPGGVWAAEAISDARENIQRFFGHGAEDSLADQAANEWGRSGKDPNH"
            "FRPAGLPEKY"
        ),
        apr_regions=[
            (1, 11, "N-terminal APR"),
            (69, 76, "Central APR"),
        ],
        disease="AA amyloidosis (secondary/reactive)",
        pdb_ids=["6MST"],
        polymorph="N-terminal proteolytic fragment deposits",
        notes="Acute phase protein. Chronic inflammation leads to AA amyloid. "
              "N-terminal 76-residue fragment forms fibrils.",
    ),
]


# =============================================================================
# Functional (Non-pathological) Amyloids
# =============================================================================

@dataclass
class FunctionalAmyloid:
    """
    Functional amyloid protein.
    
    These form beneficial amyloid structures in normal biology.
    """
    name: str
    organism: str
    sequence: str
    apr_regions: list[tuple[int, int, str]]
    function: str
    fold_type: str
    pdb_ids: list[str] = field(default_factory=list)
    notes: str = ""


FUNCTIONAL_AMYLOIDS = [
    FunctionalAmyloid(
        name="CsgA (Curli major subunit)",
        organism="Escherichia coli",
        sequence=(
            "MKLLKVAAIAAIVFSGSALAGVVPQYGGGGNHGGGGNNSGPNSELNIYQYGGGNSALALQT"
            "DARNSDLTITQHGGGNGADVGQGSDDSSIDLTQRGFGNSATLDQWNGKNSEMTVKQFGGGN"
            "GAAVDQTASNSSVNVTQVGFGNNATAHQY"
        ),
        apr_regions=[
            (22, 43, "R1 repeat"),
            (44, 65, "R2 repeat"),
            (66, 87, "R3 repeat"),
            (88, 109, "R4 repeat"),
            (110, 131, "R5 repeat"),
        ],
        function="Biofilm matrix structural protein",
        fold_type="β-solenoid",
        pdb_ids=["6G8C"],
        notes="Five imperfect repeats form β-helix. Major component of "
              "bacterial biofilm extracellular matrix. Cross-seeding concern.",
    ),
    FunctionalAmyloid(
        name="HET-s prion domain",
        organism="Podospora anserina",
        sequence=(
            "VIDAKLKATGANGQTNIGAKIGSNSVGWATGAATAIATALQSAREANQKGNEA"
            "IVAKIGSNAGIGVLIGAQVGAMATVATAL"
        ),
        apr_regions=[
            (1, 43, "First β-solenoid unit"),
            (44, 85, "Second β-solenoid unit"),
        ],
        function="Heterokaryon incompatibility",
        fold_type="β-solenoid",
        pdb_ids=["2KJ3", "2RNM"],  # NMR and ssNMR structures
        notes="Two-rung β-solenoid. Fungal prion with biological function. "
              "Controls cell death in heterokaryon incompatibility.",
    ),
    FunctionalAmyloid(
        name="Pmel17 repeat domain",
        organism="Homo sapiens",
        sequence=(
            "PGSGSGSGPSGTGGGSTGPATPGAATTDPTTTGPGASVHVNGTSPSNGTNGATNPGI"
            "GASTVNGTSQNGATNPGIGATPNGTSQNGATNPGIGATPNGTSPNGATNPGIGATPN"
        ),
        apr_regions=[
            (40, 70, "First repeat cluster"),
            (80, 110, "Second repeat cluster"),
        ],
        function="Melanin synthesis scaffold in melanosomes",
        fold_type="β-sheet",
        pdb_ids=[],  # No high-resolution structure
        notes="Forms amyloid scaffold for melanin polymerization. "
              "Essential for normal pigmentation. Repeat-rich sequence.",
    ),
    FunctionalAmyloid(
        name="Sup35 NM domain",
        organism="Saccharomyces cerevisiae",
        sequence=(
            "MSDSNQGNNQQNYQQYSQNGNQQQGNNRYQGYQAYNAQAQPAGGYYQNYQGYSGYQQG"
            "GYQQYNPQGGYQQYNPQGGYQQQFNPQGGRGNYKNFNYNNNLQGYQ"
        ),
        apr_regions=[
            (1, 40, "N domain - aggregation core"),
            (41, 97, "M domain - middle charged region"),
        ],
        function="Translation termination factor / yeast prion [PSI+]",
        fold_type="β-sheet",
        pdb_ids=["5KJ5"],
        notes="[PSI+] prion. N-rich region drives aggregation. "
              "M region modulates solubility. Classic yeast prion model.",
    ),
]


# =============================================================================
# Negative Controls (Non-amyloidogenic)
# =============================================================================

@dataclass
class NegativeControl:
    """Non-amyloidogenic protein for negative control."""
    name: str
    uniprot_id: str
    sequence: str
    fold_type: str
    notes: str = ""


NEGATIVE_CONTROLS = [
    NegativeControl(
        name="Ubiquitin",
        uniprot_id="P0CG48",
        sequence=(
            "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQ"
            "KESTLHLVLRLRGG"
        ),
        fold_type="β-grasp fold",
        notes="Extremely stable. Forms amyloid only under harsh denaturing conditions.",
    ),
    NegativeControl(
        name="Lysozyme (hen egg)",
        uniprot_id="P00698",
        sequence=(
            "MRSLLILVLCFLPLAALGKVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQAT"
            "NRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGN"
            "GMNAWVAWRNRCKGTDVQAWIRGCRL"
        ),
        fold_type="α+β",
        notes="Stable native fold. Can form amyloid via specific mutations "
              "(e.g., D67H associated with hereditary amyloidosis).",
    ),
    NegativeControl(
        name="Green fluorescent protein",
        uniprot_id="P42212",
        sequence=(
            "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLV"
            "TTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVN"
            "RIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADH"
            "YQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
        ),
        fold_type="β-barrel",
        notes="11-stranded β-barrel. Chromophore protected. Very stable.",
    ),
    NegativeControl(
        name="Bovine serum albumin",
        uniprot_id="P02769",
        sequence=(
            "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFD"
            "EHVKLVNELTEFAKTCVADESHAGCEKSLHTLFGDELCKVASLRETYGDMADCCEKQEPE"
            "RNECFLSHKDDSPDLPKLKPDPNTLCDEFKADEKKFWGKYLYEIARRHPYFYAPELLLYA"
            "NKYNGVFQECCQAEDKGACLLPKIETMREKVLASSARQRLRCASIQKFGERALKAWSVAR"
            "LSQKFPKAEFVEVTKLVTDLTKVHKECCHGDLLECADDRADLAKYICDNQDTISSKLKEC"
            "CDKPLLEKSHCIAEVEKDAIPENLPPLTADFAEDKDVCKNYQEAKDAFLGSFLYEYSRRH"
            "PEYAVSVLLRLAKEYEATLEECCAKDDPHACYSTVFDKLKHLVDEPQNLIKQNCDQFEKL"
            "GEYGFQNALIVRYTRKVPQVSTPTLVEVSRSLGKVGTRCCTKPESERMPCTEDYLSLILNR"
            "LCVLHEKTPVSEKVTKCCTESLVNRRPCFSALTPDETYVPKAFDEKLFTFHADICTLPDT"
            "EKQIKKQTALVELLKHKPKATEEQLKTVMENFVAFVDKCCAADDKEACFAVEGPKLVVST"
            "QTAL"
        ),
        fold_type="All-α (heart-shaped)",
        notes="607 residues. Predominantly α-helical. Highly soluble carrier protein.",
    ),
    NegativeControl(
        name="Thioredoxin",
        uniprot_id="P10599",
        sequence=(
            "MVKQIESKTAFQEALDAAGDKLVVVDFSATWCGPCKMIKPFFHSLSEKYSNVIFLEVDVDD"
            "CQDVASECEVKCMPTFQFFKKGQKVGEFSGANKEKLEATINELV"
        ),
        fold_type="Thioredoxin fold",
        notes="Redox protein. Stable fold. Classic non-aggregating control.",
    ),
]


# =============================================================================
# Dataset Building Functions
# =============================================================================

def create_canonical_peptide_dataset() -> BenchmarkDataset:
    """
    Create benchmark dataset from canonical amyloid peptides.
    
    All entries are positive (amyloid-forming) with high confidence
    based on crystallographic evidence.
    
    Returns:
        BenchmarkDataset with canonical peptides
    """
    entries = []
    
    for peptide in CANONICAL_PEPTIDES:
        entry = BenchmarkEntry(
            id=f"canonical_{peptide.name}",
            sequence=peptide.sequence,
            amyloid_status=AmyloidStatus.POSITIVE,
            amyloid_regions=[
                Region(
                    start=0,
                    end=len(peptide.sequence),
                    sequence=peptide.sequence,
                    score=1.0,
                    metadata={"zipper_class": peptide.zipper_class},
                )
            ],
            residue_labels=[True] * len(peptide.sequence),
            experimental_method=peptide.experimental_method,
            source_database="AmyloidBench_Canonical",
            metadata={
                "source_protein": peptide.source_protein,
                "pdb_ids": peptide.pdb_ids,
                "zipper_class": peptide.zipper_class,
                "notes": peptide.notes,
            },
        )
        entries.append(entry)
    
    return BenchmarkDataset(
        name="Canonical_Peptides",
        description=(
            "Canonical amyloid-forming hexapeptides from X-ray crystallography. "
            "Gold-standard positive controls with known steric zipper classes."
        ),
        entries=entries,
        source_url="https://doi.org/10.1038/nature05695",
        citation="Sawaya et al. (2007) Nature 447:453-457",
        version="1.0",
    )


def create_disease_protein_dataset() -> BenchmarkDataset:
    """
    Create benchmark dataset from disease-associated proteins.
    
    Full-length proteins with annotated APR regions based on
    structural and biochemical evidence.
    
    Returns:
        BenchmarkDataset with disease proteins
    """
    entries = []
    
    for protein in DISEASE_PROTEINS:
        # Create residue labels
        residue_labels = [False] * len(protein.sequence)
        apr_regions = []
        
        for start, end, name in protein.apr_regions:
            # Convert to 0-indexed
            s, e = start - 1, end
            for i in range(max(0, s), min(len(protein.sequence), e)):
                residue_labels[i] = True
            
            apr_regions.append(Region(
                start=start,
                end=end,
                sequence=protein.sequence[s:e] if s < len(protein.sequence) else "",
                score=1.0,
                metadata={"name": name},
            ))
        
        entry = BenchmarkEntry(
            id=f"disease_{protein.uniprot_id}",
            sequence=protein.sequence,
            amyloid_status=AmyloidStatus.POSITIVE,
            amyloid_regions=apr_regions,
            residue_labels=residue_labels,
            experimental_method=ExperimentalMethod.CRYO_EM,
            source_database="AmyloidBench_Disease",
            metadata={
                "name": protein.name,
                "uniprot_id": protein.uniprot_id,
                "disease": protein.disease,
                "pdb_ids": protein.pdb_ids,
                "polymorph": protein.polymorph,
                "notes": protein.notes,
            },
        )
        entries.append(entry)
    
    return BenchmarkDataset(
        name="Disease_Proteins",
        description=(
            "Disease-associated amyloidogenic proteins with mapped APR regions "
            "from cryo-EM structures and mutagenesis studies."
        ),
        entries=entries,
        source_url="https://www.uniprot.org",
        citation="UniProt Consortium + primary structural studies",
        version="1.0",
    )


def create_functional_amyloid_dataset() -> BenchmarkDataset:
    """
    Create benchmark dataset from functional amyloids.
    
    Non-pathological amyloid-forming proteins. Important for
    distinguishing pathological from functional aggregation.
    
    Returns:
        BenchmarkDataset with functional amyloids
    """
    entries = []
    
    for protein in FUNCTIONAL_AMYLOIDS:
        residue_labels = [False] * len(protein.sequence)
        apr_regions = []
        
        for start, end, name in protein.apr_regions:
            s, e = start - 1, end
            for i in range(max(0, s), min(len(protein.sequence), e)):
                residue_labels[i] = True
            
            apr_regions.append(Region(
                start=start,
                end=end,
                sequence=protein.sequence[s:e] if s < len(protein.sequence) else "",
                score=1.0,
                metadata={"name": name, "function": protein.function},
            ))
        
        entry = BenchmarkEntry(
            id=f"functional_{protein.name.replace(' ', '_')}",
            sequence=protein.sequence,
            amyloid_status=AmyloidStatus.POSITIVE,
            amyloid_regions=apr_regions,
            residue_labels=residue_labels,
            experimental_method=ExperimentalMethod.NMR,
            source_database="AmyloidBench_Functional",
            metadata={
                "organism": protein.organism,
                "function": protein.function,
                "fold_type": protein.fold_type,
                "pdb_ids": protein.pdb_ids,
                "notes": protein.notes,
            },
        )
        entries.append(entry)
    
    return BenchmarkDataset(
        name="Functional_Amyloids",
        description=(
            "Functional (non-pathological) amyloid-forming proteins. "
            "Include biofilm components, prions with biological function."
        ),
        entries=entries,
        citation="Chapman et al. (2002) Science; Ritter et al. (2005) Nature",
        version="1.0",
    )


def create_negative_control_dataset() -> BenchmarkDataset:
    """
    Create benchmark dataset from negative controls.
    
    Well-folded, non-amyloidogenic proteins. Essential for
    calculating specificity and avoiding false positives.
    
    Returns:
        BenchmarkDataset with negative controls
    """
    entries = []
    
    for protein in NEGATIVE_CONTROLS:
        entry = BenchmarkEntry(
            id=f"negative_{protein.uniprot_id}",
            sequence=protein.sequence,
            amyloid_status=AmyloidStatus.NEGATIVE,
            amyloid_regions=[],
            residue_labels=[False] * len(protein.sequence),
            experimental_method=ExperimentalMethod.XRAY,
            source_database="AmyloidBench_Negative",
            metadata={
                "name": protein.name,
                "uniprot_id": protein.uniprot_id,
                "fold_type": protein.fold_type,
                "notes": protein.notes,
            },
        )
        entries.append(entry)
    
    return BenchmarkDataset(
        name="Negative_Controls",
        description=(
            "Well-characterized non-amyloidogenic proteins. "
            "Stable globular folds that resist aggregation."
        ),
        entries=entries,
        citation="Standard protein biochemistry literature",
        version="1.0",
    )


def create_comprehensive_dataset() -> BenchmarkDataset:
    """
    Create comprehensive benchmark combining all reference datasets.
    
    Balanced dataset with positive (disease + functional amyloids)
    and negative controls for unbiased evaluation.
    
    Returns:
        BenchmarkDataset with all reference sequences
    """
    canonical = create_canonical_peptide_dataset()
    disease = create_disease_protein_dataset()
    functional = create_functional_amyloid_dataset()
    negative = create_negative_control_dataset()
    
    all_entries = (
        canonical.entries +
        disease.entries +
        functional.entries +
        negative.entries
    )
    
    return BenchmarkDataset(
        name="AmyloidBench_Reference",
        description=(
            "Comprehensive reference dataset combining canonical peptides, "
            "disease-associated proteins, functional amyloids, and negative controls. "
            f"Total: {len(all_entries)} sequences "
            f"({sum(1 for e in all_entries if e.is_positive)} positive, "
            f"{sum(1 for e in all_entries if e.is_negative)} negative)."
        ),
        entries=all_entries,
        citation="AmyloidBench curated reference set v1.0",
        version="1.0",
    )


# =============================================================================
# Convenience Access
# =============================================================================

def get_canonical_peptides() -> list[CanonicalPeptide]:
    """Get list of canonical amyloid peptides."""
    return CANONICAL_PEPTIDES.copy()


def get_disease_proteins() -> list[DiseaseProtein]:
    """Get list of disease-associated proteins."""
    return DISEASE_PROTEINS.copy()


def get_functional_amyloids() -> list[FunctionalAmyloid]:
    """Get list of functional amyloid proteins."""
    return FUNCTIONAL_AMYLOIDS.copy()


def get_negative_controls() -> list[NegativeControl]:
    """Get list of negative control proteins."""
    return NEGATIVE_CONTROLS.copy()


def get_sequence_by_name(name: str) -> Optional[str]:
    """
    Get sequence by common name.
    
    Searches across all datasets.
    
    Args:
        name: Protein/peptide name (case-insensitive)
        
    Returns:
        Sequence string or None if not found
    """
    name_lower = name.lower()
    
    for peptide in CANONICAL_PEPTIDES:
        if peptide.name.lower() == name_lower:
            return peptide.sequence
    
    for protein in DISEASE_PROTEINS:
        if protein.name.lower() == name_lower:
            return protein.sequence
    
    for protein in FUNCTIONAL_AMYLOIDS:
        if protein.name.lower() == name_lower:
            return protein.sequence
    
    for protein in NEGATIVE_CONTROLS:
        if protein.name.lower() == name_lower:
            return protein.sequence
    
    return None
