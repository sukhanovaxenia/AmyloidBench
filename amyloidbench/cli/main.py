"""
AmyloidBench Command Line Interface.

This module provides a comprehensive CLI for running amyloidogenicity
predictions, benchmarking predictors, and analyzing results. Built with
Click for a user-friendly experience with proper help documentation.

Usage:
    amyloidbench predict sequence.fasta -o results/
    amyloidbench predict --predictor Aggrescan3D --structure protein.pdb
    amyloidbench benchmark --database waltz-db
    amyloidbench list-predictors
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Initialize rich console for pretty output
console = Console()


def print_banner():
    """Print the AmyloidBench banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                      AmyloidBench v0.1.0                      ║
    ║     Consensus Meta-Predictor for Protein Amyloidogenicity     ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold blue")


@click.group()
@click.version_option(version="0.1.0", prog_name="AmyloidBench")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-essential output")
@click.pass_context
def cli(ctx, verbose: bool, quiet: bool):
    """
    AmyloidBench: Comprehensive consensus predictor for protein amyloidogenicity.
    
    This tool provides:
    
    \b
    • Meta-prediction using multiple amyloidogenicity tools
    • Per-residue score profiles and APR detection
    • Structural classification of amyloid types
    • Benchmarking against curated databases
    
    Run 'amyloidbench COMMAND --help' for command-specific help.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    
    if not quiet:
        print_banner()


@cli.command("predict")
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="amyloidbench_results",
    help="Output directory for results"
)
@click.option(
    "--predictor", "-p",
    multiple=True,
    help="Specific predictor(s) to use (default: all available)"
)
@click.option(
    "--structure", "-s",
    type=click.Path(exists=True),
    help="Structure file (PDB/mmCIF) for structure-based prediction"
)
@click.option(
    "--threshold", "-t",
    type=float,
    default=None,
    help="Score threshold for APR detection (predictor-specific default if not set)"
)
@click.option(
    "--min-region-length",
    type=int,
    default=5,
    help="Minimum length for predicted regions (default: 5)"
)
@click.option(
    "--consensus-method",
    type=click.Choice(["majority_vote", "weighted", "metamyl"]),
    default="majority_vote",
    help="Method for consensus prediction"
)
@click.option(
    "--format", "-f",
    type=click.Choice(["json", "csv", "tsv", "fasta"]),
    default="json",
    help="Output format for results"
)
@click.option(
    "--plot/--no-plot",
    default=True,
    help="Generate visualization plots"
)
@click.pass_context
def predict(
    ctx,
    input_file: str,
    output: str,
    predictor: tuple,
    structure: Optional[str],
    threshold: Optional[float],
    min_region_length: int,
    consensus_method: str,
    format: str,
    plot: bool,
):
    """
    Run amyloidogenicity prediction on input sequences.
    
    INPUT_FILE should be a FASTA file containing one or more protein sequences.
    
    \b
    Examples:
        amyloidbench predict proteins.fasta
        amyloidbench predict query.fasta -p Aggrescan3D -s structure.pdb
        amyloidbench predict proteome.fasta -o results/ --consensus-method weighted
    """
    from ..core.sequence import parse_fasta
    from ..core.models import ProteinRecord
    from ..predictors.base import get_predictor, list_predictors, PredictorConfig
    
    input_path = Path(input_file)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse input sequences
    console.print(f"\n[bold]Loading sequences from:[/bold] {input_path}")
    
    try:
        proteins = list(parse_fasta(input_path))
        console.print(f"[green]✓[/green] Loaded {len(proteins)} sequence(s)")
    except Exception as e:
        console.print(f"[red]✗ Error loading sequences:[/red] {e}")
        sys.exit(1)
    
    # If structure provided, assign to first protein (or all if single seq)
    if structure:
        structure_path = Path(structure)
        if len(proteins) == 1:
            proteins[0].structure_path = structure_path
            console.print(f"[green]✓[/green] Using structure: {structure_path}")
        else:
            console.print(
                "[yellow]Warning:[/yellow] Structure provided but multiple sequences. "
                "Applying to first sequence only."
            )
            proteins[0].structure_path = structure_path
    
    # Configure predictors
    config = PredictorConfig(
        threshold=threshold,
        min_region_length=min_region_length,
    )
    
    # Determine which predictors to use
    available = list_predictors()
    available_names = [p["name"] for p in available]
    
    if predictor:
        selected_predictors = list(predictor)
        invalid = set(selected_predictors) - set(available_names)
        if invalid:
            console.print(f"[red]Unknown predictor(s):[/red] {invalid}")
            console.print(f"Available: {available_names}")
            sys.exit(1)
    else:
        selected_predictors = available_names
    
    console.print(f"\n[bold]Using predictors:[/bold] {', '.join(selected_predictors)}")
    
    # Run predictions
    all_results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        for protein in proteins:
            task = progress.add_task(
                f"Predicting: {protein.id}", total=len(selected_predictors)
            )
            
            protein_results = {"id": protein.id, "predictions": {}}
            
            for pred_name in selected_predictors:
                try:
                    pred = get_predictor(pred_name, config)
                    result = pred.predict(protein)
                    
                    protein_results["predictions"][pred_name] = {
                        "success": result.success,
                        "is_amyloidogenic": result.is_amyloidogenic,
                        "n_regions": len(result.predicted_regions),
                        "regions": [
                            {
                                "start": r.start,
                                "end": r.end,
                                "sequence": r.sequence,
                                "score": r.score,
                            }
                            for r in result.predicted_regions
                        ],
                        "scores": (
                            result.per_residue_scores.scores
                            if result.per_residue_scores else None
                        ),
                    }
                    
                except Exception as e:
                    protein_results["predictions"][pred_name] = {
                        "success": False,
                        "error": str(e),
                    }
                    if ctx.obj.get("verbose"):
                        console.print(f"[yellow]Warning:[/yellow] {pred_name} failed: {e}")
                
                progress.update(task, advance=1)
            
            all_results.append(protein_results)
    
    # Save results
    results_file = output_dir / f"predictions.{format}"
    
    if format == "json":
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
    elif format in ("csv", "tsv"):
        delimiter = "," if format == "csv" else "\t"
        with open(results_file, "w") as f:
            # Header
            f.write(delimiter.join([
                "protein_id", "predictor", "is_amyloidogenic", 
                "n_regions", "regions"
            ]) + "\n")
            
            for protein_result in all_results:
                for pred_name, pred_data in protein_result["predictions"].items():
                    if pred_data.get("success"):
                        regions_str = ";".join([
                            f"{r['start']}-{r['end']}"
                            for r in pred_data.get("regions", [])
                        ])
                        f.write(delimiter.join([
                            protein_result["id"],
                            pred_name,
                            str(pred_data.get("is_amyloidogenic", "")),
                            str(pred_data.get("n_regions", 0)),
                            regions_str or "none",
                        ]) + "\n")
    
    console.print(f"\n[green]✓[/green] Results saved to: {results_file}")
    
    # Generate plots if requested
    if plot:
        try:
            from ..visualization.profiles import plot_score_profiles
            
            plot_dir = output_dir / "plots"
            plot_dir.mkdir(exist_ok=True)
            
            for protein_result in all_results:
                # Collect scores from all predictors
                scores_dict = {}
                for pred_name, pred_data in protein_result["predictions"].items():
                    if pred_data.get("success") and pred_data.get("scores"):
                        scores_dict[pred_name] = pred_data["scores"]
                
                if scores_dict:
                    plot_file = plot_dir / f"{protein_result['id']}_profile.png"
                    # plot_score_profiles will be implemented in visualization module
                    console.print(f"[dim]Plot saved: {plot_file}[/dim]")
        
        except ImportError:
            console.print("[yellow]Plotting requires matplotlib. Skipping plots.[/yellow]")
    
    # Print summary
    console.print("\n" + "=" * 60)
    console.print("[bold]Summary[/bold]")
    
    summary_table = Table(show_header=True, header_style="bold")
    summary_table.add_column("Protein")
    summary_table.add_column("Predictors OK")
    summary_table.add_column("Amyloidogenic")
    summary_table.add_column("Total APRs")
    
    for protein_result in all_results:
        preds = protein_result["predictions"]
        n_ok = sum(1 for p in preds.values() if p.get("success"))
        n_amyloid = sum(1 for p in preds.values() if p.get("is_amyloidogenic"))
        total_aprs = sum(p.get("n_regions", 0) for p in preds.values() if p.get("success"))
        
        summary_table.add_row(
            protein_result["id"],
            f"{n_ok}/{len(preds)}",
            f"{n_amyloid}/{n_ok}" if n_ok > 0 else "-",
            str(total_aprs),
        )
    
    console.print(summary_table)


@cli.command("list-predictors")
@click.option("--detailed", "-d", is_flag=True, help="Show detailed information")
def list_predictors_cmd(detailed: bool):
    """
    List all available amyloidogenicity predictors.
    
    Shows name, type, and capabilities for each registered predictor.
    """
    from ..predictors.base import list_predictors
    
    predictors = list_predictors()
    
    if not predictors:
        console.print("[yellow]No predictors registered.[/yellow]")
        return
    
    table = Table(
        title="Available Predictors",
        show_header=True,
        header_style="bold cyan",
    )
    
    table.add_column("Name", style="bold")
    table.add_column("Version")
    table.add_column("Type")
    table.add_column("Capabilities")
    
    if detailed:
        table.add_column("Threshold")
        table.add_column("URL")
    
    for pred in predictors:
        caps = ", ".join(pred["capabilities"][:3])
        if len(pred["capabilities"]) > 3:
            caps += f" (+{len(pred['capabilities']) - 3})"
        
        row = [
            pred["name"],
            pred["version"],
            pred["type"],
            caps,
        ]
        
        if detailed:
            row.extend([
                str(pred["threshold"]),
                pred.get("url", "-"),
            ])
        
        table.add_row(*row)
    
    console.print(table)
    
    if detailed:
        console.print("\n[bold]Predictor Types:[/bold]")
        console.print("  • sequence_heuristic: Physics-based sequence analysis")
        console.print("  • sequence_ml: Machine learning on sequence features")
        console.print("  • structure_based: Uses 3D protein structure")
        console.print("  • threading: Template-based structural modeling")
        console.print("  • consensus: Meta-predictor combining multiple tools")
        console.print("  • fallback: Our biophysical/contextual predictor")


@cli.command("benchmark")
@click.option(
    "--database", "-d",
    type=click.Choice(["waltz-db", "crossbeta-db", "amypro", "all"]),
    default="all",
    help="Benchmark database(s) to use"
)
@click.option(
    "--predictor", "-p",
    multiple=True,
    help="Specific predictor(s) to benchmark"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="benchmark_results",
    help="Output directory for benchmark results"
)
@click.option(
    "--cv-folds",
    type=int,
    default=5,
    help="Number of cross-validation folds"
)
@click.pass_context
def benchmark(ctx, database: str, predictor: tuple, output: str, cv_folds: int):
    """
    Benchmark predictors against curated amyloid databases.
    
    Calculates sensitivity, specificity, accuracy, F1-score, and MCC
    for each predictor on the selected database(s).
    
    \b
    Examples:
        amyloidbench benchmark --database waltz-db
        amyloidbench benchmark -p Aggrescan3D -p FoldAmyloid --cv-folds 10
    """
    console.print(
        "\n[yellow]Benchmark functionality will be implemented in Phase 6.[/yellow]"
    )
    console.print("This will include:")
    console.print("  • Database loaders for WALTZ-DB, Cross-Beta DB, AmyPro")
    console.print("  • Stratified cross-validation")
    console.print("  • Metrics: Sensitivity, Specificity, F1, MCC, AUC")
    console.print("  • Statistical comparison between predictors")


@cli.command("info")
@click.argument("predictor_name")
def info(predictor_name: str):
    """
    Show detailed information about a specific predictor.
    
    PREDICTOR_NAME is the name of the predictor (case-insensitive).
    """
    from ..predictors.base import get_predictor, PredictorConfig
    
    try:
        pred = get_predictor(predictor_name, PredictorConfig(use_cache=False))
        info = pred.get_info()
        
        panel_content = f"""
[bold]Name:[/bold] {info['name']}
[bold]Version:[/bold] {info['version']}
[bold]Type:[/bold] {info['type']}

[bold]Capabilities:[/bold]
{chr(10).join('  • ' + c for c in info['capabilities'])}

[bold]Parameters:[/bold]
  • Default threshold: {info['threshold']}
  • Window size: {info['window_size']}
  • Score range: {info['score_range']}

[bold]Description:[/bold]
{info.get('description', 'No description available.')}

[bold]Citation:[/bold]
{info.get('citation', 'No citation available.')}

[bold]URL:[/bold] {info.get('url', '-')}
"""
        
        console.print(Panel(
            panel_content.strip(),
            title=f"[bold]{predictor_name}[/bold]",
            border_style="blue",
        ))
        
    except KeyError:
        console.print(f"[red]Predictor '{predictor_name}' not found.[/red]")
        console.print("Run 'amyloidbench list-predictors' to see available options.")
        sys.exit(1)


@cli.command("validate-sequence")
@click.argument("sequence", required=False)
@click.option("--file", "-f", type=click.Path(exists=True), help="FASTA file to validate")
def validate_sequence(sequence: Optional[str], file: Optional[str]):
    """
    Validate protein sequence(s) for amyloidogenicity prediction.
    
    Checks for valid amino acid characters, appropriate length,
    and potential issues that might affect prediction.
    
    \b
    Examples:
        amyloidbench validate-sequence MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH
        amyloidbench validate-sequence -f proteins.fasta
    """
    from ..core.sequence import SequenceValidator, parse_fasta
    
    validator = SequenceValidator(allow_ambiguous=True)
    
    sequences_to_check = []
    
    if sequence:
        sequences_to_check.append(("command_line", sequence))
    
    if file:
        try:
            for record in parse_fasta(file, validate=False):
                sequences_to_check.append((record.id, record.sequence))
        except Exception as e:
            console.print(f"[red]Error reading file:[/red] {e}")
            sys.exit(1)
    
    if not sequences_to_check:
        console.print("[yellow]No sequence provided. Use --help for usage.[/yellow]")
        sys.exit(1)
    
    all_valid = True
    
    for seq_id, seq in sequences_to_check:
        is_valid, errors = validator.validate(seq)
        
        if is_valid:
            console.print(f"[green]✓[/green] {seq_id}: Valid ({len(seq)} residues)")
        else:
            all_valid = False
            console.print(f"[red]✗[/red] {seq_id}: Invalid")
            for error in errors:
                console.print(f"    - {error}")
    
    sys.exit(0 if all_valid else 1)


def main():
    """Main entry point for CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
