"""
APPNN R Script Wrapper.

APPNN (Amyloidogenic Pattern Prediction Neural Network) is an R package that uses
a neural network to predict amyloidogenic hexapeptides based on 14 physicochemical
features. This wrapper provides a Python interface to the R package.

Reference:
Família C, et al. (2015) PLoS ONE 10(8):e0134355
DOI: 10.1371/journal.pone.0134355

Neural Network Architecture:
- Input: 14 physicochemical features per hexapeptide
- Features include: hydrophobicity, β-propensity, α-propensity, turn propensity,
  accessibility, mutability, and others
- Hidden layer with sigmoid activation
- Output: Probability of amyloid formation

R Package: appnn (CRAN)

This wrapper:
1. Generates an R script that processes the input sequence
2. Runs the R script and captures output
3. Parses the results into standardized format
4. Generates visualizations using R's plotting capabilities
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from ..output_models import (
        PredictorOutput,
        ResidueScore,
        PredictedRegion,
        ClassificationLabel,
        ScoreType,
        normalize_scores,
    )
except ImportError:
    from amyloidbench.predictors.output_models import (
        PredictorOutput,
        ResidueScore,
        PredictedRegion,
        ClassificationLabel,
        ScoreType,
        normalize_scores,
    )


logger = logging.getLogger(__name__)


# R script template for APPNN analysis
APPNN_SCRIPT_TEMPLATE = '''
# APPNN Amyloid Prediction Script
# Generates per-residue scores, classifications, and visualizations

# Install packages if needed
if (!require("appnn", quietly = TRUE)) {{
    install.packages("appnn", repos = "https://cloud.r-project.org/")
}}

library(appnn)

# Input parameters
sequence <- "{sequence}"
sequence_id <- "{sequence_id}"
output_dir <- "{output_dir}"
threshold <- {threshold}

# Create output directory
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Define function to scan sequence with hexapeptide windows
scan_sequence <- function(seq, threshold = 0.5) {{
    n <- nchar(seq)
    if (n < 6) {{
        stop("Sequence must be at least 6 amino acids long")
    }}
    
    # Initialize results
    positions <- 1:n
    residues <- strsplit(seq, "")[[1]]
    scores <- rep(NA, n)
    raw_predictions <- list()
    
    # Scan with hexapeptide windows
    for (i in 1:(n - 5)) {{
        hexapeptide <- substr(seq, i, i + 5)
        
        # Get APPNN prediction
        tryCatch({{
            pred <- appnn(hexapeptide)
            hex_score <- pred$probability
            
            # Store raw prediction
            raw_predictions[[i]] <- list(
                start = i,
                end = i + 5,
                hexapeptide = hexapeptide,
                probability = hex_score,
                classification = pred$classification
            )
            
            # Assign maximum score to each position in the hexapeptide
            for (j in i:(i + 5)) {{
                if (is.na(scores[j]) || hex_score > scores[j]) {{
                    scores[j] <- hex_score
                }}
            }}
        }}, error = function(e) {{
            warning(paste("Error predicting hexapeptide", i, ":", e$message))
        }})
    }}
    
    # Fill any NA values (edge cases) with 0
    scores[is.na(scores)] <- 0
    
    # Classify residues
    classifications <- ifelse(scores >= threshold, "amyloidogenic", "non_amyloidogenic")
    
    # Identify regions
    regions <- identify_regions(positions, scores, threshold, seq)
    
    return(list(
        positions = positions,
        residues = residues,
        scores = scores,
        classifications = classifications,
        regions = regions,
        raw_predictions = raw_predictions
    ))
}}

# Function to identify continuous amyloidogenic regions
identify_regions <- function(positions, scores, threshold, sequence, min_length = 6) {{
    n <- length(positions)
    regions <- list()
    in_region <- FALSE
    region_start <- 0
    
    for (i in 1:n) {{
        if (scores[i] >= threshold) {{
            if (!in_region) {{
                region_start <- i
                in_region <- TRUE
            }}
        }} else {{
            if (in_region) {{
                region_end <- i - 1
                if (region_end - region_start + 1 >= min_length) {{
                    region_seq <- substr(sequence, region_start, region_end)
                    region_scores <- scores[region_start:region_end]
                    regions[[length(regions) + 1]] <- list(
                        start = region_start,
                        end = region_end,
                        sequence = region_seq,
                        mean_score = mean(region_scores),
                        max_score = max(region_scores)
                    )
                }}
                in_region <- FALSE
            }}
        }}
    }}
    
    # Handle region at end
    if (in_region) {{
        region_end <- n
        if (region_end - region_start + 1 >= min_length) {{
            region_seq <- substr(sequence, region_start, region_end)
            region_scores <- scores[region_start:region_end]
            regions[[length(regions) + 1]] <- list(
                start = region_start,
                end = region_end,
                sequence = region_seq,
                mean_score = mean(region_scores),
                max_score = max(region_scores)
            )
        }}
    }}
    
    return(regions)
}}

# Run analysis
results <- scan_sequence(sequence, threshold)

# === OUTPUT FILES ===

# 1. Per-residue scores table (CSV)
scores_df <- data.frame(
    position = results$positions,
    residue = results$residues,
    raw_score = round(results$scores, 4),
    normalized_score = round(results$scores, 4),  # Already 0-1
    classification = results$classifications
)
write.csv(scores_df, file.path(output_dir, "per_residue_scores.csv"), row.names = FALSE)

# 2. Regions table (CSV)
if (length(results$regions) > 0) {{
    regions_df <- data.frame(
        start = sapply(results$regions, function(r) r$start),
        end = sapply(results$regions, function(r) r$end),
        sequence = sapply(results$regions, function(r) r$sequence),
        mean_score = round(sapply(results$regions, function(r) r$mean_score), 4),
        max_score = round(sapply(results$regions, function(r) r$max_score), 4)
    )
    write.csv(regions_df, file.path(output_dir, "regions.csv"), row.names = FALSE)
}}

# 3. Summary statistics (JSON-like format)
summary_file <- file.path(output_dir, "summary.txt")
cat("predictor_name: APPNN\\n", file = summary_file)
cat("predictor_version: 1.0\\n", file = summary_file, append = TRUE)
cat(paste0("sequence_id: ", sequence_id, "\\n"), file = summary_file, append = TRUE)
cat(paste0("sequence_length: ", nchar(sequence), "\\n"), file = summary_file, append = TRUE)
cat(paste0("threshold: ", threshold, "\\n"), file = summary_file, append = TRUE)
cat(paste0("n_regions: ", length(results$regions), "\\n"), file = summary_file, append = TRUE)
cat(paste0("n_amyloidogenic: ", sum(results$classifications == "amyloidogenic"), "\\n"), file = summary_file, append = TRUE)
cat(paste0("max_score: ", round(max(results$scores), 4), "\\n"), file = summary_file, append = TRUE)
cat(paste0("mean_score: ", round(mean(results$scores), 4), "\\n"), file = summary_file, append = TRUE)
cat(paste0("is_amyloidogenic: ", length(results$regions) > 0, "\\n"), file = summary_file, append = TRUE)

# === VISUALIZATIONS ===

# 4. Score profile plot
png(file.path(output_dir, "score_profile.png"), width = 1200, height = 400, res = 100)
par(mar = c(5, 4, 4, 2) + 0.1)

# Create color gradient for bars
colors <- ifelse(results$scores >= threshold, "#E74C3C", "#3498DB")

barplot(results$scores,
        names.arg = results$residues,
        col = colors,
        border = NA,
        ylim = c(0, 1),
        main = paste("APPNN Amyloidogenicity Profile:", sequence_id),
        xlab = "Residue",
        ylab = "Amyloid Probability",
        las = 2,
        cex.names = 0.7)

# Add threshold line
abline(h = threshold, col = "#E67E22", lwd = 2, lty = 2)

# Add legend
legend("topright", 
       legend = c("Amyloidogenic", "Non-amyloidogenic", "Threshold"),
       fill = c("#E74C3C", "#3498DB", NA),
       border = c(NA, NA, NA),
       lty = c(NA, NA, 2),
       lwd = c(NA, NA, 2),
       col = c(NA, NA, "#E67E22"),
       bty = "n")

dev.off()

# 5. Heatmap-style sequence view
png(file.path(output_dir, "sequence_heatmap.png"), width = 1200, height = 200, res = 100)
par(mar = c(2, 4, 3, 2) + 0.1)

# Create color palette
n_colors <- 100
color_palette <- colorRampPalette(c("#2ECC71", "#F1C40F", "#E74C3C"))(n_colors)

# Map scores to colors
score_colors <- color_palette[ceiling(results$scores * (n_colors - 1)) + 1]

# Plot heatmap
image(matrix(results$scores, nrow = 1), 
      col = color_palette,
      axes = FALSE,
      main = paste("APPNN Score Heatmap:", sequence_id))

# Add residue labels (every 10th for long sequences)
n <- length(results$residues)
if (n <= 50) {{
    label_pos <- 1:n
}} else {{
    label_pos <- seq(1, n, by = 10)
}}

axis(1, at = (label_pos - 1) / (n - 1), 
     labels = paste0(results$residues[label_pos], label_pos), 
     las = 2, cex.axis = 0.7)

# Add color bar
legend("right", legend = c("1.0", "0.5", "0.0"),
       fill = c("#E74C3C", "#F1C40F", "#2ECC71"),
       title = "Score", bty = "n", cex = 0.8)

dev.off()

# 6. Region summary plot (if regions exist)
if (length(results$regions) > 0) {{
    png(file.path(output_dir, "regions_summary.png"), width = 800, height = 400, res = 100)
    par(mar = c(5, 10, 4, 2) + 0.1)
    
    region_names <- sapply(1:length(results$regions), function(i) {{
        r <- results$regions[[i]]
        paste0("Region ", i, ": ", r$start, "-", r$end)
    }})
    
    region_scores <- sapply(results$regions, function(r) r$mean_score)
    
    barplot(region_scores,
            names.arg = region_names,
            horiz = TRUE,
            col = "#E74C3C",
            xlim = c(0, 1),
            main = "Predicted Amyloidogenic Regions",
            xlab = "Mean APPNN Score",
            las = 1)
    
    abline(v = threshold, col = "#E67E22", lwd = 2, lty = 2)
    
    dev.off()
}}

cat("\\nAPPNN analysis complete. Results saved to:", output_dir, "\\n")
'''


class AppnnPredictor:
    """
    Python wrapper for APPNN R package.
    
    APPNN uses a neural network trained on 14 physicochemical features
    to predict hexapeptide amyloidogenicity. This wrapper:
    
    1. Generates an R script with the input sequence
    2. Executes the R script using Rscript
    3. Parses the CSV output into standardized format
    4. Returns visualizations as file paths
    
    Requirements:
    - R installed and accessible via 'Rscript' command
    - appnn package installed in R
    """
    
    predictor_name = "APPNN"
    predictor_version = "1.0"
    score_type = ScoreType.PROBABILITY
    default_threshold = 0.5
    
    def __init__(
        self,
        threshold: float = 0.5,
        r_executable: str = "Rscript",
        output_dir: Optional[str] = None,
    ):
        """
        Initialize APPNN predictor.
        
        Args:
            threshold: Classification threshold (default 0.5)
            r_executable: Path to Rscript executable
            output_dir: Directory for output files (temp if not specified)
        """
        self.threshold = threshold
        self.r_executable = r_executable
        self.output_dir = output_dir
        
        # Verify R is available
        self._check_r_installation()
    
    def _check_r_installation(self) -> bool:
        """Check if R/Rscript is available."""
        try:
            result = subprocess.run(
                [self.r_executable, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                logger.info(f"R available: {result.stdout.split()[0:3]}")
                return True
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.warning(f"R not available: {e}")
        return False
    
    def predict(
        self,
        sequence: str,
        sequence_id: str = "query",
    ) -> PredictorOutput:
        """
        Run APPNN prediction on a sequence.
        
        Args:
            sequence: Protein sequence
            sequence_id: Identifier for the sequence
            
        Returns:
            PredictorOutput with per-residue scores and regions
        """
        # Validate sequence
        sequence = self._validate_sequence(sequence)
        
        # Set up output directory
        if self.output_dir:
            output_dir = Path(self.output_dir) / sequence_id
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(tempfile.mkdtemp(prefix="appnn_"))
        
        # Generate and run R script
        script_content = APPNN_SCRIPT_TEMPLATE.format(
            sequence=sequence,
            sequence_id=sequence_id,
            output_dir=str(output_dir),
            threshold=self.threshold,
        )
        
        script_path = output_dir / "appnn_analysis.R"
        script_path.write_text(script_content)
        
        try:
            result = subprocess.run(
                [self.r_executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=str(output_dir),
            )
            
            if result.returncode != 0:
                logger.error(f"R script failed: {result.stderr}")
                raise RuntimeError(f"APPNN R script failed: {result.stderr}")
            
            logger.info(f"APPNN analysis complete: {result.stdout}")
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("APPNN analysis timed out")
        
        # Parse results
        return self._parse_results(sequence, sequence_id, output_dir)
    
    def _validate_sequence(self, sequence: str) -> str:
        """Validate and clean sequence."""
        sequence = "".join(sequence.split()).upper()
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        
        invalid = set(sequence) - valid_aa
        if invalid:
            logger.warning(f"Removing invalid characters: {invalid}")
            sequence = "".join(aa for aa in sequence if aa in valid_aa)
        
        if len(sequence) < 6:
            raise ValueError("Sequence must be at least 6 amino acids")
        
        return sequence
    
    def _parse_results(
        self,
        sequence: str,
        sequence_id: str,
        output_dir: Path,
    ) -> PredictorOutput:
        """Parse APPNN output files into PredictorOutput."""
        
        # Parse per-residue scores
        scores_file = output_dir / "per_residue_scores.csv"
        residue_scores = []
        
        if scores_file.exists():
            import csv
            with open(scores_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    classification = (
                        ClassificationLabel.AMYLOIDOGENIC 
                        if row['classification'] == 'amyloidogenic'
                        else ClassificationLabel.NON_AMYLOIDOGENIC
                    )
                    
                    residue_scores.append(ResidueScore(
                        position=int(row['position']),
                        residue=row['residue'],
                        raw_score=float(row['raw_score']),
                        normalized_score=float(row['normalized_score']),
                        classification=classification,
                    ))
        
        # Parse regions
        regions_file = output_dir / "regions.csv"
        predicted_regions = []
        
        if regions_file.exists():
            import csv
            with open(regions_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    predicted_regions.append(PredictedRegion(
                        start=int(row['start']),
                        end=int(row['end']),
                        sequence=row['sequence'],
                        mean_score=float(row['mean_score']),
                        max_score=float(row['max_score']),
                        mean_normalized=float(row['mean_score']),
                        region_type="neural_network",
                    ))
        
        # Parse summary
        summary_file = output_dir / "summary.txt"
        raw_output = {'output_dir': str(output_dir)}
        
        if summary_file.exists():
            for line in summary_file.read_text().strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    raw_output[key.strip()] = value.strip()
        
        # Add visualization paths
        raw_output['visualizations'] = {
            'score_profile': str(output_dir / "score_profile.png"),
            'sequence_heatmap': str(output_dir / "sequence_heatmap.png"),
            'regions_summary': str(output_dir / "regions_summary.png"),
        }
        
        # Determine overall classification
        is_amyloidogenic = len(predicted_regions) > 0
        raw_values = [r.raw_score for r in residue_scores]
        
        return PredictorOutput(
            predictor_name=self.predictor_name,
            predictor_version=self.predictor_version,
            sequence_id=sequence_id,
            sequence=sequence,
            residue_scores=residue_scores,
            predicted_regions=predicted_regions,
            overall_classification=(
                ClassificationLabel.AMYLOIDOGENIC if is_amyloidogenic
                else ClassificationLabel.NON_AMYLOIDOGENIC
            ),
            overall_score=float(np.max(raw_values)) if raw_values else 0.0,
            overall_probability=float(np.max(raw_values)) if raw_values else 0.0,
            score_type=self.score_type,
            threshold=self.threshold,
            source="standalone",
            raw_output=raw_output,
        )
    
    def get_visualization_paths(
        self,
        sequence: str,
        sequence_id: str = "query",
    ) -> dict[str, str]:
        """
        Run prediction and return paths to visualization files.
        
        Returns:
            Dictionary with keys 'score_profile', 'sequence_heatmap', 'regions_summary'
        """
        result = self.predict(sequence, sequence_id)
        return result.raw_output.get('visualizations', {})


def predict_with_appnn(
    sequence: str,
    sequence_id: str = "query",
    threshold: float = 0.5,
    output_dir: Optional[str] = None,
) -> PredictorOutput:
    """
    Predict amyloidogenicity using APPNN R package.
    
    Args:
        sequence: Protein sequence
        sequence_id: Identifier for the sequence
        threshold: Classification threshold (default 0.5)
        output_dir: Directory for output files
        
    Returns:
        PredictorOutput with per-residue scores, regions, and visualization paths
    """
    predictor = AppnnPredictor(threshold=threshold, output_dir=output_dir)
    return predictor.predict(sequence, sequence_id)
